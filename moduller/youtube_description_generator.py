import json
import os
import re
import time
from pathlib import Path
from typing import Optional, Tuple

from moduller._module_alias import load_numbered_module
from moduller.llm_manager import (
    CentralLLM,
    get_module_recommended_llm_config,
    print_module_llm_choice_summary,
    prompt_module_llm_plan,
    select_llm,
)
from moduller.logger import get_logger
from moduller.metadata_translation_utils import (
    create_metadata_translation_llm,
    metadata_translation_model_name,
    translate_text,
)
from moduller.output_paths import grouped_output_path, stem_json_output_path
from moduller.social_media_utils import prepare_transcript, select_primary_srt
from moduller.srt_utils import parse_srt_blocks, read_srt_file
from moduller.trend_cache_utils import build_trend_summary, extract_trend_keywords, load_latest_trend_data
from moduller.youtube_llm_profiles import call_with_youtube_profile

_VIDEO_CRITIC_MODULE = load_numbered_module("202_video_critic.py")
extract_json_response = _VIDEO_CRITIC_MODULE.extract_json_response
normalize_whitespace = _VIDEO_CRITIC_MODULE.normalize_whitespace

logger = get_logger("description")


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return value if value >= minimum else default


ALL_LANGUAGES = ["Türkçe", "İngilizce", "Almanca"]
LABELS = {"Türkçe": "TR", "İngilizce": "EN", "Almanca": "DE"}
CHAPTER_HEADERS = {"Türkçe": "KISIMLAR", "İngilizce": "CHAPTERS", "Almanca": "KAPITEL"}

MAX_TRANSCRIPT_CHARS = _env_int("DESCRIPTION_MAX_TRANSCRIPT_CHARS", 35000, 8000)
MIN_DESCRIPTION_TEXT_CHARS = _env_int("DESCRIPTION_MIN_TEXT_CHARS", 1500, 1500)
MAX_DESCRIPTION_TEXT_CHARS = _env_int("DESCRIPTION_MAX_TEXT_CHARS", 2000, MIN_DESCRIPTION_TEXT_CHARS)
MAX_HASHTAG_COUNT = 10
TITLE_SUGGESTION_COUNT = 5
MIN_CHAPTER_COUNT = _env_int("DESCRIPTION_MIN_CHAPTER_COUNT", 6, 3)
MAX_CHAPTER_COUNT = max(MIN_CHAPTER_COUNT, _env_int("DESCRIPTION_MAX_CHAPTER_COUNT", 12, MIN_CHAPTER_COUNT))
CHAPTER_TARGET_SECONDS = _env_int("DESCRIPTION_CHAPTER_TARGET_SECONDS", 150, 60)
LLM_RETRIES = _env_int("DESCRIPTION_LLM_RETRIES", 3, 1)


def _description_txt_path(language: str) -> Path:
    return grouped_output_path("youtube", f"YT-Metadata_{LABELS.get(language, 'TR')}.txt")


def _compress_transcript(raw_transcript: str) -> str:
    """
    LLM token israfini onlemek icin saniye sonlarindaki milisaniyeleri (,000 vb.) temizler
    ve genel bir sikistirma uygular.
    """
    # Ornegin 00:00:00,000 formatindaki ,000 kismini atar.
    clean = re.sub(r"[,.]\d{3}", "", raw_transcript)
    return clean


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    result: list[str] = []
    for item in items:
        clean = normalize_whitespace(item)
        if not clean:
            continue
        key = clean.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(clean)
    return result


def _sentences(text: str) -> list[str]:
    clean = normalize_whitespace(text)
    return [item.strip() for item in re.split(r"(?<=[.!?])\s+", clean) if item.strip()]


def _terms(text: str, max_items: int = 20) -> list[str]:
    return _dedupe(
        re.findall(r"[A-Za-z0-9À-ÖØ-öø-ÿĞğİıŞşÇçÖöÜüÄäßẞ]{4,}", text or "")
    )[:max_items]


def _timestamp_to_seconds(value: str) -> Optional[float]:
    text = normalize_whitespace(value).replace(",", ".")
    parts = text.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except Exception:
        return None
    return None


def _seconds_to_timestamp(value: float) -> str:
    total = max(0, int(round(value)))
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def _format_timestamp(value: object) -> str:
    seconds = _timestamp_to_seconds(str(value or ""))
    return _seconds_to_timestamp(seconds) if seconds is not None else ""


def _block_start_end_seconds(block: object) -> tuple[Optional[float], Optional[float]]:
    timing_line = normalize_whitespace(getattr(block, "timing_line", ""))
    if "-->" not in timing_line:
        return None, None
    start_text, end_text = [part.strip() for part in timing_line.split("-->", 1)]
    return _timestamp_to_seconds(start_text), _timestamp_to_seconds(end_text)


def _normalize_hashtag(token: object) -> str:
    pieces = re.findall(
        r"[A-Za-z0-9À-ÖØ-öø-ÿĞğİıŞşÇçÖöÜüÄäßẞ]+",
        normalize_whitespace(token),
    )
    return "#" + "".join(piece[:1].upper() + piece[1:] for piece in pieces[:4]) if pieces else ""


def _request_json(
    llm: CentralLLM,
    prompt: str,
    log_prefix: str,
    retries: int = LLM_RETRIES,
    profile: str = "description_json",
) -> Optional[dict]:
    logger.info(f"{log_prefix} LLM cagrisi baslatiliyor.")
    for attempt in range(1, retries + 1):
        try:
            raw = call_with_youtube_profile(llm, prompt, profile=profile)
            parsed = extract_json_response(raw, logger_override=logger)
            if isinstance(parsed, dict):
                logger.info(
                    f"{log_prefix} JSON alindi ({attempt}/{retries}) | "
                    f"keys={list(parsed.keys())[:8]}"
                )
                return parsed
            logger.warning(f"{log_prefix} JSON parse edilemedi ({attempt}/{retries}).")
        except Exception as exc:
            logger.warning(f"{log_prefix} LLM hatasi ({attempt}/{retries}): {exc}")
        time.sleep(attempt)
    return None


def _build_prompt(
    role_description: str,
    task_description: str,
    json_schema: str,
    transcript: str,
    trend_summary: str,
    extra_context: str = "",
) -> str:
    # "Lost in the Middle" (Baglam Kaybi) etkisini kirmak icin ters yapi
    return (
        f"Sen {role_description}.\n\n"
        f"--- TREND OZETI ---\n{trend_summary}\n\n"
        f"--- TRANSKRIPT ---\n{transcript}\n\n"
        "------------------\n"
        "ANA HEDEF:\n"
        "Yukaridaki transkripti ve trend özetini kullanarak, YouTube algoritmasinda "
        "en yuksek retention, CTR ve SEO uyumu getirecek ciktiyi uret.\n\n"
        f"GOREV:\n{task_description}\n\n"
        f"JSON SEMASI:\n{json_schema}\n\n"
        f"{extra_context}"
        "KESİN KURALLAR:\n"
        "- Sadece ve sadece yukaridaki JSON semasina tam uyan gecerli bir JSON dondur.\n"
        "- Asla markdown (```json) kullanma, direkt { ile basla.\n"
        "- Ekstra aciklama, yorum veya sohbet metni ekleme.\n"
        "- Kısımlar (chapters) icin '00:00' uydurma, transkriptin icindeki GERCEK zaman damgalarini (ornek: 01:24, 05:10) bularak yaz."
    ).strip()


def _build_summary_prompt(
    role_description: str,
    task_description: str,
    json_schema: str,
    summary_bundle: str,
    trend_summary: str,
) -> str:
    # Ayni sekilde kisa/net kurallar en sonda
    return (
        f"Sen {role_description}.\n\n"
        f"--- TREND OZETI ---\n{trend_summary}\n\n"
        f"--- VIDEO OZET PAKETI ---\n{summary_bundle}\n\n"
        "------------------\n"
        "ANA HEDEF:\n"
        "Bu videonun YouTube tarafinda en yuksek CTR, retention beklentisi ve SEO uyumu ile "
        "algoritma tarafindan daha iyi anlasilip one cikmasina yardimci olacak ciktiyi uret.\n\n"
        f"GOREV:\n{task_description}\n\n"
        f"JSON SEMASI:\n{json_schema}\n\n"
        "KESİN KURALLAR:\n"
        "- Sadece tek bir gecerli JSON nesnesi ver.\n"
        "- Asla markdown (```json) kullanma, direkt { ile basla.\n"
        "- Aciklama ekleme, yorum ekleme.\n"
        "- Elindeki ozet paketten cikma; yeni konu uydurma."
    ).strip()


def _fallback_hook_lines(transcript: str) -> list[str]:
    fallback = _dedupe(_sentences(transcript)[:3])
    if len(fallback) >= 2:
        return fallback[:2]
    words = _terms(transcript, 8)
    if words:
        core = " ".join(words[:3])
        return [f"{core} konusunda en kritik noktalar.", f"{core} icin dogru yaklasim nedir?"]
    return ["Videonun en kritik noktalarini anlatiyoruz.", "En dogru yaklasimi sade sekilde acikliyoruz."]


def _fallback_description_body(transcript: str) -> str:
    sentences = _sentences(transcript)
    if sentences:
        return " ".join(sentences[: min(10, len(sentences))]).strip()
    return normalize_whitespace(transcript[:1200])


def _transcript_segments_for_description(transcript: str) -> list[str]:
    cleaned_lines: list[str] = []
    seen = set()

    for raw_line in str(transcript or "").splitlines():
        line = re.sub(r"^\[[^\]]+\]\s*", "", raw_line)
        line = normalize_whitespace(line)
        if len(line) < 24:
            continue
        key = line.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned_lines.append(line)

    if not cleaned_lines:
        fallback = normalize_whitespace(re.sub(r"\[[^\]]+\]", " ", str(transcript or "")))
        return [item.strip() for item in re.split(r"(?<=[.!?])\s+", fallback) if len(item.strip()) >= 24]

    segments: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in cleaned_lines:
        projected_len = current_len + (1 if current else 0) + len(line)
        if current and projected_len > 260:
            segments.append(" ".join(current).strip())
            current = [line]
            current_len = len(line)
            continue
        current.append(line)
        current_len = projected_len

    if current:
        segments.append(" ".join(current).strip())

    return segments


def _fallback_hashtags(transcript: str, trend_data: Optional[dict]) -> list[str]:
    hashtags = [
        _normalize_hashtag(item)
        for item in extract_trend_keywords(trend_data or {}, max_items=MAX_HASHTAG_COUNT)
    ]
    hashtags.extend(_normalize_hashtag(item) for item in _terms(transcript, MAX_HASHTAG_COUNT * 2))
    return _dedupe([item for item in hashtags if item])[:MAX_HASHTAG_COUNT]


def _fallback_titles(description_body: str, transcript: str) -> list[str]:
    core_terms = _terms(description_body or transcript, 4)
    core = " ".join(core_terms[:3]).strip() or "Bu Konu"
    candidates = [
        f"{core}: En Kritik Noktalar",
        f"{core}: Dogru Yaklasim",
        f"{core}: En Buyuk Hatalar",
        f"{core}: Bilmen Gerekenler",
        f"{core}: Gercekten Ne Onemli?",
    ]
    return _dedupe(candidates)[:TITLE_SUGGESTION_COUNT]


def _fallback_keywords(description_body: str, transcript: str) -> list[str]:
    return _dedupe(_terms(description_body + "\n" + transcript, 8))[:6]


def _load_valid_srt_blocks(srt_path: Path) -> list:
    try:
        blocks = parse_srt_blocks(read_srt_file(srt_path))
    except Exception:
        return []
    return [block for block in blocks if normalize_whitespace(getattr(block, "text_content", ""))]


def _chapter_title_from_block(block: object, fallback_title: str = "Bolum") -> str:
    text = normalize_whitespace(getattr(block, "text_content", ""))
    title = " ".join(_terms(text, 3)).strip()
    if title:
        return title.title()
    words = text.split()[:4]
    return " ".join(words).strip().title() or fallback_title


def _chapter_time_title_pairs(chapters: list[dict]) -> list[tuple[float, str]]:
    pairs: list[tuple[float, str]] = []
    seen = set()
    for item in chapters:
        if not isinstance(item, dict):
            continue
        seconds = _timestamp_to_seconds(item.get("timestamp", ""))
        title = " ".join(normalize_whitespace(item.get("title", "")).split()[:4]).strip()
        if seconds is None or not title:
            continue
        key = (int(round(seconds)), title.casefold())
        if key in seen:
            continue
        seen.add(key)
        pairs.append((seconds, title))
    return sorted(pairs, key=lambda item: item[0])


def _prune_chapter_pairs(pairs: list[tuple[float, str]], keep_count: int) -> list[tuple[float, str]]:
    if len(pairs) <= keep_count:
        return pairs
    if keep_count <= 1:
        return [pairs[0]]
    if keep_count == 2:
        return [pairs[0], pairs[-1]]

    middle = pairs[1:-1]
    slots = keep_count - 2
    if len(middle) <= slots:
        return [pairs[0], *middle, pairs[-1]]

    selected_middle: list[tuple[float, str]] = []
    middle_count = len(middle)
    for index in range(slots):
        pick = round(index * (middle_count - 1) / max(slots - 1, 1))
        candidate = middle[pick]
        if candidate not in selected_middle:
            selected_middle.append(candidate)

    if len(selected_middle) < slots:
        for candidate in middle:
            if candidate in selected_middle:
                continue
            selected_middle.append(candidate)
            if len(selected_middle) >= slots:
                break

    return [pairs[0], *selected_middle[:slots], pairs[-1]]


def _normalize_chapters_for_full_coverage(srt_path: Path, chapters: list[dict]) -> list[dict]:
    valid_blocks = _load_valid_srt_blocks(srt_path)
    if not valid_blocks:
        return chapters[:MAX_CHAPTER_COUNT]

    first_start, _ = _block_start_end_seconds(valid_blocks[0])
    _, last_end = _block_start_end_seconds(valid_blocks[-1])
    first_start = first_start or 0
    last_end = last_end or first_start
    total_seconds = max(0, last_end - first_start)
    target_count = max(
        MIN_CHAPTER_COUNT,
        min(MAX_CHAPTER_COUNT, int(max(total_seconds, 1) / CHAPTER_TARGET_SECONDS) + 1),
    )
    target_spacing = max(45, int(max(total_seconds, 1) / max(target_count, 1)))
    tail_window = max(45, min(CHAPTER_TARGET_SECONDS, target_spacing))

    pairs = _chapter_time_title_pairs(chapters)

    start_anchor = (
        first_start,
        _chapter_title_from_block(valid_blocks[0], "Giris"),
    )
    if not pairs or pairs[0][0] > first_start + 10:
        pairs.insert(0, start_anchor)

    end_threshold = max(first_start, last_end - tail_window)
    end_block = valid_blocks[-1]
    for block in valid_blocks:
        start_seconds, _ = _block_start_end_seconds(block)
        if start_seconds is None:
            continue
        if start_seconds >= end_threshold:
            end_block = block
            break
    end_anchor_seconds, _ = _block_start_end_seconds(end_block)
    end_anchor_seconds = end_anchor_seconds or first_start
    end_anchor = (
        end_anchor_seconds,
        _chapter_title_from_block(end_block, "Kapanis"),
    )
    if not pairs or pairs[-1][0] < end_anchor_seconds - 5:
        pairs.append(end_anchor)

    pairs = sorted(pairs, key=lambda item: item[0])
    deduped_pairs: list[tuple[float, str]] = []
    seen_seconds = set()
    for seconds, title in pairs:
        rounded = int(round(seconds))
        if rounded in seen_seconds:
            continue
        seen_seconds.add(rounded)
        deduped_pairs.append((seconds, title))

    pairs = _prune_chapter_pairs(deduped_pairs, MAX_CHAPTER_COUNT)
    return [
        {"timestamp": _seconds_to_timestamp(seconds), "title": title}
        for seconds, title in pairs
        if title
    ][:MAX_CHAPTER_COUNT]


def _fallback_chapters(srt_path: Path) -> list[dict]:
    valid = _load_valid_srt_blocks(srt_path)
    if not valid:
        return []

    first_start, _ = _block_start_end_seconds(valid[0])
    _, last_end = _block_start_end_seconds(valid[-1])
    first_start = first_start or 0
    last_end = last_end or first_start
    total_seconds = max(0, last_end - first_start)
    target = max(MIN_CHAPTER_COUNT, min(MAX_CHAPTER_COUNT, int(max(total_seconds, 1) / CHAPTER_TARGET_SECONDS) + 1))
    step = max(1, len(valid) // max(target, 1))

    chapters: list[dict] = []
    for index in range(0, len(valid), step):
        block = valid[index]
        start_seconds, _ = _block_start_end_seconds(block)
        timestamp = _seconds_to_timestamp(start_seconds) if start_seconds is not None else ""
        title = " ".join(_terms(getattr(block, "text_content", ""), 3)).strip()
        if timestamp and title:
            chapters.append({"timestamp": timestamp, "title": title.title()})
        if len(chapters) >= MAX_CHAPTER_COUNT:
            break

    return _normalize_chapters_for_full_coverage(srt_path, chapters[:MAX_CHAPTER_COUNT])


def _build_seo_summary(
    hook_lines: list[str],
    description_body: str,
    chapters: list[dict],
    topic_keywords: list[str],
    transcript_head: str = "",
) -> str:
    chapter_lines = [
        f"{item.get('timestamp', '')} - {item.get('title', '')}"
        for item in chapters
        if item.get("timestamp") and item.get("title")
    ]
    return "\n".join(
        [
            "--- BAGLAM ICIN VİDEO GİRİŞİ ---",
            transcript_head,
            "",
            "HOOK",
            "\n".join(hook_lines[:2]).strip(),
            "",
            "DESCRIPTION",
            description_body.strip(),
            "",
            "TOPIC KEYWORDS",
            ", ".join(topic_keywords[:6]),
            "",
            "CHAPTERS",
            "\n".join(chapter_lines),
        ]
    ).strip()


def _build_description_text(language: str, data: dict) -> str:
    chapter_lines = [
        f"{item.get('timestamp', '')} - {item.get('title', '')}"
        for item in data.get("chapters", [])
        if item.get("timestamp") and item.get("title")
    ]
    return _assemble_description(
        language,
        data.get("hook_lines", []),
        data.get("description_body", ""),
        chapter_lines,
    )


def _assemble_description(language: str, hook_lines: list[str], body: str, chapter_lines: list[str]) -> str:
    parts = ["\n".join(hook_lines[:2]).strip(), normalize_whitespace(body)]
    if chapter_lines:
        parts.append(CHAPTER_HEADERS.get(language, "CHAPTERS") + "\n" + "\n".join(chapter_lines))
    return "\n\n".join(part for part in parts if normalize_whitespace(part)).strip()


def _trim_text_to_char_limit(text: str, limit: int) -> str:
    clean = normalize_whitespace(text)
    if len(clean) <= limit:
        return clean

    sentences = _sentences(clean)
    trimmed_parts: list[str] = []
    current_length = 0
    for sentence in sentences:
        projected = current_length + (1 if trimmed_parts else 0) + len(sentence)
        if projected > limit:
            break
        trimmed_parts.append(sentence)
        current_length = projected

    if trimmed_parts:
        return " ".join(trimmed_parts).strip()

    cutoff = max(0, limit - 1)
    return clean[:cutoff].rstrip(" ,.;:-") + "..."


def _enforce_max_description_text(
    language: str,
    hook_lines: list[str],
    description_body: str,
    chapter_lines: list[str],
    maximum_chars: int = MAX_DESCRIPTION_TEXT_CHARS,
) -> tuple[str, str]:
    clean_body = normalize_whitespace(description_body)
    description_text = _assemble_description(language, hook_lines, clean_body, chapter_lines)
    if len(description_text) <= maximum_chars:
        return clean_body, description_text

    fixed_length = len(_assemble_description(language, hook_lines, "", chapter_lines))
    available_for_body = max(0, maximum_chars - fixed_length)
    clean_body = _trim_text_to_char_limit(clean_body, available_for_body)
    description_text = _assemble_description(language, hook_lines, clean_body, chapter_lines)
    if len(description_text) <= maximum_chars:
        return clean_body, description_text

    return clean_body, description_text[:maximum_chars].rstrip()


def _ensure_min_description_text(
    language: str,
    hook_lines: list[str],
    description_body: str,
    chapter_lines: list[str],
    transcript: str,
    minimum_chars: int = MIN_DESCRIPTION_TEXT_CHARS,
) -> tuple[str, str]:
    clean_body = normalize_whitespace(description_body)
    description_text = _assemble_description(language, hook_lines, clean_body, chapter_lines)
    if len(description_text) >= minimum_chars:
        return clean_body, description_text

    body_parts = [clean_body] if clean_body else []
    body_lower = clean_body.casefold()

    for segment in _transcript_segments_for_description(transcript):
        segment_clean = normalize_whitespace(segment)
        if not segment_clean:
            continue
        segment_key = segment_clean.casefold()
        if segment_key in body_lower:
            continue
        body_parts.append(segment_clean)
        clean_body = " ".join(part for part in body_parts if part).strip()
        body_lower = clean_body.casefold()
        description_text = _assemble_description(language, hook_lines, clean_body, chapter_lines)
        if len(description_text) >= minimum_chars:
            return clean_body, description_text

    transcript_fallback = normalize_whitespace(re.sub(r"\[[^\]]+\]", " ", str(transcript or "")))
    if transcript_fallback and transcript_fallback.casefold() not in body_lower:
        needed_chars = max(0, minimum_chars - len(description_text)) + 200
        clean_body = " ".join(
            part for part in [clean_body, transcript_fallback[:needed_chars].strip()] if part
        ).strip()
        description_text = _assemble_description(language, hook_lines, clean_body, chapter_lines)

    return clean_body, description_text


def _translate_items(items: list[str], target_language: str, llm: CentralLLM, field_prefix: str) -> list[str]:
    translated: list[str] = []
    for index, item in enumerate(items, start=1):
        clean = normalize_whitespace(item)
        if not clean:
            continue
        translated.append(
            normalize_whitespace(
                translate_text(
                    clean,
                    target_language,
                    llm,
                    active_logger=logger,
                    field_label=f"{field_prefix}_{index}",
                )
            )
        )
    return translated


def _translate_hashtag_items(hashtags: list[str], target_language: str, llm: CentralLLM) -> list[str]:
    translated: list[str] = []
    seen = set()
    for index, hashtag in enumerate(hashtags, start=1):
        source_term = normalize_whitespace(str(hashtag or "").lstrip("#"))
        if not source_term:
            continue
        translated_term = translate_text(
            source_term,
            target_language,
            llm,
            active_logger=logger,
            field_label=f"hashtag_{index}",
        )
        normalized = _normalize_hashtag(translated_term)
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        translated.append(normalized)
    return translated[:MAX_HASHTAG_COUNT]


def _parse_report_sections(text: str, section_names: list[str]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {name: [] for name in section_names}
    current_section = ""

    for raw_line in str(text or "").splitlines():
        stripped = raw_line.strip()
        if stripped in sections:
            current_section = stripped
            continue
        if stripped and set(stripped) == {"-"}:
            continue
        if current_section:
            sections[current_section].append(raw_line.rstrip())

    return sections


def _load_turkish_txt_seed() -> dict:
    description_path = _description_txt_path("Türkçe")
    if not description_path.exists():
        raise FileNotFoundError(f"Turkce description raporu bulunamadi: {description_path.name}")

    description_sections = _parse_report_sections(
        description_path.read_text(encoding="utf-8"),
        ["HOOK", "DESCRIPTION", "KISIMLAR", "BASLIK ONERILERI", "HASHTAG SATIRI"],
    )

    hook_lines = _dedupe(
        [normalize_whitespace(line) for line in description_sections.get("HOOK", []) if normalize_whitespace(line)]
    )[:2]
    description_body = "\n".join(description_sections.get("DESCRIPTION", [])).strip()

    chapters: list[dict] = []
    for line in description_sections.get("KISIMLAR", []):
        match = re.match(r"^\s*([0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?)\s*-\s*(.+?)\s*$", line)
        if not match:
            continue
        chapters.append(
            {
                "timestamp": match.group(1).strip(),
                "title": normalize_whitespace(match.group(2)),
            }
        )

    title_suggestions = _dedupe(
        [
            normalize_whitespace(line)
            for line in description_sections.get("BASLIK ONERILERI", [])
            if normalize_whitespace(line)
        ]
    )[:TITLE_SUGGESTION_COUNT]
    best_title = title_suggestions[0] if title_suggestions else ""

    hashtag_text = normalize_whitespace(" ".join(description_sections.get("HASHTAG SATIRI", [])))
    hashtags = _dedupe(_normalize_hashtag(item) for item in re.findall(r"#[^\s#]+", hashtag_text))[:MAX_HASHTAG_COUNT]

    return {
        "hook_lines": hook_lines,
        "description_body": description_body,
        "chapters": chapters[:MAX_CHAPTER_COUNT],
        "title_suggestions": title_suggestions,
        "best_title_suggestion": best_title,
        "hashtags": hashtags,
        "hashtag_line": " ".join(hashtags).strip(),
    }


def _translate_from_turkish_txt(
    turkce_payload: dict,
    target_language: str,
    llm: CentralLLM,
) -> dict:
    txt_seed = _load_turkish_txt_seed()

    hook_lines = _translate_items(txt_seed.get("hook_lines", []), target_language, llm, "hook")[:2]
    description_body = normalize_whitespace(
        translate_text(
            txt_seed.get("description_body", ""),
            target_language,
            llm,
            active_logger=logger,
            field_label="description_body_txt",
        )
    )
    chapters = []
    for index, item in enumerate(txt_seed.get("chapters", []), start=1):
        if not isinstance(item, dict):
            continue
        translated_title = normalize_whitespace(
            translate_text(
                item.get("title", ""),
                target_language,
                llm,
                active_logger=logger,
                field_label=f"chapter_title_{index}",
            )
        )
        timestamp = _format_timestamp(item.get("timestamp", ""))
        if timestamp and translated_title:
            chapters.append({"timestamp": timestamp, "title": translated_title})

    title_suggestions = _dedupe(
        _translate_items(
            txt_seed.get("title_suggestions", []),
            target_language,
            llm,
            "title",
        )
    )[:TITLE_SUGGESTION_COUNT]
    hashtags = _translate_hashtag_items(txt_seed.get("hashtags", []), target_language, llm)
    hashtag_line = " ".join(hashtags).strip()

    search_terms = _dedupe(
        _translate_items(
            turkce_payload.get("search_terms", []),
            target_language,
            llm,
            "search_term",
        )
    )
    topic_keywords = _dedupe(
        _translate_items(
            turkce_payload.get("topic_keywords", []),
            target_language,
            llm,
            "topic_keyword",
        )
    )[:6]

    best_title_suggestion = title_suggestions[0] if title_suggestions else normalize_whitespace(
        translate_text(
            txt_seed.get("best_title_suggestion", ""),
            target_language,
            llm,
            active_logger=logger,
            field_label="best_title",
        )
    )

    description_text = _assemble_description(
        target_language,
        hook_lines,
        description_body,
        [f"{item['timestamp']} - {item['title']}" for item in chapters],
    )
    description_body, description_text = _enforce_max_description_text(
        target_language,
        hook_lines,
        description_body,
        [f"{item['timestamp']} - {item['title']}" for item in chapters],
    )

    return {
        "language": target_language,
        "source_srt": turkce_payload.get("source_srt", ""),
        "hook_lines": hook_lines,
        "description_body": description_body,
        "chapters": chapters[:MAX_CHAPTER_COUNT],
        "description_text": description_text,
        "description_char_count": len(description_text),
        "hashtags": hashtags,
        "hashtag_line": hashtag_line,
        "description_with_hashtags": f"{description_text}\n\n{hashtag_line}".strip() if hashtag_line else description_text,
        "search_terms": search_terms,
        "optimization_note_tr": turkce_payload.get("optimization_note_tr", ""),
        "topic_keywords": topic_keywords,
        "title_suggestions": title_suggestions,
        "best_title_suggestion": best_title_suggestion,
    }


def _generate_turkish(
    srt_path: Path,
    llm: CentralLLM,
    trend_data: Optional[dict],
    prepared_transcript: Optional[str],
    trend_summary_override: Optional[str],
) -> Optional[dict]:
    
    transcript = (prepared_transcript or prepare_transcript(srt_path, max_karakter=MAX_TRANSCRIPT_CHARS)).strip()
    transcript = _compress_transcript(transcript) # Token temizligi
    transcript = transcript[:MAX_TRANSCRIPT_CHARS]
    if not normalize_whitespace(transcript):
        logger.error("Description icin transcript bulunamadi.")
        return None

    trend_data = trend_data or load_latest_trend_data()
    trend_summary = trend_summary_override or build_trend_summary(trend_data)

    logger.info(f"[Türkçe] Metadata uretimi baslatiliyor. Kaynak SRT: {srt_path.name}")

    core_json = _request_json(
        llm,
        _build_prompt(
            "ust duzey bir YouTube description ve chapter stratejistisin",
            "Tek seferde su alanlari uret: tam 2 hook satiri, ana description govdesi, "
            "00:00 formatinda videonun ana kisimlari ve videonun ana konusunu temsil eden kisa topic keyword listesi. "
            "Metnin dili samimi, akici ve merak uyandirici olsun. Izleyiciyle dogrudan konusuyormus gibi "
            "(sen/siz diliyle) profesyonel ama sikici olmayan bir uslup kullan. "
            "Ilk hook satiri izleyicinin aci noktasina dokunan merak uyandirici bir soru olsun. "
            "Ikinci hook satiri ise videonun bu sorunu nasil cozecegini anlatan net bir vaat icersin. "
            f"Description retention ve SEO dengesini korusun; description govdesi en az {MIN_DESCRIPTION_TEXT_CHARS} karakter, "
            f"toplam description metni ise en fazla {MAX_DESCRIPTION_TEXT_CHARS} karakter olsun; "
            "kisimlar 2-3 kelimelik net bolum basliklari olsun ve videonun basindan finaline kadar tum akisi kapsasin.",
            '{"hook_lines":["", ""],"description_body":"","chapters":[{"timestamp":"02:15","title":"Gercek Bolum Basligi"}],"topic_keywords":["ana konu"]}',
            transcript,
            trend_summary,
        ),
        "Description Core",
    ) or {}

    hook_lines = _dedupe(
        [normalize_whitespace(item) for item in core_json.get("hook_lines", []) if normalize_whitespace(item)]
    )[:2]
    if len(hook_lines) < 2:
        hook_lines = _fallback_hook_lines(transcript)

    description_body = normalize_whitespace(core_json.get("description_body", ""))
    if not description_body:
        description_body = _fallback_description_body(transcript)

    chapters: list[dict] = []
    for item in core_json.get("chapters", []):
        if not isinstance(item, dict):
            continue
        timestamp = _format_timestamp(item.get("timestamp", ""))
        title = " ".join(normalize_whitespace(item.get("title", "")).split()[:4]).strip()
        if timestamp and title:
            chapters.append({"timestamp": timestamp, "title": title})
    chapters = _normalize_chapters_for_full_coverage(srt_path, chapters[:MAX_CHAPTER_COUNT])
    if len(chapters) < MIN_CHAPTER_COUNT:
        chapters = _fallback_chapters(srt_path)

    topic_keywords = _dedupe(
        [normalize_whitespace(item) for item in core_json.get("topic_keywords", []) if normalize_whitespace(item)]
    )[:6]
    if not topic_keywords:
        topic_keywords = _fallback_keywords(description_body, transcript)

    # SEO cagrisina ilk 3000 karakteri (Video kancasi) baglam olarak ekliyoruz
    seo_summary = _build_seo_summary(hook_lines, description_body, chapters, topic_keywords, transcript[:3000])
    
    seo_json = _request_json(
        llm,
        _build_summary_prompt(
            "ust duzey bir YouTube SEO ve CTR editörüsün",
            f"Bu video ozet paketine bakarak tam olarak {TITLE_SUGGESTION_COUNT} baslik onerisi ve "
            f"tam olarak {MAX_HASHTAG_COUNT} hashtag uret. Basliklar CTR odakli ama dogru olmali; "
            "hashtagler ise SEO ve konu netligi tasimali. "
            "5 baslik onerisi birbirinden belirgin sekilde farkli olsun ve ayni fikrin kelimelerini sadece yer degistirerek tekrar etme. "
            "Basliklardan biri soru formatinda, biri iddiali/kesin bir ifade, biri gizem uyandiran bir cumle, "
            "biri SEO odakli liste formati ve biri de dogrudan fayda odakli bir vaat kullansin.",
            '{"titles":["","","","",""],"hashtags":["#ornek"]}',
            seo_summary,
            trend_summary,
        ),
        "Description SEO Package",
    ) or {}

    title_suggestions = _dedupe(
        [normalize_whitespace(item) for item in seo_json.get("titles", []) if normalize_whitespace(item)]
    )[:TITLE_SUGGESTION_COUNT]
    if len(title_suggestions) < TITLE_SUGGESTION_COUNT:
        title_suggestions = _dedupe(title_suggestions + _fallback_titles(description_body, transcript))[:TITLE_SUGGESTION_COUNT]

    hashtags = _dedupe(
        [_normalize_hashtag(item) for item in seo_json.get("hashtags", []) if _normalize_hashtag(item)]
    )[:MAX_HASHTAG_COUNT]
    if len(hashtags) < MAX_HASHTAG_COUNT:
        hashtags = _dedupe(hashtags + _fallback_hashtags(transcript, trend_data))[:MAX_HASHTAG_COUNT]

    chapter_lines = [f"{item['timestamp']} - {item['title']}" for item in chapters]
    description_body, description_text = _ensure_min_description_text(
        "Türkçe",
        hook_lines,
        description_body,
        chapter_lines,
        transcript,
    )
    description_body, description_text = _enforce_max_description_text(
        "Türkçe",
        hook_lines,
        description_body,
        chapter_lines,
    )
    hashtag_line = " ".join(hashtags[:MAX_HASHTAG_COUNT]).strip()

    return {
        "language": "Türkçe",
        "source_srt": srt_path.name,
        "hook_lines": hook_lines[:2],
        "description_body": description_body,
        "chapters": chapters[:MAX_CHAPTER_COUNT],
        "description_text": description_text,
        "description_char_count": len(description_text),
        "hashtags": hashtags[:MAX_HASHTAG_COUNT],
        "hashtag_line": hashtag_line,
        "description_with_hashtags": f"{description_text}\n\n{hashtag_line}".strip() if hashtag_line else description_text,
        "search_terms": [],
        "optimization_note_tr": "",
        "topic_keywords": topic_keywords[:6],
        "title_suggestions": title_suggestions[:TITLE_SUGGESTION_COUNT],
        "best_title_suggestion": title_suggestions[0] if title_suggestions else "",
    }


def _write_language_report_files(language: str, data: dict, model_name: str) -> Path:
    description_path = _description_txt_path(language)
    description_path.write_text(_build_description_report(language, data, model_name), encoding="utf-8")
    return description_path


def analyze(
    srt_path: Path,
    llm: CentralLLM,
    hedef_diller: Optional[list[str]] = None,
    trend_data: Optional[dict] = None,
    draft_llm: Optional[CentralLLM] = None,
    prepared_transcript: Optional[str] = None,
    trend_ozeti_override: Optional[str] = None,
) -> dict:
    _ = draft_llm
    hedef_diller = [dil for dil in ALL_LANGUAGES if dil in (hedef_diller or ALL_LANGUAGES)]
    turkce = _generate_turkish(
        srt_path,
        llm,
        trend_data=trend_data,
        prepared_transcript=prepared_transcript,
        trend_summary_override=trend_ozeti_override,
    )
    if not turkce:
        return {}

    results = {"Türkçe": turkce} if "Türkçe" in hedef_diller else {}
    needs_translation = any(language != "Türkçe" for language in hedef_diller)
    if not needs_translation:
        return results

    try:
        translation_llm = create_metadata_translation_llm()
    except Exception as exc:
        logger.error(f"Description ceviri modeli hazirlanamadi: {exc}")
        return results

    try:
        _write_language_report_files("Türkçe", turkce, f"TR Final: {llm.model_name}")
        logger.info("Turkce description TXT raporu ceviri oncesi kaydedildi.")
    except Exception as exc:
        logger.error(f"Turkce TXT seed raporlari kaydedilemedi: {exc}")
        return results

    for language in hedef_diller:
        if language == "Türkçe":
            continue
        try:
            translated = _translate_from_turkish_txt(turkce, language, translation_llm)
            results[language] = translated
        except Exception as exc:
            logger.error(f"[{language}] Description cevirisi basarisiz oldu: {exc}")

    return results


def _chapter_lines(chapters: list[dict]) -> list[str]:
    return [
        f"{item.get('timestamp', '')} - {item.get('title', '')}"
        for item in chapters
        if item.get("timestamp") and item.get("title")
    ]


def _build_description_report(language: str, data: dict, model_name: str) -> str:
    return "\n".join(
        [
            f"=== {language.upper()} VIDEO DESCRIPTION ===",
            f"Kullanilan Model: {model_name}",
            f"Kaynak SRT: {data.get('source_srt', '')}",
            f"Karakter Sayisi: {data.get('description_char_count', 0)}",
            "",
            "HOOK",
            "-" * 40,
            "\n".join(data.get("hook_lines", [])),
            "",
            "DESCRIPTION",
            "-" * 40,
            data.get("description_body", ""),
            "",
            "KISIMLAR",
            "-" * 40,
            "\n".join(_chapter_lines(data.get("chapters", []))),
            "",
            "BASLIK ONERILERI",
            "-" * 40,
            "\n".join(data.get("title_suggestions", [])),
            "",
            "HASHTAG SATIRI",
            "-" * 40,
            data.get("hashtag_line", ""),
        ]
    ).strip() + "\n"


def _build_title_json(results: dict) -> dict:
    payload = {}
    for language, data in results.items():
        payload[language] = {
            "best_title": data.get("best_title_suggestion", ""),
            "titles": [
                {
                    "rank": index,
                    "title": title,
                    "score": 0,
                    "reason_tr": "Description modulu icinde bagimsiz baslik onerisi olarak uretildi.",
                    "angle": "description_title_suggestion",
                }
                for index, title in enumerate(data.get("title_suggestions", []), start=1)
            ],
        }
    return payload


def save_reports(srt_path: Path, results: dict, model_name: str) -> Tuple[Path, Path, list[str]]:
    description_json = stem_json_output_path(srt_path.stem, "_video_description.json", group="youtube")
    title_json = stem_json_output_path(srt_path.stem, "_video_titles.json", group="youtube")

    description_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    title_json.write_text(json.dumps(_build_title_json(results), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Description JSON kaydedildi: {description_json.name}")
    logger.info(f"Title JSON kaydedildi: {title_json.name}")

    output_paths: list[str] = []
    for language, data in results.items():
        description_path = _write_language_report_files(language, data, model_name)
        output_paths.append(str(description_path))

    try:
        from moduller.metadata_olusturucu import update_combined_metadata

        update_combined_metadata(srt_path, model_name)
    except Exception as exc:
        logger.warning(f"Birlesik metadata senkronize edilemedi: {exc}")

    return description_json, title_json, output_paths


def _select_srt() -> Optional[Path]:
    return select_primary_srt(logger, "Description")


def run() -> None:
    print("\n" + "=" * 60)
    print("VIDEO ACIKLAMA VE BASLIK OLUSTURUCU")
    print("=" * 60)
    print("Akis: Smart LLM ile Turkce metadata uretilir, EN/DE TR TXT raporlarindan TranslateGemma ile olusturulur.")

    srt_path = _select_srt()
    if not srt_path:
        return

    use_recommended = prompt_module_llm_plan("201", needs_smart=True)
    if use_recommended:
        provider, model_name = get_module_recommended_llm_config("201", "smart")
        print_module_llm_choice_summary("201", {"smart": (provider, model_name)})
    else:
        provider, model_name = select_llm("smart")
    llm = CentralLLM(provider=provider, model_name=model_name)
    results = analyze(srt_path, llm, hedef_diller=ALL_LANGUAGES)
    if not results:
        logger.error("Video metadata uretimi basarisiz oldu.")
        return

    model_summary = f"TR Final: {llm.model_name}"
    if any(language != "Türkçe" for language in results.keys()):
        model_summary += f" | EN/DE Ceviri: {metadata_translation_model_name()}"
    save_reports(srt_path, results, model_summary)
    logger.info("Video metadata uretimi tamamlandi.")


def run_automatic(
    girdi_dosyasi: Path,
    llm: CentralLLM,
    hedef_diller: Optional[list[str]] = None,
    trend_data: Optional[dict] = None,
    draft_llm: Optional[CentralLLM] = None,
    prepared_transcript: Optional[str] = None,
    trend_ozeti_override: Optional[str] = None,
) -> Optional[dict]:
    _ = draft_llm
    logger.info(f"OTOMASYON: {girdi_dosyasi.name} icin sade metadata paketi uretiliyor...")
    results = analyze(
        girdi_dosyasi,
        llm,
        hedef_diller=hedef_diller or ALL_LANGUAGES,
        trend_data=trend_data,
        prepared_transcript=prepared_transcript,
        trend_ozeti_override=trend_ozeti_override,
    )
    if not results:
        logger.error("Otomatik description uretimi basarisiz oldu.")
        return None

    model_summary = f"TR Final: {llm.model_name}"
    if any(language != "Türkçe" for language in results.keys()):
        model_summary += f" | EN/DE Ceviri: {metadata_translation_model_name()}"
    description_json, title_json, language_txt_paths = save_reports(girdi_dosyasi, results, model_summary)
    return {
        "data": results,
        "json_path": description_json,
        "title_json_path": title_json,
        "txt_path": None,
        "language_txt_paths": language_txt_paths,
    }
