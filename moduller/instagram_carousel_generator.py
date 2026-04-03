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
from moduller.output_paths import grouped_output_path, stem_json_output_path, txt_output_path
from moduller.social_media_utils import (
    build_broll_summary,
    build_critic_summary,
    build_metadata_summary,
    build_trim_summary,
    load_related_json,
    select_primary_srt,
    select_metadata_language,
)
from moduller.srt_utils import parse_srt_blocks, read_srt_file
from moduller.youtube_llm_profiles import call_with_youtube_profile

_VIDEO_CRITIC_MODULE = load_numbered_module("202_video_critic.py")
extract_json_response = _VIDEO_CRITIC_MODULE.extract_json_response
normalize_whitespace = _VIDEO_CRITIC_MODULE.normalize_whitespace

logger = get_logger("carousel")
LLM_RETRIES = 3
MIN_SLIDES = 5
MAX_SLIDES = 10
MIN_CAROUSEL_CANDIDATES = 3
MAX_CAROUSEL_CANDIDATES = 5
COVER_TITLE_LIMIT = 35
COVER_SUBTITLE_LIMIT = 50
SLIDE_TITLE_LIMIT = 40
SLIDE_BODY_LIMIT = 90
CTA_TITLE_LIMIT = 35
CTA_BODY_LIMIT = 60
CTA_BUTTON_LIMIT = 20
DEFAULT_COVER_SUBTITLE = "Asil cevabi 2. slaytta goreceksin."
DEFAULT_CTA_TITLE = "Bunu kaydet"
DEFAULT_CTA_BODY = "Sonra tekrar bakmak icin bu carouseli kaydet."
DEFAULT_CTA_BUTTON = "Kaydet"
MAX_BODY_SENTENCES = 2
CAROUSEL_DEBUG_DIR = grouped_output_path("instagram", "_llm_debug")
CAROUSEL_ITEM_TXT_DIR = grouped_output_path("instagram", "301_IG-Carousel_Fikirleri")
LEGACY_CAROUSEL_ITEM_TXT_DIR = CAROUSEL_ITEM_TXT_DIR.parent / "_tekil_carousel_txt"
CAROUSEL_PROMPT_FOOTER = (
    "Metin buyuk, mobilde rahat okunur olsun, 2-3 satiri gecmesin ve guclu gorsel hiyerarsiyle ana mesaja "
    "odaklansin."
)
CAROUSEL_FORMAT_SPEC_TR = "Format/Cozunurluk: 4:5 (dikey) en-boy orani, 1080x1350."
ADAPTIVE_TRANSCRIPT_MAX_CHARS = 35000
ADAPTIVE_TRANSCRIPT_TARGET_ANCHORS = 8
ADAPTIVE_TRANSCRIPT_WINDOW_RADIUS = 1
CAROUSEL_LIGHT_MODE_MAX_SECONDS = 150
CAROUSEL_LIGHT_MODE_MAX_BLOCKS = 16
CAROUSEL_LIGHT_MODE_MAX_TRANSCRIPT_CHARS = 1800
DEFAULT_CAROUSEL_GOAL_EN = "Kaydetme ve kaydirma istegi uyandiran dikkat cekici bir Instagram carousel tasarimi."
DEFAULT_CAROUSEL_BACKGROUND_EN = "Temiz, modern ve acik tonlu bir arka plan; profesyonel sosyal medya estetikli."
DEFAULT_CAROUSEL_SUBJECT_EN = "Konuyla dogrudan baglantili gercekci bir kisi, nesne veya sahne."
DEFAULT_CAROUSEL_SUPPORTING_ELEMENTS_EN = (
    "Ince oklar, euro ikonlari, bilgi kutulari ve ana mesaji vurgulayan dikkat ceken detaylar."
)
DEFAULT_CAROUSEL_STYLE_EN = "Modern, temiz, yuksek kontrastli ve mobilde kolay okunur bir carousel dili."
GENERIC_COVER_SUPPORTING_EN = "Merak uyandiran kisa bir teaser cizgisi, ince oklar ve tek bir vurgulu deger."
GENERIC_SLIDE_SUPPORTING_EN = "Ana fikri netlestiren kucuk etiketler, oklar, ikonlar veya aciklayici callout'lar."
GENERIC_CTA_SUPPORTING_EN = "Kaydet ikonunu, tek bir belirgin CTA butonunu ve temiz bir yonlendirme isaretini kullan."
TURKISH_PROMPT_MARKERS = {
    "acik",
    "amac",
    "ana",
    "arka",
    "arkaplan",
    "baslik",
    "bir",
    "bu",
    "cok",
    "daha",
    "gibi",
    "gider",
    "gorsel",
    "icin",
    "ile",
    "kadar",
    "maliyet",
    "neden",
    "net",
    "olsun",
    "ozne",
    "ruh",
    "sonuc",
    "stil",
    "tasarim",
    "unsur",
    "vergi",
    "yatay",
    "dikey",
    "cozunurluk",
    "format",
    "kapak",
    "carousel",
    "kaydet",
    "maas",
    "almanya",
}


def _env_int(name: str, default: int) -> int:
    raw_value = normalize_whitespace(os.getenv(name, ""))
    if not raw_value:
        return default
    try:
        return max(1, int(raw_value))
    except ValueError:
        return default


def _carousel_profile_timeout(llm: CentralLLM, profile: str) -> Optional[int]:
    if llm.provider != "OLLAMA":
        return None
    if profile in {"analytic_json", "strict_json"}:
        return _env_int("CAROUSEL_OLLAMA_REPAIR_TIMEOUT_SECONDS", 240)
    return _env_int("CAROUSEL_OLLAMA_TIMEOUT_SECONDS", 300)


def _parse_timecode(value: str) -> int:
    text = str(value or "").strip().replace(",", ".")
    if not text:
        return 0

    if re.match(r"^\d{2}:\d{2}$", text):
        minutes, seconds = text.split(":")
        return int(minutes) * 60 + int(seconds)

    match = re.match(r"(?:(\d+):)?(\d{2}):(\d{2})(?:\.\d+)?$", text)
    if not match:
        return 0

    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    return (hours * 3600) + (minutes * 60) + seconds


def _seconds_to_mmss(value: int) -> str:
    total_seconds = max(0, int(value))
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


def _parse_timing_line(timing_line: str) -> tuple[int, int]:
    if "-->" not in str(timing_line):
        return 0, 0
    start_raw, end_raw = [part.strip() for part in str(timing_line).split("-->", 1)]
    return _parse_timecode(start_raw), _parse_timecode(end_raw)


def _timed_blocks_from_srt(girdi_dosyasi: Path) -> list[dict]:
    blocks = parse_srt_blocks(read_srt_file(girdi_dosyasi))
    timed_blocks = []
    for block in blocks:
        if not block.is_processable:
            continue
        start_sec, end_sec = _parse_timing_line(block.timing_line or "")
        if end_sec <= start_sec:
            continue
        text = normalize_whitespace(block.text_content)
        if not text:
            continue
        timed_blocks.append(
            {
                "start_sec": start_sec,
                "end_sec": end_sec,
                "start": _seconds_to_mmss(start_sec),
                "end": _seconds_to_mmss(end_sec),
                "text": text,
            }
        )
    return timed_blocks


def _extract_reference_moments(
    metadata_data: Optional[dict],
    critic_data: Optional[dict],
    trim_data: Optional[dict],
    broll_data: Optional[list],
) -> list[int]:
    moments: list[int] = []

    secilen = select_metadata_language(metadata_data)
    if secilen:
        for item in secilen.get("chapters", [])[:6]:
            if not isinstance(item, dict):
                continue
            seconds = _parse_timecode(item.get("timestamp", ""))
            if seconds > 0:
                moments.append(seconds)

    if isinstance(critic_data, dict):
        for key in ("timeline_notes", "rewrite_opportunities"):
            for item in critic_data.get(key, [])[:4]:
                if not isinstance(item, dict):
                    continue
                seconds = _parse_timecode(item.get("timestamp", ""))
                if seconds > 0:
                    moments.append(seconds)

    if isinstance(trim_data, dict):
        for item in trim_data.get("trim_targets", [])[:4]:
            if not isinstance(item, dict):
                continue
            seconds = _parse_timecode(item.get("timestamp", ""))
            if seconds > 0:
                moments.append(seconds)

    if isinstance(broll_data, list):
        for item in broll_data[:5]:
            if not isinstance(item, dict):
                continue
            seconds = _parse_timecode(item.get("timestamp", ""))
            if seconds > 0:
                moments.append(seconds)

    return moments


def _find_closest_block_index(start_times: list[int], target_seconds: int) -> Optional[int]:
    best_idx = None
    best_distance = None
    for idx, start_sec in enumerate(start_times):
        distance = abs(start_sec - target_seconds)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_idx = idx
    return best_idx


def _build_adaptive_transcript(
    timed_blocks: list[dict],
    metadata_data: Optional[dict] = None,
    critic_data: Optional[dict] = None,
    trim_data: Optional[dict] = None,
    broll_data: Optional[list] = None,
    max_chars: int = ADAPTIVE_TRANSCRIPT_MAX_CHARS,
) -> str:
    if not timed_blocks:
        return ""

    lines = [f"[{item['start']} - {item['end']}] {item['text']}" for item in timed_blocks]
    full_text = "\n".join(lines)
    if len(full_text) <= max_chars:
        return full_text

    anchor_indices = {0, max(0, len(timed_blocks) - 1)}
    if len(timed_blocks) > 1:
        anchor_indices.add(1)
        anchor_indices.add(max(0, len(timed_blocks) - 2))
    if len(timed_blocks) > 2:
        for idx in range(ADAPTIVE_TRANSCRIPT_TARGET_ANCHORS):
            fraction = idx / max(ADAPTIVE_TRANSCRIPT_TARGET_ANCHORS - 1, 1)
            anchor_indices.add(round((len(timed_blocks) - 1) * fraction))

    start_times = [int(item["start_sec"]) for item in timed_blocks]
    for moment in _extract_reference_moments(metadata_data, critic_data, trim_data, broll_data):
        closest = _find_closest_block_index(start_times, moment)
        if closest is not None:
            anchor_indices.add(closest)

    ordered_indices = []
    seen = set()
    for idx in sorted(anchor_indices):
        for expanded in range(
            max(0, idx - ADAPTIVE_TRANSCRIPT_WINDOW_RADIUS),
            min(len(timed_blocks), idx + ADAPTIVE_TRANSCRIPT_WINDOW_RADIUS + 1),
        ):
            if expanded in seen:
                continue
            seen.add(expanded)
            ordered_indices.append(expanded)

    selected_lines: list[str] = []
    current_length = 0
    for idx in ordered_indices:
        line = lines[idx]
        candidate_length = current_length + len(line) + (1 if selected_lines else 0)
        if candidate_length > max_chars and current_length >= int(max_chars * 0.72):
            break
        selected_lines.append(line)
        current_length = candidate_length

    if len(selected_lines) < 8:
        return full_text[:max_chars]
    return "\n".join(selected_lines)


def _should_use_light_mode(timed_blocks: list[dict], transcript_text: str) -> bool:
    if not timed_blocks:
        return False
    total_duration_seconds = max(int(timed_blocks[-1]["end_sec"]), 0)
    if total_duration_seconds <= CAROUSEL_LIGHT_MODE_MAX_SECONDS:
        return True
    if len(timed_blocks) <= CAROUSEL_LIGHT_MODE_MAX_BLOCKS:
        return True
    if len(transcript_text) <= CAROUSEL_LIGHT_MODE_MAX_TRANSCRIPT_CHARS:
        return True
    return False


def _safe_debug_label(value: object, default: str) -> str:
    normalized = normalize_whitespace(value).casefold()
    cleaned = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return cleaned or default


def _debug_response_path(debug_stem: str, stage_label: str, kind: str) -> Path:
    CAROUSEL_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    stem = _safe_debug_label(debug_stem, "carousel")
    stage = _safe_debug_label(stage_label, "stage")
    label = _safe_debug_label(kind, "response")
    return CAROUSEL_DEBUG_DIR / f"{stem}_{stage}_{label}.txt"


def _save_debug_response(
    debug_stem: str,
    stage_label: str,
    kind: str,
    response_text: object,
    *,
    note: str = "",
) -> Path | None:
    content = str(response_text or "").strip()
    if not content:
        return None

    path = _debug_response_path(debug_stem, stage_label, kind)
    parts = []
    if note:
        parts.extend([note, ""])
    parts.append(content)
    path.write_text("\n".join(parts).strip() + "\n", encoding="utf-8")
    logger.warning(f"Carousel debug cevabi kaydedildi: {path.name}")
    return path


def _json_repair_schema_hint(stage_label: str) -> str:
    stage = _safe_debug_label(stage_label, "stage")
    if stage == "ideation":
        return """
Expected root shape:
- `why_this_many_carousels_tr`: string
- `carousel_candidates`: array

Each candidate should remain an object that includes:
- `carousel_title_tr`
- `cover_prompt_en`
- `angle_tr`
- `why_selected_tr`
- `target_slide_count`
- `save_trigger_tr`
- `slide_outline_tr`
- `virality_rationale_tr`
- `cta_prompt_en`
""".strip()

    return """
Expected root shape:
- `why_this_many_carousels_tr`: string
- `selected_carousel_count`: integer
- `carousel_candidates`: array

Each candidate should remain an object with metadata, prompt fields, and a `slides` array.
Each slide should remain an object with:
- `slide_no`
- `slide_goal_tr`
- `visual_plan_tr`
- `image_prompt_en`
- `design_prompt_en`
- `caption_tr`
""".strip()


def _build_invalid_json_repair_prompt(raw_response: str, stage_label: str) -> str:
    schema_hint = _json_repair_schema_hint(stage_label)
    return f"""
You are repairing malformed Instagram carousel JSON produced by another model.

Task:
Convert the raw assistant output below into one valid JSON object.

Rules:
- Return only JSON. No markdown fences. No explanation.
- Preserve the original meaning and structure as much as possible.
- Keep all audience-facing fields in Turkish.
- Keep the legacy `_en` prompt fields in Turkish too; only the key names stay the same.
- Remove any brainstorm prose, duplicate wrappers, or trailing commentary outside the JSON.
- Fix invalid commas, brackets, quotes, and escaping.
- If a field is obviously cut off, complete it minimally and conservatively so the JSON becomes valid.
- Do not invent extra carousel candidates unless required to preserve the existing structure.

Schema guidance:
{schema_hint}

RAW OUTPUT:
{raw_response}
""".strip()


def _build_strict_json_retry_prompt(original_prompt: str, stage_label: str) -> str:
    return f"""
The previous response for the Instagram carousel `{stage_label}` stage was not valid JSON.

Return a fresh answer from scratch.

Hard rules:
- Return only one valid JSON object.
- Do not include analysis, brainstorm text, markdown fences, or any extra text before/after JSON.
- Escape all double quotes correctly inside string values.
- Ensure commas, arrays, and braces are fully valid.
- If you are unsure, keep values shorter and simpler rather than verbose.

Follow these original task instructions exactly:

{original_prompt}
""".strip()


def _repair_invalid_json_response(
    llm: CentralLLM,
    raw_response: str,
    *,
    stage_label: str,
    debug_stem: str,
    attempt_no: int,
) -> Optional[dict]:
    if not normalize_whitespace(raw_response):
        return None

    try:
        repaired_response = call_with_youtube_profile(
            llm,
            _build_invalid_json_repair_prompt(raw_response, stage_label),
            profile="analytic_json",
            timeout=_carousel_profile_timeout(llm, "analytic_json"),
            max_retries=1,
        )
    except Exception as exc:
        logger.warning(f"Carousel JSON repair istegi basarisiz oldu ({stage_label} / deneme {attempt_no}): {exc}")
        return None

    _save_debug_response(
        debug_stem,
        stage_label,
        f"attempt_{attempt_no}_repair_response",
        repaired_response,
        note="Otomatik JSON repair cevabi",
    )

    repaired = extract_json_response(repaired_response, logger_override=logger, log_errors=False)
    if repaired:
        logger.info(f"Carousel JSON repair basarili oldu: {stage_label} (deneme {attempt_no})")
    else:
        logger.warning(f"Carousel JSON repair gecersiz cevap dondu: {stage_label} (deneme {attempt_no})")
    return repaired


def _retry_with_strict_json_profile(
    llm: CentralLLM,
    prompt: str,
    *,
    stage_label: str,
    debug_stem: str,
    attempt_no: int,
) -> Optional[dict]:
    try:
        strict_response = call_with_youtube_profile(
            llm,
            _build_strict_json_retry_prompt(prompt, stage_label),
            profile="strict_json",
            timeout=_carousel_profile_timeout(llm, "strict_json"),
            max_retries=1,
        )
    except Exception as exc:
        logger.warning(f"Carousel strict_json retry basarisiz oldu ({stage_label} / deneme {attempt_no}): {exc}")
        return None

    _save_debug_response(
        debug_stem,
        stage_label,
        f"attempt_{attempt_no}_strict_raw",
        strict_response,
        note="strict_json profili ile yeniden uretilen ham cevap",
    )

    parsed = extract_json_response(strict_response, logger_override=logger, log_errors=False)
    if parsed:
        logger.info(f"Carousel strict_json retry basarili oldu: {stage_label} (deneme {attempt_no})")
        return parsed

    repaired = _repair_invalid_json_response(
        llm,
        strict_response,
        stage_label=f"{stage_label}_strict",
        debug_stem=debug_stem,
        attempt_no=attempt_no,
    )
    if repaired:
        return repaired

    logger.warning(f"Carousel strict_json retry da parse edilemedi: {stage_label} (deneme {attempt_no})")
    return None


def _truncate_words(text: str, limit: int) -> str:
    content = normalize_whitespace(text)
    if limit <= 0 or len(content) <= limit:
        return content

    shortened = content[: max(1, limit - 1)].rstrip()
    if " " in shortened:
        shortened = shortened.rsplit(" ", 1)[0].rstrip()
    return (shortened or content[: max(1, limit - 1)]).rstrip(" ,;:") + "…"


def _limit_sentences(text: str, max_sentences: int = MAX_BODY_SENTENCES) -> str:
    content = normalize_whitespace(text)
    if not content:
        return ""

    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", content) if part.strip()]
    if len(parts) <= max_sentences:
        return content
    return " ".join(parts[:max_sentences]).strip()


def _fit_title(text: str, limit: int) -> str:
    content = normalize_whitespace(text)
    if not content:
        return ""
    content = re.split(r"[.!?]", content, maxsplit=1)[0].strip()
    return _truncate_words(content.rstrip(" ,;:"), limit).rstrip(".!?")


def _fit_body(text: str, limit: int) -> str:
    content = _limit_sentences(text, MAX_BODY_SENTENCES)
    return _truncate_words(content, limit)


def _fit_button(text: str, limit: int) -> str:
    content = normalize_whitespace(text)
    if not content:
        return ""
    content = re.split(r"[.!?]", content, maxsplit=1)[0].strip()
    return _truncate_words(content, limit).rstrip(".!?")


def _fallback_cta_potential(raw_value: object, *fallbacks: object) -> str:
    fitted = _fit_body(raw_value, max(CTA_BODY_LIMIT, 75))
    if fitted:
        return fitted
    for fallback in fallbacks:
        fitted_fallback = _fit_body(fallback, max(CTA_BODY_LIMIT, 75))
        if fitted_fallback:
            return fitted_fallback
    return "Kaydetme ve CTA tepki ihtimali guclu."


def _default_goal_en(angle_tr: str) -> str:
    angle = normalize_whitespace(angle_tr)
    if not angle:
        return DEFAULT_CAROUSEL_GOAL_EN
    return f'"{angle}" acisini ilk bakista anlasilir ve kaydetmeye deger hale getiren dikkat cekici bir carousel tasarimi.'


def _first_non_empty(*values: object) -> str:
    for value in values:
        text = normalize_whitespace(value)
        if text:
            return text
    return ""


def _looks_turkish(text: object) -> bool:
    content = normalize_whitespace(text)
    if not content:
        return False
    if any(ch in content for ch in "çğıöşüÇĞİÖŞÜİı"):
        return True
    lowered = content.casefold()
    words = set(re.findall(r"[a-zA-Zçğıöşü]+", lowered))
    matches = sum(1 for marker in TURKISH_PROMPT_MARKERS if marker in words)
    return matches >= 2


def _needs_turkish_prompt_repair(text: object) -> bool:
    content = normalize_whitespace(text)
    if not content:
        return False
    words = re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ]+", content)
    if not words:
        return False
    if len(words) <= 2:
        return (not _looks_turkish(content)) and any(len(word) > 3 for word in words)
    return not _looks_turkish(content)


def _normalize_prompt_fields(
    item: object,
    *,
    primary_text_fallback: str,
    angle_tr: str = "",
) -> dict:
    payload = item if isinstance(item, dict) else {}
    return {
        "goal_en": _first_non_empty(payload.get("goal_en", ""), _default_goal_en(angle_tr)),
        "background_en": _first_non_empty(payload.get("background_en", ""), DEFAULT_CAROUSEL_BACKGROUND_EN),
        "subject_en": _first_non_empty(
            payload.get("subject_en", ""),
            payload.get("people_en", ""),
            DEFAULT_CAROUSEL_SUBJECT_EN,
        ),
        "primary_text_en": _first_non_empty(payload.get("primary_text_en", ""), primary_text_fallback),
        "supporting_elements_en": _first_non_empty(
            payload.get("supporting_elements_en", ""),
            payload.get("supporting_en", ""),
            DEFAULT_CAROUSEL_SUPPORTING_ELEMENTS_EN,
        ),
        "style_mood_en": _first_non_empty(
            payload.get("style_mood_en", ""),
            payload.get("style_en", ""),
            DEFAULT_CAROUSEL_STYLE_EN,
        ),
    }


def build_carousel_design_prompt(prompt_fields: dict) -> str:
    lines = [
        "Bir Instagram Carousel tasarimi olustur.",
        f"Amac: {prompt_fields.get('goal_en', '')}",
        f"Arka Plan: {prompt_fields.get('background_en', '')}",
        f"Ozne: {prompt_fields.get('subject_en', '')}",
        f'Ana Metin: "{prompt_fields.get("primary_text_en", "")}"',
        f"Destekleyici Unsurlar: {prompt_fields.get('supporting_elements_en', '')}",
        f"Stil/Ruh Hali: {prompt_fields.get('style_mood_en', '')}",
        CAROUSEL_FORMAT_SPEC_TR,
        CAROUSEL_PROMPT_FOOTER,
    ]
    return "\n".join(line for line in lines if normalize_whitespace(line))


def _iter_carousel_prompt_texts(data: dict):
    for candidate in data.get("carousel_candidates", []):
        if not isinstance(candidate, dict):
            continue
        for prompt_key in ("cover_prompt_en", "cta_prompt_en"):
            prompt = candidate.get(prompt_key, {})
            if isinstance(prompt, dict):
                for value in prompt.values():
                    yield value
        for slide in candidate.get("slides", []):
            if not isinstance(slide, dict):
                continue
            image_prompt = slide.get("image_prompt_en", {})
            if isinstance(image_prompt, dict):
                for value in image_prompt.values():
                    yield value
            design_prompt = slide.get("design_prompt_en", {})
            if isinstance(design_prompt, dict):
                for value in design_prompt.values():
                    yield value


def _needs_carousel_prompt_repair(data: dict) -> bool:
    return any(_needs_turkish_prompt_repair(value) for value in _iter_carousel_prompt_texts(data))


def _build_prompt_language_repair_prompt(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    return f"""
You are repairing Instagram carousel JSON.

Task:
Rewrite only the image-generation prompt fields into fluent, natural Turkish while preserving the JSON structure.

Rules:
- Return only one valid JSON object.
- Keep every `*_tr` field in Turkish.
- Keep ranks, scores, slide counts, ordering, captions, and CTA metadata unchanged unless needed to preserve valid JSON.
- Rewrite these fields in Turkish only. Keep the legacy `_en` key names unchanged:
  - `cover_prompt_en.*`
  - `cta_prompt_en.*`
  - each slide `image_prompt_en.background`
  - each slide `image_prompt_en.people`
  - each slide `image_prompt_en.overlay_text`
  - each slide `design_prompt_en.*`
- Do not leave English sentences inside those prompt fields unless a brand name or technical term truly requires it.
- Make the Turkish prompts detailed enough for image-generation tools.
- Preserve the meaning of the existing content while rewriting the prompt content into natural Turkish.
- The six-field design prompts must align with this structure:
  Bir Instagram Carousel tasarimi olustur.
  Amac: ...
  Arka Plan: ...
  Ozne: ...
  Ana Metin: "..."
  Destekleyici Unsurlar: ...
  Stil/Ruh Hali: ...
  {CAROUSEL_FORMAT_SPEC_TR}
  {CAROUSEL_PROMPT_FOOTER}

JSON TO REPAIR:
{payload}
""".strip()


def _repair_carousel_prompt_language(data: dict, llm: CentralLLM) -> dict:
    repaired = _request_llm(
        llm,
        _build_prompt_language_repair_prompt(data),
        retries=1,
        profile="analytic_json",
        stage_label="prompt_language_repair",
    )
    if isinstance(repaired, dict) and isinstance(repaired.get("carousel_candidates"), list):
        return repaired
    return data


def _request_llm(
    llm: CentralLLM,
    prompt: str,
    retries: int = LLM_RETRIES,
    profile: str = "creative_ranker",
    profile_timeout: Optional[int] = None,
    provider_retries: Optional[int] = None,
    debug_stem: str = "carousel",
    stage_label: str = "generation",
) -> Optional[dict]:
    for deneme in range(1, retries + 1):
        try:
            cevap = call_with_youtube_profile(
                llm,
                prompt,
                profile=profile,
                timeout=profile_timeout,
                max_retries=provider_retries,
            )
            parsed = extract_json_response(cevap, logger_override=logger, log_errors=False)
            if parsed:
                return parsed
            _save_debug_response(
                debug_stem,
                stage_label,
                f"attempt_{deneme}_invalid_raw",
                cevap,
                note="LLM'den gelen ancak parse edilemeyen ham cevap",
            )
            repaired = _repair_invalid_json_response(
                llm,
                cevap,
                stage_label=stage_label,
                debug_stem=debug_stem,
                attempt_no=deneme,
            )
            if repaired:
                return repaired
            strict_retry = _retry_with_strict_json_profile(
                llm,
                prompt,
                stage_label=stage_label,
                debug_stem=debug_stem,
                attempt_no=deneme,
            )
            if strict_retry:
                return strict_retry
            logger.warning(f"Carousel JSON parse edilemedi, yeni deneme yapiliyor: {stage_label} ({deneme}/{retries})")
        except Exception as exc:
            logger.warning(f"Carousel LLM hatasi ({deneme}/{retries}): {exc}")
        time.sleep(deneme)
    return None


def build_ideation_prompt(
    transkript: str,
    critic_ozeti: str,
    trim_ozeti: str,
    metadata_ozeti: str = "YouTube metadata verisi yok.",
    broll_ozeti: str = "B-roll verisi yok.",
) -> str:
    return f"""
Sen viral Instagram carousel stratejisti, icerik editoru ve gorsel yonlendirme uzmani bir sosyal medya kreatif direktorusun.

Gorevin:
Ana videonun transcriptini oku, videoyu derinlemesine anla ve bu videodan kac adet anlamli Instagram carousel cikmasi gerektigine karar ver.

KRITIK KURALLAR:
- Dusunme surecini icten yap; ciktiya asla brainstorm, analiz, aciklama veya markdown ekleme.
- Sadece tek bir gecerli JSON nesnesi don.
- Tum aciklamalar Turkce olsun.
- Gorsel prompt alanlari da dahil olmak uzere tum prompt icerikleri akici ve net Turkce olsun.
- `_en` eki yalnizca eski alan ismidir; `cover_prompt_en`, `design_prompt_en`, `cta_prompt_en` ve benzeri tum alanlarin degerlerini Turkce yaz.
- Transcriptin zenginligine gore kac carousel adayi gerektigine sen karar ver.
- En az {MIN_CAROUSEL_CANDIDATES} anlamli carousel adayi uret.
- En fazla {MAX_CAROUSEL_CANDIDATES} guclu carousel adayi uret.
- Her carousel adayi {MIN_SLIDES} ile {MAX_SLIDES} slayt arasinda olmali.
- Tum adaylari otomatik olarak {MIN_SLIDES} slayta sabitleme; konu derinligi ve anlatim ihtiyacina gore daha uzun alternatifler de cikar.
- Malzeme uygunsa adaylar arasinda slayt sayisi cesitliligi olsun.
- Her aday farkli bir viral aci kullansin.
- Slayt yapilari Instagram'da kaydetme, paylasma ve son slayta kadar kaydirma istegi dogursun.
- En kritik retention metriği: cover/slide 1'den slide 2'ye gecis orani.
- Cover sayfasi bilgi vermesin; merak yaratsin.
- Altin kural: cevabi 2. slayta sakla.
- Cover basligi kisa, dikkat cekici ve max {COVER_TITLE_LIMIT} karakter olsun.
- Cover alt metni max {COVER_SUBTITLE_LIMIT} karakter olsun.
- Slayt basliklari max {SLIDE_TITLE_LIMIT} karakter olsun.
- Slayt metinleri max {SLIDE_BODY_LIMIT} karakter olsun.
- Her slayt tek fikir icersin.
- Cumleler basit, net ve mobilde rahat okunur olsun.
- Metinler 2-3 satiri gecmeyecek kadar kisa tasarlansin.
- Cover, slaytlar ve CTA ayni gorsel aileden geliyormus gibi tutarli olsun.
- Ayni carousel icindeki tum slaytlar ortak tema, renk paleti, tipografi ve kompozisyon dili tasisin.
- Cover icin `cover_prompt_en`, her slayt icin `design_prompt_en`, CTA icin `cta_prompt_en` ver.
- Her promptta su katmanlar mutlaka bulunsun:
  Amac
  Arka Plan
  Ozne
  Ana Metin
  Destekleyici Unsurlar
  Stil/Ruh Hali
- Her promptta format/cozunurluk bilgisini acikca belirt:
  {CAROUSEL_FORMAT_SPEC_TR}
- Her prompt mantigina su cümleyi dahil et:
  {CAROUSEL_PROMPT_FOOTER}

JSON SEMASI:
{{
  "why_this_many_carousels_tr": "",
  "carousel_candidates": [
    {{
      "carousel_title_tr": "",
      "cover_prompt_en": {{
        "goal_en": "",
        "background_en": "",
        "subject_en": "",
        "primary_text_en": "",
        "supporting_elements_en": "",
        "style_mood_en": ""
      }},
      "angle_tr": "",
      "why_selected_tr": "",
      "target_slide_count": 8,
      "save_trigger_tr": "",
      "slide_outline_tr": ["", "", ""],
      "virality_rationale_tr": "",
      "cta_prompt_en": {{
        "goal_en": "",
        "background_en": "",
        "subject_en": "",
        "primary_text_en": "",
        "supporting_elements_en": "",
        "style_mood_en": ""
      }}
    }}
  ]
}}

Video Elestirmeni Ozeti:
{critic_ozeti}

Kesim Onerileri Ozeti:
{trim_ozeti}

YouTube Metadata Ozeti:
{metadata_ozeti}

B-Roll Ozeti:
{broll_ozeti}

Transcript:
{transkript}
""".strip()


def build_selection_prompt(
    transkript: str,
    critic_ozeti: str,
    trim_ozeti: str,
    ideation_payload: dict,
    metadata_ozeti: str = "YouTube metadata verisi yok.",
    broll_ozeti: str = "B-roll verisi yok.",
) -> str:
    ideation_text = json.dumps(ideation_payload, ensure_ascii=False, indent=2)
    return f"""
Sen viral Instagram carousel stratejisti, tasarim yoneticisi ve final packaging editorusun.

Gorevin:
Verilen carousel adaylarini incele, en guclu olanlari sec, viral olma ihtimallerine gore skorla ve final carousel paketlerini olustur.

KRITIK KURALLAR:
- Dusunme surecini icten yap; ciktiya analiz, brainstorm, aciklama veya markdown ekleme.
- Sadece tek bir gecerli JSON nesnesi don.
- Tum Turkce alanlar yalnizca Turkce olsun.
- `image_prompt_en.background`, `image_prompt_en.people` ve `image_prompt_en.overlay_text` alanlari akici Turkce olsun.
- `cover_prompt_en`, `design_prompt_en` ve `cta_prompt_en` alanlari da akici Turkce olsun.
- `_en` eki yalnizca eski alan ismidir; bu alanlarin degerlerini Turkce yaz.
- Gorsel promptlari DALL-E, Midjourney, Nano Banana Pro gibi sistemlerin anlayacagi kadar detayli yaz.
- Final listede en az {MIN_CAROUSEL_CANDIDATES} guclu carousel adayi olsun.
- Final listede en fazla {MAX_CAROUSEL_CANDIDATES} guclu carousel adayi olsun.
- Her carousel adayi {MIN_SLIDES} ile {MAX_SLIDES} slayt arasinda olsun.
- En kritik retention metriği: cover/slide 1'den slide 2'ye gecis orani.
- Cover sayfasi bilgi vermesin; merak yaratsin.
- Altin kural: cevabi 2. slayta sakla.
- Her slaytta iki katman bulunmali:
  1. Goruntude ne olmali? Bunu `visual_plan_tr` ve detayli `image_prompt_en` ile anlat.
  2. Slaytin uzerindeki/aciklama metni ne olmali? Bunu `caption_tr` ile ver.
- `image_prompt_en` mutlaka 3 katman icermeli:
  1. `background`
  2. `people`
  3. `overlay_text`
- Tumu viralite, retention ve kaydetme ihtimaline gore optimize et.
- Tum carousel adaylarini otomatik olarak {MIN_SLIDES} slaytta birakma; konu derinligi uygunsa bazilarini daha uzun tasarla.
- Malzeme yeterliyse minimumda takilma; farkli adaylarda slayt sayisi cesitliligi olustur.
- `carousel_title_tr` cover basligi olarak dusunulsun ve max {COVER_TITLE_LIMIT} karakter olsun.
- `cover_subtitle_tr` max {COVER_SUBTITLE_LIMIT} karakter olsun.
- `slide_goal_tr` kullanicinin gorecegi slayt basligi olarak dusunulsun ve max {SLIDE_TITLE_LIMIT} karakter olsun.
- `caption_tr` max {SLIDE_BODY_LIMIT} karakter olsun.
- `cta_title_tr` max {CTA_TITLE_LIMIT} karakter, `cta_body_tr` max {CTA_BODY_LIMIT} karakter, `cta_button_tr` max {CTA_BUTTON_LIMIT} karakter olsun.
- `cta_potential_tr` alaninda bu carouselin kaydetme/CTA olasiligini bir cümleyle acikla.
- Basliklar kisa, dikkat cekici ve merak uyandirici olsun.
- Cumleler basit ve net olsun.
- Her slayt tek fikir icersin.
- Metinler mobilde 2-3 satiri gecmeyecek yogunlukta olsun.
- Cover, slaytlar ve CTA ayni gorsel aileden geliyormus gibi tutarli olsun.
- Ayni carousel icindeki tum slaytlar ortak tema, renk paleti, tipografi ve kompozisyon dili tasisin.
- Cover icin `cover_prompt_en`, her slayt icin `design_prompt_en`, CTA icin `cta_prompt_en` ver.
- Her promptta su katmanlar mutlaka bulunsun:
  Amac
  Arka Plan
  Ozne
  Ana Metin
  Destekleyici Unsurlar
  Stil/Ruh Hali
- Her promptta format/cozunurluk bilgisini acikca belirt:
  {CAROUSEL_FORMAT_SPEC_TR}
- Her prompt mantigina su cümleyi dahil et:
  {CAROUSEL_PROMPT_FOOTER}

JSON SEMASI:
{{
  "why_this_many_carousels_tr": "",
  "selected_carousel_count": 0,
  "carousel_candidates": [
    {{
      "rank": 1,
      "viral_score": 95,
      "carousel_title_tr": "",
      "cover_subtitle_tr": "",
      "cover_prompt_en": {{
        "goal_en": "",
        "background_en": "",
        "subject_en": "",
        "primary_text_en": "",
        "supporting_elements_en": "",
        "style_mood_en": ""
      }},
      "angle_tr": "",
      "why_selected_tr": "",
      "audience_value_tr": "",
      "cta_potential_tr": "",
      "cta_title_tr": "",
      "cta_body_tr": "",
      "cta_button_tr": "",
      "cta_prompt_en": {{
        "goal_en": "",
        "background_en": "",
        "subject_en": "",
        "primary_text_en": "",
        "supporting_elements_en": "",
        "style_mood_en": ""
      }},
      "slides": [
        {{
          "slide_no": 1,
          "slide_goal_tr": "",
          "visual_plan_tr": "",
          "image_prompt_en": {{
            "background": "",
            "people": "",
            "overlay_text": ""
          }},
          "design_prompt_en": {{
            "goal_en": "",
            "background_en": "",
            "subject_en": "",
            "primary_text_en": "",
            "supporting_elements_en": "",
            "style_mood_en": ""
          }},
          "caption_tr": ""
        }}
      ]
    }}
  ]
}}

Video Elestirmeni Ozeti:
{critic_ozeti}

Kesim Onerileri Ozeti:
{trim_ozeti}

YouTube Metadata Ozeti:
{metadata_ozeti}

B-Roll Ozeti:
{broll_ozeti}

Carousel Aday Havuzu:
{ideation_text}

Transcript:
{transkript}
""".strip()


def build_prompt(
    transkript: str,
    critic_ozeti: str,
    trim_ozeti: str,
    metadata_ozeti: str = "YouTube metadata verisi yok.",
    broll_ozeti: str = "B-roll verisi yok.",
) -> str:
    return f"""
Sen viral Instagram carousel stratejisti, tasarim yoneticisi ve kreatif sosyal medya editorusun.

Gorevin:
Transcripti okuyup videonun en guclu acilarindan birden fazla Instagram carousel adayi cikar, bunlari viral olma ihtimallerine gore skorla ve final detaylarini uret.

KRITIK KURALLAR:
- Dusunme surecini icten yap; ciktiya asla brainstorm, analiz, aciklama veya markdown ekleme.
- Sadece tek bir gecerli JSON nesnesi don.
- Tum Turkce alanlar yalnizca Turkce olsun.
- `image_prompt_en.background`, `image_prompt_en.people` ve `image_prompt_en.overlay_text` alanlari akici Turkce olsun.
- `cover_prompt_en`, `design_prompt_en` ve `cta_prompt_en` alanlari da akici Turkce olsun.
- `_en` eki yalnizca eski alan ismidir; bu alanlarin degerlerini Turkce yaz.
- Gorsel promptlari DALL-E, Midjourney, Nano Banana Pro gibi sistemlerin anlayacagi kadar detayli yaz.
- Transcriptin zenginligine gore kac carousel adayi gerektigine sen karar ver.
- En az {MIN_CAROUSEL_CANDIDATES} guclu carousel adayi uret.
- En fazla {MAX_CAROUSEL_CANDIDATES} guclu carousel adayi uret.
- Her carousel adayi {MIN_SLIDES} ile {MAX_SLIDES} slayt arasinda olsun.
- En kritik retention metriği: cover/slide 1'den slide 2'ye gecis orani.
- Cover sayfasi bilgi vermesin; merak yaratsin.
- Altin kural: cevabi 2. slayta sakla.
- Tum adaylari otomatik olarak {MIN_SLIDES} slayta sabitleme; konu derinligi uygunsa bazilarini daha uzun tasarla.
- Malzeme yeterliyse slide sayilarinda cesitlilik olustur.
- Tumu viralite, retention ve kaydetme ihtimaline gore optimize et.
- `carousel_title_tr` cover basligi olarak dusunulsun ve max {COVER_TITLE_LIMIT} karakter olsun.
- `cover_subtitle_tr` max {COVER_SUBTITLE_LIMIT} karakter olsun.
- `slide_goal_tr` max {SLIDE_TITLE_LIMIT} karakter olsun.
- `caption_tr` max {SLIDE_BODY_LIMIT} karakter olsun.
- `cta_title_tr` max {CTA_TITLE_LIMIT} karakter, `cta_body_tr` max {CTA_BODY_LIMIT} karakter, `cta_button_tr` max {CTA_BUTTON_LIMIT} karakter olsun.
- `cta_potential_tr` alaninda bu carouselin kaydetme/CTA olasiligini bir cümleyle acikla.
- Her baslik kisa ve dikkat cekici olsun.
- Cumleler basit ve net olsun.
- Her slayt tek fikir icersin.
- Metinler mobilde 2-3 satiri gecmeyecek yogunlukta olsun.
- Cover, slaytlar ve CTA ayni gorsel aileden geliyormus gibi tutarli olsun.
- Ayni carousel icindeki tum slaytlar ortak tema, renk paleti, tipografi ve kompozisyon dili tasisin.
- Cover icin `cover_prompt_en`, her slayt icin `design_prompt_en`, CTA icin `cta_prompt_en` ver.
- Her promptta su katmanlar mutlaka bulunsun:
  Amac
  Arka Plan
  Ozne
  Ana Metin
  Destekleyici Unsurlar
  Stil/Ruh Hali
- Her promptta format/cozunurluk bilgisini acikca belirt:
  {CAROUSEL_FORMAT_SPEC_TR}
- Her prompt mantigina su cümleyi dahil et:
  {CAROUSEL_PROMPT_FOOTER}

JSON SEMASI:
{{
  "why_this_many_carousels_tr": "",
  "selected_carousel_count": 0,
  "carousel_candidates": [
    {{
      "rank": 1,
      "viral_score": 95,
      "carousel_title_tr": "",
      "cover_subtitle_tr": "",
      "cover_prompt_en": {{
        "goal_en": "",
        "background_en": "",
        "subject_en": "",
        "primary_text_en": "",
        "supporting_elements_en": "",
        "style_mood_en": ""
      }},
      "angle_tr": "",
      "why_selected_tr": "",
      "audience_value_tr": "",
      "cta_potential_tr": "",
      "cta_title_tr": "",
      "cta_body_tr": "",
      "cta_button_tr": "",
      "cta_prompt_en": {{
        "goal_en": "",
        "background_en": "",
        "subject_en": "",
        "primary_text_en": "",
        "supporting_elements_en": "",
        "style_mood_en": ""
      }},
      "slides": [
        {{
          "slide_no": 1,
          "slide_goal_tr": "",
          "visual_plan_tr": "",
          "image_prompt_en": {{
            "background": "",
            "people": "",
            "overlay_text": ""
          }},
          "design_prompt_en": {{
            "goal_en": "",
            "background_en": "",
            "subject_en": "",
            "primary_text_en": "",
            "supporting_elements_en": "",
            "style_mood_en": ""
          }},
          "caption_tr": ""
        }}
      ]
    }}
  ]
}}

Video Elestirmeni Ozeti:
{critic_ozeti}

Kesim Onerileri Ozeti:
{trim_ozeti}

YouTube Metadata Ozeti:
{metadata_ozeti}

B-Roll Ozeti:
{broll_ozeti}

Transcript:
{transkript}
""".strip()


def _normalize_image_prompt(item: object) -> dict:
    prompt = item if isinstance(item, dict) else {}
    return {
        "background": normalize_whitespace(prompt.get("background", "")),
        "people": normalize_whitespace(prompt.get("people", "")),
        "overlay_text": normalize_whitespace(prompt.get("overlay_text", "")),
    }


def _fallback_cover_subtitle(raw_value: str) -> str:
    fitted = _fit_body(raw_value, COVER_SUBTITLE_LIMIT)
    return fitted or DEFAULT_COVER_SUBTITLE


def _fallback_cta_title(raw_value: str) -> str:
    fitted = _fit_title(raw_value, CTA_TITLE_LIMIT)
    return fitted or DEFAULT_CTA_TITLE


def _fallback_cta_body(raw_value: str) -> str:
    fitted = _fit_body(raw_value, CTA_BODY_LIMIT)
    return fitted or DEFAULT_CTA_BODY


def _fallback_cta_button(raw_value: str) -> str:
    fitted = _fit_button(raw_value, CTA_BUTTON_LIMIT)
    return fitted or DEFAULT_CTA_BUTTON


def _normalize_carousel_prompt_fields(
    item: object,
    *,
    angle_tr: str,
    primary_text_fallback: str,
    background_fallback: str = "",
    subject_fallback: str = "",
    supporting_fallback: str = "",
) -> dict:
    payload = item if isinstance(item, dict) else {}
    return _normalize_prompt_fields(
        {
            "goal_en": payload.get("goal_en", ""),
            "background_en": payload.get("background_en", "") or background_fallback,
            "subject_en": payload.get("subject_en", "") or subject_fallback,
            "primary_text_en": payload.get("primary_text_en", "") or primary_text_fallback,
            "supporting_elements_en": payload.get("supporting_elements_en", "") or supporting_fallback,
            "style_mood_en": payload.get("style_mood_en", ""),
        },
        primary_text_fallback=primary_text_fallback,
        angle_tr=angle_tr,
    )


def _candidate_slide_count(candidate: object) -> int:
    if not isinstance(candidate, dict):
        return 0
    slides = candidate.get("slides", [])
    if not isinstance(slides, list):
        return 0
    count = 0
    for slide in slides:
        if not isinstance(slide, dict):
            continue
        slide_goal = normalize_whitespace(slide.get("slide_goal_tr", ""))
        caption = normalize_whitespace(slide.get("caption_tr", ""))
        visual_plan = normalize_whitespace(slide.get("visual_plan_tr", ""))
        if slide_goal or caption or visual_plan:
            count += 1
    return count


def _candidate_target_slide_count(candidate: dict, ideation_candidate: Optional[dict] = None) -> int:
    target_candidates = [
        candidate.get("target_slide_count"),
        len(candidate.get("slide_outline_tr", [])) if isinstance(candidate.get("slide_outline_tr"), list) else None,
    ]
    if isinstance(ideation_candidate, dict):
        target_candidates.extend(
            [
                ideation_candidate.get("target_slide_count"),
                len(ideation_candidate.get("slide_outline_tr", []))
                if isinstance(ideation_candidate.get("slide_outline_tr"), list)
                else None,
            ]
        )

    for value in target_candidates:
        try:
            numeric = int(value or 0)
        except (TypeError, ValueError):
            continue
        if numeric > 0:
            return max(MIN_SLIDES, min(MAX_SLIDES, numeric))
    return MIN_SLIDES


def _match_ideation_candidate(candidate: dict, ideation_payload: Optional[dict], fallback_index: int) -> Optional[dict]:
    if not isinstance(ideation_payload, dict):
        return None
    ideation_candidates = ideation_payload.get("carousel_candidates", [])
    if not isinstance(ideation_candidates, list):
        return None

    title = normalize_whitespace(candidate.get("carousel_title_tr", "")).casefold()
    angle = normalize_whitespace(candidate.get("angle_tr", "")).casefold()

    if title:
        for ideation_candidate in ideation_candidates:
            if not isinstance(ideation_candidate, dict):
                continue
            if normalize_whitespace(ideation_candidate.get("carousel_title_tr", "")).casefold() == title:
                return ideation_candidate

    if angle:
        for ideation_candidate in ideation_candidates:
            if not isinstance(ideation_candidate, dict):
                continue
            if normalize_whitespace(ideation_candidate.get("angle_tr", "")).casefold() == angle:
                return ideation_candidate

    zero_based = fallback_index - 1
    if 0 <= zero_based < len(ideation_candidates):
        matched = ideation_candidates[zero_based]
        if isinstance(matched, dict):
            return matched
    return None


def _build_underfilled_candidate_repair_prompt(
    candidate: dict,
    *,
    target_slide_count: int,
    transkript: str,
    critic_ozeti: str,
    trim_ozeti: str,
    metadata_ozeti: str,
    broll_ozeti: str,
    ideation_candidate: Optional[dict] = None,
) -> str:
    current_slide_count = _candidate_slide_count(candidate)
    candidate_text = json.dumps(candidate, ensure_ascii=False, indent=2)
    ideation_text = (
        json.dumps(ideation_candidate, ensure_ascii=False, indent=2)
        if isinstance(ideation_candidate, dict)
        else "null"
    )
    transcript_excerpt = transkript
    return f"""
You are repairing one underfilled Instagram carousel candidate.

Problem:
The candidate currently has only {current_slide_count} slide(s), but it must contain exactly {target_slide_count} slides.

Task:
Rewrite and complete this candidate into one production-ready carousel candidate with exactly {target_slide_count} slides.

Rules:
- Return only one valid JSON object for a single carousel candidate.
- Keep `rank` and `viral_score` unless a tiny adjustment is required for valid JSON.
- Preserve the core angle, audience promise, CTA direction, and strongest existing slide ideas whenever possible.
- Reuse strong existing slides, but rewrite weak, repetitive, or incomplete slides if needed.
- All `*_tr` fields must stay in Turkish.
- All legacy `_en` prompt fields must also be fluent Turkish; keep only the key names unchanged.
- `carousel_title_tr` max {COVER_TITLE_LIMIT} chars.
- `cover_subtitle_tr` max {COVER_SUBTITLE_LIMIT} chars.
- Every `slide_goal_tr` max {SLIDE_TITLE_LIMIT} chars.
- Every `caption_tr` max {SLIDE_BODY_LIMIT} chars.
- `cta_title_tr` max {CTA_TITLE_LIMIT} chars.
- `cta_body_tr` max {CTA_BODY_LIMIT} chars.
- `cta_button_tr` max {CTA_BUTTON_LIMIT} chars.
- `cta_potential_tr` should briefly explain the CTA/save potential in Turkish.
- Cover must create curiosity and should not fully reveal the answer.
- The strongest answer/payoff should appear on slide 2.
- Every slide must include:
  - `slide_no`
  - `slide_goal_tr`
  - `visual_plan_tr`
  - `image_prompt_en.background`
  - `image_prompt_en.people`
  - `image_prompt_en.overlay_text`
  - `design_prompt_en.goal_en`
  - `design_prompt_en.background_en`
  - `design_prompt_en.subject_en`
  - `design_prompt_en.primary_text_en`
  - `design_prompt_en.supporting_elements_en`
  - `design_prompt_en.style_mood_en`
  - `caption_tr`
- Slide numbers must be consecutive from 1 to {target_slide_count}.
- Keep the visual family coherent across cover, slides, and CTA.
- All slides in the same carousel should share the same theme, palette, typography, and design system.
- Every structured design prompt should align with this Turkish format:
  Bir Instagram Carousel tasarimi olustur.
  Amac: ...
  Arka Plan: ...
  Ozne: ...
  Ana Metin: "..."
  Destekleyici Unsurlar: ...
  Stil/Ruh Hali: ...
  {CAROUSEL_FORMAT_SPEC_TR}
  {CAROUSEL_PROMPT_FOOTER}

Reference summaries:
Video Elestirmeni Ozeti:
{critic_ozeti}

Kesim Onerileri Ozeti:
{trim_ozeti}

YouTube Metadata Ozeti:
{metadata_ozeti}

B-Roll Ozeti:
{broll_ozeti}

Transcript excerpt:
{transcript_excerpt}

Original ideation candidate:
{ideation_text}

Current incomplete candidate:
{candidate_text}
""".strip()


def _repair_underfilled_candidate(
    candidate: dict,
    *,
    llm: CentralLLM,
    transkript: str,
    critic_ozeti: str,
    trim_ozeti: str,
    metadata_ozeti: str,
    broll_ozeti: str,
    ideation_candidate: Optional[dict] = None,
    debug_stem: str = "carousel",
) -> dict:
    target_slide_count = _candidate_target_slide_count(candidate, ideation_candidate)
    repaired = _request_llm(
        llm,
        _build_underfilled_candidate_repair_prompt(
            candidate,
            target_slide_count=target_slide_count,
            transkript=transkript,
            critic_ozeti=critic_ozeti,
            trim_ozeti=trim_ozeti,
            metadata_ozeti=metadata_ozeti,
            broll_ozeti=broll_ozeti,
            ideation_candidate=ideation_candidate,
        ),
        retries=2,
        profile="analytic_json",
        profile_timeout=_carousel_profile_timeout(llm, "analytic_json"),
        provider_retries=2,
        debug_stem=debug_stem,
        stage_label=f"underfilled_candidate_{candidate.get('rank', target_slide_count)}",
    )
    if isinstance(repaired, dict) and _candidate_slide_count(repaired) >= MIN_SLIDES:
        return repaired
    return candidate


def _repair_underfilled_candidates(
    data: dict,
    *,
    llm: CentralLLM,
    transkript: str,
    critic_ozeti: str,
    trim_ozeti: str,
    metadata_ozeti: str,
    broll_ozeti: str,
    ideation_payload: Optional[dict] = None,
    debug_stem: str = "carousel",
) -> dict:
    if not isinstance(data, dict):
        return {}

    raw_candidates = data.get("carousel_candidates", [])
    if not isinstance(raw_candidates, list):
        return data

    repaired_candidates = []
    any_repaired = False

    for index, candidate in enumerate(raw_candidates, start=1):
        if not isinstance(candidate, dict):
            continue
        slide_count = _candidate_slide_count(candidate)
        if slide_count < MIN_SLIDES:
            logger.warning(
                f"Carousel adayi eksik slaytla geldi ({slide_count}). Otomatik tamamlama deneniyor..."
            )
            ideation_candidate = _match_ideation_candidate(candidate, ideation_payload, index)
            repaired_candidate = _repair_underfilled_candidate(
                candidate,
                llm=llm,
                transkript=transkript,
                critic_ozeti=critic_ozeti,
                trim_ozeti=trim_ozeti,
                metadata_ozeti=metadata_ozeti,
                broll_ozeti=broll_ozeti,
                ideation_candidate=ideation_candidate,
                debug_stem=debug_stem,
            )
            repaired_count = _candidate_slide_count(repaired_candidate)
            if repaired_count >= MIN_SLIDES:
                any_repaired = True
            else:
                logger.warning(
                    f"Carousel adayi otomatik tamamlama sonrasinda da yetersiz kaldi: {repaired_count}"
                )
            repaired_candidates.append(repaired_candidate)
            continue
        repaired_candidates.append(candidate)

    if not any_repaired:
        return data

    payload = dict(data)
    payload["carousel_candidates"] = repaired_candidates
    payload["selected_carousel_count"] = len([item for item in repaired_candidates if isinstance(item, dict)])
    return payload


def normalize_data(data: dict) -> dict:
    if not isinstance(data, dict):
        return {}

    candidates = []
    raw_candidates = data.get("carousel_candidates", [])
    if not isinstance(raw_candidates, list):
        raw_candidates = []

    for index, item in enumerate(raw_candidates, start=1):
        if not isinstance(item, dict):
            continue

        slides = []
        raw_slides = item.get("slides", [])
        if not isinstance(raw_slides, list):
            raw_slides = []

        for slide_index, slide in enumerate(raw_slides, start=1):
            if not isinstance(slide, dict):
                continue
            slide_goal_raw = normalize_whitespace(slide.get("slide_goal_tr", ""))
            caption_tr = normalize_whitespace(slide.get("caption_tr", ""))
            visual_plan_tr = normalize_whitespace(slide.get("visual_plan_tr", ""))
            if not caption_tr and not visual_plan_tr and not slide_goal_raw:
                continue
            user_facing_caption = caption_tr or slide_goal_raw or visual_plan_tr
            image_prompt = _normalize_image_prompt(slide.get("image_prompt_en", {}))
            slides.append(
                {
                    "slide_no": int(slide.get("slide_no", slide_index) or slide_index),
                    "slide_goal_tr": _fit_title(slide_goal_raw, SLIDE_TITLE_LIMIT),
                    "visual_plan_tr": visual_plan_tr,
                    "image_prompt_en": image_prompt,
                    "design_prompt_en": _normalize_carousel_prompt_fields(
                        slide.get("design_prompt_en", {}),
                        angle_tr=normalize_whitespace(item.get("angle_tr", "")),
                        primary_text_fallback=image_prompt.get("overlay_text", "") or user_facing_caption,
                        background_fallback=image_prompt.get("background", ""),
                        subject_fallback=image_prompt.get("people", ""),
                        supporting_fallback=GENERIC_SLIDE_SUPPORTING_EN,
                    ),
                    "caption_tr": _fit_body(user_facing_caption, SLIDE_BODY_LIMIT),
                }
            )

        if not slides:
            continue

        if len(slides) > MAX_SLIDES:
            slides = slides[:MAX_SLIDES]
        if len(slides) < MIN_SLIDES:
            logger.warning(
                f"Carousel adayi beklenen {MIN_SLIDES}-{MAX_SLIDES} slayt bandinin altinda geldi: {len(slides)}"
            )
            continue

        carousel_title = _fit_title(item.get("carousel_title_tr", ""), COVER_TITLE_LIMIT)
        if not carousel_title and slides:
            carousel_title = _fit_title(slides[0].get("slide_goal_tr", ""), COVER_TITLE_LIMIT)
        if not carousel_title:
            carousel_title = "Bunu herkes atliyor"

        candidates.append(
            {
                "rank": int(item.get("rank", index) or index),
                "viral_score": max(0, min(100, int(item.get("viral_score", 0) or 0))),
                "carousel_title_tr": carousel_title,
                "cover_subtitle_tr": _fallback_cover_subtitle(item.get("cover_subtitle_tr", "")),
                "cover_prompt_en": _normalize_carousel_prompt_fields(
                    item.get("cover_prompt_en", {}),
                    angle_tr=normalize_whitespace(item.get("angle_tr", "")),
                    primary_text_fallback=carousel_title,
                    supporting_fallback=GENERIC_COVER_SUPPORTING_EN,
                ),
                "angle_tr": normalize_whitespace(item.get("angle_tr", "")),
                "why_selected_tr": normalize_whitespace(item.get("why_selected_tr", "")),
                "audience_value_tr": normalize_whitespace(item.get("audience_value_tr", "")),
                "cta_potential_tr": _fallback_cta_potential(
                    item.get("cta_potential_tr", ""),
                    item.get("save_trigger_tr", ""),
                    item.get("audience_value_tr", ""),
                    item.get("cta_body_tr", ""),
                ),
                "cta_title_tr": _fallback_cta_title(item.get("cta_title_tr", "")),
                "cta_body_tr": _fallback_cta_body(item.get("cta_body_tr", "")),
                "cta_button_tr": _fallback_cta_button(item.get("cta_button_tr", "")),
                "cta_prompt_en": _normalize_carousel_prompt_fields(
                    item.get("cta_prompt_en", {}),
                    angle_tr=normalize_whitespace(item.get("angle_tr", "")),
                    primary_text_fallback=_fallback_cta_title(item.get("cta_title_tr", "")),
                    supporting_fallback=GENERIC_CTA_SUPPORTING_EN,
                ),
                "slides": slides,
            }
        )

    candidates.sort(key=lambda item: (-item.get("viral_score", 0), item.get("rank", 999)))
    if len(candidates) > MAX_CAROUSEL_CANDIDATES:
        candidates = candidates[:MAX_CAROUSEL_CANDIDATES]
    for rank, item in enumerate(candidates, start=1):
        item["rank"] = rank

    return {
        "why_this_many_carousels_tr": normalize_whitespace(data.get("why_this_many_carousels_tr", "")),
        "selected_carousel_count": len(candidates),
        "carousel_candidates": candidates,
    }


def build_report_text(girdi_stem: str, data: dict, model_adi: str) -> str:
    lines = [
        f"=== {girdi_stem} ICIN INSTAGRAM CAROUSEL FIKIRLERI ===",
        f"Kullanilan Model: {model_adi}",
        "",
    ]

    for candidate in data.get("carousel_candidates", []):
        slide_count = len(candidate.get("slides", []))
        lines.extend(
            [
                f"CAROUSEL #{candidate.get('rank', '')}",
                f"Viral Skoru: {candidate.get('viral_score', 0)}/100",
                f"Neden Tutar?: {candidate.get('why_selected_tr', '')}",
                f"CTA ihtimali Nedir?: {candidate.get('cta_potential_tr', '')}",
                f"Slayt Sayisi: {slide_count}",
                "",
            ]
        )

        for slide in candidate.get("slides", []):
            lines.extend(
                [
                    f"SLAYT #{slide.get('slide_no', '')}",
                    f"Amac: {slide.get('slide_goal_tr', '')}",
                    f"Goruntude ne olmali: {slide.get('visual_plan_tr', '')}",
                    "Goruntu promptu (TR):",
                    build_carousel_design_prompt(slide.get("design_prompt_en", {})),
                    "",
                ]
            )

        lines.extend(["=" * 60, ""])

    return "\n".join(lines).strip() + "\n"


def _safe_file_fragment(value: object, default: str) -> str:
    normalized = normalize_whitespace(value)
    cleaned = re.sub(r'[<>:"/\\\\|?*]+', "", normalized).strip().strip(".")
    cleaned = re.sub(r"\s+", "-", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or default


def _build_single_carousel_text(girdi_stem: str, candidate: dict, model_adi: str) -> str:
    slide_count = len(candidate.get("slides", []))
    lines = [
        f"CAROUSEL #{candidate.get('rank', '')}",
        f"Kaynak: {girdi_stem}",
        f"Kullanilan Model: {model_adi}",
        "",
        f"Viral Skoru: {candidate.get('viral_score', 0)}/100",
        f"Neden Tutar?: {candidate.get('why_selected_tr', '')}",
        f"CTA ihtimali Nedir?: {candidate.get('cta_potential_tr', '')}",
        f"Slayt Sayisi: {slide_count}",
        "",
    ]

    for slide in candidate.get("slides", []):
        lines.extend(
            [
                f"SLAYT #{slide.get('slide_no', '')}",
                f"Amac: {slide.get('slide_goal_tr', '')}",
                f"Goruntude ne olmali: {slide.get('visual_plan_tr', '')}",
                "Goruntu promptu (TR):",
                build_carousel_design_prompt(slide.get("design_prompt_en", {})),
                "",
            ]
        )

    return "\n".join(lines).strip() + "\n"


def _write_individual_carousel_txts(girdi_stem: str, data: dict, model_adi: str) -> None:
    if LEGACY_CAROUSEL_ITEM_TXT_DIR.exists():
        for old_file in LEGACY_CAROUSEL_ITEM_TXT_DIR.glob("*.txt"):
            old_file.unlink()
    CAROUSEL_ITEM_TXT_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in CAROUSEL_ITEM_TXT_DIR.glob("*.txt"):
        old_file.unlink()

    count = 0
    for candidate in data.get("carousel_candidates", []):
        rank = int(candidate.get("rank", count + 1) or (count + 1))
        title_fragment = _safe_file_fragment(candidate.get("carousel_title_tr", ""), "Baslik")
        path = CAROUSEL_ITEM_TXT_DIR / f"{rank:02d}-IG_Carousel-{title_fragment}.txt"
        path.write_text(_build_single_carousel_text(girdi_stem, candidate, model_adi), encoding="utf-8")
        count += 1

    logger.info(f"Tekil carousel TXT dosyalari kaydedildi: {count} adet")


def save_reports(girdi_dosyasi: Path, data: dict, model_adi: str) -> Tuple[Path, Optional[Path]]:
    json_yolu = stem_json_output_path(girdi_dosyasi.stem, "_instagram_carousel.json", group="instagram")
    txt_yolu = txt_output_path("instagram_carousel")

    with open(json_yolu, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)

    txt_yolu.unlink(missing_ok=True)
    _write_individual_carousel_txts(girdi_dosyasi.stem, data, model_adi)
    logger.info(f"Instagram carousel JSON kaydedildi: {json_yolu.name}")
    logger.info(f"Tekil carousel TXT klasoru guncellendi: {CAROUSEL_ITEM_TXT_DIR.name}")
    return json_yolu, None


def analyze(
    girdi_dosyasi: Path,
    llm: CentralLLM,
    critic_data: Optional[dict] = None,
    trim_data: Optional[dict] = None,
    metadata_data: Optional[dict] = None,
    broll_data: Optional[list] = None,
    draft_llm: Optional[CentralLLM] = None,
    prepared_transcript: Optional[str] = None,
) -> dict:
    timed_blocks = _timed_blocks_from_srt(girdi_dosyasi)
    transkript = normalize_whitespace(prepared_transcript)
    if not transkript:
        transkript = _build_adaptive_transcript(
            timed_blocks,
            metadata_data=metadata_data,
            critic_data=critic_data,
            trim_data=trim_data,
            broll_data=broll_data,
        )
    if not normalize_whitespace(transkript):
        logger.error("Carousel olusturmak icin transcript bulunamadi.")
        return {}

    metadata_data = metadata_data or load_related_json(girdi_dosyasi, "_metadata.json")
    critic_data = critic_data or load_related_json(girdi_dosyasi, "_video_critic.json")
    trim_data = trim_data or load_related_json(girdi_dosyasi, "_trim_suggestions.json")
    broll_data = broll_data or load_related_json(girdi_dosyasi, "_B_roll_fikirleri.json")
    if not prepared_transcript and timed_blocks:
        transkript = _build_adaptive_transcript(
            timed_blocks,
            metadata_data=metadata_data,
            critic_data=critic_data,
            trim_data=trim_data,
            broll_data=broll_data,
        )
        if not normalize_whitespace(transkript):
            logger.error("Carousel olusturmak icin transcript bulunamadi.")
            return {}

    metadata_ozeti = build_metadata_summary(metadata_data)
    critic_ozeti = build_critic_summary(critic_data)
    trim_ozeti = build_trim_summary(trim_data)
    broll_ozeti = build_broll_summary(broll_data)
    draft_engine = draft_llm or llm

    logger.info(
        f"Instagram carousel adaylari yaratiliyor... (aday hedefi: {MIN_CAROUSEL_CANDIDATES}-{MAX_CAROUSEL_CANDIDATES}, slide hedefi: {MIN_SLIDES}-{MAX_SLIDES})"
    )
    parsed = None
    debug_stem = girdi_dosyasi.stem
    ideation = None
    if _should_use_light_mode(timed_blocks, transkript):
        logger.info("Carousel icin hafif mod devrede: dogrudan tek gecis final uretim deneniyor.")
        parsed = _request_llm(
            llm,
            build_prompt(transkript, critic_ozeti, trim_ozeti, metadata_ozeti, broll_ozeti),
            profile="creative_ranker",
            profile_timeout=_carousel_profile_timeout(llm, "creative_ranker"),
            provider_retries=1,
            debug_stem=debug_stem,
            stage_label="light_mode",
        )

    if not parsed:
        ideation = _request_llm(
            draft_engine,
            build_ideation_prompt(transkript, critic_ozeti, trim_ozeti, metadata_ozeti, broll_ozeti),
            profile="creative_ideation",
            profile_timeout=_carousel_profile_timeout(draft_engine, "creative_ideation"),
            provider_retries=1,
            debug_stem=debug_stem,
            stage_label="ideation",
        )
        if isinstance(ideation, dict) and isinstance(ideation.get("carousel_candidates"), list):
            parsed = _request_llm(
                llm,
                build_selection_prompt(transkript, critic_ozeti, trim_ozeti, ideation, metadata_ozeti, broll_ozeti),
                profile="creative_ranker",
                profile_timeout=_carousel_profile_timeout(llm, "creative_ranker"),
                provider_retries=1,
                debug_stem=debug_stem,
                stage_label="selection",
            )
    if not parsed:
        parsed = _request_llm(
            llm,
            build_prompt(transkript, critic_ozeti, trim_ozeti, metadata_ozeti, broll_ozeti),
            profile="creative_ranker",
            profile_timeout=_carousel_profile_timeout(llm, "creative_ranker"),
            provider_retries=1,
            debug_stem=debug_stem,
            stage_label="fallback",
        )
    if not parsed:
        logger.error("Carousel cevabi parse edilemedi.")
        return {}
    parsed = _repair_underfilled_candidates(
        parsed,
        llm=llm,
        transkript=transkript,
        critic_ozeti=critic_ozeti,
        trim_ozeti=trim_ozeti,
        metadata_ozeti=metadata_ozeti,
        broll_ozeti=broll_ozeti,
        ideation_payload=ideation,
        debug_stem=debug_stem,
    )
    normalized = normalize_data(parsed)
    if normalized.get("carousel_candidates") and _needs_carousel_prompt_repair(normalized):
        logger.warning("Carousel prompt alanlarinda Turkce olmayan icerik bulundu; Turkce repair asamasi calistiriliyor.")
        normalized = normalize_data(_repair_carousel_prompt_language(normalized, llm))
        if _needs_carousel_prompt_repair(normalized):
            logger.warning("Carousel prompt repair sonrasinda bazi prompt alanlari hala Turkceye tam oturmadi.")
    if len(normalized.get("carousel_candidates", [])) < MIN_CAROUSEL_CANDIDATES and isinstance(ideation, dict):
        logger.warning(
            f"Carousel aday sayisi minimum hedefin altinda kaldi ({len(normalized.get('carousel_candidates', []))}/{MIN_CAROUSEL_CANDIDATES}). Secim asamasi tekrar deneniyor..."
        )
        retried = _request_llm(
            llm,
            build_selection_prompt(transkript, critic_ozeti, trim_ozeti, ideation, metadata_ozeti, broll_ozeti),
            profile="creative_ranker",
            profile_timeout=_carousel_profile_timeout(llm, "creative_ranker"),
            provider_retries=1,
            debug_stem=debug_stem,
            stage_label="selection_retry",
        )
        if retried:
            retried = _repair_underfilled_candidates(
                retried,
                llm=llm,
                transkript=transkript,
                critic_ozeti=critic_ozeti,
                trim_ozeti=trim_ozeti,
                metadata_ozeti=metadata_ozeti,
                broll_ozeti=broll_ozeti,
                ideation_payload=ideation,
                debug_stem=debug_stem,
            )
            retried_normalized = normalize_data(retried)
            if retried_normalized.get("carousel_candidates") and _needs_carousel_prompt_repair(retried_normalized):
                retried_normalized = normalize_data(_repair_carousel_prompt_language(retried_normalized, llm))
            if len(retried_normalized.get("carousel_candidates", [])) > len(normalized.get("carousel_candidates", [])):
                normalized = retried_normalized
    if not normalized.get("carousel_candidates"):
        logger.error(f"Carousel icin {MIN_SLIDES}-{MAX_SLIDES} slayt bandina uyan gecerli aday bulunamadi.")
        return {}
    return normalized


def run():
    print("\n" + "=" * 60)
    print("CAROUSEL FIKIR URETICI")
    print("=" * 60)

    girdi = select_primary_srt(logger, "Carousel Fikir Uretici")
    if not girdi:
        return

    use_recommended = prompt_module_llm_plan("301", needs_main=True, needs_smart=True)
    if use_recommended:
        saglayici_ana, model_adi_ana = get_module_recommended_llm_config("301", "main")
        saglayici, model_adi = get_module_recommended_llm_config("301", "smart")
        print_module_llm_choice_summary(
            "301",
            {"main": (saglayici_ana, model_adi_ana), "smart": (saglayici, model_adi)},
        )
    else:
        saglayici_ana, model_adi_ana = select_llm("main")
        saglayici, model_adi = select_llm("smart")
    draft_llm = CentralLLM(provider=saglayici_ana, model_name=model_adi_ana)
    llm = CentralLLM(provider=saglayici, model_name=model_adi)

    data = analyze(girdi, llm, draft_llm=draft_llm)
    if not data:
        return logger.error("❌ Carousel uretilemedi.")

    save_reports(girdi, data, f"Draft: {model_adi_ana} | Final: {model_adi}")
    logger.info("🎉 Instagram carousel olusturma islemi tamamlandi.")


def run_automatic(
    girdi_dosyasi: Path,
    llm: CentralLLM,
    critic_data: Optional[dict] = None,
    trim_data: Optional[dict] = None,
    metadata_data: Optional[dict] = None,
    broll_data: Optional[list] = None,
    draft_llm: Optional[CentralLLM] = None,
    prepared_transcript: Optional[str] = None,
) -> Optional[dict]:
    logger.info(f"🔄 OTOMASYON: {girdi_dosyasi.name} icin Instagram carousel uretiliyor...")
    data = analyze(
        girdi_dosyasi,
        llm,
        critic_data=critic_data,
        trim_data=trim_data,
        metadata_data=metadata_data,
        broll_data=broll_data,
        draft_llm=draft_llm,
        prepared_transcript=prepared_transcript,
    )
    if not data:
        logger.error("❌ Otomatik Instagram carousel uretimi basarisiz oldu.")
        return None

    json_yolu, txt_yolu = save_reports(girdi_dosyasi, data, llm.model_name)
    return {
        "data": data,
        "json_path": json_yolu,
        "txt_path": txt_yolu,
    }
