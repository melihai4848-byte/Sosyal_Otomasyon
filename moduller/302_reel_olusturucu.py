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
from moduller.output_paths import grouped_output_path, json_output_path, txt_output_path
from moduller.social_media_utils import (
    build_broll_summary,
    build_critic_summary,
    build_metadata_summary,
    build_trim_summary,
    load_related_json,
    select_metadata_language,
    select_primary_srt,
)
from moduller.srt_utils import parse_srt_blocks, read_srt_file
from moduller.video_edit_utils import (
    duration_between,
    seconds_to_timestamp,
    timestamp_to_seconds,
    total_duration_from_segments,
)
from moduller.youtube_llm_profiles import call_with_youtube_profile

_VIDEO_CRITIC_MODULE = load_numbered_module("202_video_critic.py")
extract_json_response = _VIDEO_CRITIC_MODULE.extract_json_response
normalize_whitespace = _VIDEO_CRITIC_MODULE.normalize_whitespace

logger = get_logger("reels")
LLM_RETRIES = 3
MIN_REEL_DURATION_SECONDS = 30
MAX_REEL_DURATION_SECONDS = 70
MIN_SEGMENT_DURATION_SECONDS = 3
PREFERRED_MIN_REEL_SEGMENTS = 2
PREFERRED_MAX_REEL_SEGMENTS = 4
DEFAULT_MIN_REEL_IDEA_COUNT = 5
DEFAULT_TARGET_REEL_IDEA_COUNT = 8
MAX_REEL_IDEA_COUNT = 8
ADAPTIVE_TRANSCRIPT_MAX_CHARS = 35000
ADAPTIVE_TRANSCRIPT_TARGET_ANCHORS = 8
ADAPTIVE_TRANSCRIPT_WINDOW_RADIUS = 1
REELS_LIGHT_MODE_MAX_SECONDS = 150
REELS_LIGHT_MODE_MAX_BLOCKS = 16
REELS_LIGHT_MODE_MAX_TRANSCRIPT_CHARS = 1800
REEL_PROMPT_FOOTER = (
    "Metin buyuk, mobilde rahat okunur olsun, 2-3 satiri gecmesin ve guclu gorsel hiyerarsiyle ana mesaja "
    "odaklansin."
)
REEL_FORMAT_SPEC_TR = "Format/Cozunurluk: 9:16 (dikey) en-boy orani, 1080x1920."
DEFAULT_REEL_GOAL_EN = "Ana hook'u ilk bakista sattiran dikkat cekici bir Instagram Reel kapak tasarimi."
DEFAULT_REEL_BACKGROUND_EN = "Temiz, modern ve sosyal medya estetikli acik tonlu bir arka plan."
DEFAULT_REEL_SUBJECT_EN = "Ana hook ile dogrudan baglantili gercekci bir kisi, nesne veya sahne."
DEFAULT_REEL_SUPPORTING_ELEMENTS_EN = "Guclu sayisal vurgu, ince oklar, kucuk ikonlar ve tek bir dikkat kutusu."
DEFAULT_REEL_STYLE_EN = "Modern, temiz, yuksek kontrastli ve mobilde kolay okunur bir sosyal medya dili."
REELS_DEBUG_DIR = grouped_output_path("instagram", "_llm_debug")
REELS_ITEM_TXT_DIR = grouped_output_path("instagram", "302_IG-Reels_Fikirleri")
LEGACY_REELS_ITEM_TXT_DIR = REELS_ITEM_TXT_DIR.parent / "_tekil_reels_txt"
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
    "maas",
    "almanya",
}


def _safe_debug_label(value: object, default: str = "stage") -> str:
    text = normalize_whitespace(value).casefold()
    cleaned = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return cleaned or default


def _debug_response_path(debug_stem: str, stage_label: str, kind: str) -> Path:
    REELS_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    stem = _safe_debug_label(debug_stem, "reels")
    stage = _safe_debug_label(stage_label, "stage")
    suffix = _safe_debug_label(kind, "raw")
    return REELS_DEBUG_DIR / f"{stem}_{stage}_{suffix}.txt"


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
    logger.warning(f"Reels debug cevabi kaydedildi: {path.name}")
    return path


def _json_repair_schema_hint(stage_label: str) -> str:
    stage = _safe_debug_label(stage_label, "stage")
    if stage == "ideation":
        return """
Expected root shape:
- `why_this_many_reels_tr`: string
- `reel_candidates`: array

Each reel candidate should remain an object that includes:
- `reel_title_tr`
- `virality_angle_tr`
- `why_selected_tr`
- `description_direction_tr`
- `cover_direction_tr`
- `rough_segments`
""".strip()

    return """
Expected root shape:
- `why_this_many_reels_tr`: string
- `selected_reel_count`: integer
- `reel_candidates`: array

Each reel candidate should remain an object that includes:
- `rank`
- `viral_score`
- `reel_title_tr`
- `why_selected_tr`
- `description_tr`
- `cover_plan_tr`
- `cover_prompt_en`
- `cover_design_prompt_en`
- `segments`
- `editing_notes_tr`
""".strip()


def _build_invalid_json_repair_prompt(raw_response: str, stage_label: str) -> str:
    schema_hint = _json_repair_schema_hint(stage_label)
    return f"""
You are repairing malformed Instagram reels JSON produced by another model.

Task:
Convert the raw assistant output below into one valid JSON object.

Rules:
- Return only JSON. No markdown fences. No explanation.
- Preserve the original meaning and structure as much as possible.
- Keep all audience-facing fields in Turkish.
- Keep the legacy `_en` prompt fields in Turkish too; only the key names stay the same.
- Remove brainstorm prose, duplicate wrappers, and trailing commentary outside JSON.
- Fix invalid commas, brackets, quotes, and escaping.
- If a field is obviously cut off, complete it minimally and conservatively so the JSON becomes valid.
- Do not invent extra reel candidates unless required to preserve valid structure.

Schema guidance:
{schema_hint}

RAW OUTPUT:
{raw_response}
""".strip()


def _build_strict_json_retry_prompt(original_prompt: str, stage_label: str) -> str:
    return f"""
The previous response for the Instagram reels `{stage_label}` stage was not valid JSON.

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
        )
    except Exception as exc:
        logger.warning(f"Reels JSON repair istegi basarisiz oldu ({stage_label} / deneme {attempt_no}): {exc}")
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
        logger.info(f"Reels JSON repair basarili oldu: {stage_label} (deneme {attempt_no})")
    else:
        logger.warning(f"Reels JSON repair gecersiz cevap dondu: {stage_label} (deneme {attempt_no})")
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
        )
    except Exception as exc:
        logger.warning(f"Reels strict_json retry basarisiz oldu ({stage_label} / deneme {attempt_no}): {exc}")
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
        logger.info(f"Reels strict_json retry basarili oldu: {stage_label} (deneme {attempt_no})")
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

    logger.warning(f"Reels strict_json retry da parse edilemedi: {stage_label} (deneme {attempt_no})")
    return None


def _parse_timecode(value: str) -> float:
    text = str(value or "").strip().replace(",", ".")
    if not text:
        return 0.0

    if re.match(r"^\d{2}:\d{2}$", text):
        minutes, seconds = text.split(":")
        return float(int(minutes) * 60 + int(seconds))

    match = re.match(r"(?:(\d+):)?(\d{2}):(\d{2})(?:\.(\d+))?$", text)
    if not match:
        return 0.0

    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    millis_raw = match.group(4) or "0"
    millis = int((millis_raw + "000")[:3])
    return float(hours * 3600 + minutes * 60 + seconds) + (millis / 1000.0)


def _seconds_to_mmss(value: float) -> str:
    total_seconds = max(0, int(value))
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


def _parse_timing_line(timing_line: str) -> tuple[float, float]:
    if "-->" not in str(timing_line):
        return 0.0, 0.0
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
) -> list[float]:
    moments: list[float] = []

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


def _find_closest_block_index(start_times: list[float], target_seconds: float) -> Optional[int]:
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

    start_times = [float(item["start_sec"]) for item in timed_blocks]
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
    total_duration_seconds = max(float(timed_blocks[-1]["end_sec"]), 0.0)
    if total_duration_seconds <= REELS_LIGHT_MODE_MAX_SECONDS:
        return True
    if len(timed_blocks) <= REELS_LIGHT_MODE_MAX_BLOCKS:
        return True
    if len(transcript_text) <= REELS_LIGHT_MODE_MAX_TRANSCRIPT_CHARS:
        return True
    return False


def _routing_decision(critic_data: Optional[dict]) -> tuple[Optional[bool], str]:
    if not isinstance(critic_data, dict):
        return None, ""
    routing = critic_data.get("routing_decisions", {})
    if not isinstance(routing, dict):
        return None, ""
    item = routing.get("reels_shorts", {})
    if not isinstance(item, dict):
        return None, ""
    if "run" not in item:
        return None, normalize_whitespace(item.get("reason", ""))
    return bool(item.get("run")), normalize_whitespace(item.get("reason", ""))


def _request_llm(
    llm: CentralLLM,
    prompt: str,
    retries: int = LLM_RETRIES,
    profile: str = "creative_ranker",
    debug_stem: str = "reels",
    stage_label: str = "response",
) -> Optional[dict]:
    for deneme in range(1, retries + 1):
        try:
            cevap = call_with_youtube_profile(llm, prompt, profile=profile)
            parsed = extract_json_response(cevap, logger_override=logger, log_errors=False)
            if parsed:
                return parsed
            _save_debug_response(
                debug_stem,
                stage_label,
                f"attempt_{deneme}_invalid_raw",
                cevap,
                note="Ilk ham cevap parse edilemedi",
            )
            repaired = _repair_invalid_json_response(
                llm,
                str(cevap or ""),
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
            logger.error("LLM cevabından geçerli JSON çıkarılamadı.")
        except Exception as exc:
            logger.warning(f"Reels LLM hatasi ({deneme}/{retries}): {exc}")
        time.sleep(deneme)
    return None


def _safe_reel_count(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default)
    return max(1, min(MAX_REEL_IDEA_COUNT, parsed))


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


def _default_reel_goal_en(reel_title_tr: str) -> str:
    title = normalize_whitespace(reel_title_tr)
    if not title:
        return DEFAULT_REEL_GOAL_EN
    return f'"{title}" fikrinin ana hookunu ilk bakista sattiran dikkat cekici bir Instagram Reel kapak tasarimi.'


def _normalize_reel_design_prompt_fields(
    item: object,
    *,
    reel_title_tr: str,
    primary_text_fallback: str,
    background_fallback: str = "",
    subject_fallback: str = "",
    supporting_fallback: str = "",
) -> dict:
    payload = item if isinstance(item, dict) else {}
    return {
        "goal_en": _first_non_empty(payload.get("goal_en", ""), _default_reel_goal_en(reel_title_tr)),
        "background_en": _first_non_empty(payload.get("background_en", ""), background_fallback, DEFAULT_REEL_BACKGROUND_EN),
        "subject_en": _first_non_empty(
            payload.get("subject_en", ""),
            payload.get("people_en", ""),
            subject_fallback,
            DEFAULT_REEL_SUBJECT_EN,
        ),
        "primary_text_en": _first_non_empty(payload.get("primary_text_en", ""), primary_text_fallback),
        "supporting_elements_en": _first_non_empty(
            payload.get("supporting_elements_en", ""),
            payload.get("supporting_en", ""),
            supporting_fallback,
            DEFAULT_REEL_SUPPORTING_ELEMENTS_EN,
        ),
        "style_mood_en": _first_non_empty(
            payload.get("style_mood_en", ""),
            payload.get("style_en", ""),
            DEFAULT_REEL_STYLE_EN,
        ),
    }


def build_reel_cover_design_prompt(prompt_fields: dict) -> str:
    lines = [
        "Bir Instagram Reel kapak tasarimi olustur.",
        f"Amac: {prompt_fields.get('goal_en', '')}",
        f"Arka Plan: {prompt_fields.get('background_en', '')}",
        f"Ozne: {prompt_fields.get('subject_en', '')}",
        f'Ana Metin: "{prompt_fields.get("primary_text_en", "")}"',
        f"Destekleyici Unsurlar: {prompt_fields.get('supporting_elements_en', '')}",
        f"Stil/Ruh Hali: {prompt_fields.get('style_mood_en', '')}",
        REEL_FORMAT_SPEC_TR,
        REEL_PROMPT_FOOTER,
    ]
    return "\n".join(line for line in lines if normalize_whitespace(line))


def _iter_reel_prompt_texts(data: dict):
    for reel in data.get("reel_candidates", []):
        if not isinstance(reel, dict):
            continue
        cover_prompt = reel.get("cover_prompt_en", {})
        if isinstance(cover_prompt, dict):
            for value in cover_prompt.values():
                yield value
        cover_design_prompt = reel.get("cover_design_prompt_en", {})
        if isinstance(cover_design_prompt, dict):
            for value in cover_design_prompt.values():
                yield value


def _needs_reel_prompt_repair(data: dict) -> bool:
    return any(_needs_turkish_prompt_repair(value) for value in _iter_reel_prompt_texts(data))


def _build_reel_prompt_language_repair_prompt(data: dict) -> str:
    candidate_text = json.dumps(data, ensure_ascii=False, indent=2)
    return f"""
You are repairing Instagram Reels JSON.

Task:
Rewrite only the image-generation prompt fields into fluent Turkish and enrich the reel cover prompt so it matches a structured Turkish design prompt.

Rules:
- Return only one valid JSON object.
- Keep every `*_tr` field in Turkish.
- Keep ranks, scores, ordering, durations, segments, and editing notes unchanged unless needed to preserve valid JSON.
- Rewrite these fields in Turkish only. Keep the legacy `_en` key names unchanged:
  - `cover_prompt_en.background`
  - `cover_prompt_en.people`
  - `cover_prompt_en.overlay_text`
  - `cover_design_prompt_en.goal_en`
  - `cover_design_prompt_en.background_en`
  - `cover_design_prompt_en.subject_en`
  - `cover_design_prompt_en.primary_text_en`
  - `cover_design_prompt_en.supporting_elements_en`
  - `cover_design_prompt_en.style_mood_en`
- Do not leave English sentences inside those prompt fields unless a brand name or technical terim gercekten gerekiyorsa.
- Preserve the meaning of the existing content while rewriting the prompt content into natural Turkish.
- The structured design prompt must align with this format:
  Bir Instagram Reel kapak tasarimi olustur.
  Amac: ...
  Arka Plan: ...
  Ozne: ...
  Ana Metin: "..."
  Destekleyici Unsurlar: ...
  Stil/Ruh Hali: ...
  {REEL_FORMAT_SPEC_TR}
  {REEL_PROMPT_FOOTER}

JSON TO REPAIR:
{candidate_text}
""".strip()


def _repair_reel_prompt_language(data: dict, llm: CentralLLM) -> dict:
    repaired = _request_llm(
        llm,
        _build_reel_prompt_language_repair_prompt(data),
        retries=1,
        profile="analytic_json",
    )
    if isinstance(repaired, dict) and isinstance(repaired.get("reel_candidates"), list):
        return repaired
    return data


def get_min_reel_idea_count() -> int:
    return _safe_reel_count(
        os.getenv("INSTAGRAM_REELS_MIN_IDEA_COUNT", DEFAULT_MIN_REEL_IDEA_COUNT),
        DEFAULT_MIN_REEL_IDEA_COUNT,
    )


def get_default_reel_target_count() -> int:
    return _safe_reel_count(
        os.getenv("INSTAGRAM_REELS_TARGET_IDEA_COUNT", DEFAULT_TARGET_REEL_IDEA_COUNT),
        DEFAULT_TARGET_REEL_IDEA_COUNT,
    )


def _resolve_reel_targets(reel_sayisi: int) -> tuple[int, int]:
    env_minimum = get_min_reel_idea_count()
    requested_cap = _safe_reel_count(reel_sayisi or env_minimum, env_minimum)
    reel_cap = max(env_minimum, requested_cap)
    return env_minimum, reel_cap


def build_ideation_prompt(
    srt_metni: str,
    min_reel_count: int,
    reel_cap: int,
    critic_ozeti: str = "Video Elestirmeni verisi yok.",
    trim_ozeti: str = "Trim verisi yok.",
    metadata_ozeti: str = "YouTube metadata verisi yok.",
    broll_ozeti: str = "B-roll verisi yok.",
    routing_notu: str = "",
) -> str:
    return f"""
Sen Instagram Reels stratejisti, short-form retention editoru ve viral packaging kreatif direktorusun.

Gorevin:
Uzun videonun transcriptini oku, videoyu iyice anla ve bundan kac adet anlamli Reel cikmasi gerektigine karar ver.

KRITIK KURALLAR:
- Final JSON'dan once kisa bir <brainstorm> bolumunde hangi anlarin neden reel olmasi gerektigini, hangi hook tiplerinin daha viral olabilecegini ve hangi kesitlerin birbirini tamamladigini dusun.
- <brainstorm> bolumunde yalnizca duz metin kullan; {{ }}, [ ] veya kod blogu kullanma.
- Final cevabinin sonunda tek bir gecerli JSON nesnesi ya da tek bir ```json``` blogu ver.
- Tum Turkce alanlar yalnizca Turkce olsun.
- Transcriptin zenginligine gore kac reel adayi gerektigine sen karar ver ama {reel_cap} adedi gecme.
- Minimum hedefin {min_reel_count} farkli reel adayi cikarmak olsun.
- Malzeme elverisliyse {min_reel_count} ile {reel_cap} arasinda reel adayi uret; sayiyi gereksiz yere dusuk tutma.
- Her reel icin neden secildigini, virality acisini ve cover mantigini dusun.
- Kesit zamanlari yalnizca transcriptte gecen gercek zaman araliklarindan olussun.
- Her reelin toplam suresi minimum {MIN_REEL_DURATION_SECONDS} saniye, maksimum {MAX_REEL_DURATION_SECONDS} saniye olmali.
- Tercihen her reel {PREFERRED_MIN_REEL_SEGMENTS}-{PREFERRED_MAX_REEL_SEGMENTS} anlamli segmentten olussun; boylece hook, gelisme ve payoff hissi korunsun.
- Eger tek bir blok zaten cok guclu, akici ve kendi basina retention tasiyorsa tek segment kullanmak serbest.
- Cok fazla mikro kesit kullanma; ancak gercekten gerekli degilse 5+ parca yapma.
- Bir aday dogal halinde kisa ya da uzun kalirsa cope atma; gerekli yerlerde kisa lead-in/lead-out ekleyerek veya zayif kenarlari kirparak 30-70 saniye bandina sigdir.

JSON SEMASI:
{{
  "why_this_many_reels_tr": "",
  "reel_candidates": [
    {{
      "reel_title_tr": "",
      "virality_angle_tr": "",
      "why_selected_tr": "",
      "description_direction_tr": "",
      "cover_direction_tr": "",
      "rough_segments": [
        {{
          "start": "00:01:10,000",
          "end": "00:01:18,000",
          "purpose_tr": ""
        }}
      ]
    }}
  ]
}}

Video Elestirmeni Ozeti:
{critic_ozeti}

Trim Ozeti:
{trim_ozeti}

YouTube Metadata Ozeti:
{metadata_ozeti}

B-Roll Ozeti:
{broll_ozeti}

Routing Notu:
{routing_notu or 'Reels adimi normal modda calisiyor.'}

Altyazi / Transcript:
{srt_metni}
""".strip()


def build_selection_prompt(
    srt_metni: str,
    min_reel_count: int,
    reel_cap: int,
    candidate_payload: dict,
    critic_ozeti: str = "Video Elestirmeni verisi yok.",
    trim_ozeti: str = "Trim verisi yok.",
    metadata_ozeti: str = "YouTube metadata verisi yok.",
    broll_ozeti: str = "B-roll verisi yok.",
    routing_notu: str = "",
) -> str:
    candidate_text = json.dumps(candidate_payload, ensure_ascii=False, indent=2)
    return f"""
Sen Instagram Reels stratejisti, jump-cut edit planlayicisi ve final short-form editorusun.

Gorevin:
Verilen reel adaylarini incele, en guclu olanlari sec, viral olma ihtimallerine gore skorla ve final Reel paketlerini olustur.

KRITIK KURALLAR:
- Once adaylarin guclu, zayif ve fazla benzer yonlerini kisaca analiz edebilirsin.
- Analiz kismini yalnizca duz metin veya <brainstorm> etiketiyle ver; {{ }}, [ ] veya kod blogu kullanma.
- Analizden sonra cevabini tek bir gecerli JSON nesnesi ya da tek bir ```json``` blogu olarak bitir.
- Tum Turkce alanlar yalnizca Turkce olsun.
- `cover_prompt_en.background`, `cover_prompt_en.people` ve `cover_prompt_en.overlay_text` alanlari akici ve net Turkce olsun.
- `cover_design_prompt_en.goal_en`, `background_en`, `subject_en`, `primary_text_en`, `supporting_elements_en`, `style_mood_en` alanlari da Turkce olsun.
- `_en` eki yalnizca eski alan ismidir; bu alanlarin degerlerini Turkce yaz.
- Cover promptlari DALL-E, Midjourney, Nano Banana Pro gibi sistemlerin anlayacagi kadar detayli yaz.
- Transcriptin zenginligine gore kac reel adayi gerektigine sen karar ver ama {reel_cap} adedi gecme.
- Minimum hedefin {min_reel_count} farkli reel adayi secmek olsun.
- Malzeme elverisliyse {min_reel_count} ile {reel_cap} arasinda final reel adayi cikart; sayiyi gereksiz yere dusuk tutma.
- Her reel icin sunlari ver:
  1. Neden secildi?
  2. Reel altina yazilacak aciklama metni
  3. Kapak fotografi mantigi, 3 katmanli Turkce prompt ve reel'e ozel tam gorsel uretim promptu
  4. Ana videodan kesilecek kisimlar
- Tam gorsel uretim promptu su formatta olsun:
  Bir Instagram Reel kapak tasarimi olustur.
  Amac: ...
  Arka Plan: ...
  Ozne: ...
  Ana Metin: "..."
  Destekleyici Unsurlar: ...
  Stil/Ruh Hali: ...
- Her promptta format/cozunurluk bilgisini acikca belirt:
  {REEL_FORMAT_SPEC_TR}
- Asagidaki cümleyi prompt mantigina mutlaka yedir:
  {REEL_PROMPT_FOOTER}
- Kesit zamanlari yalnizca transcriptte gecen gercek zaman araliklarindan olussun.
- Viral olma ihtimaline gore her adayi skorla ve sirala.
- Her reelin toplam suresi minimum {MIN_REEL_DURATION_SECONDS} saniye, maksimum {MAX_REEL_DURATION_SECONDS} saniye olmali.
- Tercihen her reel {PREFERRED_MIN_REEL_SEGMENTS}-{PREFERRED_MAX_REEL_SEGMENTS} anlamli segmentten olussun; boylece hook, gelisme ve payoff hissi korunsun.
- Eger tek bir blok zaten cok guclu, akici ve kendi basina retention tasiyorsa tek segment kullanmak serbest.
- Cok fazla mikro kesit kullanma; ancak gercekten gerekli degilse 5+ parca yapma.
- Bir aday dogal halinde kisa ya da uzun kalirsa cope atma; gerekli yerlerde kisa lead-in/lead-out ekleyerek veya zayif kenarlari kirparak 30-70 saniye bandina sigdir.

JSON SEMASI:
{{
  "why_this_many_reels_tr": "",
  "selected_reel_count": 0,
  "reel_candidates": [
    {{
      "rank": 1,
      "viral_score": 95,
      "reel_title_tr": "",
      "why_selected_tr": "",
      "description_tr": "",
      "cover_plan_tr": "",
      "cover_prompt_en": {{
        "background": "",
        "people": "",
        "overlay_text": ""
      }},
      "cover_design_prompt_en": {{
        "goal_en": "",
        "background_en": "",
        "subject_en": "",
        "primary_text_en": "",
        "supporting_elements_en": "",
        "style_mood_en": ""
      }},
      "segments": [
        {{
          "order": 1,
          "start": "00:01:10,000",
          "end": "00:01:18,000",
          "text": "",
          "purpose_tr": "",
          "transition_hint_tr": ""
        }}
      ],
      "editing_notes_tr": ["", ""]
    }}
  ]
}}

Aday Havuzu:
{candidate_text}

Video Elestirmeni Ozeti:
{critic_ozeti}

Trim Ozeti:
{trim_ozeti}

YouTube Metadata Ozeti:
{metadata_ozeti}

B-Roll Ozeti:
{broll_ozeti}

Routing Notu:
{routing_notu or 'Reels adimi normal modda calisiyor.'}

Altyazi / Transcript:
{srt_metni}
""".strip()


def build_prompt(
    srt_metni: str,
    min_reel_count: int,
    reel_cap: int,
    critic_ozeti: str = "Video Elestirmeni verisi yok.",
    trim_ozeti: str = "Trim verisi yok.",
    metadata_ozeti: str = "YouTube metadata verisi yok.",
    broll_ozeti: str = "B-roll verisi yok.",
    routing_notu: str = "",
) -> str:
    return f"""
Sen Instagram Reels stratejisti, jump-cut edit planlayicisi ve viral short-form producer'sin.

Gorevin:
Uzun videonun transcriptinden en guclu Reel adaylarini sec, viral olma ihtimallerine gore skorla ve final detaylarini uret.

KRITIK KURALLAR:
- Final JSON'dan once kisa bir <brainstorm> bolumunde hangi kesitlerin neden daha guclu oldugunu dusun.
- <brainstorm> bolumunde yalnizca duz metin kullan; {{ }}, [ ] veya kod blogu kullanma.
- Analizden sonra cevabini tek bir gecerli JSON nesnesi ya da tek bir ```json``` blogu olarak bitir.
- Tum Turkce alanlar yalnizca Turkce olsun.
- `cover_prompt_en.background`, `cover_prompt_en.people` ve `cover_prompt_en.overlay_text` alanlari akici ve net Turkce olsun.
- `cover_design_prompt_en.goal_en`, `background_en`, `subject_en`, `primary_text_en`, `supporting_elements_en`, `style_mood_en` alanlari da Turkce olsun.
- `_en` eki yalnizca eski alan ismidir; bu alanlarin degerlerini Turkce yaz.
- Cover promptlari DALL-E, Midjourney, Nano Banana Pro gibi sistemlerin anlayacagi kadar detayli yaz.
- Transcriptin zenginligine gore kac reel adayi gerektigine sen karar ver ama {reel_cap} adedi gecme.
- Minimum hedefin {min_reel_count} farkli reel adayi cikarmak olsun.
- Malzeme elverisliyse {min_reel_count} ile {reel_cap} arasinda reel adayi uret; sayiyi gereksiz yere dusuk tutma.
- Her reel icin neden secildigini, aciklamayi, kapak promptunu, reel'e ozel tam gorsel uretim promptunu ve kesilecek kisimlari ver.
- Tam gorsel uretim promptu su formatta olsun:
  Bir Instagram Reel kapak tasarimi olustur.
  Amac: ...
  Arka Plan: ...
  Ozne: ...
  Ana Metin: "..."
  Destekleyici Unsurlar: ...
  Stil/Ruh Hali: ...
- Her promptta format/cozunurluk bilgisini acikca belirt:
  {REEL_FORMAT_SPEC_TR}
- Asagidaki cümleyi prompt mantigina mutlaka ekle:
  {REEL_PROMPT_FOOTER}
- Viral olma ihtimaline gore her adayi skorla ve sirala.
- Her reelin toplam suresi minimum {MIN_REEL_DURATION_SECONDS} saniye, maksimum {MAX_REEL_DURATION_SECONDS} saniye olmali.
- Tercihen her reel {PREFERRED_MIN_REEL_SEGMENTS}-{PREFERRED_MAX_REEL_SEGMENTS} anlamli segmentten olussun; boylece hook, gelisme ve payoff hissi korunsun.
- Eger tek bir blok zaten cok guclu, akici ve kendi basina retention tasiyorsa tek segment kullanmak serbest.
- Cok fazla mikro kesit kullanma; ancak gercekten gerekli degilse 5+ parca yapma.
- Bir aday dogal halinde kisa ya da uzun kalirsa cope atma; gerekli yerlerde kisa lead-in/lead-out ekleyerek veya zayif kenarlari kirparak 30-70 saniye bandina sigdir.

JSON SEMASI:
{{
  "why_this_many_reels_tr": "",
  "selected_reel_count": 0,
  "reel_candidates": [
    {{
      "rank": 1,
      "viral_score": 95,
      "reel_title_tr": "",
      "why_selected_tr": "",
      "description_tr": "",
      "cover_plan_tr": "",
      "cover_prompt_en": {{
        "background": "",
        "people": "",
        "overlay_text": ""
      }},
      "cover_design_prompt_en": {{
        "goal_en": "",
        "background_en": "",
        "subject_en": "",
        "primary_text_en": "",
        "supporting_elements_en": "",
        "style_mood_en": ""
      }},
      "segments": [
        {{
          "order": 1,
          "start": "00:01:10,000",
          "end": "00:01:18,000",
          "text": "",
          "purpose_tr": "",
          "transition_hint_tr": ""
        }}
      ],
      "editing_notes_tr": ["", ""]
    }}
  ]
}}

Video Elestirmeni Ozeti:
{critic_ozeti}

Trim Ozeti:
{trim_ozeti}

YouTube Metadata Ozeti:
{metadata_ozeti}

B-Roll Ozeti:
{broll_ozeti}

Routing Notu:
{routing_notu or 'Reels adimi normal modda calisiyor.'}

Altyazi / Transcript:
{srt_metni}
""".strip()


def _normalize_segment(item: dict, order: int) -> Optional[dict]:
    if not isinstance(item, dict):
        return None

    start = normalize_whitespace(item.get("start", ""))
    end = normalize_whitespace(item.get("end", ""))
    if not start or not end:
        return None

    duration_seconds = round(duration_between(start, end), 2)
    if duration_seconds <= 0:
        return None

    return {
        "order": int(item.get("order", order) or order),
        "start": start,
        "end": end,
        "text": normalize_whitespace(item.get("text", "")),
        "purpose_tr": normalize_whitespace(item.get("purpose_tr", item.get("purpose", ""))),
        "transition_hint_tr": normalize_whitespace(item.get("transition_hint_tr", item.get("transition_hint", ""))),
        "duration_seconds": duration_seconds,
    }


def _normalize_cover_prompt(item: object) -> dict:
    prompt = item if isinstance(item, dict) else {}
    return {
        "background": normalize_whitespace(prompt.get("background", "")),
        "people": normalize_whitespace(prompt.get("people", "")),
        "overlay_text": normalize_whitespace(prompt.get("overlay_text", "")),
    }


def _clone_segment(segment: dict, start_seconds: float, end_seconds: float, order: int) -> Optional[dict]:
    duration_seconds = round(max(0.0, float(end_seconds) - float(start_seconds)), 2)
    if duration_seconds <= 0:
        return None
    cloned = dict(segment)
    cloned["order"] = order
    cloned["start"] = seconds_to_timestamp(start_seconds)
    cloned["end"] = seconds_to_timestamp(end_seconds)
    cloned["duration_seconds"] = duration_seconds
    return cloned


def _rebuild_segments(segments: list[dict]) -> list[dict]:
    sortable = []
    for segment in segments:
        start_seconds = timestamp_to_seconds(segment.get("start", ""))
        end_seconds = timestamp_to_seconds(segment.get("end", ""))
        if end_seconds <= start_seconds:
            continue
        sortable.append((start_seconds, end_seconds, segment))

    rebuilt = []
    for order, (start_seconds, end_seconds, segment) in enumerate(
        sorted(sortable, key=lambda item: (item[0], item[1])),
        start=1,
    ):
        cloned = _clone_segment(segment, start_seconds, end_seconds, order)
        if cloned:
            rebuilt.append(cloned)
    return rebuilt


def _fit_segments_to_duration_window(segments: list[dict]) -> tuple[list[dict], float, float, str]:
    fitted = _rebuild_segments(segments)
    if not fitted:
        return [], 0.0, 0.0, ""

    raw_duration = round(total_duration_from_segments(fitted), 2)
    adjustment_note = ""
    mutable = []
    for segment in fitted:
        mutable.append(
            {
                **segment,
                "_start_seconds": timestamp_to_seconds(segment.get("start", "")),
                "_end_seconds": timestamp_to_seconds(segment.get("end", "")),
            }
        )

    if raw_duration > MAX_REEL_DURATION_SECONDS:
        excess = raw_duration - MAX_REEL_DURATION_SECONDS
        for segment in reversed(mutable):
            current_duration = segment["_end_seconds"] - segment["_start_seconds"]
            available_trim = max(0.0, current_duration - MIN_SEGMENT_DURATION_SECONDS)
            if available_trim <= 0:
                continue
            trim_amount = min(excess, available_trim)
            segment["_end_seconds"] -= trim_amount
            excess -= trim_amount
            if excess <= 0:
                break

        if excess > 0:
            for segment in mutable:
                current_duration = segment["_end_seconds"] - segment["_start_seconds"]
                available_trim = max(0.0, current_duration - MIN_SEGMENT_DURATION_SECONDS)
                if available_trim <= 0:
                    continue
                trim_amount = min(excess, available_trim)
                segment["_start_seconds"] += trim_amount
                excess -= trim_amount
                if excess <= 0:
                    break

        adjustment_note = (
            "Ham aday uzun geldigi icin zayif kenarlari kirpilarak 30-70 saniye bandina uyarlandi."
        )
    elif raw_duration < MIN_REEL_DURATION_SECONDS:
        shortfall = MIN_REEL_DURATION_SECONDS - raw_duration
        first_segment = mutable[0]
        left_padding = min(shortfall / 2, max(0.0, first_segment["_start_seconds"]))
        first_segment["_start_seconds"] -= left_padding
        shortfall -= left_padding
        mutable[-1]["_end_seconds"] += shortfall
        adjustment_note = (
            "Ham aday kisa geldigi icin giris/cikis payi eklenerek 30-70 saniye bandina uyarlandi."
        )

    rebuilt = []
    for order, segment in enumerate(mutable, start=1):
        cloned = _clone_segment(segment, segment["_start_seconds"], segment["_end_seconds"], order)
        if cloned:
            rebuilt.append(cloned)

    adjusted_duration = round(total_duration_from_segments(rebuilt), 2)
    return rebuilt, raw_duration, adjusted_duration, adjustment_note


def _prefer_segment_count(segments: list[dict]) -> tuple[list[dict], str]:
    rebuilt = _rebuild_segments(segments)
    if len(rebuilt) <= PREFERRED_MAX_REEL_SEGMENTS:
        return rebuilt, ""

    strongest_segments = sorted(
        rebuilt,
        key=lambda segment: (-segment.get("duration_seconds", 0), segment.get("order", 999)),
    )[:PREFERRED_MAX_REEL_SEGMENTS]
    strongest_segments = sorted(
        strongest_segments,
        key=lambda segment: timestamp_to_seconds(segment.get("start", "")),
    )
    reduced = _rebuild_segments(strongest_segments)
    reduced_duration = round(total_duration_from_segments(reduced), 2)
    if reduced_duration >= MIN_REEL_DURATION_SECONDS:
        return (
            reduced,
            f"Asiri parcalanmayi azaltmak icin en guclu {PREFERRED_MAX_REEL_SEGMENTS} segment korundu.",
        )
    return rebuilt, ""


def normalize_reels_data(data: dict) -> dict:
    if not isinstance(data, dict):
        return {}

    final = []
    raw_candidates = data.get("reel_candidates", [])
    if not isinstance(raw_candidates, list):
        raw_candidates = []

    for index, item in enumerate(raw_candidates, start=1):
        if not isinstance(item, dict):
            continue

        segments = []
        for seg_index, seg in enumerate(item.get("segments", item.get("rough_segments", [])), start=1):
            normalized = _normalize_segment(seg, seg_index)
            if normalized:
                segments.append(normalized)

        if not segments:
            continue

        fitted_segments, raw_duration, actual_duration, duration_adjustment_note = _fit_segments_to_duration_window(segments)
        if not fitted_segments:
            continue

        segment_adjustment_note = ""
        preferred_segments, segment_adjustment_note = _prefer_segment_count(fitted_segments)
        if preferred_segments != fitted_segments:
            fitted_segments, _, actual_duration, post_reduce_duration_note = _fit_segments_to_duration_window(
                preferred_segments
            )
            if post_reduce_duration_note:
                duration_adjustment_note = " ".join(
                    item for item in [duration_adjustment_note, post_reduce_duration_note] if item
                )

        duration_target_ok = MIN_REEL_DURATION_SECONDS <= actual_duration <= MAX_REEL_DURATION_SECONDS
        if duration_adjustment_note:
            logger.warning(
                f"Reel adayi sure bandi disinda geldi ve uyarlandi: {raw_duration} sn -> {actual_duration} sn "
                f"(hedef {MIN_REEL_DURATION_SECONDS}-{MAX_REEL_DURATION_SECONDS})"
            )
        elif not duration_target_ok:
            logger.warning(
                f"Reel adayi hedef sure bandina tam oturtulamadi ama korundu: {actual_duration} sn "
                f"(hedef {MIN_REEL_DURATION_SECONDS}-{MAX_REEL_DURATION_SECONDS})"
            )

        editing_notes = [
            normalize_whitespace(note)
            for note in item.get("editing_notes_tr", item.get("editing_notes", []))
            if normalize_whitespace(note)
        ]
        if duration_adjustment_note:
            editing_notes.insert(0, duration_adjustment_note)
        if segment_adjustment_note:
            editing_notes.insert(0, segment_adjustment_note)

        reel_title_tr = normalize_whitespace(item.get("reel_title_tr", item.get("concept", "")))
        cover_prompt = _normalize_cover_prompt(item.get("cover_prompt_en", {}))
        final.append(
            {
                "rank": int(item.get("rank", index) or index),
                "viral_score": max(0, min(100, int(item.get("viral_score", item.get("score", 0)) or 0))),
                "reel_title_tr": reel_title_tr,
                "concept": reel_title_tr,
                "why_selected_tr": normalize_whitespace(item.get("why_selected_tr", item.get("reasoning", ""))),
                "description_tr": normalize_whitespace(item.get("description_tr", "")),
                "cover_plan_tr": normalize_whitespace(item.get("cover_plan_tr", "")),
                "cover_prompt_en": cover_prompt,
                "cover_design_prompt_en": _normalize_reel_design_prompt_fields(
                    item.get("cover_design_prompt_en", {}),
                    reel_title_tr=reel_title_tr,
                    primary_text_fallback=cover_prompt.get("overlay_text", "") or reel_title_tr,
                    background_fallback=cover_prompt.get("background", ""),
                    subject_fallback=cover_prompt.get("people", ""),
                ),
                "segments": fitted_segments,
                "segment_count": len(fitted_segments),
                "editing_notes_tr": editing_notes,
                "raw_total_duration_seconds": raw_duration,
                "actual_total_duration_seconds": actual_duration,
                "duration_target_ok": duration_target_ok,
            }
        )

    final.sort(key=lambda reel: (-reel.get("viral_score", 0), reel.get("rank", 999)))
    for rank, reel in enumerate(final, start=1):
        reel["rank"] = rank

    return {
        "why_this_many_reels_tr": normalize_whitespace(data.get("why_this_many_reels_tr", "")),
        "selected_reel_count": len(final),
        "reel_candidates": final,
        "ideas": final,
        "skipped_by_routing": bool(data.get("skipped_by_routing")),
        "skip_reason": normalize_whitespace(data.get("skip_reason", "")),
    }


def build_report_text(source_name: str, data: dict, model_adi: str) -> str:
    lines = [
        "=== INSTAGRAM REELS FIKIRLERI ===",
        f"Kaynak SRT: {source_name}",
        f"Kullanilan Model: {model_adi}",
        "",
        "NEDEN BU KADAR REEL SECILDI?",
        "-" * 60,
        data.get("why_this_many_reels_tr", ""),
        "",
        f"TOPLAM REEL ADAYI: {data.get('selected_reel_count', 0)}",
        "",
    ]

    if data.get("skipped_by_routing") and not data.get("reel_candidates"):
        lines.extend(
            [
                "ROUTING KARARI",
                "-" * 60,
                data.get("skip_reason", "AI Critic bu video icin ayri Reels adimini gerekli gormedi."),
                "",
                "Bu rapor hafif skip modunda olusturuldu.",
            ]
        )
        return "\n".join(lines).strip() + "\n"

    for reel in data.get("reel_candidates", []):
        cover_prompt = reel.get("cover_prompt_en", {})
        lines.extend(
            [
                f"REEL #{reel.get('rank', '')}",
                "-" * 60,
                f"Viral Skoru: {reel.get('viral_score', 0)}/100",
                f"Reel Basligi: {reel.get('reel_title_tr', '')}",
                f"Segment Sayisi: {reel.get('segment_count', len(reel.get('segments', [])))}",
                (
                    f"Toplam Sure: {reel.get('actual_total_duration_seconds', 0)} sn "
                    f"(ham aday: {reel.get('raw_total_duration_seconds', reel.get('actual_total_duration_seconds', 0))} sn)"
                ),
                f"Neden Secildi: {reel.get('why_selected_tr', '')}",
                f"Reels Aciklamasi (TR): {reel.get('description_tr', '')}",
                f"Kapak Fotografi Mantigi: {reel.get('cover_plan_tr', '')}",
                "Kapak Promptu Katmanlari (TR):",
                f"- Arka Plan: {cover_prompt.get('background', '')}",
                f"- Kisiler/Ozne: {cover_prompt.get('people', '')}",
                f"- Ana Metin: {cover_prompt.get('overlay_text', '')}",
                "",
                "Reel Kapak Gorsel Uretim Promptu (TR):",
                build_reel_cover_design_prompt(reel.get("cover_design_prompt_en", {})),
                "",
                "KESILECEK KISIMLAR",
                "-" * 40,
            ]
        )

        for segment in reel.get("segments", []):
            lines.extend(
                [
                    f"[{segment.get('order')}] {segment.get('start')} --> {segment.get('end')} | {segment.get('duration_seconds')} sn",
                    f"Amac: {segment.get('purpose_tr', '')}",
                    f"Metin: {segment.get('text', '')}",
                    f"Gecis Notu: {segment.get('transition_hint_tr', '')}",
                    "",
                ]
            )

        lines.extend(["EDIT NOTLARI", "-" * 40])
        for item in reel.get("editing_notes_tr", []):
            lines.append(f"- {item}")
        lines.extend(["", "=" * 60, ""])

    return "\n".join(lines).strip() + "\n"


def _safe_file_fragment(value: object, default: str) -> str:
    normalized = normalize_whitespace(value)
    cleaned = re.sub(r'[<>:"/\\\\|?*]+', "", normalized).strip().strip(".")
    cleaned = re.sub(r"\s+", "-", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or default


def _build_single_reel_text(source_name: str, reel: dict, model_adi: str) -> str:
    cover_prompt = reel.get("cover_prompt_en", {})
    lines = [
        f"REEL #{reel.get('rank', '')}",
        f"Kaynak SRT: {source_name}",
        f"Kullanilan Model: {model_adi}",
        "",
        f"Viral Skoru: {reel.get('viral_score', 0)}/100",
        f"Reel Basligi: {reel.get('reel_title_tr', '')}",
        f"Segment Sayisi: {reel.get('segment_count', len(reel.get('segments', [])))}",
        (
            f"Toplam Sure: {reel.get('actual_total_duration_seconds', 0)} sn "
            f"(ham aday: {reel.get('raw_total_duration_seconds', reel.get('actual_total_duration_seconds', 0))} sn)"
        ),
        f"Neden Secildi: {reel.get('why_selected_tr', '')}",
        f"Reels Aciklamasi (TR): {reel.get('description_tr', '')}",
        f"Kapak Fotografi Mantigi: {reel.get('cover_plan_tr', '')}",
        "Kapak Promptu Katmanlari (TR):",
        f"- Arka Plan: {cover_prompt.get('background', '')}",
        f"- Kisiler/Ozne: {cover_prompt.get('people', '')}",
        f"- Ana Metin: {cover_prompt.get('overlay_text', '')}",
        "",
        "Reel Kapak Gorsel Uretim Promptu (TR):",
        build_reel_cover_design_prompt(reel.get("cover_design_prompt_en", {})),
        "",
        "KESILECEK KISIMLAR",
        "-" * 40,
    ]

    for segment in reel.get("segments", []):
        lines.extend(
            [
                f"[{segment.get('order')}] {segment.get('start')} --> {segment.get('end')} | {segment.get('duration_seconds')} sn",
                f"Amac: {segment.get('purpose_tr', '')}",
                f"Metin: {segment.get('text', '')}",
                f"Gecis Notu: {segment.get('transition_hint_tr', '')}",
                "",
            ]
        )

    lines.extend(["EDIT NOTLARI", "-" * 40])
    for item in reel.get("editing_notes_tr", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def _build_single_reel_srt(blocks: list, reel: dict, output_srt: Path) -> Optional[Path]:
    subtitle_entries: list[str] = []
    clip_offset = 0.0
    subtitle_index = 1

    for segment in reel.get("segments", []):
        segment_start = timestamp_to_seconds(segment.get("start", ""))
        segment_end = timestamp_to_seconds(segment.get("end", ""))
        if segment_end <= segment_start:
            continue

        for block in blocks:
            if not block.is_processable or not block.timing_line or not block.text_content:
                continue
            block_start, block_end = _parse_timing_line(block.timing_line)
            if block_end <= segment_start or block_start >= segment_end:
                continue

            clipped_start = max(block_start, segment_start)
            clipped_end = min(block_end, segment_end)
            if clipped_end <= clipped_start:
                continue

            local_start = clip_offset + (clipped_start - segment_start)
            local_end = clip_offset + (clipped_end - segment_start)
            subtitle_entries.append(
                "\n".join(
                    [
                        str(subtitle_index),
                        f"{seconds_to_timestamp(local_start)} --> {seconds_to_timestamp(local_end)}",
                        block.text_content,
                    ]
                )
            )
            subtitle_index += 1

        clip_offset += segment_end - segment_start

    if not subtitle_entries:
        clip_offset = 0.0
        for segment in reel.get("segments", []):
            segment_start = timestamp_to_seconds(segment.get("start", ""))
            segment_end = timestamp_to_seconds(segment.get("end", ""))
            if segment_end <= segment_start:
                continue
            text = normalize_whitespace(segment.get("text") or segment.get("purpose_tr") or "")
            if not text:
                clip_offset += segment_end - segment_start
                continue
            subtitle_entries.append(
                "\n".join(
                    [
                        str(subtitle_index),
                        f"{seconds_to_timestamp(clip_offset)} --> {seconds_to_timestamp(clip_offset + (segment_end - segment_start))}",
                        text,
                    ]
                )
            )
            subtitle_index += 1
            clip_offset += segment_end - segment_start

    if not subtitle_entries:
        return None

    output_srt.parent.mkdir(parents=True, exist_ok=True)
    output_srt.write_text("\n\n".join(subtitle_entries).strip() + "\n", encoding="utf-8")
    return output_srt


def _write_individual_reel_outputs(source_name: str, source_blocks: list, data: dict, model_adi: str) -> None:
    if LEGACY_REELS_ITEM_TXT_DIR.exists():
        for old_file in LEGACY_REELS_ITEM_TXT_DIR.glob("*.txt"):
            old_file.unlink()
    REELS_ITEM_TXT_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in REELS_ITEM_TXT_DIR.glob("*.txt"):
        old_file.unlink()
    for old_file in REELS_ITEM_TXT_DIR.glob("*.srt"):
        old_file.unlink()

    count = 0
    srt_count = 0
    for reel in data.get("reel_candidates", []):
        rank = int(reel.get("rank", count + 1) or (count + 1))
        title_fragment = _safe_file_fragment(reel.get("reel_title_tr", ""), "Baslik")
        txt_path = REELS_ITEM_TXT_DIR / f"{rank:02d}-IG_Reels-{title_fragment}.txt"
        srt_path = REELS_ITEM_TXT_DIR / f"{rank:02d}-IG_Reels-{title_fragment}.srt"
        txt_path.write_text(_build_single_reel_text(source_name, reel, model_adi), encoding="utf-8")
        if _build_single_reel_srt(source_blocks, reel, srt_path):
            srt_count += 1
        count += 1

    logger.info(f"Tekil reel TXT dosyalari kaydedildi: {count} adet")
    logger.info(f"Tekil reel SRT dosyalari kaydedildi: {srt_count} adet")


def save_reports(girdi_dosyasi: Path, data: dict, model_adi: str) -> Tuple[Path, Optional[Path]]:
    json_yolu = json_output_path("reels_ideas")
    txt_yolu = txt_output_path("reels_ideas")
    source_blocks = parse_srt_blocks(read_srt_file(girdi_dosyasi))

    payload = {
        "source_srt": girdi_dosyasi.name,
        "source_stem": girdi_dosyasi.stem,
        "model_name": model_adi,
        "why_this_many_reels_tr": data.get("why_this_many_reels_tr", ""),
        "selected_reel_count": data.get("selected_reel_count", 0),
        "reel_candidates": data.get("reel_candidates", []),
        "ideas": data.get("ideas", []),
        "skipped_by_routing": bool(data.get("skipped_by_routing")),
        "skip_reason": normalize_whitespace(data.get("skip_reason", "")),
    }

    with open(json_yolu, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    txt_yolu.unlink(missing_ok=True)
    _write_individual_reel_outputs(girdi_dosyasi.name, source_blocks, payload, model_adi)
    logger.info(f"🎉 Reel fikirleri JSON kaydedildi: {json_yolu.name}")
    logger.info(f"Tekil reel cikti klasoru guncellendi: {REELS_ITEM_TXT_DIR.name}")
    return json_yolu, None


def load_latest_reels_data() -> Optional[dict]:
    json_yolu = json_output_path("reels_ideas")
    if not json_yolu.exists():
        return None
    try:
        return json.loads(json_yolu.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Reels fikir JSON'u okunamadi: {exc}")
        return None


def analyze(
    girdi_dosyasi: Path,
    llm: CentralLLM,
    reel_sayisi: int = 7,
    metadata_data: Optional[dict] = None,
    critic_data: Optional[dict] = None,
    trim_data: Optional[dict] = None,
    broll_data: Optional[list] = None,
    draft_llm: Optional[CentralLLM] = None,
    ranker_llm: Optional[CentralLLM] = None,
    prepared_transcript: Optional[str] = None,
    respect_routing: bool = False,
) -> dict:
    active_metadata = metadata_data or load_related_json(girdi_dosyasi, "_metadata.json")
    active_critic = critic_data or load_related_json(girdi_dosyasi, "_video_critic.json")
    active_trim = trim_data or load_related_json(girdi_dosyasi, "_trim_suggestions.json")
    active_broll = broll_data or load_related_json(girdi_dosyasi, "_B_roll_fikirleri.json")
    routing_run, routing_reason = _routing_decision(active_critic)
    if respect_routing and routing_run is False:
        logger.info("AI Critic Reels adimini gereksiz gordu; hafif skip raporu olusturuluyor.")
        return {
            "why_this_many_reels_tr": routing_reason or "AI Critic bu video icin ayri Reels adimini gerekli gormedi.",
            "selected_reel_count": 0,
            "reel_candidates": [],
            "ideas": [],
            "skipped_by_routing": True,
            "skip_reason": routing_reason or "AI Critic bu video icin ayri Reels adimini gerekli gormedi.",
        }

    timed_blocks = _timed_blocks_from_srt(girdi_dosyasi)
    srt_metni = normalize_whitespace(prepared_transcript)
    if not srt_metni:
        srt_metni = _build_adaptive_transcript(
            timed_blocks,
            metadata_data=active_metadata,
            critic_data=active_critic,
            trim_data=active_trim,
            broll_data=active_broll,
        )
    if not normalize_whitespace(srt_metni):
        logger.error("Reels fikirleri icin transcript bulunamadi.")
        return {}

    min_reel_count, reel_cap = _resolve_reel_targets(reel_sayisi)
    draft_engine = draft_llm or llm
    ranker_engine = ranker_llm or llm
    critic_ozeti = build_critic_summary(active_critic)
    trim_ozeti = build_trim_summary(active_trim)
    metadata_ozeti = build_metadata_summary(active_metadata)
    broll_ozeti = build_broll_summary(active_broll)
    routing_notu = routing_reason or "AI Critic tarafindan Reels adimi uygun goruldu."
    logger.info(
        f"Instagram Reels adaylari planlaniyor... (minimum hedef: {min_reel_count}, ust sinir: {reel_cap})"
    )
    parsed = None
    ideation = None
    if _should_use_light_mode(timed_blocks, srt_metni):
        logger.info("Reels icin hafif mod devrede: dogrudan tek gecis final uretim deneniyor.")
        parsed = _request_llm(
            ranker_engine,
            build_prompt(
                srt_metni,
                min_reel_count,
                reel_cap,
                critic_ozeti,
                trim_ozeti,
                metadata_ozeti,
                broll_ozeti,
                routing_notu,
            ),
            profile="creative_ranker",
            debug_stem=girdi_dosyasi.stem,
            stage_label="fallback",
        )
    if not parsed:
        ideation = _request_llm(
            draft_engine,
            build_ideation_prompt(
                srt_metni,
                min_reel_count,
                reel_cap,
                critic_ozeti,
                trim_ozeti,
                metadata_ozeti,
                broll_ozeti,
                routing_notu,
            ),
            profile="creative_ideation",
            debug_stem=girdi_dosyasi.stem,
            stage_label="ideation",
        )
        if isinstance(ideation, dict) and isinstance(ideation.get("reel_candidates"), list):
            parsed = _request_llm(
                ranker_engine,
                build_selection_prompt(
                    srt_metni,
                    min_reel_count,
                    reel_cap,
                    ideation,
                    critic_ozeti,
                    trim_ozeti,
                    metadata_ozeti,
                    broll_ozeti,
                    routing_notu,
                ),
                profile="creative_ranker",
                debug_stem=girdi_dosyasi.stem,
                stage_label="selection",
            )
    if not parsed:
        parsed = _request_llm(
            ranker_engine,
            build_prompt(
                srt_metni,
                min_reel_count,
                reel_cap,
                critic_ozeti,
                trim_ozeti,
                metadata_ozeti,
                broll_ozeti,
                routing_notu,
            ),
            profile="creative_ranker",
            debug_stem=girdi_dosyasi.stem,
            stage_label="fallback",
        )
    if not parsed:
        logger.error("Reels cevabi parse edilemedi.")
        return {}
    normalized = normalize_reels_data(parsed)
    if normalized.get("reel_candidates") and _needs_reel_prompt_repair(normalized):
        logger.warning("Reels prompt alanlarinda Turkce olmayan icerik bulundu; Turkce repair asamasi calistiriliyor.")
        normalized = normalize_reels_data(_repair_reel_prompt_language(normalized, ranker_engine))
        if _needs_reel_prompt_repair(normalized):
            logger.warning("Reels prompt repair sonrasinda bazi prompt alanlari hala Turkceye tam oturmadi.")
    if len(normalized.get("reel_candidates", [])) < min_reel_count and isinstance(ideation, dict):
        logger.warning(
            f"Reels aday sayisi minimum hedefin altinda kaldi "
            f"({len(normalized.get('reel_candidates', []))}/{min_reel_count}). Tekrar deneniyor..."
        )
        retried = _request_llm(
            ranker_engine,
            build_selection_prompt(
                srt_metni,
                min_reel_count,
                reel_cap,
                ideation,
                critic_ozeti,
                trim_ozeti,
                metadata_ozeti,
                broll_ozeti,
                routing_notu,
            ),
            profile="creative_ranker",
            debug_stem=girdi_dosyasi.stem,
            stage_label="selection_retry",
        )
        if retried:
            retried_normalized = normalize_reels_data(retried)
            if retried_normalized.get("reel_candidates") and _needs_reel_prompt_repair(retried_normalized):
                retried_normalized = normalize_reels_data(_repair_reel_prompt_language(retried_normalized, ranker_engine))
            if len(retried_normalized.get("reel_candidates", [])) > len(normalized.get("reel_candidates", [])):
                normalized = retried_normalized
    if not normalized.get("reel_candidates"):
        logger.error("Reels icin gecerli aday bulunamadi.")
        return {}
    return normalized


def run():
    print("\n" + "=" * 60)
    print("REELS FIKIR URETICI")
    print("=" * 60)

    secilen_srt = select_primary_srt(logger, "Reels Fikir Uretici")
    if not secilen_srt:
        return

    use_recommended = prompt_module_llm_plan("302", needs_main=True, needs_smart=True)
    if use_recommended:
        saglayici_ana, model_adi_ana = get_module_recommended_llm_config("302", "main")
        saglayici, model_adi = get_module_recommended_llm_config("302", "smart")
        print_module_llm_choice_summary(
            "302",
            {"main": (saglayici_ana, model_adi_ana), "smart": (saglayici, model_adi)},
        )
    else:
        saglayici_ana, model_adi_ana = select_llm("main")
        saglayici, model_adi = select_llm("smart")
    draft_llm = CentralLLM(provider=saglayici_ana, model_name=model_adi_ana)
    llm = CentralLLM(provider=saglayici, model_name=model_adi)

    reels_data = analyze(
        secilen_srt,
        llm,
        reel_sayisi=get_default_reel_target_count(),
        draft_llm=draft_llm,
        respect_routing=True,
    )
    if not reels_data:
        return logger.error("❌ Reel fikirleri uretilemedi.")

    save_reports(secilen_srt, reels_data, f"Draft: {model_adi_ana} | Final: {model_adi}")


def run_automatic(
    girdi_dosyasi: Path,
    llm: CentralLLM,
    reel_sayisi: int = DEFAULT_TARGET_REEL_IDEA_COUNT,
    metadata_data: Optional[dict] = None,
    critic_data: Optional[dict] = None,
    trim_data: Optional[dict] = None,
    broll_data: Optional[list] = None,
    draft_llm: Optional[CentralLLM] = None,
    ranker_llm: Optional[CentralLLM] = None,
    prepared_transcript: Optional[str] = None,
    respect_routing: bool = True,
) -> Optional[dict]:
    logger.info(f"🔄 OTOMASYON: {girdi_dosyasi.name} icin viral Reel fikirleri uretiliyor...")
    reels_data = analyze(
        girdi_dosyasi,
        llm,
        reel_sayisi=reel_sayisi,
        metadata_data=metadata_data,
        critic_data=critic_data,
        trim_data=trim_data,
        broll_data=broll_data,
        draft_llm=draft_llm,
        ranker_llm=ranker_llm,
        prepared_transcript=prepared_transcript,
        respect_routing=respect_routing,
    )
    if not reels_data:
        logger.error("❌ Otomatik Reel fikir uretimi basarisiz oldu.")
        return None

    json_yolu, txt_yolu = save_reports(girdi_dosyasi, reels_data, llm.model_name)
    return {
        "data": reels_data.get("ideas", []),
        "payload": reels_data,
        "json_path": json_yolu,
        "txt_path": txt_yolu,
        "detail": reels_data.get("skip_reason") or "Instagram Reels fikirleri olusturuldu.",
        "skipped_by_routing": bool(reels_data.get("skipped_by_routing")),
    }
