from __future__ import annotations

import json
import os
import re
from typing import Optional

from moduller.llm_manager import CentralLLM
from moduller.logger import get_logger

logger = get_logger("metadata_translate")

TRANSLATEGEMMA_DEFAULT_MODEL = "translategemma:12b-it-q8_0"
TRANSLATION_PROVIDER = "OLLAMA"
TARGET_LANGUAGE_SPECS = {
    "İngilizce": ("English", "en"),
    "Almanca": ("German", "de"),
}
CHAPTER_HEADERS = {
    "Türkçe": "KISIMLAR",
    "İngilizce": "CHAPTERS",
    "Almanca": "KAPITEL",
}
META_LEAK_MARKERS = (
    "here is the translation",
    "translation:",
    "translated text:",
    "sure,",
    "i translated",
    "the german translation",
    "the english translation",
)

TITLE_SCORE_WEIGHTS = {
    "smart": 0.45,
    "main": 0.25,
    "heuristic": 0.30,
}


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return value if value >= minimum else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _env_text(name: str, default: str) -> str:
    value = str(os.getenv(name, default) or "").strip()
    return value or default


def _env_auto_or_int(name: str, minimum: int) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    normalized = str(raw).strip().lower()
    if normalized in {"", "auto"}:
        return None
    try:
        value = int(normalized)
        return value if value >= minimum else None
    except (TypeError, ValueError):
        logger.warning(
            f"Gecersiz {name} degeri bulundu: {raw}. Metadata cevirisi icin model bazli varsayilan kullanilacak."
        )
        return None


def _env_auto_or_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    normalized = str(raw).strip().lower()
    if normalized in {"", "auto"}:
        return None
    try:
        return float(normalized)
    except (TypeError, ValueError):
        logger.warning(
            f"Gecersiz {name} degeri bulundu: {raw}. Metadata cevirisi icin model bazli varsayilan kullanilacak."
        )
        return None


def _env_auto_or_text(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = str(raw).strip()
    if not value or value.lower() == "auto":
        return None
    return value


def _optional_clamped_score(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        number = int(float(value))
    except Exception:
        return None
    return max(0, min(100, number))


def _compose_title_score(smart_score: int | None, main_score: int | None, heuristic_score: int) -> int:
    weighted_parts = [(heuristic_score, TITLE_SCORE_WEIGHTS["heuristic"])]
    if smart_score is not None:
        weighted_parts.append((smart_score, TITLE_SCORE_WEIGHTS["smart"]))
    if main_score is not None:
        weighted_parts.append((main_score, TITLE_SCORE_WEIGHTS["main"]))

    total_weight = sum(weight for _score, weight in weighted_parts)
    if total_weight <= 0:
        return heuristic_score
    return round(sum(score * weight for score, weight in weighted_parts) / total_weight)


def _normalize_whitespace(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _strip_code_fences(text: str) -> str:
    return str(text or "").replace("```json", "").replace("```", "").strip()


def _clean_translation_response(raw_response: str) -> str:
    cleaned = _strip_code_fences(raw_response)
    if not cleaned:
        return ""

    if cleaned.startswith('"') and cleaned.endswith('"'):
        try:
            cleaned = json.loads(cleaned)
        except Exception:
            pass

    cleaned = str(cleaned).strip()
    lowered = cleaned.lower()
    for marker in META_LEAK_MARKERS:
        if lowered.startswith(marker):
            raise ValueError(f"Metadata ceviri cevabi meta icerik iceriyor: {marker}")
    return cleaned


def _metadata_translation_ollama_profile(llm: CentralLLM) -> tuple[int, int, int, str, float]:
    model_name = str(getattr(llm, "model_name", "")).strip().lower()

    if "translategemma" in model_name:
        return 240, 1, 8192, "20m", 0.0
    if "gemma3:12b-it-q8_0" in model_name:
        return 300, 1, 8192, "25m", 0.0
    if model_name.startswith("gemma3:12b"):
        return 280, 1, 8192, "25m", 0.0
    if model_name.startswith("gemma3:"):
        return 240, 1, 7168, "20m", 0.0
    if "qwen3:14b" in model_name:
        return 280, 1, 8192, "20m", 0.0
    return 240, 1, 6144, "15m", 0.0


def _translation_call_settings(llm: CentralLLM) -> tuple[int, int, dict | None, str | None]:
    provider = str(getattr(llm, "provider", "")).strip().upper()
    if provider != "OLLAMA":
        return 120, 1, None, None

    (
        default_timeout,
        default_max_retries,
        default_num_ctx,
        default_keep_alive,
        default_temperature,
    ) = _metadata_translation_ollama_profile(llm)

    timeout_seconds = _env_auto_or_int("TRANSLATION_OLLAMA_TIMEOUT_SECONDS", 30) or default_timeout
    max_retries = _env_auto_or_int("TRANSLATION_OLLAMA_MAX_RETRIES", 0)
    if max_retries is None:
        max_retries = default_max_retries
    num_ctx = _env_auto_or_int("TRANSLATION_OLLAMA_NUM_CTX", 2048) or default_num_ctx
    keep_alive = _env_auto_or_text("TRANSLATION_OLLAMA_KEEP_ALIVE") or default_keep_alive
    temperature = _env_auto_or_float("TRANSLATION_OLLAMA_TEMPERATURE")
    if temperature is None:
        temperature = default_temperature

    return timeout_seconds, max_retries, {"temperature": temperature, "num_ctx": num_ctx}, keep_alive


def metadata_translation_model_name() -> str:
    return _env_auto_or_text("TRANSLATEGEMMA_MODEL_NAME") or TRANSLATEGEMMA_DEFAULT_MODEL


def create_metadata_translation_llm() -> CentralLLM:
    model_name = metadata_translation_model_name()
    logger.info(
        f"Metadata cevirisi env uzerinden secilen TranslateGemma modeliyle yapiliyor. "
        f"Saglayici={TRANSLATION_PROVIDER} | Model={model_name}"
    )
    return CentralLLM(provider=TRANSLATION_PROVIDER, model_name=model_name)


def _build_translate_prompt(text: str, target_lang_en: str, target_lang_code: str) -> str:
    return (
        f"You are a professional Turkish (tr) to {target_lang_en} ({target_lang_code}) translator. "
        f"Your goal is to accurately convey the meaning and nuances of the original Turkish text while "
        f"adhering to {target_lang_en} grammar, vocabulary, and cultural sensitivities. "
        f"Produce only the {target_lang_en} translation, without any additional explanations or commentary. "
        f"Please translate the following Turkish text into {target_lang_en}:\n\n\n"
        f"{text}"
    )


def translate_text(
    text: str,
    target_language: str,
    llm: CentralLLM,
    active_logger=None,
    field_label: str = "",
) -> str:
    clean_text = str(text or "").strip()
    if not clean_text:
        return ""

    logger_override = active_logger or logger
    target_lang_en, target_lang_code = TARGET_LANGUAGE_SPECS[target_language]
    timeout_seconds, max_retries, ollama_options, ollama_keep_alive = _translation_call_settings(llm)
    attempts = _env_int("TRANSLATION_STRUCTURED_MAX_ATTEMPTS", 3, 1)
    last_error = ""

    for attempt in range(1, attempts + 1):
        try:
            raw_response = llm.uret(
                _build_translate_prompt(clean_text, target_lang_en, target_lang_code),
                timeout=timeout_seconds,
                max_retries=max_retries,
                ollama_options=ollama_options,
                ollama_keep_alive=ollama_keep_alive,
            )
            translated = _clean_translation_response(str(raw_response or ""))
            if translated:
                return translated
            last_error = "bos cevap"
        except Exception as exc:
            last_error = str(exc)
            logger_override.warning(
                f"[{target_language}] Metadata cevirisi reddedildi"
                f"{f' ({field_label})' if field_label else ''} ({attempt}/{attempts}): {exc}"
            )

    raise ValueError(last_error or "metadata translation failed")


def _assemble_description_text(language: str, hook_lines: list[str], description_body: str, chapters: list[dict]) -> str:
    parts: list[str] = []
    if hook_lines:
        parts.append("\n".join(item for item in hook_lines[:2] if _normalize_whitespace(item)).strip())
    if description_body:
        parts.append(description_body.strip())
    chapter_lines = [
        f"{item.get('timestamp', '')} - {item.get('title', '')}"
        for item in chapters
        if _normalize_whitespace(item.get("timestamp", "")) and _normalize_whitespace(item.get("title", ""))
    ]
    if chapter_lines:
        parts.append(CHAPTER_HEADERS.get(language, "CHAPTERS") + "\n" + "\n".join(chapter_lines))
    return "\n\n".join(part for part in parts if _normalize_whitespace(part)).strip()


def _normalize_hashtag(translated_text: str) -> str:
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿĞğİıŞşÇçÖöÜüÄäßẞ0-9]+", str(translated_text or ""))
    if not tokens:
        return ""
    return "#" + "".join(token[:1].upper() + token[1:] for token in tokens[:4])


def _translate_hashtags(hashtags: list[str], target_language: str, llm: CentralLLM, active_logger=None) -> list[str]:
    translated_hashtags: list[str] = []
    seen = set()
    for index, hashtag in enumerate(hashtags or [], start=1):
        source_term = str(hashtag or "").lstrip("#").strip()
        if not source_term:
            continue
        translated_term = translate_text(
            source_term,
            target_language,
            llm,
            active_logger=active_logger,
            field_label=f"hashtag_{index}",
        )
        normalized = _normalize_hashtag(translated_term)
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        translated_hashtags.append(normalized)
    return translated_hashtags


def _extract_json_object(raw_text: str) -> dict:
    cleaned = _strip_code_fences(str(raw_text or ""))
    if not cleaned:
        raise ValueError("bos structured metadata cevabi")

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            parsed, _end = decoder.raw_decode(cleaned[index:])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue

    raise ValueError("structured metadata JSON parse edilemedi")


def _prepare_description_translation_source(payload: dict) -> dict:
    return {
        "hook_lines": [
            _normalize_whitespace(item)
            for item in payload.get("hook_lines", [])
            if _normalize_whitespace(item)
        ],
        "description_body": str(payload.get("description_body", "") or "").strip(),
        "chapters": [
            {
                "timestamp": _normalize_whitespace(item.get("timestamp", "")),
                "title": _normalize_whitespace(item.get("title", "")),
            }
            for item in payload.get("chapters", [])
            if isinstance(item, dict)
            and _normalize_whitespace(item.get("timestamp", ""))
            and _normalize_whitespace(item.get("title", ""))
        ],
        "search_terms": [
            _normalize_whitespace(item)
            for item in payload.get("search_terms", [])
            if _normalize_whitespace(item)
        ],
        "hashtags": [
            _normalize_whitespace(item)
            for item in payload.get("hashtags", [])
            if _normalize_whitespace(item)
        ],
    }


def _build_description_payload_translation_prompt(
    source_payload: dict,
    target_lang_en: str,
    target_lang_code: str,
) -> str:
    payload_json = json.dumps(source_payload, ensure_ascii=False, indent=2)
    return (
        f"You are a professional Turkish (tr) to {target_lang_en} ({target_lang_code}) translator. "
        f"Translate the Turkish metadata JSON below into natural and fluent {target_lang_en}. "
        "Return only one valid JSON object with the exact same schema and field names. "
        "Translate only the natural language values. Preserve every chapter timestamp exactly as-is. "
        "Keep arrays aligned with the input whenever possible. Keep hashtag strings as hashtags. "
        "Do not add explanations, notes, markdown, or extra fields. "
        "optimization_note_tr is intentionally excluded and must not be invented.\n\n"
        "JSON schema:\n"
        '{"hook_lines":["",""],"description_body":"","chapters":[{"timestamp":"00:00","title":""}],"search_terms":[""],"hashtags":["#..."]}\n\n'
        "Turkish metadata JSON:\n\n\n"
        f"{payload_json}"
    )


def _normalize_description_translation_payload(
    parsed: dict,
    source_payload: dict,
) -> dict:
    if not isinstance(parsed, dict):
        raise ValueError("structured description translation payload gecersiz")

    hook_lines = [
        _normalize_whitespace(item)
        for item in parsed.get("hook_lines", [])
        if _normalize_whitespace(item)
    ]
    if source_payload.get("hook_lines") and len(hook_lines) < min(2, len(source_payload.get("hook_lines", []))):
        raise ValueError("structured description translation hook_lines eksik")

    description_body = str(parsed.get("description_body", "") or "").strip()
    if not description_body:
        raise ValueError("structured description translation description_body eksik")

    source_chapters = source_payload.get("chapters", [])
    raw_chapters = parsed.get("chapters", [])
    normalized_chapters = []
    if source_chapters:
        if not isinstance(raw_chapters, list) or len(raw_chapters) != len(source_chapters):
            raise ValueError("structured description translation chapter sayisi bozuk")
        for source_item, translated_item in zip(source_chapters, raw_chapters):
            if not isinstance(translated_item, dict):
                raise ValueError("structured description translation chapter tipi bozuk")
            title = _normalize_whitespace(translated_item.get("title", ""))
            if not title:
                raise ValueError("structured description translation chapter title eksik")
            normalized_chapters.append(
                {
                    "timestamp": source_item.get("timestamp", ""),
                    "title": title,
                }
            )

    search_terms = [
        _normalize_whitespace(item)
        for item in parsed.get("search_terms", [])
        if _normalize_whitespace(item)
    ]
    hashtags = []
    for item in parsed.get("hashtags", []) if isinstance(parsed.get("hashtags", []), list) else []:
        normalized = _normalize_hashtag(item)
        if normalized and normalized.casefold() not in {tag.casefold() for tag in hashtags}:
            hashtags.append(normalized)
    if source_payload.get("hashtags") and not hashtags:
        raise ValueError("structured description translation hashtags eksik")

    return {
        "hook_lines": hook_lines,
        "description_body": description_body,
        "chapters": normalized_chapters,
        "search_terms": search_terms,
        "hashtags": hashtags,
    }


def _translate_description_payload_structured(
    payload: dict,
    target_language: str,
    llm: CentralLLM,
    active_logger=None,
) -> dict:
    logger_override = active_logger or logger
    source_payload = _prepare_description_translation_source(payload)
    target_lang_en, target_lang_code = TARGET_LANGUAGE_SPECS[target_language]
    timeout_seconds, max_retries, ollama_options, ollama_keep_alive = _translation_call_settings(llm)
    attempts = _env_int("TRANSLATION_STRUCTURED_MAX_ATTEMPTS", 3, 1)
    last_error = ""

    for attempt in range(1, attempts + 1):
        try:
            raw_response = llm.uret(
                _build_description_payload_translation_prompt(
                    source_payload,
                    target_lang_en,
                    target_lang_code,
                ),
                timeout=timeout_seconds,
                max_retries=max_retries,
                ollama_options=ollama_options,
                ollama_keep_alive=ollama_keep_alive,
            )
            parsed = _extract_json_object(str(raw_response or ""))
            normalized = _normalize_description_translation_payload(parsed, source_payload)
            description_text = _assemble_description_text(
                target_language,
                normalized["hook_lines"],
                normalized["description_body"],
                normalized["chapters"],
            )
            hashtag_line = ", ".join(normalized["hashtags"])
            description_with_hashtags = (
                f"{description_text}\n\n{hashtag_line}".strip() if hashtag_line else description_text
            )
            return {
                "language": target_language,
                "hook_lines": normalized["hook_lines"],
                "description_body": normalized["description_body"],
                "chapters": normalized["chapters"],
                "description_text": description_text,
                "description_char_count": len(description_text),
                "hashtags": normalized["hashtags"],
                "hashtag_line": hashtag_line,
                "search_terms": normalized["search_terms"],
                "optimization_note_tr": payload.get("optimization_note_tr", ""),
                "description_with_hashtags": description_with_hashtags,
                "source_srt": payload.get("source_srt", ""),
            }
        except Exception as exc:
            last_error = str(exc)
            logger_override.warning(
                f"[{target_language}] Structured description payload cevirisi reddedildi ({attempt}/{attempts}): {exc}"
            )

    raise ValueError(last_error or "structured description translation failed")


def _translate_description_payload_fieldwise(payload: dict, target_language: str, llm: CentralLLM, active_logger=None) -> dict:
    translated_hook_lines = [
        translate_text(line, target_language, llm, active_logger=active_logger, field_label=f"hook_{idx}")
        for idx, line in enumerate(payload.get("hook_lines", []), start=1)
        if _normalize_whitespace(line)
    ]
    translated_body = translate_text(
        payload.get("description_body", ""),
        target_language,
        llm,
        active_logger=active_logger,
        field_label="description_body",
    )
    translated_chapters = []
    for index, item in enumerate(payload.get("chapters", []), start=1):
        if not isinstance(item, dict):
            continue
        translated_chapters.append(
            {
                "timestamp": item.get("timestamp", ""),
                "title": translate_text(
                    item.get("title", ""),
                    target_language,
                    llm,
                    active_logger=active_logger,
                    field_label=f"chapter_{index}",
                ),
            }
        )
    translated_search_terms = [
        translate_text(term, target_language, llm, active_logger=active_logger, field_label=f"search_term_{idx}")
        for idx, term in enumerate(payload.get("search_terms", []), start=1)
        if _normalize_whitespace(term)
    ]
    translated_hashtags = _translate_hashtags(payload.get("hashtags", []), target_language, llm, active_logger=active_logger)
    description_text = _assemble_description_text(target_language, translated_hook_lines, translated_body, translated_chapters)
    hashtag_line = ", ".join(translated_hashtags)
    description_with_hashtags = f"{description_text}\n\n{hashtag_line}".strip() if hashtag_line else description_text

    return {
        "language": target_language,
        "hook_lines": translated_hook_lines,
        "description_body": translated_body,
        "chapters": translated_chapters,
        "description_text": description_text,
        "description_char_count": len(description_text),
        "hashtags": translated_hashtags,
        "hashtag_line": hashtag_line,
        "search_terms": translated_search_terms,
        "optimization_note_tr": payload.get("optimization_note_tr", ""),
        "description_with_hashtags": description_with_hashtags,
        "source_srt": payload.get("source_srt", ""),
    }


def translate_description_payload(payload: dict, target_language: str, llm: CentralLLM, active_logger=None) -> dict:
    logger_override = active_logger or logger
    try:
        return _translate_description_payload_structured(payload, target_language, llm, active_logger=logger_override)
    except Exception as exc:
        logger_override.warning(
            f"[{target_language}] Structured description payload cevirisi basarisiz oldu; "
            f"alan-bazli fallback kullaniliyor: {exc}"
        )
        return _translate_description_payload_fieldwise(payload, target_language, llm, active_logger=logger_override)


def translate_title_payload(
    payload: dict,
    target_language: str,
    llm: CentralLLM,
    score_callback,
    active_logger=None,
) -> dict:
    translated_items = []
    for index, item in enumerate(payload.get("titles", []), start=1):
        if not isinstance(item, dict):
            continue
        translated_title = translate_text(
            item.get("title", ""),
            target_language,
            llm,
            active_logger=active_logger,
            field_label=f"title_{index}",
        )
        score_breakdown = score_callback(translated_title, target_language)
        smart_score = _optional_clamped_score(item.get("smart_ctr_score", item.get("llm_ctr_score")))
        main_score = _optional_clamped_score(item.get("main_ctr_score"))
        final_score = _compose_title_score(smart_score, main_score, score_breakdown["toplam"])
        translated_items.append(
            {
                "title": translated_title,
                "score": final_score,
                "llm_ctr_score": int(smart_score or 0),
                "smart_ctr_score": int(smart_score or 0),
                "main_ctr_score": int(main_score or 0),
                "heuristic_score": score_breakdown["toplam"],
                "reason_tr": item.get("reason_tr", ""),
                "structural_reason_tr": item.get("structural_reason_tr", ""),
                "angle": item.get("angle", ""),
                "score_breakdown": score_breakdown,
                "rank": item.get("rank", len(translated_items) + 1),
            }
        )

    translated_items.sort(
        key=lambda item: (item.get("score", 0), item.get("heuristic_score", 0)),
        reverse=True,
    )
    for rank, item in enumerate(translated_items, start=1):
        item["rank"] = rank

    return {
        "language": target_language,
        "best_title": translated_items[0]["title"] if translated_items else "",
        "titles": translated_items,
    }
