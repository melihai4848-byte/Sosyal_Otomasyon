# moduller/gramer_duzenleyici.py
import concurrent.futures
import difflib
import json
import os
import re
import time
from dataclasses import dataclass
from moduller.logger import get_logger
from moduller.runtime_utils import format_elapsed
from moduller.srt_utils import (
    SrtBlock,
    parse_srt_blocks,
    read_srt_file,
    serialize_srt_blocks,
    write_srt_file,
)
from moduller.llm_manager import (
    CentralLLM,
    get_module_recommended_llm_config,
    print_module_llm_choice_summary,
    prompt_module_llm_plan,
    select_llm,
)
from moduller.subtitle_llm_utils import (
    append_debug_response,
    dump_subtitle_block_payload,
    normalize_text_lines,
    prepare_debug_file,
    rebuild_srt_from_replacements,
    validate_structured_subtitle_response,
)
from moduller.subtitle_output_utils import (
    find_subtitle_artifact,
    list_subtitle_files,
    relocate_known_subtitle_intermediates,
    subtitle_intermediate_output_path,
    subtitle_output_path,
)
from pathlib import Path

logger = get_logger("grammar")
GRAMMAR_INPUT_NAME = "subtitle_raw_tr.srt"
GRAMMAR_OUTPUT_NAME = "subtitle_tr.srt"
GRAMMAR_EN_INPUT_NAME = "subtitle_raw_en.srt"
GRAMMAR_EN_OUTPUT_NAME = "subtitle_en.srt"
GRAMMAR_SHORTS_INPUT_NAME = "subtitle_raw_shorts.srt"
GRAMMAR_SHORTS_OUTPUT_NAME = "subtitle_shorts.srt"
GRAMMAR_DEBUG_OUTPUT_NAME = "grammar_llm_debug.txt"
GRAMMAR_GLOSSARY_JSON_NAME = "grammar_video_glossary.json"
GRAMMAR_GLOSSARY_FIXED_INPUT_NAME = "subtitle_raw_tr_glossary_fixed.srt"
WHISPER_ENGLISH_HINT_NAME = "subtitle_raw_en.srt"
GRAMMAR_LINE_RE = re.compile(r"^\s*(\d+)\s*(?:\t|\||:|-)\s*(.+?)\s*$")
GRAMMAR_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿĞğİıŞşÇçÖöÜüÄäßẞ]+")
GLOSSARY_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿĞğİıŞşÇçÖöÜüÄäßẞ][A-Za-zÀ-ÖØ-öø-ÿĞğİıŞşÇçÖöÜüÄäßẞ'’-]*")
GRAMMAR_META_MARKERS = (
    "could you",
    "i'm here to help",
    "i am here to help",
    "let me know",
    "grammatically correct",
    "please provide",
    "your message got cut off",
    "yardimci olayim",
    "yardimci olabilirim",
)


@dataclass
class GrammarChunk:
    context_blocks: list
    target_blocks: list
    target_start_index: int
    target_end_index: int


@dataclass
class GrammarLineEntry:
    block_index: int
    block_id: str
    source_block: SrtBlock
    cleaned_text: str


def _resolve_worker_count(llm: CentralLLM) -> int:
    default_workers = 1 if llm.provider == "OLLAMA" else 3
    allow_ollama_parallel = os.getenv("GRAMMAR_ALLOW_OLLAMA_PARALLEL", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    raw = os.getenv("GRAMMAR_MAX_WORKERS")
    if raw is None:
        return default_workers
    try:
        value = int(raw.strip())
        if value <= 0:
            return default_workers
        if llm.provider == "OLLAMA" and value > 1 and not allow_ollama_parallel:
            logger.warning(
                "OLLAMA icin gramer modulu varsayilan olarak tekli calisir. "
                f"GRAMMAR_MAX_WORKERS={value} bulundu ama 1'e dusuruldu. "
                "Paralel calisma istiyorsan GRAMMAR_ALLOW_OLLAMA_PARALLEL=true yap."
            )
            return 1
        return value
    except (TypeError, ValueError):
        logger.warning(
            f"Gecersiz GRAMMAR_MAX_WORKERS degeri bulundu: {raw}. Varsayilan {default_workers} kullanilacak."
        )
        return default_workers


def _env_auto_or_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    normalized = str(raw).strip().lower()
    if normalized in {"", "auto"}:
        return None
    try:
        value = int(normalized)
        return value if value >= 500 else None
    except (TypeError, ValueError):
        logger.warning(
            f"Gecersiz {name} degeri bulundu: {raw}. Model bazli varsayilan kullanilacak."
        )
        return None


def _resolve_single_pass_override() -> int | None:
    raw = os.getenv("GRAMMAR_SINGLE_PASS_MAX_CHARS")
    if raw is None:
        return None
    normalized = str(raw).strip().lower()
    if normalized in {"", "auto"}:
        return None
    try:
        value = int(normalized)
        return value if value >= 1000 else None
    except (TypeError, ValueError):
        logger.warning(
            f"Gecersiz GRAMMAR_SINGLE_PASS_MAX_CHARS degeri bulundu: {raw}. Model bazli varsayilan kullanilacak."
        )
        return None


def _resolve_overlap_chars_override() -> int | None:
    raw = os.getenv("GRAMMAR_CHUNK_OVERLAP_CHARS")
    if raw is None:
        return None
    normalized = str(raw).strip().lower()
    if normalized in {"", "auto"}:
        return None
    try:
        value = int(normalized)
        return value if value >= 0 else None
    except (TypeError, ValueError):
        logger.warning(
            f"Gecersiz GRAMMAR_CHUNK_OVERLAP_CHARS degeri bulundu: {raw}. Model bazli varsayilan kullanilacak."
        )
        return None


def _resolve_overlap_blocks_override() -> int | None:
    raw = os.getenv("GRAMMAR_CHUNK_OVERLAP_BLOCKS")
    if raw is None:
        return None
    normalized = str(raw).strip().lower()
    if normalized in {"", "auto"}:
        return None
    try:
        value = int(normalized)
        return value if value >= 0 else None
    except (TypeError, ValueError):
        logger.warning(
            f"Gecersiz GRAMMAR_CHUNK_OVERLAP_BLOCKS degeri bulundu: {raw}. Model bazli varsayilan kullanilacak."
        )
        return None


def _env_auto_or_int_at_least(name: str, minimum: int) -> int | None:
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
            f"Gecersiz {name} degeri bulundu: {raw}. Model bazli varsayilan kullanilacak."
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
            f"Gecersiz {name} degeri bulundu: {raw}. Model bazli varsayilan kullanilacak."
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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    logger.warning(f"Gecersiz {name} degeri bulundu: {raw}. Varsayilan {default} kullanilacak.")
    return default


def _grammar_diff_report_enabled() -> bool:
    return _env_bool("GRAMMAR_WRITE_DIFF_REPORT", True)


def _grammar_video_glossary_enabled() -> bool:
    return _env_bool("GRAMMAR_ENABLE_VIDEO_GLOSSARY", True)


def _grammar_process_english_enabled() -> bool:
    return _env_bool("GRAMMAR_PROCESS_ENGLISH_SUBTITLE", True)


def _grammar_process_shorts_enabled() -> bool:
    return _env_bool("GRAMMAR_PROCESS_SHORTS_SUBTITLE", True)


def _grammar_glossary_max_terms() -> int:
    return _env_auto_or_int_at_least("GRAMMAR_GLOSSARY_MAX_TERMS", 4) or 24


def _grammar_glossary_source_chars() -> int:
    return _env_auto_or_int_at_least("GRAMMAR_GLOSSARY_SOURCE_MAX_CHARS", 1000) or 12000


def _grammar_glossary_fuzzy_threshold() -> float:
    value = _env_auto_or_float("GRAMMAR_GLOSSARY_FUZZY_THRESHOLD")
    if value is None:
        return 0.88
    return min(0.98, max(0.70, value))


def _resolve_structured_attempts() -> int:
    return _env_auto_or_int_at_least("GRAMMAR_STRUCTURED_MAX_ATTEMPTS", 1) or 3


def _resolve_rescue_split_depth() -> int:
    value = _env_auto_or_int_at_least("GRAMMAR_RESCUE_MAX_SPLIT_DEPTH", 0)
    return 3 if value is None else value


def _serialized_length(blocks: list) -> int:
    if not blocks:
        return 0
    return len(serialize_srt_blocks(blocks))


def _grammar_chunk_profile(llm: CentralLLM) -> tuple[int, int, int, int]:
    provider = str(getattr(llm, "provider", "")).strip().upper()
    model_name = str(getattr(llm, "model_name", "")).strip().lower()

    if provider == "OLLAMA":
        if "gemma3:12b-it-q8_0" in model_name:
            return 5000, 6500, 480, 2
        if model_name.startswith("gemma3:12b"):
            return 4500, 6000, 450, 2
        if model_name.startswith("gemma3:"):
            return 4000, 5500, 380, 2
        if "qwen3.5" in model_name:
            return 5000, 6500, 500, 2
        if "qwen3:14b" in model_name:
            return 4500, 6000, 450, 2
        return 3500, 5000, 320, 2

    return 4000, 8000, 400, 2


def _normalize_glossary_key(text: str) -> str:
    return re.sub(r"[^a-z0-9çğıöşüäöüß]+", "", str(text or "").strip().lower())


def _extract_json_payload(raw_response: str):
    response = str(raw_response or "").strip()
    if not response:
        return None
    response = re.sub(r"```json\s*|```", "", response).strip()
    for loader in (lambda value: json.loads(value),):
        try:
            return loader(response)
        except Exception:
            pass

    for open_char, close_char in (("{", "}"), ("[", "]")):
        start_positions = [index for index, char in enumerate(response) if char == open_char]
        for start in start_positions:
            depth = 0
            in_string = False
            escape = False
            for idx in range(start, len(response)):
                char = response[idx]
                if escape:
                    escape = False
                    continue
                if char == "\\":
                    escape = True
                    continue
                if char == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if char == open_char:
                    depth += 1
                elif char == close_char:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(response[start : idx + 1])
                        except Exception:
                            break
    return None


def _load_whisper_english_hint() -> str:
    path = find_subtitle_artifact(WHISPER_ENGLISH_HINT_NAME)
    if path is None:
        logger.info("Whisper Ingilizce ipucu bulunamadi. Glossary sadece ana transcript ile hazirlanacak.")
        return ""
    if not path.exists():
        logger.info("Whisper Ingilizce ipucu yolu bulundu ama dosya mevcut degil. Glossary ipucusuz devam edecek.")
        return ""
    try:
        logger.info(f"Whisper Ingilizce ipucu yukleniyor: {path.name}")
        text = read_srt_file(path)
        blocks = parse_srt_blocks(text)
        excerpt = "\n".join(block.text_content for block in blocks if block.is_processable and block.text_content).strip()
        logger.info(f"Whisper Ingilizce ipucu hazirlandi. Karakter={min(len(excerpt), _grammar_glossary_source_chars())}")
        return excerpt[: _grammar_glossary_source_chars()]
    except Exception as exc:
        logger.warning(f"Whisper İngilizce ipucu okunamadi: {exc}")
        return ""


def _build_glossary_prompt(girdi_dosyasi: Path, bloklar: list[SrtBlock]) -> str:
    source_excerpt_lines: list[str] = []
    current_length = 0
    max_chars = _grammar_glossary_source_chars()
    for block in bloklar:
        if not block.is_processable or not block.text_content:
            continue
        snippet = f"[{block.id}] {block.text_content}"
        source_excerpt_lines.append(snippet)
        current_length += len(snippet) + 1
        if current_length >= max_chars:
            break

    whisper_hint = _load_whisper_english_hint()
    video_hint = girdi_dosyasi.stem
    return (
        "Sen karisik dilli YouTube altyazilarinda yabanci terim ve ozel isim kurtarma uzmani bir dil editorusun.\n"
        "Amacin, Turkce konusma icine serpilmis Ingilizce/Almanca terimleri ve ozel isimleri tespit edip "
        "Whisper'in yanlis duymus olabilecegi varyantlarla birlikte video-ozel bir sozluk cikarmak.\n"
        "Sadece gercekten duzeltmeye deger, videoya ozel ve yabanci/ozel isim niteliginde olan maddeleri ver.\n"
        "Turkce genel kelimeleri listeleme. Emin degilsen madde ekleme.\n"
        f"En fazla {_grammar_glossary_max_terms()} madde ver.\n"
        "Sadece JSON don. Semasi su olsun:\n"
        "{\n"
        '  "terms": [\n'
        '    {\n'
        '      "correct": "Anschreiben",\n'
        '      "language": "de|en|proper",\n'
        '      "category": "term|company|city|person|brand|concept",\n'
        '      "variants": ["Unshriven", "Anschraiben"]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Video/dosya ipucu:\n{video_hint}\n\n"
        f"Ham Turkce transcript onizlemesi:\n{chr(10).join(source_excerpt_lines).strip()}\n\n"
        f"Opsiyonel Whisper Ingilizce ipucu:\n{whisper_hint or 'Yok'}\n"
    )


def _extract_glossary_terms(payload) -> list[dict]:
    if isinstance(payload, dict):
        raw_terms = payload.get("terms", [])
    elif isinstance(payload, list):
        raw_terms = payload
    else:
        raw_terms = []

    terms: list[dict] = []
    seen = set()
    for item in raw_terms:
        if not isinstance(item, dict):
            continue
        correct = str(item.get("correct") or "").strip()
        if len(correct) < 3:
            continue
        variants = []
        for variant in item.get("variants", []) or []:
            text = str(variant or "").strip()
            if len(text) >= 3 and text.lower() != correct.lower():
                variants.append(text)
        key = _normalize_glossary_key(correct)
        if not key or key in seen:
            continue
        seen.add(key)
        terms.append(
            {
                "correct": correct,
                "language": str(item.get("language") or "").strip().lower(),
                "category": str(item.get("category") or "").strip().lower(),
                "variants": list(dict.fromkeys(variants)),
            }
        )
    return terms[: _grammar_glossary_max_terms()]


def _replace_exact_variants(text: str, variants: list[str], correct: str) -> tuple[str, int]:
    updated = text
    replacements = 0
    for variant in variants:
        pattern = re.compile(rf"(?<!\w){re.escape(variant)}(?!\w)", flags=re.IGNORECASE)
        updated, count = pattern.subn(correct, updated)
        replacements += count
    return updated, replacements


def _replace_fuzzy_tokens(text: str, variants: list[str], correct: str) -> tuple[str, int]:
    if " " in correct.strip():
        return text, 0
    tokens = re.split(r"(\W+)", text)
    replacements = 0
    threshold = _grammar_glossary_fuzzy_threshold()
    normalized_variants = [_normalize_glossary_key(item) for item in variants if _normalize_glossary_key(item)]
    if not normalized_variants:
        return text, 0
    for index, token in enumerate(tokens):
        if not GLOSSARY_TOKEN_RE.fullmatch(token or ""):
            continue
        normalized_token = _normalize_glossary_key(token)
        if len(normalized_token) < 5:
            continue
        if normalized_token == _normalize_glossary_key(correct):
            continue
        best_ratio = 0.0
        for variant in normalized_variants:
            if not variant or abs(len(variant) - len(normalized_token)) > 4:
                continue
            ratio = difflib.SequenceMatcher(None, normalized_token, variant).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
        if best_ratio >= threshold:
            tokens[index] = correct
            replacements += 1
    return "".join(tokens), replacements


def _apply_video_glossary(bloklar: list[SrtBlock], glossary_terms: list[dict]) -> tuple[list[SrtBlock], dict]:
    if not glossary_terms:
        return list(bloklar), {"replacements": 0, "matched_terms": []}

    corrected_blocks: list[SrtBlock] = []
    total_replacements = 0
    matched_terms: set[str] = set()
    for block in bloklar:
        if not block.is_processable or not block.text_lines:
            corrected_blocks.append(block)
            continue
        updated_lines = []
        for line in block.text_lines:
            updated_line = line
            for term in glossary_terms:
                correct = term.get("correct", "")
                variants = term.get("variants", []) or []
                updated_line, exact_count = _replace_exact_variants(updated_line, variants, correct)
                updated_line, fuzzy_count = _replace_fuzzy_tokens(updated_line, variants, correct)
                replacement_count = exact_count + fuzzy_count
                if replacement_count:
                    total_replacements += replacement_count
                    matched_terms.add(correct)
            updated_lines.append(updated_line)
        corrected_blocks.append(SrtBlock(block.raw, block.index_line, block.timing_line, updated_lines))

    return corrected_blocks, {
        "replacements": total_replacements,
        "matched_terms": sorted(matched_terms),
    }


def _prepare_glossary_fixed_blocks(girdi_dosyasi: Path, llm: CentralLLM, debug_path: Path) -> tuple[list[SrtBlock], dict]:
    icerik = read_srt_file(girdi_dosyasi)
    bloklar = parse_srt_blocks(icerik)

    if not _grammar_video_glossary_enabled():
        logger.info("Video-ozel glossary katmani env geregi kapali. Direkt gramer islemine gecilecek.")
        return bloklar, {"enabled": False, "terms": [], "replacements": 0}

    logger.info(
        f"Video-ozel glossary hazirligi basliyor: {girdi_dosyasi.name} | "
        f"Islenebilir blok={sum(1 for block in bloklar if block.is_processable)}"
    )
    prompt = _build_glossary_prompt(girdi_dosyasi, bloklar)
    timeout_seconds, max_retries, ollama_options, ollama_keep_alive = _resolve_grammar_llm_call_settings(llm)
    raw_response = ""
    glossary_started_at = time.perf_counter()
    try:
        logger.info(
            f"Video-ozel glossary icin LLM cagrisi baslatiliyor... "
            f"(prompt={len(prompt)} karakter, timeout={timeout_seconds}s)"
        )
        raw_response = llm.uret(
            prompt,
            timeout=timeout_seconds,
            max_retries=max_retries,
            ollama_options=ollama_options,
            ollama_keep_alive=ollama_keep_alive,
        )
    except Exception as exc:
        elapsed = format_elapsed(time.perf_counter() - glossary_started_at)
        logger.warning(f"Video-ozel glossary uretilemedi: {exc} | Sure={elapsed}")
        append_debug_response(
            debug_path,
            "VIDEO GLOSSARY HATASI",
            str(exc),
            raw_response,
            source_excerpt=prompt[:3000],
        )
        return bloklar, {"enabled": True, "terms": [], "replacements": 0, "error": str(exc)}

    payload = _extract_json_payload(raw_response)
    terms = _extract_glossary_terms(payload)
    elapsed = format_elapsed(time.perf_counter() - glossary_started_at)
    logger.info(f"Video-ozel glossary cevabi alindi. Terim adayi={len(terms)} | Sure={elapsed}")
    if payload is None:
        append_debug_response(
            debug_path,
            "VIDEO GLOSSARY PARSE HATASI",
            "LLM cevabindan gecerli JSON glossary cikarilamadi.",
            raw_response,
            source_excerpt=prompt[:3000],
        )

    corrected_blocks, stats = _apply_video_glossary(bloklar, terms)
    glossary_json_path = subtitle_intermediate_output_path(GRAMMAR_GLOSSARY_JSON_NAME)
    glossary_json_path.write_text(
        json.dumps(
            {
                "source_file": girdi_dosyasi.name,
                "term_count": len(terms),
                "terms": terms,
                "replacement_stats": stats,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    glossary_fixed_path = subtitle_intermediate_output_path(GRAMMAR_GLOSSARY_FIXED_INPUT_NAME)
    write_srt_file(glossary_fixed_path, serialize_srt_blocks(corrected_blocks))
    logger.info(
        f"Video-ozel glossary hazirlandi. Terim={len(terms)} | "
        f"Toplam duzeltme={stats.get('replacements', 0)} | "
        f"Ara dosya={glossary_fixed_path.name}"
    )
    return corrected_blocks, {
        "enabled": True,
        "terms": terms,
        "replacements": stats.get("replacements", 0),
        "glossary_json_path": glossary_json_path,
        "glossary_fixed_path": glossary_fixed_path,
        "matched_terms": stats.get("matched_terms", []),
    }


def _grammar_ollama_profile(llm: CentralLLM) -> tuple[int, int, int, str, float]:
    model_name = str(getattr(llm, "model_name", "")).strip().lower()

    if "gemma3:12b-it-q8_0" in model_name:
        return 420, 1, 12288, "30m", 0.0
    if model_name.startswith("gemma3:12b"):
        return 360, 1, 12288, "30m", 0.0
    if model_name.startswith("gemma3:"):
        return 360, 1, 10240, "20m", 0.0
    if "qwen3:14b" in model_name:
        return 360, 1, 10240, "20m", 0.0
    return 300, 1, 8192, "15m", 0.0


def _resolve_grammar_llm_call_settings(llm: CentralLLM) -> tuple[int, int, dict | None, str | None]:
    provider = str(getattr(llm, "provider", "")).strip().upper()
    if provider != "OLLAMA":
        return 120, 2, None, None

    (
        default_timeout,
        default_max_retries,
        default_num_ctx,
        default_keep_alive,
        default_temperature,
    ) = _grammar_ollama_profile(llm)

    timeout_override = _env_auto_or_int_at_least("GRAMMAR_OLLAMA_TIMEOUT_SECONDS", 60)
    timeout_seconds = default_timeout if timeout_override is None else timeout_override

    retries_override = _env_auto_or_int_at_least("GRAMMAR_OLLAMA_MAX_RETRIES", 0)
    max_retries = default_max_retries if retries_override is None else retries_override

    num_ctx_override = _env_auto_or_int_at_least("GRAMMAR_OLLAMA_NUM_CTX", 2048)
    num_ctx = default_num_ctx if num_ctx_override is None else num_ctx_override

    keep_alive_override = _env_auto_or_text("GRAMMAR_OLLAMA_KEEP_ALIVE")
    keep_alive = default_keep_alive if keep_alive_override is None else keep_alive_override

    temperature_override = _env_auto_or_float("GRAMMAR_OLLAMA_TEMPERATURE")
    temperature = default_temperature if temperature_override is None else temperature_override

    return timeout_seconds, max_retries, {"temperature": temperature, "num_ctx": num_ctx}, keep_alive


def _select_overlap_blocks(
    bloklar: list,
    target_start_index: int,
    overlap_chars: int,
    overlap_blocks: int,
) -> list:
    if target_start_index <= 0 or overlap_chars <= 0 or overlap_blocks <= 0:
        return []

    selected = []
    total_length = 0

    for idx in range(target_start_index - 1, -1, -1):
        block = bloklar[idx]
        block_length = _serialized_length([block])
        if selected and total_length + block_length > overlap_chars:
            break
        selected.append(block)
        total_length += block_length
        if len(selected) >= overlap_blocks:
            break

    selected.reverse()
    return selected


def _effective_target_chunk_budget(chunk_size: int, overlap_chars: int) -> int:
    if overlap_chars <= 0:
        return chunk_size
    return max(500, chunk_size - overlap_chars)


def _split_target_chunks(
    bloklar: list,
    chunk_size: int,
    overlap_chars: int = 0,
) -> list[tuple[int, int, list]]:
    if not bloklar:
        return []

    subsequent_budget = _effective_target_chunk_budget(chunk_size, overlap_chars)

    def _budget_for_chunk(start_index: int) -> int:
        return chunk_size if start_index == 0 else subsequent_budget

    chunks = []
    current_chunk = []
    current_length = 0
    current_start_index = 0

    for idx, block in enumerate(bloklar):
        block_len = _serialized_length([block])
        current_budget = _budget_for_chunk(current_start_index)
        if current_chunk and current_length + block_len > current_budget:
            chunks.append((current_start_index, idx, current_chunk))
            current_chunk = []
            current_length = 0
            current_start_index = idx
        current_chunk.append(block)
        current_length += block_len

    if current_chunk:
        chunks.append((current_start_index, len(bloklar), current_chunk))

    return chunks


def _resolve_rescue_overlap_settings(llm: CentralLLM) -> tuple[int, int]:
    _, _, default_overlap_chars, default_overlap_blocks = _grammar_chunk_profile(llm)
    overlap_chars = _resolve_overlap_chars_override()
    if overlap_chars is None:
        overlap_chars = default_overlap_chars
    overlap_blocks = _resolve_overlap_blocks_override()
    if overlap_blocks is None:
        overlap_blocks = default_overlap_blocks
    rescue_chars = max(140, overlap_chars // 2) if overlap_chars > 0 else 0
    rescue_blocks = 1 if overlap_blocks > 0 else 0
    return rescue_chars, rescue_blocks


def _split_grammar_chunk_for_rescue(chunk: GrammarChunk, llm: CentralLLM) -> list[GrammarChunk]:
    if len(chunk.target_blocks) <= 1:
        return [chunk]

    midpoint = max(1, len(chunk.target_blocks) // 2)
    overlap_chars, overlap_blocks = _resolve_rescue_overlap_settings(llm)
    child_ranges = (
        (0, midpoint),
        (midpoint, len(chunk.target_blocks)),
    )
    child_chunks: list[GrammarChunk] = []

    for child_start, child_end in child_ranges:
        prefix_blocks = list(chunk.target_blocks[:child_start])
        if child_start == 0 and chunk.context_blocks:
            parent_seed_context = _select_overlap_blocks(
                chunk.context_blocks,
                len(chunk.context_blocks),
                overlap_chars=overlap_chars,
                overlap_blocks=overlap_blocks,
            )
            prefix_blocks = parent_seed_context
        child_context_blocks = _select_overlap_blocks(
            prefix_blocks,
            len(prefix_blocks),
            overlap_chars=overlap_chars,
            overlap_blocks=overlap_blocks,
        )
        child_chunks.append(
            GrammarChunk(
                context_blocks=child_context_blocks,
                target_blocks=chunk.target_blocks[child_start:child_end],
                target_start_index=chunk.target_start_index + child_start,
                target_end_index=chunk.target_start_index + child_end,
            )
        )

    return child_chunks


def _strip_context_echo(result_text: str, context_blocks: list) -> str:
    if not result_text or not context_blocks:
        return result_text

    parsed_blocks = parse_srt_blocks(result_text)
    if not parsed_blocks:
        return result_text

    context_ids = [block.id for block in context_blocks if block.id]
    if not context_ids:
        return result_text

    drop_count = 0
    for block in parsed_blocks:
        if drop_count < len(context_ids) and block.id == context_ids[drop_count]:
            drop_count += 1
            continue
        break

    if drop_count == 0:
        return result_text

    trimmed_blocks = parsed_blocks[drop_count:]
    if not trimmed_blocks:
        return ""
    return serialize_srt_blocks(trimmed_blocks).strip()


def _resolve_chunk_strategy(bloklar: list, llm: CentralLLM) -> tuple[list[GrammarChunk], int, int, int, int, int]:
    total_serialized_chars = _serialized_length(bloklar)
    (
        default_chunk_size,
        default_single_pass_limit,
        default_overlap_chars,
        default_overlap_blocks,
    ) = _grammar_chunk_profile(llm)
    chunk_size = _env_auto_or_int("GRAMMAR_CHUNK_MAX_CHARS") or default_chunk_size
    single_pass_limit = _resolve_single_pass_override() or default_single_pass_limit
    overlap_chars = _resolve_overlap_chars_override()
    if overlap_chars is None:
        overlap_chars = default_overlap_chars
    overlap_blocks = _resolve_overlap_blocks_override()
    if overlap_blocks is None:
        overlap_blocks = default_overlap_blocks

    if total_serialized_chars and total_serialized_chars <= single_pass_limit:
        logger.info(
            f"Gramer modulu bu dosyayi tek geciste isleyecek. "
            f"Toplam boyut: {total_serialized_chars} karakter | Tek gecis limiti: {single_pass_limit} karakter."
        )
        return [
            GrammarChunk(
                context_blocks=[],
                target_blocks=bloklar,
                target_start_index=0,
                target_end_index=len(bloklar),
            )
        ], total_serialized_chars, total_serialized_chars, single_pass_limit, 0, 0

    effective_target_budget = _effective_target_chunk_budget(chunk_size, overlap_chars)
    logger.info(
        f"Gramer chunk toplam butcesi {chunk_size} karakter; "
        f"referans baglam rezerve edildikten sonra hedef blok butcesi yaklasik {effective_target_budget} karakter."
    )

    target_chunks = _split_target_chunks(bloklar, chunk_size, overlap_chars=overlap_chars)
    parcalar = []
    for start_index, end_index, target_blocks in target_chunks:
        context_blocks = _select_overlap_blocks(
            bloklar,
            start_index,
            overlap_chars=overlap_chars,
            overlap_blocks=overlap_blocks,
        )
        parcalar.append(
            GrammarChunk(
                context_blocks=context_blocks,
                target_blocks=target_blocks,
                target_start_index=start_index,
                target_end_index=end_index,
            )
        )

    return parcalar, total_serialized_chars, chunk_size, single_pass_limit, overlap_chars, overlap_blocks


def _build_grammar_prompt(system_prompt: str, chunk: GrammarChunk) -> tuple[str, str]:
    target_payload = dump_subtitle_block_payload(chunk.target_blocks)
    if chunk.context_blocks:
        context_payload = dump_subtitle_block_payload(chunk.context_blocks)
        context_section = (
            "REFERANS BAGLAM JSON:\n"
            f"{context_payload}\n\n"
            "Bu baglam sadece uslup ve akis icindir. Referans bloklari ASLA ciktiya ekleme.\n\n"
        )
    else:
        context_section = ""

    prompt = (
        f"{system_prompt}\n\n"
        "KRITIK CIKTI KONTRATI:\n"
        "- SADECE tek bir gecerli JSON nesnesi dondur.\n"
        '- JSON semasi: {"blocks":[{"id":"1","text_lines":["..."]}]}\n'
        "- `blocks` dizisinin uzunlugu HEDEF BLOKLAR ile birebir ayni olmali.\n"
        "- Her cikti blogundaki `id`, ayni siradaki hedef blok id'si ile birebir ayni olmali.\n"
        "- `text_lines` icinde SADECE duzeltilmis Turkce altyazi satirlari olmali.\n"
        "- Aciklama, analiz, yardimci-asistan cevabi, ozur, soru, not, markdown veya kod blogu yazma.\n"
        "- Asla Cin karakteri veya Turkce disi aciklama metni uretme.\n"
        "- Zaman kodu ya da index satiri yazma; onlar kod tarafinda korunuyor.\n"
        "- Anlami koru, gereksiz yere yeniden yazma.\n\n"
        f"{context_section}"
        "HEDEF BLOKLAR JSON:\n"
        f"{target_payload}\n\n"
        "Simdi SADECE gecerli JSON dondur."
    )
    return prompt, target_payload


def _build_retry_prompt(base_prompt: str, previous_raw: str, validation_error: str) -> str:
    preview = str(previous_raw or "").strip()[:4000]
    return (
        f"{base_prompt}\n\n"
        "ONCEKI CEVAP REDDEDILDI.\n"
        f"Ret nedeni: {validation_error}\n"
        "Asagidaki onceki cevap kurallari ihlal etti. Onu tekrar etme.\n"
        "Gecersiz onceki cevap:\n"
        f"{preview}\n\n"
        "Simdi kurallara birebir uyan tek bir gecerli JSON nesnesi dondur."
    )


def process_chunk(
    index: int,
    total_chunks: int,
    chunk: GrammarChunk,
    llm: CentralLLM,
    system_prompt: str,
    generate_diff_report: bool,
    debug_path: Path,
    chunk_label: str | None = None,
) -> tuple:
    target_str = serialize_srt_blocks(chunk.target_blocks)
    context_str = serialize_srt_blocks(chunk.context_blocks).strip() if chunk.context_blocks else ""
    prompt, source_payload = _build_grammar_prompt(system_prompt, chunk)
    display_label = chunk_label or f"{index + 1}/{total_chunks}"

    logger.info(
        f"Parça {display_label} LLM'e gönderiliyor... "
        f"(Hedef: {len(target_str)} karakter, Referans: {len(context_str)} karakter)"
    )
    timeout_seconds, max_retries, ollama_options, ollama_keep_alive = _resolve_grammar_llm_call_settings(llm)
    if ollama_options:
        logger.info(
            f"Parca {display_label} Ollama ayarlari: "
            f"timeout={timeout_seconds}s, max_retries={max_retries}, "
            f"num_ctx={ollama_options.get('num_ctx')}, keep_alive={ollama_keep_alive}, "
            f"temperature={ollama_options.get('temperature')}"
        )
    validation_attempts = _resolve_structured_attempts()
    sonuc = None
    last_error = ""
    last_raw_response = ""
    for attempt in range(1, validation_attempts + 1):
        active_prompt = prompt if attempt == 1 else _build_retry_prompt(prompt, last_raw_response, last_error)
        raw_response = llm.uret(
            active_prompt,
            timeout=timeout_seconds,
            max_retries=max_retries,
            ollama_options=ollama_options,
            ollama_keep_alive=ollama_keep_alive,
        )
        last_raw_response = str(raw_response or "").strip()

        try:
            replacement_lines = validate_structured_subtitle_response(
                last_raw_response,
                chunk.target_blocks,
                target_language_code="tr",
            )
            sonuc = rebuild_srt_from_replacements(chunk.target_blocks, replacement_lines)
            break
        except Exception as exc:
            last_error = str(exc)
            logger.warning(
                f"Parca {display_label} structured cevap reddedildi "
                f"({attempt}/{validation_attempts}): {last_error}"
            )
            append_debug_response(
                debug_path,
                f"Grammar | Parca {display_label} | Deneme {attempt}",
                last_error,
                last_raw_response,
                source_excerpt=source_payload[:2000],
            )

    if sonuc is None:
        raise ValueError(f"Structured grammar cevabi dogrulanamadi: {last_error or 'bilinmeyen hata'}")
    
    diff_text = None
    if generate_diff_report:
        diff_gen = difflib.unified_diff(
            target_str.splitlines(),
            sonuc.splitlines(),
            fromfile=f"Orijinal_Parca_{display_label}",
            tofile=f"Duzeltilmis_Parca_{display_label}",
            lineterm=""
        )
        diff_text = "\n".join(list(diff_gen))
    
    logger.info(f"✅ Parça {display_label} başarıyla düzenlendi.")
    return index, sonuc, diff_text


def process_chunk_with_rescue(
    index: int,
    total_chunks: int,
    chunk: GrammarChunk,
    llm: CentralLLM,
    system_prompt: str,
    generate_diff_report: bool,
    debug_path: Path,
    chunk_label: str | None = None,
    rescue_depth: int = 0,
) -> tuple:
    display_label = chunk_label or f"{index + 1}/{total_chunks}"

    try:
        return process_chunk(
            index,
            total_chunks,
            chunk,
            llm,
            system_prompt,
            generate_diff_report,
            debug_path,
            chunk_label=display_label,
        )
    except Exception as exc:
        max_rescue_depth = _resolve_rescue_split_depth()
        if len(chunk.target_blocks) <= 1 or rescue_depth >= max_rescue_depth:
            raise

        child_chunks = _split_grammar_chunk_for_rescue(chunk, llm)
        logger.warning(
            f"Parca {display_label} dogrudan duzeltilemedi ({exc}). "
            f"Kurtarma icin {len(chunk.target_blocks)} blokluk hedef "
            f"{len(child_chunks)} alt parcaya bolunuyor "
            f"(derinlik {rescue_depth + 1}/{max_rescue_depth})."
        )

        rescued_outputs: list[str] = []
        for child_index, child_chunk in enumerate(child_chunks, start=1):
            child_label = f"{display_label}.{child_index}"
            try:
                _, child_result, _ = process_chunk_with_rescue(
                    index,
                    total_chunks,
                    child_chunk,
                    llm,
                    system_prompt,
                    False,
                    debug_path,
                    chunk_label=child_label,
                    rescue_depth=rescue_depth + 1,
                )
                rescued_outputs.append(child_result.strip())
            except Exception as child_exc:
                raise ValueError(
                    f"{exc} | rescue_failed:{child_label}:{child_exc}"
                ) from child_exc

        rescued_text = "\n\n".join(item for item in rescued_outputs if item.strip())
        diff_text = None
        if generate_diff_report and rescued_text:
            target_str = serialize_srt_blocks(chunk.target_blocks)
            diff_gen = difflib.unified_diff(
                target_str.splitlines(),
                rescued_text.splitlines(),
                fromfile=f"Orijinal_Parca_{display_label}",
                tofile=f"Duzeltilmis_Parca_{display_label}",
                lineterm="",
            )
            diff_text = "\n".join(list(diff_gen))

        logger.info(f"Parca {display_label} alt parcalara bolunerek kurtarildi.")
        return index, rescued_text, diff_text


def _process_chunks(
    parcalar: list,
    llm: CentralLLM,
    sistem_talimati: str,
    max_workers: int,
    generate_diff_report: bool,
    debug_path: Path,
) -> tuple[list, list]:
    islenmis_parcalar = [None] * len(parcalar)
    rapor_icerikleri = [None] * len(parcalar)
    total_chunks = len(parcalar)

    if max_workers <= 1:
        logger.info("Gramer modulu sirali modda calisiyor. Ctrl+C aktif parcada daha hizli yanit verecek.")
        for idx, parca in enumerate(parcalar):
            try:
                index, sonuc, diff_text = process_chunk_with_rescue(
                    idx,
                    total_chunks,
                    parca,
                    llm,
                    sistem_talimati,
                    generate_diff_report,
                    debug_path,
                    chunk_label=f"{idx + 1}/{total_chunks}",
                )
                islenmis_parcalar[index] = sonuc
                rapor_icerikleri[index] = diff_text
            except KeyboardInterrupt:
                logger.warning("Kullanici tarafindan durdurma istendi. Gramer isleme guvenli sekilde sonlandiriliyor.")
                raise
            except Exception as e:
                logger.error(f"❌ Parça {idx + 1}/{total_chunks} işlenirken hata oluştu: {e}")
                islenmis_parcalar[idx] = serialize_srt_blocks(parcalar[idx].target_blocks)
                rapor_icerikleri[idx] = f"Parça {idx + 1} için hata oluştu (Fehler aufgetreten): {e}"
        return islenmis_parcalar, rapor_icerikleri

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            gelecek_gorevler = {
                executor.submit(
                    process_chunk_with_rescue,
                    i,
                    total_chunks,
                    parca,
                    llm,
                    sistem_talimati,
                    generate_diff_report,
                    debug_path,
                    f"{i + 1}/{total_chunks}",
                ): i
                for i, parca in enumerate(parcalar)
            }

            for future in concurrent.futures.as_completed(gelecek_gorevler):
                idx = gelecek_gorevler[future]
                try:
                    index, sonuc, diff_text = future.result()
                    islenmis_parcalar[index] = sonuc
                    rapor_icerikleri[index] = diff_text
                except Exception as e:
                    logger.error(f"❌ Parça {idx + 1}/{total_chunks} işlenirken hata oluştu: {e}")
                    islenmis_parcalar[idx] = serialize_srt_blocks(parcalar[idx].target_blocks)
                    rapor_icerikleri[idx] = f"Parça {idx + 1} için hata oluştu (Fehler aufgetreten): {e}"
    except KeyboardInterrupt:
        logger.warning("Kullanici tarafindan durdurma istendi. Bekleyen parcalar iptal ediliyor.")
        executor.shutdown(wait=False, cancel_futures=True)
        raise

    return islenmis_parcalar, rapor_icerikleri


def _write_grammar_report(girdi_dosyasi: Path, rapor_icerikleri: list, enabled: bool) -> None:
    if not enabled:
        logger.info("📄 Gramer degisiklik raporu kapali. Performans icin rapor olusturma atlandi.")
        return

    rapor_dosyasi = subtitle_intermediate_output_path("Gramer_Duzenleyici_Raporu.txt")
    with open(rapor_dosyasi, "w", encoding="utf-8") as f:
        f.write(f"=== GRAMER DÜZELTME RAPORU ({girdi_dosyasi.name}) ===\n")
        f.write("Açıklama: '-' işareti ile başlayan satırlar silinen/eski metni, '+' işareti ile başlayan satırlar eklenen/yeni metni gösterir.\n\n")
        for rapor in rapor_icerikleri:
            if rapor:
                f.write(rapor + "\n\n")

    logger.info(f"📄 Değişiklik raporu kaydedildi: {rapor_dosyasi.name}")


def _env_auto_or_positive_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    normalized = str(raw).strip().lower()
    if normalized in {"", "auto"}:
        return None
    try:
        value = int(normalized)
        return value if value > 0 else None
    except (TypeError, ValueError):
        logger.warning(f"Gecersiz {name} degeri bulundu: {raw}. Hızlı gramer varsayilanı kullanilacak.")
        return None


def _grammar_fast_batch_profile(llm: CentralLLM) -> tuple[int, int]:
    model_name = str(getattr(llm, "model_name", "")).strip().lower()
    if "qwen3:14b" in model_name or "gemma3:12b" in model_name:
        default_chars, default_blocks = 2400, 18
    elif "qwen3.5" in model_name:
        default_chars, default_blocks = 1800, 14
    else:
        default_chars, default_blocks = 1600, 12

    batch_chars = _env_auto_or_positive_int("GRAMMAR_FAST_BATCH_MAX_CHARS") or default_chars
    batch_blocks = _env_auto_or_positive_int("GRAMMAR_FAST_BATCH_MAX_BLOCKS") or default_blocks
    return batch_chars, batch_blocks


def _resolve_fast_validation_attempts() -> int:
    value = _env_auto_or_positive_int("GRAMMAR_FAST_MAX_ATTEMPTS")
    if value is None:
        return 2
    return max(1, min(value, 2))


def _resolve_fast_rescue_depth() -> int:
    requested = _resolve_rescue_split_depth()
    return min(requested, 1)


def _strip_code_fences(text: str) -> str:
    return re.sub(r"```(?:json)?\s*|```", "", str(text or ""), flags=re.IGNORECASE).strip()


def _replace_block_text(block: SrtBlock, lines: list[str]) -> SrtBlock:
    normalized_lines = normalize_text_lines(lines) or [""]
    raw = "\n".join([block.index_line or "", block.timing_line or "", *normalized_lines]).strip()
    return SrtBlock(
        raw=raw,
        index_line=block.index_line,
        timing_line=block.timing_line,
        text_lines=normalized_lines,
    )


def _deterministic_cleanup_text(text: str) -> str:
    cleaned = " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split())
    cleaned = re.sub(r"\s+([,.;!?%:])", r"\1", cleaned)
    cleaned = re.sub(r"([,.;!?])([A-Za-zÇĞİÖŞÜçğıöşü])", r"\1 \2", cleaned)
    cleaned = re.sub(r"\(\s+", "(", cleaned)
    cleaned = re.sub(r"\s+\)", ")", cleaned)
    cleaned = re.sub(r"\[\s+", "[", cleaned)
    cleaned = re.sub(r"\s+\]", "]", cleaned)
    cleaned = re.sub(r"\s+([”’])", r"\1", cleaned)
    cleaned = re.sub(r"([“‘])\s+", r"\1", cleaned)
    cleaned = re.sub(r"\.\.\.+", "...", cleaned)
    cleaned = re.sub(r"%([A-Za-zÇĞİÖŞÜçğıöşü])", r"% \1", cleaned)
    return cleaned.strip()


def _rebalance_text_to_lines(text: str, original_lines: list[str] | None) -> list[str]:
    cleaned = _deterministic_cleanup_text(text)
    if not cleaned:
        return normalize_text_lines(original_lines) or [""]

    original_lines = normalize_text_lines(original_lines) or [cleaned]
    line_count = max(1, len(original_lines))
    if line_count == 1:
        return [cleaned]

    words = cleaned.split()
    if len(words) <= line_count:
        return [" ".join([word]).strip() for word in words] + [""] * max(0, line_count - len(words))

    target_lengths = [max(8, len(line)) for line in original_lines]
    total_target = sum(target_lengths) or line_count
    remaining_words = list(words)
    output_lines: list[str] = []

    for idx in range(line_count):
        if idx == line_count - 1:
            output_lines.append(" ".join(remaining_words).strip())
            break

        remaining_lines = line_count - idx
        desired_ratio = target_lengths[idx] / total_target
        desired_words = max(1, round(len(words) * desired_ratio))
        max_take = max(1, len(remaining_words) - (remaining_lines - 1))
        take = min(max_take, desired_words)
        output_lines.append(" ".join(remaining_words[:take]).strip())
        del remaining_words[:take]

    return normalize_text_lines(output_lines) or [cleaned]


def _word_count(text: str) -> int:
    return len(GRAMMAR_WORD_RE.findall(str(text or "")))


def _block_needs_llm(block: SrtBlock, cleaned_text: str) -> bool:
    if not block.is_processable:
        return False
    if not cleaned_text or not GRAMMAR_WORD_RE.search(cleaned_text):
        return False

    word_count = _word_count(cleaned_text)
    text_length = len(cleaned_text)
    line_count = len(normalize_text_lines(block.text_lines))

    if word_count <= 4 and text_length <= 28:
        return False

    score = 0
    if text_length >= 42:
        score += 2
    if word_count >= 7:
        score += 2
    if line_count > 1:
        score += 1
    if cleaned_text and cleaned_text[0].islower():
        score += 1
    if not re.search(r"[.!?…]$", cleaned_text):
        score += 1
    if any(token in cleaned_text for token in ('"', "'", "(", ")", " - ", " / ")):
        score += 1
    if cleaned_text.count(",") >= 2:
        score += 1

    return score >= 3 or (text_length >= 52 and word_count >= 8)


def _build_fast_entries(blocks: list[SrtBlock]) -> tuple[list[SrtBlock], list[GrammarLineEntry]]:
    cleaned_blocks: list[SrtBlock] = []
    entries: list[GrammarLineEntry] = []

    for idx, block in enumerate(blocks):
        if not block.is_processable:
            cleaned_blocks.append(block)
            continue

        cleaned_text = _deterministic_cleanup_text(block.text_content)
        cleaned_lines = _rebalance_text_to_lines(cleaned_text, block.text_lines)
        cleaned_block = _replace_block_text(block, cleaned_lines)
        cleaned_blocks.append(cleaned_block)

        if _block_needs_llm(cleaned_block, cleaned_text):
            entries.append(
                GrammarLineEntry(
                    block_index=idx,
                    block_id=block.id,
                    source_block=block,
                    cleaned_text=cleaned_text,
                )
            )

    return cleaned_blocks, entries


def _build_fast_batches(entries: list[GrammarLineEntry], llm: CentralLLM) -> list[list[GrammarLineEntry]]:
    if not entries:
        return []

    max_chars, max_blocks = _grammar_fast_batch_profile(llm)
    batches: list[list[GrammarLineEntry]] = []
    current_batch: list[GrammarLineEntry] = []
    current_chars = 0

    for entry in entries:
        entry_len = len(entry.cleaned_text) + len(entry.block_id) + 4
        if current_batch and (len(current_batch) >= max_blocks or current_chars + entry_len > max_chars):
            batches.append(current_batch)
            current_batch = []
            current_chars = 0
        current_batch.append(entry)
        current_chars += entry_len

    if current_batch:
        batches.append(current_batch)
    return batches


def _build_fast_grammar_prompt(
    batch: list[GrammarLineEntry],
    target_language_code: str = "tr",
    short_form: bool = False,
) -> tuple[str, str]:
    payload_lines = [f"{entry.block_id}\t{entry.cleaned_text}" for entry in batch]
    payload = "\n".join(payload_lines)
    if target_language_code == "en":
        language_title = "English"
        role_text = "fast professional English subtitle editor"
        fix_rules = (
            "- Fix only spelling, punctuation, casing and obvious grammar issues.\n"
            "- Keep the meaning and subtitle brevity.\n"
            "- Do not translate, localize or rewrite into another language.\n"
            "- Output natural English subtitle lines only.\n"
        )
        example_in = "12\tthis is actualy a really good idea"
        example_out = "12\tThis is actually a really good idea."
        example_in_2 = "13\twe should probly send the email today"
        example_out_2 = "13\tWe should probably send the email today."
        leak_rule = "- Do not output Turkish explanations or assistant commentary.\n"
    else:
        language_title = "Turkce"
        role_text = "hizli calisan profesyonel bir Turkce redaktor"
        fix_rules = (
            "- Yalnizca yazim, noktalama ve gramer duzelt.\n"
            "- Anlami koru, gereksiz yeniden yazim yapma.\n"
            "- Turkce altyazi dogalligini koru.\n"
        )
        example_in = "12\tbugun hava cok guzel oldu"
        example_out = "12\tBugun hava cok guzel oldu."
        example_in_2 = "13\tbu gercekten ilginc bir durum"
        example_out_2 = "13\tBu gercekten ilginc bir durum."
        leak_rule = "- Turkce disi aciklama uretme.\n"

    short_rule = (
        "- Bu girdiler shorts/reel altyazisidir; satirlari kisa ve okunur tut.\n"
        if short_form
        else ""
    )
    prompt = (
        f"Sen {role_text}sun.\n"
        f"Asagidaki her satir bir {language_title} altyazi blogunu temsil ediyor.\n"
        "Her girdi satiri su formattadir: ID<TAB>METIN\n\n"
        "Gorevin:\n"
        f"{fix_rules}"
        f"{short_rule}"
        "- Her satir icin yalnizca TEK SATIRLIK sonuc don.\n"
        "- Cikti formati birebir: ID<TAB>DUZELTILMIS_METIN\n"
        "- Ayni ID'leri ve ayni sirayi koru.\n"
        "- Markdown, JSON, analiz, aciklama, kod blogu, not veya asistan cevabi yazma.\n"
        f"{leak_rule}\n"
        "ORNEK:\n"
        f"{example_in}\n"
        f"{example_in_2}\n\n"
        "DOGRU CIKTI ORNEGI:\n"
        f"{example_out}\n"
        f"{example_out_2}\n\n"
        "SIMDI CIKTI URET:\n"
        f"{payload}"
    )
    return prompt, payload


def _build_fast_retry_prompt(base_prompt: str, previous_raw: str, validation_error: str) -> str:
    preview = str(previous_raw or "").strip()[:2500]
    return (
        f"{base_prompt}\n\n"
        "ONEMLI UYARI:\n"
        f"- Onceki cevap reddedildi: {validation_error}\n"
        "- Sadece ID<TAB>METIN satirlarini dondur.\n"
        "- Toplam satir sayisi girdideki kadar olmali.\n"
        "- Her satir tek satirlik olmali.\n"
        "- Gecersiz onceki cevap ozeti:\n"
        f"{preview}"
    )


def _parse_fast_grammar_response(raw_text: str, batch: list[GrammarLineEntry]) -> dict[int, str]:
    cleaned = _strip_code_fences(raw_text)
    expected_ids = [entry.block_id for entry in batch]
    matched_lines = []

    for line in cleaned.splitlines():
        line = str(line).strip()
        if not line:
            continue
        match = GRAMMAR_LINE_RE.match(line)
        if match:
            matched_lines.append((match.group(1).strip(), _deterministic_cleanup_text(match.group(2))))

    if len(matched_lines) == len(batch):
        parsed_items = matched_lines
    else:
        raw_lines = [str(line).strip() for line in cleaned.splitlines() if str(line).strip()]
        if len(raw_lines) != len(batch):
            raise ValueError(f"line_count_mismatch:{len(raw_lines)}!={len(batch)}")
        parsed_items = []
        for expected_id, raw_line in zip(expected_ids, raw_lines):
            line = re.sub(r"^\s*\d+\s*(?:[\t|:.-])\s*", "", raw_line).strip()
            parsed_items.append((expected_id, _deterministic_cleanup_text(line)))

    corrected: dict[int, str] = {}
    issues: list[str] = []
    for entry, (item_id, text) in zip(batch, parsed_items):
        if item_id != entry.block_id:
            issues.append(f"{entry.block_id}:id_mismatch:{item_id}")
        if not text:
            issues.append(f"{entry.block_id}:empty")
            text = entry.cleaned_text
        lowered = text.casefold()
        if any(marker in lowered for marker in GRAMMAR_META_MARKERS):
            issues.append(f"{entry.block_id}:meta_leak")
        if re.search(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]", text):
            issues.append(f"{entry.block_id}:cjk_chars")
        if len(text) > max(120, len(entry.cleaned_text) * 3):
            issues.append(f"{entry.block_id}:length_blowup")
        corrected[entry.block_index] = text

    if issues:
        raise ValueError("; ".join(issues))
    return corrected


def _resolve_fast_llm_call_settings(llm: CentralLLM) -> tuple[int, int, dict | None, str | None]:
    timeout_seconds, max_retries, ollama_options, ollama_keep_alive = _resolve_grammar_llm_call_settings(llm)
    timeout_seconds = min(timeout_seconds, 120)
    max_retries = min(max_retries, 1)
    if ollama_options:
        ollama_options = dict(ollama_options)
        ollama_options["temperature"] = 0.0
        ollama_options["num_ctx"] = min(int(ollama_options.get("num_ctx", 6144)), 6144)
    return timeout_seconds, max_retries, ollama_options, ollama_keep_alive


def _process_fast_batch(
    batch: list[GrammarLineEntry],
    batch_label: str,
    llm: CentralLLM,
    debug_path: Path,
    target_language_code: str = "tr",
    short_form: bool = False,
) -> dict[int, str]:
    prompt, source_payload = _build_fast_grammar_prompt(batch, target_language_code=target_language_code, short_form=short_form)
    timeout_seconds, max_retries, ollama_options, ollama_keep_alive = _resolve_fast_llm_call_settings(llm)
    validation_attempts = _resolve_fast_validation_attempts()

    logger.info(
        f"Hizli gramer batch {batch_label} LLM'e gonderiliyor... "
        f"({len(batch)} blok, {len(source_payload)} karakter)"
    )
    if ollama_options:
        logger.info(
            f"Gramer hizli batch {batch_label} Ollama ayarlari: "
            f"timeout={timeout_seconds}s, max_retries={max_retries}, "
            f"num_ctx={ollama_options.get('num_ctx')}, keep_alive={ollama_keep_alive}, "
            f"temperature={ollama_options.get('temperature')}"
        )

    last_error = ""
    last_raw_response = ""
    for attempt in range(1, validation_attempts + 1):
        active_prompt = prompt if attempt == 1 else _build_fast_retry_prompt(prompt, last_raw_response, last_error)
        raw_response = llm.uret(
            active_prompt,
            timeout=timeout_seconds,
            max_retries=max_retries,
            ollama_options=ollama_options,
            ollama_keep_alive=ollama_keep_alive,
        )
        last_raw_response = str(raw_response or "").strip()
        try:
            return _parse_fast_grammar_response(last_raw_response, batch)
        except Exception as exc:
            last_error = str(exc)
            logger.warning(
                f"Gramer hizli batch {batch_label} reddedildi "
                f"({attempt}/{validation_attempts}): {last_error}"
            )
            append_debug_response(
                debug_path,
                f"GrammarFast | Batch {batch_label} | Deneme {attempt}",
                last_error,
                last_raw_response,
                source_excerpt=source_payload[:2000],
            )

    raise ValueError(f"Hizli gramer batch dogrulanamadi: {last_error or 'bilinmeyen hata'}")


def _process_fast_batch_with_fallback(
    batch: list[GrammarLineEntry],
    batch_label: str,
    llm: CentralLLM,
    debug_path: Path,
    rescue_depth: int = 0,
    target_language_code: str = "tr",
    short_form: bool = False,
) -> dict[int, str]:
    try:
        return _process_fast_batch(
            batch,
            batch_label,
            llm,
            debug_path,
            target_language_code=target_language_code,
            short_form=short_form,
        )
    except Exception as exc:
        max_rescue_depth = _resolve_fast_rescue_depth()
        if len(batch) <= 2 or rescue_depth >= max_rescue_depth:
            logger.warning(
                f"Gramer hizli batch {batch_label} kurtarilamadi ({exc}). "
                "Bu batch deterministik duzeltme ile devam edecek."
            )
            return {entry.block_index: entry.cleaned_text for entry in batch}

        midpoint = max(1, len(batch) // 2)
        logger.warning(
            f"Gramer hizli batch {batch_label} dogrudan gecemedi ({exc}). "
            f"Kurtarma icin {len(batch)} blok 2 alt batch'e bolunuyor."
        )
        merged: dict[int, str] = {}
        for child_index, child_batch in enumerate((batch[:midpoint], batch[midpoint:]), start=1):
            child_label = f"{batch_label}.{child_index}"
            merged.update(
                _process_fast_batch_with_fallback(
                    child_batch,
                    child_label,
                    llm,
                    debug_path,
                    rescue_depth=rescue_depth + 1,
                    target_language_code=target_language_code,
                    short_form=short_form,
                )
            )
        return merged


def _build_global_diff_report(original_blocks: list[SrtBlock], final_blocks: list[SrtBlock]) -> list[str]:
    original_text = serialize_srt_blocks(original_blocks)
    final_text = serialize_srt_blocks(final_blocks)
    if original_text == final_text:
        return []
    diff_text = "\n".join(
        difflib.unified_diff(
            original_text.splitlines(),
            final_text.splitlines(),
            fromfile="Orijinal_Altyazi",
            tofile="Duzeltilmis_Altyazi",
            lineterm="",
        )
    )
    return [diff_text] if diff_text else []


def _run_fast_grammar_pipeline(
    bloklar: list[SrtBlock],
    llm: CentralLLM,
    generate_diff_report: bool,
    debug_path: Path,
    target_language_code: str = "tr",
    short_form: bool = False,
) -> tuple[str, list]:
    cleaned_blocks, llm_entries = _build_fast_entries(bloklar)
    final_blocks = list(cleaned_blocks)
    processable_count = sum(1 for block in bloklar if block.is_processable)
    deterministic_only = processable_count - len(llm_entries)

    logger.info(
        f"Hizli gramer modu etkin. Toplam {processable_count} islenebilir bloktan "
        f"{len(llm_entries)} blok LLM duzeltmesine aday, {deterministic_only} blok sadece "
        "deterministik temizlikle geciyor."
    )

    batches = _build_fast_batches(llm_entries, llm)
    batch_chars, batch_blocks = _grammar_fast_batch_profile(llm)
    logger.info(
        f"Hizli gramer LLM modu {len(batches)} batch ile calisacak. "
        f"Batch hedefi {batch_chars} karakter / {batch_blocks} blok."
    )
    if str(getattr(llm, "provider", "")).strip().upper() == "OLLAMA":
        logger.info("Hizli gramer batch'leri OLLAMA icin sirali olarak islenecek.")

    for batch_index, batch in enumerate(batches, start=1):
        corrections = _process_fast_batch_with_fallback(
            batch,
            f"{batch_index}/{len(batches)}",
            llm,
            debug_path,
            target_language_code=target_language_code,
            short_form=short_form,
        )
        for entry in batch:
            corrected_text = corrections.get(entry.block_index, entry.cleaned_text)
            reflowed_lines = _rebalance_text_to_lines(corrected_text, entry.source_block.text_lines)
            final_blocks[entry.block_index] = _replace_block_text(entry.source_block, reflowed_lines)

    report_entries = _build_global_diff_report(bloklar, final_blocks) if generate_diff_report else []
    return serialize_srt_blocks(final_blocks), report_entries


def _resolve_manual_input_file() -> Path | None:
    preferred_input = find_subtitle_artifact(GRAMMAR_INPUT_NAME)
    if preferred_input and preferred_input.exists():
        logger.info(f"Gramer modulu dogrudan 101 cikisini kullanacak: {preferred_input.name}")
        return preferred_input

    srt_dosyalari = list_subtitle_files()
    if not srt_dosyalari:
        return None

    logger.warning(
        f"Beklenen 101 cikisi bulunamadi: {GRAMMAR_INPUT_NAME}. "
        "Uyumluluk icin manuel dosya secimine donuluyor."
    )
    print("\n100_Altyazı dizininde bulunan SRT dosyaları:")
    for i, dosya in enumerate(srt_dosyalari, 1):
        print(f"[{i}] {dosya.name}")

    while True:
        try:
            secim_str = input(
                f"\nLütfen işlemek istediğiniz dosyanın numarasını girin (1-{len(srt_dosyalari)}): "
            )
            secim_idx = int(secim_str) - 1

            if 0 <= secim_idx < len(srt_dosyalari):
                girdi_dosyasi = srt_dosyalari[secim_idx]
                logger.info(f"Seçilen dosya (Ausgewählte Datei): {girdi_dosyasi.name}")
                return girdi_dosyasi

            print("❌ Geçersiz numara. Lütfen listedeki numaralardan birini girin.")
        except ValueError:
            print("❌ Geçersiz format. Lütfen sayısal bir değer girin.")

    return None


def _process_subtitle_variant(
    *,
    input_path: Path,
    output_filename: str,
    llm: CentralLLM,
    debug_path: Path,
    generate_diff_report: bool,
    target_language_code: str,
    short_form: bool = False,
    glossary_terms: list[dict] | None = None,
    glossary_fixed_filename: str | None = None,
    write_report: bool = False,
) -> Path:
    icerik = read_srt_file(input_path)
    original_blocks = parse_srt_blocks(icerik)
    working_blocks = list(original_blocks)

    if glossary_terms:
        working_blocks, glossary_stats = _apply_video_glossary(working_blocks, glossary_terms)
        if glossary_fixed_filename:
            glossary_fixed_path = subtitle_intermediate_output_path(glossary_fixed_filename)
            write_srt_file(glossary_fixed_path, serialize_srt_blocks(working_blocks))
            logger.info(
                f"Glossary uygulanmis ara dosya kaydedildi: {glossary_fixed_path.name} | "
                f"Duzeltme={glossary_stats.get('replacements', 0)}"
            )

    nihai_srt, rapor_icerikleri = _run_fast_grammar_pipeline(
        working_blocks,
        llm,
        generate_diff_report,
        debug_path,
        target_language_code=target_language_code,
        short_form=short_form,
    )

    cikti_dosyasi = subtitle_output_path(output_filename)
    write_srt_file(cikti_dosyasi, nihai_srt)
    if write_report:
        _write_grammar_report(input_path, rapor_icerikleri, generate_diff_report)
    logger.info(f"🎉 Altyazi duzeltme tamamlandi: {cikti_dosyasi.name}")
    return cikti_dosyasi

def run():
    started_at = time.perf_counter()
    print("\n" + "="*50)
    print("✨ AŞAMA 2: Yapay Zeka ile Gramer ve İmla Düzenleme")
    print("="*50)

    relocate_known_subtitle_intermediates()

    # 1. Normal akista 101 cikisini dogrudan kullan, yoksa uyumluluk icin secime don
    girdi_dosyasi = _resolve_manual_input_file()
    if girdi_dosyasi is None:
        logger.error("❌ 100_Altyazı dizininde kullanılabilir hiçbir SRT dosyası bulunamadı!")
        return

    # 2. Yapay Zeka Motoru Secimi ve Baslatilmasi
    use_recommended = prompt_module_llm_plan("102", needs_main=True, needs_smart=False)
    if use_recommended:
        saglayici, model_adi = get_module_recommended_llm_config("102", "main")
        print_module_llm_choice_summary("102", {"main": (saglayici, model_adi)})
    else:
        saglayici, model_adi = select_llm("main")

    try:
        llm = CentralLLM(provider=saglayici, model_name=model_adi)
    except Exception as e:
        logger.error(f"LLM Başlatılamadı: {e}")
        return

    max_workers = _resolve_worker_count(llm)
    generate_diff_report = _grammar_diff_report_enabled()
    debug_path = subtitle_intermediate_output_path(GRAMMAR_DEBUG_OUTPUT_NAME)
    prepare_debug_file(debug_path, f"GRAMER LLM DEBUG ({girdi_dosyasi.name})")
    logger.info(f"Gramer modulu {max_workers} paralel is parcacigi ile calisacak.")
    logger.info(f"Gramer degisiklik raporu {'acik' if generate_diff_report else 'kapali'} calisacak.")
    logger.info("Yeni hizli gramer mimarisi kullaniliyor: deterministik temizlik + secmeli hafif LLM duzeltmesi.")

    _tr_blocks, glossary_meta = _prepare_glossary_fixed_blocks(girdi_dosyasi, llm, debug_path)
    glossary_terms = glossary_meta.get("terms", []) or []
    if glossary_meta.get("enabled"):
        logger.info(
            f"Video-ozel glossary katmani aktif. Terim={len(glossary_terms)} | "
            f"Duzeltme={glossary_meta.get('replacements', 0)}"
        )

    logger.info(f"Turkce gramer duzeltmesi basliyor: {girdi_dosyasi.name}")
    cikti_dosyasi = _process_subtitle_variant(
        input_path=girdi_dosyasi,
        output_filename=GRAMMAR_OUTPUT_NAME,
        llm=llm,
        debug_path=debug_path,
        generate_diff_report=generate_diff_report,
        target_language_code="tr",
        glossary_terms=glossary_terms,
        glossary_fixed_filename=GRAMMAR_GLOSSARY_FIXED_INPUT_NAME,
        write_report=True,
    )

    raw_en_path = find_subtitle_artifact(GRAMMAR_EN_INPUT_NAME)
    if _grammar_process_english_enabled() and raw_en_path and raw_en_path.exists():
        logger.info(f"Ingilizce gramer duzeltmesi basliyor: {raw_en_path.name}")
        _process_subtitle_variant(
            input_path=raw_en_path,
            output_filename=GRAMMAR_EN_OUTPUT_NAME,
            llm=llm,
            debug_path=debug_path,
            generate_diff_report=False,
            target_language_code="en",
        )

    raw_shorts_path = find_subtitle_artifact(GRAMMAR_SHORTS_INPUT_NAME)
    if _grammar_process_shorts_enabled() and raw_shorts_path and raw_shorts_path.exists():
        logger.info(f"Shorts gramer duzeltmesi basliyor: {raw_shorts_path.name}")
        _process_subtitle_variant(
            input_path=raw_shorts_path,
            output_filename=GRAMMAR_SHORTS_OUTPUT_NAME,
            llm=llm,
            debug_path=debug_path,
            generate_diff_report=False,
            target_language_code="tr",
            short_form=True,
            glossary_terms=glossary_terms,
            glossary_fixed_filename="subtitle_raw_shorts_glossary_fixed.srt",
        )

    logger.info(f"🎉 Gramer düzenleme işlemi TAMAMLANDI! Çıktı dosyası: {cikti_dosyasi.name}")
    elapsed = format_elapsed(time.perf_counter() - started_at)
    logger.info(f"⏱️ Modul 2 toplam sure: {elapsed}")

def run_automatic(girdi_dosyasi: Path, llm: CentralLLM) -> Path:
    """Tam otomasyon (Pipeline) için dışarıdan dosya ve LLM objesi alarak çalışır (Raporlu)."""
    started_at = time.perf_counter()
    logger.info(f"🔄 OTOMASYON: {girdi_dosyasi.name} gramer kontrolüne sokuluyor...")

    relocate_known_subtitle_intermediates()
    max_workers = _resolve_worker_count(llm)
    generate_diff_report = _grammar_diff_report_enabled()
    debug_path = subtitle_intermediate_output_path(GRAMMAR_DEBUG_OUTPUT_NAME)
    prepare_debug_file(debug_path, f"GRAMER LLM DEBUG ({girdi_dosyasi.name})")
    logger.info(f"Gramer modulu {max_workers} paralel is parcacigi ile calisacak.")
    logger.info(f"Gramer degisiklik raporu {'acik' if generate_diff_report else 'kapali'} calisacak.")
    logger.info("Yeni hizli gramer mimarisi kullaniliyor: deterministik temizlik + secmeli hafif LLM duzeltmesi.")

    _tr_blocks, glossary_meta = _prepare_glossary_fixed_blocks(girdi_dosyasi, llm, debug_path)
    glossary_terms = glossary_meta.get("terms", []) or []
    if glossary_meta.get("enabled"):
        logger.info(
            f"Video-ozel glossary katmani aktif. Terim={len(glossary_terms)} | "
            f"Duzeltme={glossary_meta.get('replacements', 0)}"
        )

    logger.info(f"Turkce gramer duzeltmesi basliyor: {girdi_dosyasi.name}")
    cikti_dosyasi = _process_subtitle_variant(
        input_path=girdi_dosyasi,
        output_filename=GRAMMAR_OUTPUT_NAME,
        llm=llm,
        debug_path=debug_path,
        generate_diff_report=generate_diff_report,
        target_language_code="tr",
        glossary_terms=glossary_terms,
        glossary_fixed_filename=GRAMMAR_GLOSSARY_FIXED_INPUT_NAME,
        write_report=True,
    )

    raw_en_path = find_subtitle_artifact(GRAMMAR_EN_INPUT_NAME)
    if _grammar_process_english_enabled() and raw_en_path and raw_en_path.exists():
        logger.info(f"Ingilizce gramer duzeltmesi basliyor: {raw_en_path.name}")
        _process_subtitle_variant(
            input_path=raw_en_path,
            output_filename=GRAMMAR_EN_OUTPUT_NAME,
            llm=llm,
            debug_path=debug_path,
            generate_diff_report=False,
            target_language_code="en",
        )

    raw_shorts_path = find_subtitle_artifact(GRAMMAR_SHORTS_INPUT_NAME)
    if _grammar_process_shorts_enabled() and raw_shorts_path and raw_shorts_path.exists():
        logger.info(f"Shorts gramer duzeltmesi basliyor: {raw_shorts_path.name}")
        _process_subtitle_variant(
            input_path=raw_shorts_path,
            output_filename=GRAMMAR_SHORTS_OUTPUT_NAME,
            llm=llm,
            debug_path=debug_path,
            generate_diff_report=False,
            target_language_code="tr",
            short_form=True,
            glossary_terms=glossary_terms,
            glossary_fixed_filename="subtitle_raw_shorts_glossary_fixed.srt",
        )

    logger.info(f"🎉 Gramer düzenleme TAMAMLANDI: {cikti_dosyasi.name}")
    elapsed = format_elapsed(time.perf_counter() - started_at)
    logger.info(f"⏱️ Modul 2 toplam sure: {elapsed}")
    
    return cikti_dosyasi # Çeviriciye bu düzeltilmiş SRT'yi göndereceğiz 

