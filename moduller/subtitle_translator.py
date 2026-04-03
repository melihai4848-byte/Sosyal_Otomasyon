# moduller/altyazi_cevirmeni.py
import os
import time
from moduller.logger import get_logger
from moduller.runtime_utils import format_elapsed
from moduller.srt_utils import (
    parse_srt_blocks,
    read_srt_file,
    serialize_srt_blocks,
    write_srt_file,
)
from moduller.llm_manager import CentralLLM
from moduller.social_media_utils import build_base_stem
from moduller.subtitle_llm_utils import (
    append_debug_response,
    dump_subtitle_block_payload,
    prepare_debug_file,
    rebuild_srt_from_replacements,
    validate_structured_subtitle_response,
)
from moduller.subtitle_output_utils import (
    list_subtitle_files,
    relocate_known_subtitle_intermediates,
    subtitle_intermediate_output_path,
    subtitle_output_path,
)
from pathlib import Path

logger = get_logger("translation")

TRANSLATION_INPUT_NAME = "subtitle_tr.srt"
TRANSLATION_OUTPUT_NAMES = {
    "en": "subtitle_llm_en.srt",
    "de": "subtitle_de.srt",
}
TRANSLATION_DEBUG_OUTPUT_NAME = "translation_llm_debug.txt"
TRANSLATEGEMMA_DEFAULT_MODEL = "translategemma:12b-it-q8_0"
TRANSLATION_PROVIDER = "OLLAMA"


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


def _resolve_structured_attempts() -> int:
    return _env_auto_or_int("TRANSLATION_STRUCTURED_MAX_ATTEMPTS", 1) or 3


def _resolve_rescue_split_depth() -> int:
    return _env_auto_or_int("TRANSLATION_RESCUE_MAX_SPLIT_DEPTH", 0) or 3


def _resolve_translategemma_model_name() -> str:
    return _env_auto_or_text("TRANSLATEGEMMA_MODEL_NAME") or TRANSLATEGEMMA_DEFAULT_MODEL


def _create_translation_llm() -> CentralLLM:
    model_adi = _resolve_translategemma_model_name()
    logger.info(
        f"Ceviri modeli env uzerinden secildi. Saglayici={TRANSLATION_PROVIDER} | Model={model_adi}"
    )
    return CentralLLM(provider=TRANSLATION_PROVIDER, model_name=model_adi)


def _translation_source_fallback_allowed() -> bool:
    return _env_bool("TRANSLATION_ALLOW_SOURCE_FALLBACK", False)


def _translation_english_enabled() -> bool:
    return _env_bool("TRANSLATION_ENABLE_ENGLISH", True)


def _translation_german_enabled() -> bool:
    return _env_bool("TRANSLATION_ENABLE_GERMAN", True)


def _resolve_translation_targets() -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    if _translation_english_enabled():
        targets.append(("en", "İngilizce (Amerikan)"))
    if _translation_german_enabled():
        targets.append(("de", "Almanca"))
    return targets


def _is_translategemma_model(llm: CentralLLM) -> bool:
    model_name = str(getattr(llm, "model_name", "")).strip().lower()
    return "translategemma" in model_name


def _serialized_length(blocks: list) -> int:
    if not blocks:
        return 0
    return len(serialize_srt_blocks(blocks))


def _translation_payload_length(blocks: list) -> int:
    if not blocks:
        return 0
    return len(dump_subtitle_block_payload(blocks))


def _translation_system_prompt(target_lang_en: str) -> str:
    return (
        f"You are a professional and highly accurate subtitle translator. "
        f"Translate the following Turkish SRT subtitles into natural and fluent {target_lang_en}. "
        f"CRITICAL RULES: "
        f"1. Preserve meaning, tone, and subtitle brevity. "
        f"2. NEVER answer as a chat assistant, explain the source sentence, or ask follow-up questions. "
        f"3. WARNING: UNDER NO CIRCUMSTANCES should you output any Chinese characters. "
        f"4. The output language MUST be entirely {target_lang_en}, except for names or unavoidable brand terms."
    )


def _translation_chunk_profile(llm: CentralLLM) -> tuple[int, int, int, int]:
    provider = str(getattr(llm, "provider", "")).strip().upper()
    model_name = str(getattr(llm, "model_name", "")).strip().lower()

    if provider == "OLLAMA":
        if "translategemma" in model_name:
            return 1800, 2600, 18, 28
        if "gemma3:12b-it-q8_0" in model_name:
            return 2600, 3600, 22, 34
        if model_name.startswith("gemma3:12b"):
            return 2400, 3400, 20, 32
        if model_name.startswith("gemma3:"):
            return 2200, 3200, 18, 30
        if "qwen3:14b" in model_name:
            return 2400, 3400, 20, 32
        return 2000, 2800, 16, 26

    return 2600, 3800, 22, 36


def _translation_ollama_profile(llm: CentralLLM) -> tuple[int, int, int, str, float]:
    model_name = str(getattr(llm, "model_name", "")).strip().lower()

    if "translategemma" in model_name:
        return 360, 1, 8192, "20m", 0.0
    if "gemma3:12b-it-q8_0" in model_name:
        return 480, 1, 8192, "30m", 0.0
    if model_name.startswith("gemma3:12b"):
        return 420, 1, 8192, "30m", 0.0
    if model_name.startswith("gemma3:"):
        return 360, 1, 7168, "20m", 0.0
    if "qwen3:14b" in model_name:
        return 420, 1, 8192, "20m", 0.0
    return 300, 1, 6144, "15m", 0.0


def _resolve_translation_llm_call_settings(llm: CentralLLM) -> tuple[int, int, dict | None, str | None]:
    provider = str(getattr(llm, "provider", "")).strip().upper()
    if provider != "OLLAMA":
        return 120, 2, None, None

    (
        default_timeout,
        default_max_retries,
        default_num_ctx,
        default_keep_alive,
        default_temperature,
    ) = _translation_ollama_profile(llm)

    timeout_seconds = _env_auto_or_int("TRANSLATION_OLLAMA_TIMEOUT_SECONDS", 60) or default_timeout
    max_retries = _env_auto_or_int("TRANSLATION_OLLAMA_MAX_RETRIES", 0)
    if max_retries is None:
        max_retries = default_max_retries
    num_ctx = _env_auto_or_int("TRANSLATION_OLLAMA_NUM_CTX", 2048) or default_num_ctx
    keep_alive = _env_auto_or_text("TRANSLATION_OLLAMA_KEEP_ALIVE") or default_keep_alive
    temperature = _env_auto_or_float("TRANSLATION_OLLAMA_TEMPERATURE")
    if temperature is None:
        temperature = default_temperature

    return timeout_seconds, max_retries, {"temperature": temperature, "num_ctx": num_ctx}, keep_alive


def _translation_prompt_overhead(llm: CentralLLM) -> int:
    empty_payload = dump_subtitle_block_payload([])
    overheads = []
    for target_lang_en, target_lang_code in (("English", "en"), ("German", "de")):
        system_prompt = _translation_system_prompt(target_lang_en)
        prompt, _ = _build_translation_prompt(
            system_prompt,
            [],
            target_lang_en,
            target_lang_code,
            llm,
        )
        overheads.append(max(0, len(prompt) - len(empty_payload)))
    return max(overheads, default=0)


def _estimated_payload_budget(text_budget: int, llm: CentralLLM) -> int:
    prompt_overhead = _translation_prompt_overhead(llm)
    payload_allowance = min(prompt_overhead, max(320, text_budget // 2))
    return max(700, text_budget + payload_allowance)


def _chunk_translation_blocks(
    blocks: list,
    max_text_chars: int,
    max_payload_chars: int,
    max_blocks: int,
) -> list[list]:
    chunks: list[list] = []
    current_chunk: list = []
    current_text_length = 0
    current_payload_length = 0

    for block in blocks:
        block_text_length = len(block.text_content) if block.is_processable else 0
        block_payload_length = _translation_payload_length([block])
        exceeds_budget = (
            current_chunk
            and (
                current_text_length + block_text_length > max_text_chars
                or current_payload_length + block_payload_length > max_payload_chars
                or len(current_chunk) >= max_blocks
            )
        )
        if exceeds_budget:
            chunks.append(current_chunk)
            current_chunk = []
            current_text_length = 0
            current_payload_length = 0

        current_chunk.append(block)
        current_text_length += block_text_length
        current_payload_length += block_payload_length

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _looks_complex_translation_chunk(
    chunk: list,
    max_text_chars: int,
    max_payload_chars: int,
    max_blocks: int,
) -> bool:
    if len(chunk) <= 1:
        return False

    text_length = sum(len(block.text_content) if block.is_processable else 0 for block in chunk)
    payload_length = _translation_payload_length(chunk)
    short_blocks = sum(
        1
        for block in chunk
        if block.is_processable and len(block.text_content.strip()) <= 18
    )
    avg_text_length = text_length / max(1, len(chunk))

    if len(chunk) >= max_blocks and avg_text_length <= 35:
        return True
    if (
        payload_length >= int(max_payload_chars * 0.94)
        and len(chunk) >= max(10, int(max_blocks * 0.75))
        and avg_text_length <= 40
    ):
        return True
    if (
        text_length >= int(max_text_chars * 0.94)
        and len(chunk) >= max(10, int(max_blocks * 0.75))
        and avg_text_length <= 40
    ):
        return True
    if len(chunk) >= max(8, int(max_blocks * 0.65)) and avg_text_length <= 32:
        return True
    if len(chunk) >= 8 and short_blocks >= max(5, int(len(chunk) * 0.6)):
        return True
    return False


def _refine_translation_chunks(
    chunks: list[list],
    max_text_chars: int,
    max_payload_chars: int,
    max_blocks: int,
) -> list[list]:
    refined: list[list] = []
    smaller_text_budget = max(500, int(max_text_chars * 0.72))
    smaller_payload_budget = max(800, int(max_payload_chars * 0.72))
    smaller_block_limit = max(6, max_blocks - 4)

    for chunk in chunks:
        if _looks_complex_translation_chunk(chunk, max_text_chars, max_payload_chars, max_blocks):
            refined.extend(
                _chunk_translation_blocks(
                    chunk,
                    smaller_text_budget,
                    smaller_payload_budget,
                    smaller_block_limit,
                )
            )
        else:
            refined.append(chunk)

    return refined


def _resolve_translation_chunk_strategy(llm: CentralLLM, bloklar: list) -> tuple[list, int, int, int, int, int]:
    total_serialized_chars = _serialized_length(bloklar)
    total_payload_chars = _translation_payload_length(bloklar)
    (
        default_chunk_size,
        default_single_pass_limit,
        default_chunk_max_blocks,
        default_single_pass_max_blocks,
    ) = _translation_chunk_profile(llm)
    chunk_size = _env_auto_or_int("TRANSLATION_CHUNK_MAX_CHARS", 500) or default_chunk_size
    single_pass_limit = (
        _env_auto_or_int("TRANSLATION_SINGLE_PASS_MAX_CHARS", 1000) or default_single_pass_limit
    )
    chunk_max_blocks = (
        _env_auto_or_int("TRANSLATION_CHUNK_MAX_BLOCKS", 2) or default_chunk_max_blocks
    )
    single_pass_max_blocks = (
        _env_auto_or_int("TRANSLATION_SINGLE_PASS_MAX_BLOCKS", 2) or default_single_pass_max_blocks
    )
    chunk_payload_budget = _estimated_payload_budget(chunk_size, llm)
    single_pass_payload_budget = _estimated_payload_budget(single_pass_limit, llm)

    if (
        total_serialized_chars
        and total_serialized_chars <= single_pass_limit
        and total_payload_chars <= single_pass_payload_budget
        and len(bloklar) <= single_pass_max_blocks
    ):
        logger.info(
            f"Ceviri modulu bu dosyayi tek geciste isleyecek. "
            f"Metin boyutu: {total_serialized_chars} karakter | "
            f"Tahmini JSON/payload boyutu: {total_payload_chars} karakter | "
            f"Tek gecis limiti: {single_pass_limit} karakter / {single_pass_max_blocks} blok."
        )
        return [bloklar], total_serialized_chars, chunk_size, single_pass_limit, chunk_payload_budget, chunk_max_blocks

    parcalar = _chunk_translation_blocks(
        bloklar,
        max_text_chars=chunk_size,
        max_payload_chars=chunk_payload_budget,
        max_blocks=chunk_max_blocks,
    )
    parcalar = _refine_translation_chunks(
        parcalar,
        max_text_chars=chunk_size,
        max_payload_chars=chunk_payload_budget,
        max_blocks=chunk_max_blocks,
    )
    return parcalar, total_serialized_chars, chunk_size, single_pass_limit, chunk_payload_budget, chunk_max_blocks


def _specific_translation_output_path(kaynak_dosya_adi: str, hedef_dil_kodu: str) -> Path | None:
    base_stem = build_base_stem(Path(kaynak_dosya_adi).stem)
    if not base_stem or base_stem.casefold() == "subtitle":
        return None
    return subtitle_intermediate_output_path(f"{base_stem}_{hedef_dil_kodu}.srt")


def _translation_output_path(hedef_dil_kodu: str) -> Path:
    filename = TRANSLATION_OUTPUT_NAMES.get(hedef_dil_kodu, f"subtitle_{hedef_dil_kodu}.srt")
    if hedef_dil_kodu == "de":
        return subtitle_output_path(filename)
    return subtitle_intermediate_output_path(filename)


def _split_chunk_for_rescue(chunk: list) -> list[list]:
    if len(chunk) <= 1:
        return [chunk]

    midpoint = max(1, len(chunk) // 2)
    return [chunk[:midpoint], chunk[midpoint:]]


def _build_translation_prompt(
    system_prompt: str,
    chunk,
    target_lang_en: str,
    target_lang_code: str,
    llm: CentralLLM,
) -> tuple[str, str]:
    target_payload = dump_subtitle_block_payload(chunk)
    if _is_translategemma_model(llm):
        prompt = (
            f"You are a professional Turkish (tr) to {target_lang_en} ({target_lang_code}) translator. "
            f"Your goal is to accurately convey the meaning and nuances of the original Turkish text while "
            f"adhering to {target_lang_en} grammar, vocabulary, and cultural sensitivities. "
            "The Turkish text is provided as JSON blocks. Translate only each `text_lines` value. "
            "Return only one valid JSON object in this exact schema: "
            '{"blocks":[{"id":"1","text_lines":["..."]}]}. '
            "The `blocks` array length must exactly match the input, and each output `id` must exactly match "
            "the corresponding input id in the same order. Produce only the requested JSON object, without any "
            "additional explanations or commentary. Under no circumstances output Chinese characters. "
            f"Please translate the following Turkish text into {target_lang_en}:\n\n\n"
            f"{target_payload}"
        )
        return prompt, target_payload

    prompt = (
        f"{system_prompt}\n\n"
        "STRICT OUTPUT CONTRACT:\n"
        "- Return only one valid JSON object.\n"
        '- JSON schema: {"blocks":[{"id":"1","text_lines":["..."]}]}\n'
        "- The `blocks` array length must exactly match TARGET BLOCKS.\n"
        "- Each output `id` must exactly match the corresponding target block id in the same order.\n"
        f"- `text_lines` must contain only final {target_lang_en} subtitle text.\n"
        "- Do not add explanations, notes, apologies, questions, markdown, timestamps, or assistant chatter.\n"
        "- Under no circumstances output Chinese characters.\n"
        "- Preserve meaning, keep the subtitle natural and concise, and do not skip blocks.\n\n"
        "TARGET BLOCKS JSON:\n"
        f"{target_payload}\n\n"
        "Now return only valid JSON."
    )
    return prompt, target_payload


def _build_retry_prompt(base_prompt: str, previous_raw: str, validation_error: str) -> str:
    preview = str(previous_raw or "").strip()[:4000]
    return (
        f"{base_prompt}\n\n"
        "YOUR PREVIOUS RESPONSE WAS REJECTED.\n"
        f"Validation error: {validation_error}\n"
        "Do not repeat the invalid answer below.\n"
        "Invalid previous answer:\n"
        f"{preview}\n\n"
        "Return only one valid JSON object that follows the schema."
    )


def translate_chunk(
    index: int,
    chunk,
    llm: CentralLLM,
    system_prompt: str,
    dil_adi: str,
    timeout_seconds: int,
    max_retries: int,
    ollama_options: dict | None,
    ollama_keep_alive: str | None,
    debug_path: Path,
    hedef_dil_kodu: str,
    chunk_label: str | None = None,
) -> tuple:
    target_lang_en = "English" if hedef_dil_kodu == "en" else "German"
    prompt, source_payload = _build_translation_prompt(
        system_prompt,
        chunk,
        target_lang_en,
        hedef_dil_kodu,
        llm,
    )

    display_label = chunk_label or str(index + 1)
    chunk_started_at = time.perf_counter()
    logger.info(
        f"[{dil_adi}] Çeviri Parçası {display_label} LLM'e gönderiliyor... "
        f"(blok={len(chunk)}, payload={len(source_payload)} karakter, prompt={len(prompt)} karakter, "
        f"timeout={timeout_seconds}s)"
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
                chunk,
                target_language_code=hedef_dil_kodu,
            )
            sonuc = rebuild_srt_from_replacements(chunk, replacement_lines)
            elapsed = format_elapsed(time.perf_counter() - chunk_started_at)
            logger.info(f"[{dil_adi}] Çeviri Parçası {display_label} doğrulandı. Süre={elapsed}")
            break
        except Exception as exc:
            last_error = str(exc)
            elapsed = format_elapsed(time.perf_counter() - chunk_started_at)
            logger.warning(
                f"[{dil_adi}] Structured ceviri cevabi reddedildi "
                f"(Parca {display_label} | {attempt}/{validation_attempts}): {last_error} | "
                f"Gecen sure={elapsed}"
            )
            append_debug_response(
                debug_path,
                f"Translation {hedef_dil_kodu.upper()} | Parca {display_label} | Deneme {attempt}",
                last_error,
                last_raw_response,
                source_excerpt=source_payload[:2000],
            )

    if sonuc is None:
        raise ValueError(f"Structured ceviri cevabi dogrulanamadi: {last_error or 'bilinmeyen hata'}")

    logger.info(f"✅ [{dil_adi}] Parça {display_label} tamamlandı.")
    return index, sonuc


def translate_chunk_with_rescue(
    index: int,
    chunk,
    llm: CentralLLM,
    system_prompt: str,
    dil_adi: str,
    timeout_seconds: int,
    max_retries: int,
    ollama_options: dict | None,
    ollama_keep_alive: str | None,
    debug_path: Path,
    hedef_dil_kodu: str,
    chunk_label: str | None = None,
    rescue_depth: int = 0,
) -> tuple:
    display_label = chunk_label or str(index + 1)

    try:
        return translate_chunk(
            index,
            chunk,
            llm,
            system_prompt,
            dil_adi,
            timeout_seconds,
            max_retries,
            ollama_options,
            ollama_keep_alive,
            debug_path,
            hedef_dil_kodu,
            chunk_label=display_label,
        )
    except Exception as exc:
        max_rescue_depth = _resolve_rescue_split_depth()
        if len(chunk) <= 1 or rescue_depth >= max_rescue_depth:
            raise

        child_chunks = _split_chunk_for_rescue(chunk)
        logger.warning(
            f"[{dil_adi}] Parça {display_label} dogrudan cevrilemedi ({exc}). "
            f"Kurtarma icin {len(chunk)} blokluk parca {len(child_chunks)} alt parcaya bolunuyor "
            f"(derinlik {rescue_depth + 1}/{max_rescue_depth})."
        )

        rescued_outputs: list[str] = []
        for child_index, child_chunk in enumerate(child_chunks, start=1):
            child_label = f"{display_label}.{child_index}"
            try:
                _, child_output = translate_chunk_with_rescue(
                    index,
                    child_chunk,
                    llm,
                    system_prompt,
                    dil_adi,
                    timeout_seconds,
                    max_retries,
                    ollama_options,
                    ollama_keep_alive,
                    debug_path,
                    hedef_dil_kodu,
                    chunk_label=child_label,
                    rescue_depth=rescue_depth + 1,
                )
                rescued_outputs.append(child_output.strip())
            except Exception as child_exc:
                raise ValueError(
                    f"{exc} | rescue_failed:{child_label}:{child_exc}"
                ) from child_exc

        logger.info(f"[{dil_adi}] Parça {display_label} alt parcalara bolunerek kurtarildi.")
        return index, "\n\n".join(item for item in rescued_outputs if item.strip())


def _resolve_manual_input_file() -> Path | None:
    preferred_input = subtitle_output_path(TRANSLATION_INPUT_NAME)
    if preferred_input.exists():
        logger.info(f"Ceviri modulu dogrudan 102 cikisini kullanacak: {preferred_input.name}")
        return preferred_input

    srt_dosyalari = list_subtitle_files()
    if not srt_dosyalari:
        return None

    logger.warning(
        f"Beklenen 102 cikisi bulunamadi: {preferred_input.name}. "
        "Uyumluluk icin manuel dosya secimine donuluyor."
    )
    print("\n📂 Lütfen çevrilecek SRT dosyasını seçin:")
    for idx, dosya in enumerate(srt_dosyalari, start=1):
        print(f"  [{idx}] {dosya.name}")

    secim_str = input(f"\n👉 Dosya Seçiminiz (1-{len(srt_dosyalari)}): ").strip()
    try:
        secim_idx = int(secim_str)
        if 1 <= secim_idx <= len(srt_dosyalari):
            return srt_dosyalari[secim_idx - 1]
    except ValueError:
        pass

    logger.warning("❌ Geçersiz dosya seçimi. İşlem iptal edildi.")
    return None

def start_translation_job(
    parcalar: list,
    llm: CentralLLM,
    hedef_dil_kodu: str,
    hedef_dil_adi: str,
    kaynak_dosya_adi: str,
    debug_path: Path,
):
    """Belirlenen dil için parçaları paralel olarak çevirir ve SRT olarak kaydeder."""

    target_lang_en = "English" if hedef_dil_kodu == "en" else "German"
    sistem_talimati = _translation_system_prompt(target_lang_en)

    cevrilmis_parcalar = [None] * len(parcalar)
    failed_chunks: list[tuple[int, str]] = []
    allow_source_fallback = _translation_source_fallback_allowed()
    
    logger.info(f"🚀 {hedef_dil_adi.upper()} çevirisi başlatılıyor... ({len(parcalar)} parça)")

    timeout_seconds, max_retries, ollama_options, ollama_keep_alive = _resolve_translation_llm_call_settings(llm)
    logger.info(f"[{hedef_dil_adi}] Ceviri modulu sirali modda calisacak.")
    logger.info(
        f"[{hedef_dil_adi}] Ceviri prompt profili: "
        f"{'TranslateGemma-uyumlu' if _is_translategemma_model(llm) else 'Standart structured'}"
    )
    if ollama_options:
        logger.info(
            f"[{hedef_dil_adi}] Ollama ayarlari: timeout={timeout_seconds}s, max_retries={max_retries}, "
            f"num_ctx={ollama_options.get('num_ctx')}, keep_alive={ollama_keep_alive}, "
            f"temperature={ollama_options.get('temperature')}"
        )

    for idx, parca in enumerate(parcalar):
        try:
            index, sonuc = translate_chunk_with_rescue(
                idx,
                parca,
                llm,
                sistem_talimati,
                hedef_dil_adi,
                timeout_seconds,
                max_retries,
                ollama_options,
                ollama_keep_alive,
                debug_path,
                hedef_dil_kodu,
                chunk_label=str(idx + 1),
            )
            cevrilmis_parcalar[index] = sonuc
        except Exception as e:
            logger.error(f"❌ [{hedef_dil_adi}] Parça {idx + 1} çevrilirken hata: {e}")
            failed_chunks.append((idx, str(e)))
            if allow_source_fallback:
                logger.warning(
                    f"[{hedef_dil_adi}] TRANSLATION_ALLOW_SOURCE_FALLBACK acik oldugu icin "
                    f"Parça {idx + 1} kaynak dilde birakiliyor. Bu durum karisik dilli SRT uretebilir."
                )
                cevrilmis_parcalar[idx] = serialize_srt_blocks(parcalar[idx])

    if failed_chunks and not allow_source_fallback:
        failed_chunk_numbers = ", ".join(str(index + 1) for index, _ in failed_chunks)
        failed_details = " | ".join(
            f"Parca {index + 1}: {error}" for index, error in failed_chunks[:5]
        )
        append_debug_response(
            debug_path,
            f"Translation {hedef_dil_kodu.upper()} | Dil Bazli Sert Durus",
            f"Ceviri iptal edildi. Basarisiz chunklar: {failed_chunk_numbers}",
            failed_details,
        )
        raise RuntimeError(
            f"{hedef_dil_adi} cevirisi iptal edildi. "
            f"Cevrilemeyen chunklar oldugu icin kaynak Turkce metin korunmadi. "
            f"Basarisiz chunklar: {failed_chunk_numbers}"
        )

    missing_chunks = [idx + 1 for idx, value in enumerate(cevrilmis_parcalar) if value is None]
    if missing_chunks:
        raise RuntimeError(
            f"{hedef_dil_adi} cevirisi tamamlanamadi. Cikti olusturulamayan chunklar: "
            + ", ".join(str(item) for item in missing_chunks)
        )

    nihai_srt = "\n\n".join(cevrilmis_parcalar)
    
    cikti_dosyasi = _translation_output_path(hedef_dil_kodu)
    write_srt_file(cikti_dosyasi, nihai_srt)
    ozel_cikti = _specific_translation_output_path(kaynak_dosya_adi, hedef_dil_kodu)
    if ozel_cikti and ozel_cikti != cikti_dosyasi:
        write_srt_file(ozel_cikti, nihai_srt)
        logger.info(f"📄 {hedef_dil_adi} icin video-bazli ceviri aynasi kaydedildi: {ozel_cikti.name}")
    
    logger.info(f"🎉 {hedef_dil_adi} Çevirisi KAYDEDİLDİ: {cikti_dosyasi.name}")

def run():
    started_at = time.perf_counter()
    print("\n" + "="*50)
    print("🌍 AŞAMA 3: Yapay Zeka ile Çoklu Dil Çevirisi")
    print("="*50)

    # 1. Normal akista 102 cikisini dogrudan kullan, yoksa uyumluluk icin secime don
    girdi_dosyasi = _resolve_manual_input_file()
    if girdi_dosyasi is None:
        logger.error("❌ '100_Altyazı' klasöründe hiç SRT dosyası bulunamadı! Lütfen önce 1. Modülü çalıştırın.")
        return

    logger.info(f"Seçilen Kaynak Dosya: {girdi_dosyasi.name}")
    cevriler = _resolve_translation_targets()
    if not cevriler:
        logger.warning("Ceviri modulu icin aktif hedef dil bulunmuyor. TRANSLATION_ENABLE_* ayarlarini kontrol edin.")
        return

    try:
        llm = _create_translation_llm()
    except Exception as e:
        logger.error(f"LLM Başlatılamadı: {e}")
        return

    # 3. DOSYAYI OKU VE PARÇALA
    icerik = read_srt_file(girdi_dosyasi)
    bloklar = parse_srt_blocks(icerik)
    parcalar, total_serialized_chars, chunk_size, single_pass_limit, chunk_payload_budget, chunk_max_blocks = _resolve_translation_chunk_strategy(
        llm,
        bloklar,
    )
    
    logger.info(
        f"Toplam {len(bloklar)} blok metin, toplam serialized boyut {total_serialized_chars} karakter. "
        f"Chunk hedefi {chunk_size} karakter, tahmini JSON/payload butcesi {chunk_payload_budget} karakter, "
        f"blok limiti {chunk_max_blocks}, tek gecis limiti {single_pass_limit} karakter. "
        f"Dosya {len(parcalar)} parça halinde çeviriye gönderilecek."
    )

    # 4. SEÇİLEN DİLLER İÇİN ÇEVİRİYİ BAŞLAT
    logger.info(f"{len(cevriler)} dil için sırayla çeviri görevleri başlatılıyor.")
    relocate_known_subtitle_intermediates()
    debug_path = subtitle_intermediate_output_path(TRANSLATION_DEBUG_OUTPUT_NAME)
    prepare_debug_file(debug_path, f"CEVIRI LLM DEBUG ({girdi_dosyasi.name})")
    failed_languages: list[tuple[str, str]] = []
    for dil_kodu, dil_adi in cevriler:
        try:
            start_translation_job(parcalar, llm, dil_kodu, dil_adi, girdi_dosyasi.name, debug_path)
        except Exception as e:
            logger.error(f"'{dil_adi}' çevirisi sırasında ana hata oluştu: {e}")
            failed_languages.append((dil_adi, str(e)))

    elapsed = format_elapsed(time.perf_counter() - started_at)
    if failed_languages:
        failed_summary = " | ".join(f"{dil}: {hata}" for dil, hata in failed_languages)
        logger.error(f"❌ Ceviri modulu hatayla tamamlandi. Basarisiz diller: {failed_summary}")
        logger.info(f"⏱️ Modul 3 toplam sure: {elapsed}")
        raise RuntimeError(f"Ceviri modulu tamamlanamadi. Basarisiz diller: {failed_summary}")

    logger.info("\n✅ TÜM ÇEVİRİ İŞLEMLERİ BAŞARIYLA TAMAMLANDI!")
    logger.info(f"⏱️ Modul 3 toplam sure: {elapsed}")
    
def run_automatic(girdi_dosyasi: Path, _unused_llm: CentralLLM | None = None):
    """Tam otomasyon için env tabanli ceviri modeliyle aktif hedef dillere ceviri yapar."""
    started_at = time.perf_counter()
    logger.info(f"🔄 OTOMASYON: {girdi_dosyasi.name} dosyası aktif hedef dillere çevriliyor...")

    try:
        aktif_llm = _create_translation_llm()
    except Exception as e:
        logger.error(f"Otomasyon icin ceviri LLM baslatilamadi: {e}")
        raise
    
    icerik = read_srt_file(girdi_dosyasi)
    bloklar = parse_srt_blocks(icerik)
    parcalar, total_serialized_chars, chunk_size, single_pass_limit, chunk_payload_budget, chunk_max_blocks = _resolve_translation_chunk_strategy(
        aktif_llm,
        bloklar,
    )
    logger.info(
        f"Toplam {len(bloklar)} blok metin, toplam serialized boyut {total_serialized_chars} karakter. "
        f"Chunk hedefi {chunk_size} karakter, tahmini JSON/payload butcesi {chunk_payload_budget} karakter, "
        f"blok limiti {chunk_max_blocks}, tek gecis limiti {single_pass_limit} karakter. "
        f"Dosya {len(parcalar)} parça halinde çeviriye gönderilecek."
    )
    
    cevriler = _resolve_translation_targets()
    if not cevriler:
        logger.warning("Otomasyon ceviri adimi atlandi; aktif hedef dil bulunmuyor.")
        return
    
    logger.info(f"Otomasyon: {len(cevriler)} dil için sırayla çeviri görevleri başlatılıyor.")
    relocate_known_subtitle_intermediates()
    debug_path = subtitle_intermediate_output_path(TRANSLATION_DEBUG_OUTPUT_NAME)
    prepare_debug_file(debug_path, f"CEVIRI LLM DEBUG ({girdi_dosyasi.name})")
    failed_languages: list[tuple[str, str]] = []
    for dil_kodu, dil_adi in cevriler:
        try:
            start_translation_job(parcalar, aktif_llm, dil_kodu, dil_adi, girdi_dosyasi.name, debug_path)
        except Exception as e:
            logger.error(f"Otomasyon sırasında '{dil_adi}' çevirisinde ana hata oluştu: {e}")
            failed_languages.append((dil_adi, str(e)))

    elapsed = format_elapsed(time.perf_counter() - started_at)
    if failed_languages:
        failed_summary = " | ".join(f"{dil}: {hata}" for dil, hata in failed_languages)
        logger.error(f"❌ Otomatik coklu dil cevirisi hatayla tamamlandi. Basarisiz diller: {failed_summary}")
        logger.info(f"⏱️ Modul 3 toplam sure: {elapsed}")
        raise RuntimeError(f"Otomatik ceviri tamamlanamadi. Basarisiz diller: {failed_summary}")

    logger.info("🎉 Otomatik Çoklu Dil Çevirisi TAMAMLANDI!")    
    logger.info(f"⏱️ Modul 3 toplam sure: {elapsed}")
