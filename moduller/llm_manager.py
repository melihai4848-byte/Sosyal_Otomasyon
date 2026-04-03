import json
import os
import re
import requests
from moduller.logger import get_logger
from moduller.exceptions import LLMConnectionError
from moduller.llm_role_table import load_llm_role_table_entries
from moduller.retry_utils import retry_with_backoff

# ÖNCE logger'ı tanımla
logger = get_logger("LLMManager")

# SONRA logger'ı kullanabileceğin bloklara geç
try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("huggingface_hub kütüphanesi kurulu değil. pip install huggingface_hub")


OPENROUTER_FALLBACK_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3-27b-it:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
]

APIFREELLM_FALLBACK_MODELS = [
    "gpt-4o-mini",
    "gemini-2.0-flash",
    "claude-3.5-haiku",
]

GEMINI_MODEL_OPTIONS = [
    (
        "gemini-3.1-flash-lite-preview",
        "Yuksek hacimli temsilci gorevleri, ceviri ve basit veri isleme icin optimize edilmis model",
    ),
    (
        "gemini-3.1-flash-live-preview",
        "Akustik nuans algilama, sayisal hassasiyet ve cok formatli farkindalik ile gercek zamanli diyalog icin optimize edilmis model",
    ),
    (
        "gemini-3-flash-preview",
        "Hiz icin tasarlanmis en akilli model",
    ),
    (
        "gemini-2.5-pro",
        "Kodlama ve karmasik akil yurutme gorevlerinde ustun performans gosteren, en yeni teknoloji urunu cok amacli model",
    ),
    (
        "gemini-2.5-flash",
        "1 milyon parcalik baglam penceresini destekleyen ve dusunme butceleri olan ilk karma akil yurutme modeli",
    ),
    (
        "gemini-2.5-flash-lite",
        "Buyuk olcekli kullanim icin tasarlanan en kucuk model",
    ),
    (
        "gemini-2.5-flash-lite-preview-09-2025",
        "Yuksek gonderim hacmi ve yuksek kalite icin optimize edilmis Gemini 2.5 Flash Lite'a dayali en yeni model",
    ),
]
GEMINI_MODEL_DESCRIPTIONS = {model_name: description for model_name, description in GEMINI_MODEL_OPTIONS}
OLLAMA_TAG_CACHE: dict[str, dict] = {}
OLLAMA_SHOW_CACHE: dict[str, dict | None] = {}

ROLE_GUIDANCE = {
    "MAIN": {
        "title": "MAIN LLM",
        "short_label": "ANA",
        "summary": "Kurallara uyan, structured/JSON dostu, disiplinli ve asker gibi calisan model roludur.",
        "bullets": [
            "Yaraticilik degil; tutarlilik, format sadakati, liste/JSON uretimi ve gorev disiplini onceliklidir.",
            "Gramer duzeltme, trim, planlama, veri isleme ve net talimat takibi gerektiren isler icin uygundur.",
            "Daha buyuk olmak tek basina avantaj degildir; sakin, duzgun ve kuralli cevap veren modeller daha dogru secimdir.",
        ],
        "online_advice": "Online secilecekse once tutarlilik ve format sadakati yuksek modelleri dusun. En 'yaratici' modeli degil, en 'duzenli' modeli ara.",
    },
    "SMART": {
        "title": "SMART LLM",
        "short_label": "YARATICI",
        "summary": "Dusunen, yorum yapan, packaging/copywriting gucu olan ve yaratici fikir ureten model roludur.",
        "bullets": [
            "Baslik, aciklama, thumbnail, hook, fikir gelistirme ve karar verme gerektiren islerde kullanilir.",
            "Bu rolde buyuk online modeller genelde daha gucludur; ozellikle dil kalitesi, yargilama ve yaratici packaging icin fayda saglar.",
            "Hiz tek kriter olmamali; buyuk baglamli ve daha zeki online modeller genelde daha iyi sonuc verir.",
        ],
        "online_advice": "Mumkunse buyuk hacimli online modelleri tercih et. Ozellikle Gemini Pro/Flash, Groq uzerindeki 70B+ modeller ve OpenRouter'daki buyuk instruct modeller iyi adaylardir.",
    },
}

ROLE_TABLE_ENTRIES = load_llm_role_table_entries()
MODULE_MAIN_RECOMMENDATIONS = {
    module_number: (entry.recommended_main[0], entry.recommended_main[1], entry.notes or "llm_rol_tablosu.md kaydindan yuklendi")
    for module_number, entry in ROLE_TABLE_ENTRIES.items()
    if entry.recommended_main
}
MODULE_SMART_RECOMMENDATIONS = {
    module_number: (entry.recommended_smart[0], entry.recommended_smart[1], entry.notes or "llm_rol_tablosu.md kaydindan yuklendi")
    for module_number, entry in ROLE_TABLE_ENTRIES.items()
    if entry.recommended_smart
}
MODULE_LLM_SUMMARIES = {
    module_number: {
        key: value
        for key, value in {
            "title": entry.title,
            "main": entry.main_summary,
            "smart": entry.smart_summary,
        }.items()
        if value
    }
    for module_number, entry in ROLE_TABLE_ENTRIES.items()
}


def print_module_llm_recommendation(module_number: str) -> None:
    recommendation = MODULE_SMART_RECOMMENDATIONS.get(str(module_number or "").strip())
    if not recommendation:
        return
    provider, model_name, reason = recommendation
    print("\n" + "-" * 72)
    print(f"💡 Smart LLM Modul Onerisi | {module_number}")
    print(f"Favori: {provider}:{model_name} ({reason})")
    print("-" * 72)


def get_module_recommended_llm_config(module_number: str, role: str = "main") -> tuple[str, str]:
    role_key = _normalize_llm_role(role)
    recommendation_map = MODULE_SMART_RECOMMENDATIONS if role_key == "SMART" else MODULE_MAIN_RECOMMENDATIONS
    recommendation = recommendation_map.get(str(module_number or "").strip())
    if recommendation:
        provider, model_name, _reason = recommendation
        return provider, model_name
    return get_default_llm_config(role)


def _format_provider_model_line(provider: str, model_name: str) -> str:
    return f"{provider} | {model_name}"


def prompt_module_llm_plan(module_number: str, *, needs_main: bool = False, needs_smart: bool = False) -> bool:
    if not llm_selection_prompt_enabled():
        return False

    summary = MODULE_LLM_SUMMARIES.get(str(module_number or "").strip(), {})
    title = summary.get("title", f"Modul {module_number}")
    main_role = summary.get("main", "ilk taslak ve yapisal isleri yapar")
    smart_role = summary.get("smart", "nihai secimi, yorumu ve cilayi yapar")

    print("\n" + "=" * 60)
    print(f"{module_number} | {title}")
    print("=" * 60)
    print("Bu modülde:")
    if needs_main:
        print(f"- Main LLM: {main_role}")
    if needs_smart:
        print(f"- Smart LLM: {smart_role}")
    print("")
    print("Onerilen profil:")
    if needs_main:
        provider, model_name = get_module_recommended_llm_config(module_number, "main")
        print(f"- Main: {_format_provider_model_line(provider, model_name)}")
    if needs_smart:
        provider, model_name = get_module_recommended_llm_config(module_number, "smart")
        print(f"- Smart: {_format_provider_model_line(provider, model_name)}")
    print("")
    print("[1] Onerilen profili kullan")
    print("[2] Kendim secmek istiyorum")

    while True:
        secim = input("👉 Secim (1 veya 2): ").strip()
        if secim == "1":
            return True
        if secim == "2":
            return False
        print("Lutfen sadece 1 veya 2 girin.")


def print_module_llm_choice_summary(module_number: str, selections: dict[str, tuple[str, str]]) -> None:
    if not selections:
        return
    summary = MODULE_LLM_SUMMARIES.get(str(module_number or "").strip(), {})
    title = summary.get("title", f"Modul {module_number}")
    print("\n" + "-" * 60)
    print(f"✅ Secilen LLM Profili | {module_number} | {title}")
    if "main" in selections:
        provider, model_name = selections["main"]
        print(f"Main:  {_format_provider_model_line(provider, model_name)}")
    if "smart" in selections:
        provider, model_name = selections["smart"]
        print(f"Smart: {_format_provider_model_line(provider, model_name)}")
    print("-" * 60)

ROLE_PROVIDER_HINTS = {
    "MAIN": {
        "GEMINI": "Yapisal ve kuralli islerde once Gemini 2.5 Pro veya 2.5 Flash gibi tutarli modelleri dusun.",
        "HUGGINGFACE": "HF tarafinda Qwen gibi instruction modeller structured cevaplar icin daha guvenlidir.",
        "GROQ": "Groq'ta Qwen/Qwen3 ve dengeli buyuk instruct modeller bu rol icin daha guvenli olur.",
        "OPENROUTER": "OpenRouter'da Qwen, Gemma ve Mistral ailesi genelde JSON/format sadakati icin daha iyi adaydir.",
        "APIFREELLM": "ApiFreeLLM'de once daha duzenli ve hafif modelleri dene; gerekirse model adini manuel yaz.",
        "OLLAMA": "Lokalde kuralli ve tutarli model istiyorsan Qwen ailesi genelde iyi bir baslangictir.",
    },
    "SMART": {
        "GEMINI": "Yaratici islerde once Gemini 2.5 Pro, 3 Flash Preview ve benzeri buyuk online modelleri dusun.",
        "HUGGINGFACE": "HF'de yaratici yazim icin Gemma ve Llama instruct ailesi daha iyi his verir.",
        "GROQ": "Groq'ta 70B+ ve buyuk yaratıcı instruct modeller genelde daha iyi fikir ve packaging uretir.",
        "OPENROUTER": "OpenRouter'da buyuk ücretsiz instruct modeller ve buyuk context modelleri oncele.",
        "APIFREELLM": "ApiFreeLLM'de daha akici ve yaratici buyuk modelleri oncele; liste gelmezse model adini manuel gir.",
        "OLLAMA": "Lokalde yaratıcı kalite daha sinirli olabilir; mumkunse Smart rolde online buyuk model tercih et.",
    },
}

ROLE_MODEL_PRIORITIES = {
    "MAIN": {
        "GEMINI": [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-3.1-flash-lite-preview",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash-lite-preview-09-2025",
            "gemini-3-flash-preview",
            "gemini-3.1-flash-live-preview",
        ],
        "HUGGINGFACE": [
            "Qwen/Qwen2.5-7B-Instruct",
            "google/gemma-2-9b-it",
            "meta-llama/Llama-3.1-8B-Instruct",
        ],
        "GROQ": [
            "qwen/qwen3-32b",
            "llama-3.3-70b-versatile",
            "groq/compound",
            "moonshotai/kimi-k2-instruct",
            "openai/gpt-oss-120b",
        ],
        "OPENROUTER": [
            "qwen/qwen3-next-80b-a3b-instruct:free",
            "google/gemma-3-27b-it:free",
            "mistralai/mistral-small-3.1-24b-instruct:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
            "meta-llama/llama-3.3-70b-instruct:free",
        ],
        "APIFREELLM": [
            "gpt-4o-mini",
            "gemini-2.0-flash",
            "claude-3.5-haiku",
        ],
    },
    "SMART": {
        "GEMINI": [
            "gemini-2.5-pro",
            "gemini-3-flash-preview",
            "gemini-2.5-flash",
            "gemini-3.1-flash-live-preview",
            "gemini-3.1-flash-lite-preview",
            "gemini-2.5-flash-lite-preview-09-2025",
            "gemini-2.5-flash-lite",
        ],
        "HUGGINGFACE": [
            "google/gemma-2-9b-it",
            "meta-llama/Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
        ],
        "GROQ": [
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-120b",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "moonshotai/kimi-k2-instruct",
            "qwen/qwen3-32b",
        ],
        "OPENROUTER": [
            "meta-llama/llama-3.3-70b-instruct:free",
            "google/gemma-3-27b-it:free",
            "openai/gpt-oss-120b:free",
            "nousresearch/hermes-3-llama-3.1-405b:free",
            "qwen/qwen3-next-80b-a3b-instruct:free",
        ],
        "APIFREELLM": [
            "claude-3.5-haiku",
            "gemini-2.0-flash",
            "gpt-4o-mini",
        ],
    },
}


def _get_role_profile(role: str = "main") -> dict:
    return ROLE_GUIDANCE[_normalize_llm_role(role)]


def _print_role_guidance(role: str = "main") -> None:
    role_key = _normalize_llm_role(role)
    profile = ROLE_GUIDANCE[role_key]
    print("")
    print(f"🎯 {profile['title']} PROFILI")
    print(f"   {profile['summary']}")
    for bullet in profile["bullets"]:
        print(f"   - {bullet}")
    print(f"   Öneri: {profile['online_advice']}")


def _print_provider_guidance(role: str = "main", provider: str | None = None) -> None:
    role_key = _normalize_llm_role(role)
    hints = ROLE_PROVIDER_HINTS.get(role_key, {})
    if provider:
        hint = hints.get(provider.upper())
        if hint:
            print(f"   İpucu: {hint}")
        return
    print("")
    print("💡 ROL BAZLI SAĞLAYICI NOTU")
    if role_key == "SMART":
        print("   Smart rolde kalite kritikse online buyuk modelleri oncelemek genelde en iyi sonuc verir.")
    else:
        print("   Main rolde oncelik yaraticilik degil, format disiplini ve net talimat takibidir.")


def _extract_model_size_b(model_name: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)b\b", str(model_name).strip().lower())
    if not match:
        return 0.0
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return 0.0


def _ollama_cache_key(model_name: str) -> str:
    return str(model_name or "").strip().lower()


def _ollama_local_server_url() -> str:
    return str(os.getenv("OLLAMA_LOCAL_SERVER", "http://localhost:11434") or "http://localhost:11434").rstrip("/")


def _apifreellm_base_url() -> str:
    return str(os.getenv("APIFREELLM_BASE_URL", "https://apifreellm.com/api/v1/chat") or "https://apifreellm.com/api/v1/chat").strip()


def _apifreellm_models_url() -> str:
    base = _apifreellm_base_url().rstrip("/")
    if base.endswith("/chat/completions"):
        return base[: -len("/chat/completions")] + "/models"
    if base.endswith("/chat"):
        return base[: -len("/chat")] + "/models"
    return base.rstrip("/") + "/models"


def _ollama_tag_payload(model_name: str) -> dict:
    return OLLAMA_TAG_CACHE.get(_ollama_cache_key(model_name), {})


def _ollama_is_cloud_model(model_name: str) -> bool:
    normalized = _ollama_cache_key(model_name)
    payload = _ollama_tag_payload(model_name)
    return bool(
        normalized.endswith(":cloud")
        or ":cloud" in normalized
        or payload.get("remote_model")
        or payload.get("remote_host")
    )


def _ollama_parameter_size_b(model_name: str) -> float:
    payload = _ollama_tag_payload(model_name)
    details = payload.get("details", {}) if isinstance(payload, dict) else {}
    raw_size = details.get("parameter_size", "")
    parsed = _extract_model_size_b(str(raw_size))
    return parsed or _extract_model_size_b(model_name)


def _parse_positive_int(value) -> int:
    try:
        parsed = int(str(value).strip())
        return parsed if parsed > 0 else 0
    except (TypeError, ValueError):
        return 0


def _ollama_show_payload(model_name: str) -> dict:
    key = _ollama_cache_key(model_name)
    if key in OLLAMA_SHOW_CACHE:
        return OLLAMA_SHOW_CACHE.get(key) or {}

    try:
        response = requests.post(
            f"{_ollama_local_server_url()}/api/show",
            json={"name": model_name},
            timeout=_env_int("OLLAMA_SHOW_TIMEOUT_SECONDS", 6),
        )
        if response.status_code >= 400:
            raise _http_error_from_response("Ollama", response)
        payload = response.json()
        OLLAMA_SHOW_CACHE[key] = payload if isinstance(payload, dict) else {}
    except Exception as exc:
        logger.debug(f"Ollama show metadata alinamadi: {model_name} | {exc}")
        OLLAMA_SHOW_CACHE[key] = {}

    return OLLAMA_SHOW_CACHE.get(key) or {}


def _ollama_context_length(model_name: str) -> int:
    details = _ollama_context_fields(model_name)
    return max(details.values(), default=0)


def _ollama_context_fields(model_name: str) -> dict[str, int]:
    payload = _ollama_show_payload(model_name)
    model_info = payload.get("model_info", {}) if isinstance(payload, dict) else {}
    fields: dict[str, int] = {}
    if isinstance(model_info, dict):
        for name, value in model_info.items():
            lowered = str(name or "").lower()
            if "context_length" in lowered:
                parsed = _parse_positive_int(value)
                if parsed:
                    fields["context_length"] = max(fields.get("context_length", 0), parsed)
                continue
            if "n_ctx" in lowered:
                parsed = _parse_positive_int(value)
                if parsed:
                    fields["n_ctx"] = max(fields.get("n_ctx", 0), parsed)
                continue
            if "window" not in lowered:
                continue
            parsed = _parse_positive_int(value)
            if parsed:
                fields["window"] = max(fields.get("window", 0), parsed)
    return fields


def _ollama_context_badge(model_name: str) -> str:
    fields = _ollama_context_fields(model_name)
    if not fields:
        return ""
    preferred_order = ("context_length", "n_ctx", "window")
    parts = [f"{key}={fields[key]}" for key in preferred_order if key in fields]
    best = max(fields.values(), default=0)
    if best:
        parts.append(f"ctx={best}")
    return " | " + " | ".join(parts)


def _ollama_role_score(model_name: str, role: str = "main") -> int:
    role_key = _normalize_llm_role(role)
    normalized = str(model_name).strip().lower()
    score = 0
    size_b = _ollama_parameter_size_b(model_name)
    context_length = _ollama_context_length(model_name)
    is_cloud = _ollama_is_cloud_model(model_name)

    if role_key == "MAIN":
        if "qwen" in normalized:
            score += 70
        if "qwen3.5" in normalized:
            score += 10
        if "deepseek" in normalized:
            score += 18
        if "kimi" in normalized:
            score += 10
        if "gemma" in normalized and "translategemma" not in normalized:
            score += 35
        if "llama" in normalized:
            score += 18
        if "deepseek-r1" in normalized:
            score -= 12
        if "distill" in normalized:
            score -= 6
        if is_cloud:
            score += 8
        if "instruct" in normalized or "it" in normalized:
            score += 10
        if "q8_0" in normalized:
            score += 8
        elif "q4" in normalized:
            score += 3
        if context_length >= 131072:
            score += 12
        elif context_length >= 65536:
            score += 7
        elif context_length >= 32768:
            score += 3
        if 8 <= size_b <= 32:
            score += 14
        elif size_b > 32:
            score += 8
        elif 6 <= size_b < 8:
            score += 6
    else:
        if "gemma" in normalized and "translategemma" not in normalized:
            score += 62
        if "llama" in normalized:
            score += 52
        if "qwen" in normalized:
            score += 42
        if "qwen3.5" in normalized:
            score += 10
        if "deepseek" in normalized:
            score += 58
        if "kimi" in normalized:
            score += 55
        if "deepseek-r1" in normalized:
            score += 12
        if "claude" in normalized or "sonnet" in normalized:
            score += 25
        if is_cloud:
            score += 34
        if "instruct" in normalized or "it" in normalized:
            score += 8
        if "q8_0" in normalized:
            score += 6
        elif "q4" in normalized:
            score += 2
        if context_length >= 262144:
            score += 28
        elif context_length >= 131072:
            score += 20
        elif context_length >= 65536:
            score += 12
        elif context_length >= 32768:
            score += 6
        if size_b >= 12:
            score += 18
        elif 8 <= size_b < 12:
            score += 10

    if "translategemma" in normalized:
        score -= 20

    return score


def _ollama_role_badge(model_name: str, role: str = "main") -> str:
    score = _ollama_role_score(model_name, role)
    if score >= 85:
        return " | Rol icin guclu onerilen"
    if score >= 60:
        return " | Rol icin uygun"
    return ""


def _sort_models_for_role(provider: str, models: list[str], role: str = "main") -> list[str]:
    role_key = _normalize_llm_role(role)
    provider_key = provider.upper()
    if provider_key == "OLLAMA":
        if role_key == "SMART":
            return sorted(
                models,
                key=lambda model_name: (
                    -int(_ollama_is_cloud_model(model_name) and _ollama_role_score(model_name, role) >= 60),
                    -int(_ollama_is_cloud_model(model_name)),
                    -_ollama_context_length(model_name),
                    -_ollama_role_score(model_name, role),
                    -_ollama_parameter_size_b(model_name),
                    model_name.lower(),
                ),
            )
        if role_key == "MAIN":
            return sorted(
                models,
                key=lambda model_name: (
                    int(_ollama_is_cloud_model(model_name)),
                    -_ollama_context_length(model_name),
                    -_ollama_role_score(model_name, role),
                    -_ollama_parameter_size_b(model_name),
                    model_name.lower(),
                ),
            )
        return sorted(
            models,
            key=lambda model_name: (
                -_ollama_role_score(model_name, role),
                -_ollama_context_length(model_name),
                -_ollama_parameter_size_b(model_name),
                model_name.lower(),
            ),
        )

    preferred = ROLE_MODEL_PRIORITIES.get(role_key, {}).get(provider_key, [])
    if not preferred:
        return models

    preferred_rank = {model_name: idx for idx, model_name in enumerate(preferred)}
    return sorted(
        models,
        key=lambda model_name: (
            1 if model_name not in preferred_rank else 0,
            preferred_rank.get(model_name, 10_000),
            model_name.lower(),
        ),
    )


def _model_role_badge(provider: str, model_name: str, role: str = "main") -> str:
    role_key = _normalize_llm_role(role)
    provider_key = provider.upper()
    if provider_key == "OLLAMA":
        return _ollama_role_badge(model_name, role)

    preferred = ROLE_MODEL_PRIORITIES.get(role_key, {}).get(provider_key, [])
    if model_name in preferred[:2]:
        return " | Rol icin guclu onerilen"
    if model_name in preferred[:5]:
        return " | Rol icin uygun"
    return ""


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw.strip())
        return value if value > 0 else default
    except (TypeError, ValueError):
        logger.warning(f"Gecersiz {name} degeri bulundu: {raw}. Varsayilan {default} kullanilacak.")
        return default


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


def llm_selection_prompt_enabled() -> bool:
    return _env_bool("LLM_SELECTION_PROMPT_ENABLED", False)


def _normalize_llm_role(role: str = "main") -> str:
    normalized = str(role).strip().lower()
    if normalized in {"smart", "cila"}:
        return "SMART"
    if normalized in {"grammar", "gramer"}:
        return "MAIN"
    return "MAIN"


def get_default_llm_config(role: str = "main") -> tuple[str, str]:
    role_key = _normalize_llm_role(role)
    provider = os.getenv(f"DEFAULT_{role_key}_LLM_PROVIDER", "OLLAMA").strip().upper()
    model = os.getenv(
        f"DEFAULT_{role_key}_LLM_MODEL",
        "qwen3.5:397b-cloud"
        if role_key == "SMART"
        else "qwen3:14b",
    ).strip()
    return provider, model


def _smart_llm_fallback_enabled() -> bool:
    return _env_bool("SMART_LLM_FALLBACK_TO_LOCAL_ON_FAILURE", True)


def _smart_llm_healthcheck_enabled() -> bool:
    return _env_bool("SMART_LLM_HEALTHCHECK_ENABLED", True)


def _smart_llm_healthcheck_timeout() -> int:
    return _env_int("SMART_LLM_HEALTHCHECK_TIMEOUT_SECONDS", 45)


def _ollama_streaming_enabled() -> bool:
    return _env_bool("OLLAMA_STREAMING_ENABLED", True)


def _ollama_requests_timeout(read_timeout_seconds: int):
    connect_timeout_seconds = _env_int("OLLAMA_CONNECT_TIMEOUT_SECONDS", 10)
    return connect_timeout_seconds, read_timeout_seconds


def _select_ollama_model_interactive(role: str = "main") -> tuple[str, str]:
    modeller = fetch_dynamic_models("OLLAMA")
    modeller = _sort_models_for_role("OLLAMA", modeller, role)
    if not modeller:
        varsayilan = get_default_llm_config(role)
        logger.warning("Ollama model listesi alinamadi. Varsayilan lokal model kullanilacak.")
        return varsayilan

    profile = _get_role_profile(role)
    print("\n" + "-" * 40)
    print("🤖 LOKAL OLLAMA MODEL SECIMI")
    print("-" * 40)
    print(f"Rol: {profile['short_label']} ({profile['title']})")
    print("Varsayilan model su an kullanilamadigi icin Ollama modelleri (lokal + cloud) listeleniyor.")
    _print_provider_guidance(role, "OLLAMA")
    for idx, model in enumerate(modeller, start=1):
        print(f"  [{idx}] {_format_model_option('OLLAMA', model, role)}")

    secim = input(f"👉 Model Seciniz (1-{len(modeller)}): ").strip()
    try:
        model_adi = modeller[int(secim) - 1]
    except (ValueError, IndexError):
        print("❌ Gecersiz secim, listedeki ilk model ayarlandi.")
        model_adi = modeller[0]
    return "OLLAMA", model_adi


def _verify_default_smart_llm(provider: str, model_adi: str) -> bool:
    if not _smart_llm_healthcheck_enabled():
        return True
    try:
        test_llm = CentralLLM(provider=provider, model_name=model_adi)
        yanit = test_llm.uret(
            'Return only this exact text: {"status":"ok"}',
            timeout=_smart_llm_healthcheck_timeout(),
            max_retries=0,
        )
        return bool(str(yanit or "").strip())
    except Exception as exc:
        logger.warning(f"Varsayilan yaratici model health-check'ten gecemedi: {provider} / {model_adi} | {exc}")
        return False


def _http_error_from_response(provider: str, response: requests.Response) -> requests.HTTPError:
    try:
        payload = response.json()
    except Exception:
        payload = response.text
    error = requests.HTTPError(f"{provider} API hatasi ({response.status_code}): {payload}")
    error.response = response
    return error


def _format_model_option(provider: str, model_name: str, role: str = "main") -> str:
    formatted = model_name
    if provider == "GEMINI":
        description = GEMINI_MODEL_DESCRIPTIONS.get(model_name)
        if description:
            formatted = f"{model_name} ({description})"
    elif provider == "OLLAMA":
        meta_parts = []
        if _ollama_is_cloud_model(model_name):
            meta_parts.append("Cloud")
        context_badge = _ollama_context_badge(model_name)
        if context_badge:
            context_text = context_badge.strip()
            if context_text.startswith("|"):
                context_text = context_text[1:].strip()
            if context_text:
                meta_parts.append(context_text)
        if meta_parts:
            formatted = f"{model_name} | " + " | ".join(meta_parts)
    badge = _model_role_badge(provider, model_name, role)
    if badge:
        formatted = f"{formatted}{badge}"
    return formatted

def fetch_dynamic_models(provider: str) -> list:
    """Sadece ücretsiz ve stabil çalışan sağlayıcıları listeler."""
    modeller = []
    print(f"⏳ {provider} modelleri hazırlanıyor...")
    
    try:
        #-------OLLAMA SERVER SETTINGS-------#
        if provider == "OLLAMA":
            url = _ollama_local_server_url() + "/api/tags"
            resp = requests.get(url, timeout=3).json()
            models_payload = resp.get("models", [])
            OLLAMA_TAG_CACHE.clear()
            for item in models_payload:
                if not isinstance(item, dict):
                    continue
                model_name = str(item.get("name") or "").strip()
                if not model_name:
                    continue
                OLLAMA_TAG_CACHE[_ollama_cache_key(model_name)] = item
            modeller = [m["name"] for m in models_payload if isinstance(m, dict) and m.get("name")]
        #------------------------------------#
        
        #-------GEMINI SERVER SETTINGS-------#
        elif provider == "GEMINI":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key: return []

            modeller = [model_name for model_name, _description in GEMINI_MODEL_OPTIONS]
        #------------------------------------#
                
 #-------GROQ SERVER SETTINGS-------#
        elif provider == "GROQ":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key: 
                logger.error("GROQ_API_KEY bulunamadı! Çevre değişkenlerini kontrol edin.")
                return []
            
            try:
                url = "https://api.groq.com/openai/v1/models"
                headers = {"Authorization": f"Bearer {api_key}"}
                
                # İsteği yap ve HTTP hatalarını kontrol et
                resp_obj = requests.get(url, headers=headers)
                
                # Eğer 200 OK dönmezse, gerçek hatayı logla ve except'e düşür
                if resp_obj.status_code != 200:
                    raise Exception(f"HTTP Hata Kodu: {resp_obj.status_code} - Detay: {resp_obj.text}")
                
                resp = resp_obj.json()
                
                # 1. Groq Şampiyonlar Ligi ve Rehber
                groq_rehber = {
                    "llama-3.3-70b-versatile": "🚀 En Zeki & Popüler (Metadata ve Analiz Uzmanı)",
                    "llama-3.1-8b-instant": "⚡ Işık Hızında (Hızlı Etiket ve Başlık Üretimi)",
                    "meta-llama/llama-4-scout-17b-16e-instruct": "✨ Yeni Nesil Llama-4 (Keskin ve Yaratıcı)",
                    "openai/gpt-oss-120b": "🐘 Dev Kapasite (Akıcı Video Senaryoları)",
                    "qwen/qwen3-32b": "📊 Titiz Veri İşleyici (JSON ve Liste Dostu)",
                    "moonshotai/kimi-k2-instruct": "📖 Uzun Metin Ustası (Transkript Analizi)",
                    "groq/compound": "🎯 Dengeli Performans (Genel SEO Kontrolleri)",
                    "openai/gpt-oss-20b": "⚖️ Hız ve Zeka Dengesi (Orta Segment)",
                    "allam-2-7b": "🛠️ Teknik ve Spesifik Konular (Öz Cevaplar)",
                    "groq/compound-mini": "🕊️ Çok Hafif (Basit Düzenleme İşleri)"
                }
                
                aktif_modeller = [m["id"] for m in resp.get("data", []) if "whisper" not in m["id"]]
                
                if not aktif_modeller:
                    logger.warning("Groq API başarılı yanıt verdi ancak model listesi boş döndü.")

                final_liste = []
                print(f"\n🔥 {provider} Şampiyonlar Ligi (Önerilenler):")
                
                count = 0
                for m_id, aciklama in groq_rehber.items():
                    if m_id in aktif_modeller:
                        count += 1
                        print(f"  [{count}] ⭐ {m_id} \n      └─> {aciklama}")
                        final_liste.append(m_id)
                
                print(f"\n🌐 Diğer Aktif Groq Modelleri:")
                for m_id in aktif_modeller:
                    if m_id not in final_liste:
                        count += 1
                        print(f"  [{count}] {m_id}")
                        final_liste.append(m_id)
                
                return final_liste
                
            except Exception as e:
                logger.error(f"Groq modelleri çekilemedi: {e}")
                return ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        #------------------------------------#
        #------------------------------------#
        
        #-------OPENROUTER SERVER SETTINGS-------#
        elif provider == "OPENROUTER":
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key: return []

            populer_rehber = {
                "meta-llama/llama-3.3-70b-instruct:free": "🚀 En Dengeli & Popüler (YouTube Uzmanı)",
                "google/gemma-3-27b-it:free": "🧠 Google'ın En Yenisi (Çok Zeki)",
                "nousresearch/hermes-3-llama-3.1-405b:free": "🐘 Dev Kapasite (En Karmaşık Sorular İçin)",
                "mistralai/mistral-small-3.1-24b-instruct:free": "🎯 Net ve Kısa Cevaplar (Hızlı Metadata)",
                "google/gemma-3-12b-it:free": "⚡ Hızlı ve Akıllı (İdeal Orta Segment)",
                "qwen/qwen3-next-80b-a3b-instruct:free": "📊 Mantık ve Veri İşleme (JSON/Liste Dostu)",
                "nvidia/nemotron-3-super-120b-a12b:free": "🛡️ Güvenilir Bilgi (Doğru Analiz)",
                "openai/gpt-oss-120b:free": "✍️ Akıcı Anlatım (Senaryo Yazımı İçin)",
                "cognitivecomputations/dolphin-mistral-24b-venice-edition:free": "🔓 Sansürsüz & Yaratıcı (Esnek Yazım)",
                "meta-llama/llama-3.2-3b-instruct:free": "🕊️ Çok Hafif (Basit Etiket İşleri)"
            }

            try:
                url = "https://openrouter.ai/api/v1/models"
                resp_obj = requests.get(url, timeout=10)
                resp_obj.raise_for_status()
                resp = resp_obj.json()
                
                tum_ucretsizler = []
                for m in resp.get("data", []):
                    model_id = m.get("id")
                    pricing = m.get("pricing", {})
                    prompt_price = pricing.get("prompt", 1)

                    try:
                        if model_id and float(prompt_price) == 0:
                            tum_ucretsizler.append(model_id)
                    except (TypeError, ValueError):
                        continue

                final_liste = []
                print(f"\n🔥 {provider} Şampiyonlar Ligi (Önerilenler):")
                
                count = 0
                # Önce popüler olanları rehberdeki açıklamayla yazdır
                for m_id, aciklama in populer_rehber.items():
                    if m_id in tum_ucretsizler:
                        count += 1
                        print(f"  [{count}] ⭐ {m_id} \n      └─> {aciklama}")
                        final_liste.append(m_id)
                
                print(f"\n🌐 Diğer Aktif Ücretsiz Modeller:")
                # Geri kalanları normal şekilde ekle
                for u_model in tum_ucretsizler:
                    if u_model not in final_liste:
                        count += 1
                        # Çok uzun liste olmaması için diğerlerini sadece isim olarak yazdırıyoruz
                        if count <= 15: # İlk 15'i göster (isteğe bağlı)
                            print(f"  [{count}] {u_model}")
                        final_liste.append(u_model)
                
                return final_liste
                
            except Exception as e:
                logger.error(f"OpenRouter modelleri çekilemedi: {e}")
                return OPENROUTER_FALLBACK_MODELS.copy()
        #------------------------------------#

        #-------APIFREELLM SERVER SETTINGS-------#
        elif provider == "APIFREELLM":
            api_key = os.getenv("APIFREELLM_API_KEY")
            if not api_key:
                logger.error("APIFREELLM_API_KEY bulunamadi! Cevre degiskenlerini kontrol edin.")
                return []
            try:
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                resp_obj = requests.get(_apifreellm_models_url(), headers=headers, timeout=10)
                if resp_obj.status_code >= 400:
                    raise _http_error_from_response("ApiFreeLLM", resp_obj)
                resp = resp_obj.json()
                data = resp.get("data", []) if isinstance(resp, dict) else []
                modeller = [item.get("id") for item in data if isinstance(item, dict) and item.get("id")]
                if modeller:
                    return modeller
            except Exception as e:
                logger.warning(f"ApiFreeLLM modelleri cekilemedi: {e}")

            env_models = [
                item.strip() for item in str(os.getenv("APIFREELLM_MODEL_OPTIONS", "") or "").split(",")
                if item.strip()
            ]
            if env_models:
                return env_models
            return APIFREELLM_FALLBACK_MODELS.copy()
        #------------------------------------#
                
        ##-------HUGGINGFACE SERVER SETTINGS-------#    
        elif provider == "HUGGINGFACE":
            # Model ID ve açıklamalarını içeren liste
            hf_data = [
                ("meta-llama/Llama-3.1-8B-Instruct", "🔥 En İyi Genel Performans"),
                ("Qwen/Qwen2.5-7B-Instruct", "🧠 Çok Akıllı & Detaycı"),
                ("google/gemma-2-9b-it", "🎨 Yaratıcı Yazım")
            ]
            
            print(f"\n✨ {provider} için önerilen modeller:")
            for i, (m_id, desc) in enumerate(hf_data, 1):
                print(f"  {i}) {m_id} -> {desc}")
                modeller.append(m_id) # Sadece ID'yi listeye ekle
        #------------------------------------#
        
        #-------DEEPSEEK SERVER SETTINGS-------#
        elif provider == "DEEPSEEK":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                logger.error("DEEPSEEK_API_KEY bulunamadı! Çevre değişkenlerini kontrol edin.")
                return []
            
            deepseek_rehber = {
                "deepseek-reasoner": "🧠 Karmaşık mantık yürütme, derin analiz (R1)",
                "deepseek-chat": "⚡ Hızlı, akıcı ve günlük görevler (V3)"
            }
            
            try:
                url = "https://api.deepseek.com/models"
                headers = {"Authorization": f"Bearer {api_key}"}
                resp_obj = requests.get(url, headers=headers, timeout=10)
                if resp_obj.status_code != 200:
                    raise Exception(f"HTTP Hata Kodu: {resp_obj.status_code} - Detay: {resp_obj.text}")
                
                resp = resp_obj.json()
                aktif_modeller = [m["id"] for m in resp.get("data", [])]
                final_liste = []
                
                print(f"\n🔥 {provider} Şampiyonlar Ligi (Önerilenler):")
                count = 0
                for m_id, aciklama in deepseek_rehber.items():
                    if m_id in aktif_modeller:
                        count += 1
                        print(f"  [{count}] ⭐ {m_id} \n      └─> {aciklama}")
                        final_liste.append(m_id)
                
                diger_modeller = [m for m in aktif_modeller if m not in final_liste]
                if diger_modeller:
                    print(f"\n🌐 Diğer Aktif Modeller:")
                    for u_model in diger_modeller:
                        count += 1
                        print(f"  [{count}] {u_model}")
                        final_liste.append(u_model)
                
                return final_liste
                
            except Exception as e:
                logger.error(f"DeepSeek modelleri çekilemedi: {e}")
                return ["deepseek-chat", "deepseek-reasoner"]
        #------------------------------------#

    except Exception as e:
        logger.warning(f"Modeller otomatik olarak çekilemedi. Hata: {e}")
        
    return modeller

def select_llm(role: str = "main") -> tuple:
    """Kullanıcıya YZ sağlayıcı seçtirir ve dinamik modelleri listeler."""
    role_key = _normalize_llm_role(role)
    profile = _get_role_profile(role)
    role_label = profile["short_label"]

    if not llm_selection_prompt_enabled():
        saglayici, model_adi = get_default_llm_config(role)
        if (
            role_key == "SMART"
            and _smart_llm_fallback_enabled()
            and not _verify_default_smart_llm(saglayici, model_adi)
        ):
            print("\n⚠️ Varsayilan yaratıcı model su anda cevap vermiyor veya kullanilamiyor.")
            saglayici, model_adi = _select_ollama_model_interactive(role)
        print("\n" + "-"*40)
        print("🤖 OTOMATIK YAPAY ZEKA ATAMASI")
        print("-"*40)
        print(f"Rol: {role_label} ({profile['title']})")
        print(f"Rol Profili: {profile['summary']}")
        print(f"Sağlayıcı: {saglayici}")
        print(f"Model: {model_adi}")
        _print_provider_guidance(role, saglayici)
        print("")
        logger.info(f"Otomatik LLM secimi aktif. Rol={role} | Saglayici={saglayici} | Model={model_adi}")
        return saglayici, model_adi

    print("\n" + "-"*40)
    print("🤖 YAPAY ZEKA MOTORU SEÇİMİ")
    print("-"*40)
    print(f"Rol: {role_label} ({profile['title']})")
    _print_role_guidance(role)
    print("")
    print("[1] Online (İnternet ve API Key gerektirir)")
    print("[2] Offline (Lokal bilgisayar gücünü kullanır)")
    
    baglanti_tipi = input("👉 Seçiminiz (1 veya 2): ").strip()
    
    saglayici = ""
    if baglanti_tipi == "1":
        _print_provider_guidance(role)
        print("\n🌐 ONLINE SAĞLAYICILAR:")
        print("  [1] Gemini (Google)")
        print("  [2] HuggingFace")
        print("  [3] Groq (Hız Şampiyonu)")
        print("  [4] OpenRouter (Ücretsiz Çeşitlilik)")
        print("  [5] ApiFreeLLM")
        print("  [6] DeepSeek (Ücretli API Key)")
        
        s_secim = input("👉 Sağlayıcı Seçiniz (1-6): ").strip()
        saglayicilar = {"1": "GEMINI", "2": "HUGGINGFACE", "3": "GROQ", "4": "OPENROUTER", "5": "APIFREELLM", "6": "DEEPSEEK"}
        saglayici = saglayicilar.get(s_secim, "GEMINI")
        _print_provider_guidance(role, saglayici)
        
    elif baglanti_tipi == "2":
        print("\n💻 OLLAMA MOD SECILDI")
        print("👉 Sistem Ollama modellerini listeleyecek (lokal + cloud).")
        saglayici = "OLLAMA"
        _print_provider_guidance(role, saglayici)
    else:
        print("❌ Geçersiz seçim. Varsayılan olarak GEMINI seçildi.")
        saglayici = "GEMINI"
        _print_provider_guidance(role, saglayici)

    modeller = fetch_dynamic_models(saglayici)
    modeller = _sort_models_for_role(saglayici, modeller, role)
    
    model_adi = ""
    if modeller:
        baslik = "OLLAMA Modelleri (lokal + cloud)" if saglayici == "OLLAMA" else f"{saglayici} Üzerinde Bulunan Modeller"
        print(f"\n📦 {baslik}:")
        for idx, m in enumerate(modeller, 1):
            print(f"  [{idx}] {_format_model_option(saglayici, m, role)}")
        allow_manual_model_name = saglayici != "OLLAMA"
        if allow_manual_model_name:
            print("  [0] Listede olmayan farklı bir model adı yazmak istiyorum")
        
        secim_araligi = f"0-{len(modeller)}" if allow_manual_model_name else f"1-{len(modeller)}"
        m_secim = input(f"👉 Model Seçiniz ({secim_araligi}): ").strip()
        
        if allow_manual_model_name and m_secim == "0":
            model_adi = input("👉 Kullanmak istediğiniz modelin tam adını yazın: ").strip()
        else:
            try:
                model_adi = modeller[int(m_secim) - 1]
            except (ValueError, IndexError):
                print("❌ Geçersiz seçim, listedeki ilk model ayarlandı.")
                model_adi = modeller[0]
    else:
        print(f"\n⚠️ {saglayici} için dinamik model listesi çekilemedi.")
        model_adi = input("👉 Lütfen kullanmak istediğiniz modelin adını manuel olarak yazın (Örn: gpt-4o): ").strip()

    print(f"\n✅ Seçilen Sistem: {saglayici} | Model: {model_adi}")
    print(f"🎯 Rol Uyumu: {profile['summary']}\n")
    return saglayici, model_adi

class CentralLLM:
    """Tüm otomasyon araçları için çoklu Yapay Zeka (LLM) yöneticisi."""
    
    def __init__(self, provider: str, model_name: str):
        self.provider = provider.upper()
        self.model_name = model_name
        self.client = None
        self._setup_client()
        logger.info(f"Yapay Zeka Motoru Hazır: {self.provider} ({self.model_name})")

    def _setup_client(self):
        
        if self.provider == "GEMINI":
            from google import genai
            self.api_key = os.getenv("GEMINI_API_KEY")
            if not self.api_key: raise ValueError("GEMINI_API_KEY eksik!")
            self.client = genai.Client(api_key=self.api_key)

        elif self.provider == "GROQ":
            import groq
            self.api_key = os.getenv("GROQ_API_KEY")
            if not self.api_key: raise ValueError("GROQ_API_KEY eksik!")
            self.client = groq.Groq(api_key=self.api_key)
            
        elif self.provider == "OPENROUTER":
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            if not self.api_key: raise ValueError("OPENROUTER_API_KEY eksik!")

        elif self.provider == "APIFREELLM":
            self.api_key = os.getenv("APIFREELLM_API_KEY")
            if not self.api_key: raise ValueError("APIFREELLM_API_KEY eksik!")
            self.base_url = _apifreellm_base_url()

        elif self.provider == "DEEPSEEK":
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
            if not self.api_key: raise ValueError("DEEPSEEK_API_KEY eksik!")

        elif self.provider == "HUGGINGFACE":
            self.api_key = os.getenv("HF_API_KEY")
            if not self.api_key: raise ValueError("HF_API_KEY eksik!")
            if HF_AVAILABLE:
                self.client = InferenceClient(api_key=self.api_key)
            else:
                logger.warning("huggingface_hub kütüphanesi kurulu değil, requests kullanılacak")

        elif self.provider == "OLLAMA":
            self.server_url = os.getenv("OLLAMA_LOCAL_SERVER", "http://localhost:11434")

    def uret(
        self,
        prompt: str,
        timeout: int = 120,
        max_retries: int = 2,
        ollama_options: dict | None = None,
        ollama_keep_alive: str | None = None,
    ) -> str:
        effective_timeout = timeout
        requests_timeout = effective_timeout
        if self.provider == "OLLAMA" and timeout == 120:
            effective_timeout = _env_int("OLLAMA_REQUEST_TIMEOUT_SECONDS", 300)
        if self.provider == "OLLAMA":
            requests_timeout = _ollama_requests_timeout(effective_timeout)

        def _call_api():
            # Tum mesaj tabanli modeller icin ortak yapi burada tanimlaniyor
            messages = [{"role": "user", "content": prompt}]

            if self.provider == "GEMINI":
                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                return resp.text

            elif self.provider == "GROQ":
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.5,
                    max_tokens=2048,
                )
                return completion.choices[0].message.content
                
            elif self.provider == "OPENROUTER":
                headers = {
                    "Authorization": f"Bearer {self.api_key}", 
                    "Content-Type": "application/json",
                    # OpenRouter bu iki opsiyonel basligi sever, bos bile olsa yollamak iyidir:
                    "HTTP-Referer": "http://localhost", 
                    "X-Title": "Youtube Otomasyon"
                }
                payload = {"model": self.model_name, "messages": messages}
                resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=effective_timeout,
                )
                if resp.status_code >= 400:
                    raise _http_error_from_response("OpenRouter", resp)
                resp_json = resp.json()
                
                if "choices" in resp_json:
                    return resp_json["choices"][0]["message"]["content"]
                else:
                    hata_mesaji = resp_json.get("error", resp_json)
                    raise RuntimeError(f"OpenRouter reddetti: {hata_mesaji}")

            elif self.provider == "APIFREELLM":
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "prompt": prompt,
                }
                resp = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=effective_timeout,
                )
                if resp.status_code >= 400:
                    raise _http_error_from_response("ApiFreeLLM", resp)
                resp_json = resp.json()
                if isinstance(resp_json, dict):
                    choices = resp_json.get("choices")
                    if isinstance(choices, list) and choices:
                        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                        content = message.get("content")
                        if content:
                            return content
                    for key in ("response", "text", "message", "output", "content"):
                        value = resp_json.get(key)
                        if isinstance(value, str) and value.strip():
                            return value
                raise RuntimeError(f"ApiFreeLLM beklenmeyen cevap dondurdu: {resp_json}")

            elif self.provider == "DEEPSEEK":
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                }
                resp = requests.post(
                    "https://api.deepseek.com/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=effective_timeout,
                )
                if resp.status_code >= 400:
                    raise _http_error_from_response("DeepSeek", resp)
                resp_json = resp.json()
                if "choices" in resp_json:
                    return resp_json["choices"][0]["message"]["content"]
                else:
                    hata_mesaji = resp_json.get("error", resp_json)
                    raise RuntimeError(f"DeepSeek reddetti: {hata_mesaji}")

            elif self.provider == "HUGGINGFACE":
                if HF_AVAILABLE and hasattr(self, 'client') and self.client:
                    response = self.client.chat_completion(
                        messages=messages,
                        model=self.model_name,
                        max_tokens=1000,
                        temperature=0.5
                    )
                    return response.choices[0].message.content
                else:
                    headers = {
                        "Authorization": f"Bearer {self.api_key}", 
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": self.model_name,
                        "messages": messages,
                        "max_tokens": 1000,
                        "temperature": 0.7
                    }
                    resp = requests.post(
                        f"https://api-inference.huggingface.co/models/{self.model_name}/v1/chat/completions", 
                        headers=headers, 
                        json=payload,
                        timeout=effective_timeout,
                    )
                    if resp.status_code >= 400:
                        raise _http_error_from_response("HuggingFace", resp)
                    resp_json = resp.json()
                    if "choices" in resp_json:
                        return resp_json["choices"][0]["message"]["content"]
                    else:
                        fallback_payload = {"inputs": prompt, "parameters": {"max_new_tokens": 1000}}
                        resp = requests.post(
                            f"https://api-inference.huggingface.co/models/{self.model_name}",
                            headers=headers,
                            json=fallback_payload,
                            timeout=effective_timeout,
                        )
                        if resp.status_code >= 400:
                            raise _http_error_from_response("HuggingFace", resp)
                        return resp.json()[0]["generated_text"]

            elif self.provider == "OLLAMA":
                use_streaming = _ollama_streaming_enabled()
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": use_streaming,
                }
                if ollama_options:
                    payload["options"] = ollama_options
                if ollama_keep_alive:
                    payload["keep_alive"] = ollama_keep_alive
                resp = requests.post(
                    f"{self.server_url}/api/generate",
                    json=payload,
                    timeout=requests_timeout,
                    stream=use_streaming,
                )
                if resp.status_code >= 400:
                    raise _http_error_from_response("Ollama", resp)
                if not use_streaming:
                    return resp.json()["response"]

                response_parts = []
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if "error" in chunk:
                        raise RuntimeError(chunk["error"])
                    response_parts.append(chunk.get("response", ""))
                    if chunk.get("done"):
                        break
                return "".join(response_parts)

        try:
            return retry_with_backoff(
                _call_api,
                f"{self.provider} uretim istegi",
                logger,
                max_attempts=max_retries + 1,
                base_delay_seconds=15,
                max_delay_seconds=60,
            )
        except Exception as e:
            logger.error(f"{self.provider} istegi kalici olarak basarisiz oldu: {e}")
            raise LLMConnectionError(f"Basarisiz oldu: {e}") from e
