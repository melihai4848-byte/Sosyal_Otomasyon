import json
import os
import shutil
import time
from datetime import datetime
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Dict, Optional

from moduller.config import INPUTS_DIR, OUTPUTS_DIR
from moduller.exceptions import DependencyError
from moduller.llm_manager import (
    CentralLLM,
    get_default_llm_config,
    get_module_recommended_llm_config,
    select_llm,
)
from moduller.logger import get_logger
from moduller.module_registry import GROUP_TITLE_BY_KEY, MODULE_BY_KEY, get_pipeline_modules
from moduller.output_paths import grouped_output_path, json_output_path, txt_output_path
from moduller.runtime_utils import format_elapsed
from moduller.social_media_utils import load_related_json
from moduller.subtitle_output_utils import (
    find_subtitle_artifact,
    find_subtitle_file,
    list_subtitle_files,
    subtitle_intermediate_output_path,
    subtitle_output_path,
)

logger = get_logger("full_automation")
TERMINAL_AYIRICI = "-" * 72
CHECKPOINT_DIR = grouped_output_path("research", "checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
_PIPELINE_EVENT_HOOK = None
FULL_AUTOMATION_MAIN_LLM = ("OLLAMA", "qwen3:14b")
FULL_AUTOMATION_TRANSLATION_MODEL = os.getenv("TRANSLATEGEMMA_MODEL_NAME", "translategemma:12b-it-q4_K_M").strip()
FULL_AUTOMATION_TRIM_LLM = ("OLLAMA", "kimi-k2.5:cloud")


def _build_full_automation_step_llm_models() -> dict[str, tuple[str, str, str]]:
    return {
        "description": ("201", *get_module_recommended_llm_config("201", "smart")),
        "critic": ("202", *get_module_recommended_llm_config("202", "smart")),
        "broll": ("203", *get_module_recommended_llm_config("203", "smart")),
        "thumbnail_main": ("204", *get_module_recommended_llm_config("204", "smart")),
        "music": ("205", *get_module_recommended_llm_config("205", "main")),
        "carousel": ("301", *get_module_recommended_llm_config("301", "smart")),
        "reels": ("302", *get_module_recommended_llm_config("302", "smart")),
        "story": ("303", *get_module_recommended_llm_config("303", "smart")),
    }


FULL_AUTOMATION_STEP_LLM_MODELS = _build_full_automation_step_llm_models()
FULL_AUTOMATION_HYBRID_STEP_KEYS = {"critic", "carousel", "reels", "story"}


def _load_runtime_module(modul_yolu: str):
    son_parca = str(modul_yolu or "").split(".")[-1]
    if son_parca and son_parca[0].isdigit():
        hedef = Path(__file__).resolve().with_name(f"{son_parca}.py")
        spec = spec_from_file_location(modul_yolu, hedef)
        if spec is None or spec.loader is None:
            raise ImportError(f"Modul yuklenemedi: {modul_yolu}")
        modul = module_from_spec(spec)
        spec.loader.exec_module(modul)
        return modul
    return import_module(modul_yolu)

PIPELINE_STEPS = [
    {"key": entry.key, "number": entry.number, "report_key": entry.pipeline_report_key, "label": entry.title}
    for entry in get_pipeline_modules()
]
PIPELINE_OPTIONS = [(item["key"], item["label"]) for item in PIPELINE_STEPS]
PIPELINE_STEP_BY_KEY = {item["key"]: item for item in PIPELINE_STEPS}
PIPELINE_ORDER = [item["key"] for item in PIPELINE_STEPS]
PIPELINE_DEPENDENCIES = {
    entry.key: set(entry.pipeline_dependencies)
    for entry in get_pipeline_modules()
    if entry.pipeline_dependencies
}
MAIN_LLM_STEPS = {entry.key for entry in get_pipeline_modules() if entry.requires_main_llm}
SMART_LLM_STEPS = {entry.key for entry in get_pipeline_modules() if entry.requires_smart_llm}


def _prompt_selected_steps_recommended_profile(
    selected_steps: Optional[set[str]],
    summary_steps: Optional[set[str]] = None,
) -> bool:
    if not selected_steps:
        return False

    display_steps = summary_steps if summary_steps is not None else selected_steps
    llm_relevant_steps = [
        key for key in _ordered_steps(display_steps)
        if key in PIPELINE_STEP_BY_KEY
    ]
    if not llm_relevant_steps:
        return False

    lines = ["Secilen moduller icin onerilen LLM profili:"]
    for step_key in llm_relevant_steps:
        entry = MODULE_BY_KEY.get(step_key)
        if not entry:
            continue
        roles = []
        if entry.requires_main_llm:
            _provider, model_name = get_module_recommended_llm_config(entry.number, "main")
            roles.append(f"main={_provider}:{model_name}")
        if entry.requires_smart_llm:
            _provider, model_name = get_module_recommended_llm_config(entry.number, "smart")
            roles.append(f"smart={_provider}:{model_name}")
        if roles:
            lines.append(f"{entry.number}: {entry.title} -> " + " | ".join(roles))
        else:
            lines.append(f"{entry.number}: {entry.title} -> LLM gerekmez")

    _print_terminal_block("COKLU SECIM LLM OZETI", "🤖", "\n".join(lines))
    print("[1] Onerilen profili kullan")
    print("[2] Kendim secmek istiyorum")
    while True:
        secim = input("👉 Secim (1 veya 2): ").strip()
        if secim == "1":
            return True
        if secim == "2":
            return False
        print("Lutfen sadece 1 veya 2 girin.")


def _build_selected_step_llm_overrides(
    selected_steps: Optional[set[str]],
    get_llm,
) -> dict[str, CentralLLM]:
    overrides: dict[str, CentralLLM] = {}
    if not selected_steps:
        return overrides

    for step_key in _ordered_steps(selected_steps):
        entry = MODULE_BY_KEY.get(step_key)
        if not entry or not entry.requires_smart_llm:
            continue
        provider, model_name = get_module_recommended_llm_config(entry.number, "smart")
        overrides[step_key] = get_llm(provider, model_name)
    return overrides


def _build_base_name(srt_adi: str) -> str:
    temel_isim = srt_adi.replace(".srt", "")
    temel_isim = (
        temel_isim.replace("_grammar_fixed", "")
        .replace("_raw", "")
        .replace("_tr", "")
        .replace("_standart", "")
        .replace("_standard", "")
    )
    if temel_isim.endswith("_"):
        temel_isim = temel_isim[:-1]
    return temel_isim


GENERIC_SUBTITLE_OUTPUT_NAMES = {
    "subtitle_raw_tr.srt",
    "subtitle_raw_en.srt",
    "subtitle_raw_shorts.srt",
    "subtitle_tr.srt",
    "subtitle_en.srt",
    "subtitle_shorts.srt",
    "subtitle_llm_en.srt",
    "subtitle_de.srt",
}
GENERIC_SUBTITLE_STEMS = {
    "subtitle",
    "subtitle_raw",
    "subtitle_tr",
    "subtitle_raw_tr",
    "subtitle_raw_en",
    "subtitle_raw_shorts",
    "subtitle_en",
    "subtitle_llm_en",
    "subtitle_de",
    "subtitle_shorts",
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _translation_enabled_output_paths() -> list[Path]:
    outputs: list[Path] = []
    if _env_bool("TRANSLATION_ENABLE_ENGLISH", True):
        outputs.append(subtitle_intermediate_output_path("subtitle_llm_en.srt"))
    if _env_bool("TRANSLATION_ENABLE_GERMAN", True):
        outputs.append(subtitle_output_path("subtitle_de.srt"))
    return outputs


def _is_generic_subtitle_output(path: Optional[Path]) -> bool:
    return bool(path) and path.name.lower() in GENERIC_SUBTITLE_OUTPUT_NAMES


def _selected_srt_can_skip_grammar(path: Optional[Path]) -> bool:
    if not path or path.suffix.lower() != ".srt":
        return False

    ad = path.name.lower()
    if "raw" in ad:
        return False
    if ad in {"subtitle_raw_tr.srt"}:
        return False
    return True


def _is_output_fresh(candidate: Optional[Path], source: Optional[Path]) -> bool:
    if not candidate or not source:
        return False
    try:
        return candidate.exists() and source.exists() and candidate.stat().st_mtime >= source.stat().st_mtime
    except OSError:
        return False


def _candidate_video_stems(video_yolu: Optional[Path]) -> list[str]:
    if not video_yolu:
        return []

    adaylar = [_build_base_name(video_yolu.name)]
    if video_yolu.suffix.lower() != ".srt":
        adaylar.insert(0, video_yolu.stem)

    sonuc = []
    gorulen = set()
    for aday in adaylar:
        temiz = str(aday or "").strip()
        if not temiz:
            continue
        if temiz.casefold() in GENERIC_SUBTITLE_STEMS:
            continue
        anahtar = temiz.casefold()
        if anahtar in gorulen:
            continue
        gorulen.add(anahtar)
        sonuc.append(temiz)
    return sonuc


def _route_run(routing: dict, key: str, default: bool) -> bool:
    if not isinstance(routing, dict):
        return default
    item = routing.get(key, {})
    if isinstance(item, dict):
        return bool(item.get("run", default))
    return bool(item)


def _route_reason(routing: dict, key: str, default: str = "") -> str:
    if not isinstance(routing, dict):
        return default
    item = routing.get(key, {})
    if isinstance(item, dict):
        return str(item.get("reason", default))
    return default


def _build_forced_detail(routing: dict, key: str, varsayilan: str) -> str:
    reason = _route_reason(routing, key, varsayilan)
    if _route_run(routing, key, True):
        return reason
    return f"{reason} Coklu Secim kapsamli cikti urettigi icin bu modul yine de calistirildi."


def _legacy_routing() -> Dict[str, dict]:
    return {
        "hook_rewrite": {"run": False, "reason": "AI Critic sonucu yok, varsayilan olarak atlandi."},
        "trim_suggestions": {"run": False, "reason": "AI Critic sonucu yok, varsayilan olarak atlandi."},
        "broll_generator": {"run": True, "reason": "Eski pipeline davranisi korunuyor."},
        "reels_shorts": {"run": True, "reason": "Eski pipeline davranisi korunuyor."},
        "metadata_generator": {"run": True, "reason": "Eski pipeline davranisi korunuyor."},
    }


def _pipeline_step(key: str) -> dict:
    return PIPELINE_STEP_BY_KEY[key]


def _pipeline_label(key: str) -> str:
    return _pipeline_step(key)["label"]


def _pipeline_number(key: str) -> str:
    return _pipeline_step(key)["number"]


def _pipeline_report_key(key: str) -> str:
    return _pipeline_step(key)["report_key"]


def _step_selected(selected_steps: Optional[set], key: str) -> bool:
    return selected_steps is None or key in selected_steps


def _all_pipeline_steps() -> set[str]:
    return set(PIPELINE_ORDER)


def _ordered_steps(step_keys: set[str]) -> list[str]:
    return [key for key in PIPELINE_ORDER if key in step_keys]


def _checkpoint_path(video_stem: str) -> Path:
    return CHECKPOINT_DIR / f"{video_stem}_coklu_secim_checkpoint.json"


def _load_checkpoint(video_stem: str) -> Optional[dict]:
    path = _checkpoint_path(video_stem)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Checkpoint dosyasi okunamadi: {path.name} | {exc}")
        return None


def _successful_step_keys(steps: dict) -> set[str]:
    tamamlanan = set()
    for step in PIPELINE_STEPS:
        report = steps.get(step["report_key"], {})
        if str(report.get("status", "")).startswith("✅"):
            tamamlanan.add(step["key"])
    return tamamlanan


def _save_checkpoint(video_stem: str, rapor: dict, requested_steps: set[str], planned_steps: set[str], auto_added_dependencies: list[str], run_state: str, current_step_no: str = "", current_step_title: str = "", current_status: str = "") -> Path:
    payload = {
        "version": 1,
        "video_stem": video_stem,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "run_state": run_state,
        "requested_steps": _ordered_steps(requested_steps),
        "planned_steps": _ordered_steps(planned_steps),
        "completed_steps": _ordered_steps(_successful_step_keys(rapor.get("steps", {}))),
        "auto_added_dependencies": auto_added_dependencies,
        "current_step": {"number": current_step_no, "title": current_step_title, "status": current_status},
        "steps": rapor.get("steps", {}),
    }
    path = _checkpoint_path(video_stem)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _save_master_report(video_stem: str, rapor: dict) -> None:
    json_yolu = json_output_path("master_pipeline")
    txt_yolu = txt_output_path("master_pipeline")
    with open(json_yolu, "w", encoding="utf-8") as f:
        json.dump(rapor, f, ensure_ascii=False, indent=2)
    lines = [
        f"=== {video_stem} ICIN MASTER PIPELINE RAPORU ===",
        "",
        f"Ana Yapay Zeka Modeli: {rapor.get('models', {}).get('main_model', '')}",
        f"Gramer Adiminda Kullanilan Model: {rapor.get('models', {}).get('grammar_model', '')}",
        f"Yaratici Yapay Zekasi Modeli: {rapor.get('models', {}).get('smart_model', '')}",
        f"Toplam Pipeline Suresi: {rapor.get('execution_context', {}).get('total_duration', '')}",
        "",
        "SORUN LISTESI",
        "-" * 60,
    ]
    issues = rapor.get("critic_issues", [])
    if issues:
        lines.extend(f"- {item}" for item in issues)
    else:
        lines.append("- AI Critic tarafindan belirgin kritik sorun listelenmedi.")
    lines.extend(["", "ROUTING KARARLARI", "-" * 60])
    for key, value in rapor.get("routing_decisions", {}).items():
        lines.append(f"{key}: {'CALISTI' if value.get('run') else 'ATLANDI'}")
        lines.append(f"Neden: {value.get('reason', '')}")
        lines.append("")
    lines.extend(["ADIM RAPORU", "-" * 60])
    for step in PIPELINE_STEPS:
        bilgi = rapor.get("steps", {}).get(step["report_key"])
        if not bilgi or str(bilgi.get("status", "")).startswith("⏭️"):
            continue
        satir = f"{step['number']}. {step['label']}: {bilgi.get('status', '')}"
        if bilgi.get("duration"):
            satir += f" | Sure: {bilgi.get('duration')}"
        lines.append(satir)
        if bilgi.get("detail"):
            lines.append(f"Detay: {bilgi.get('detail')}")
        for path in bilgi.get("outputs", []):
            lines.append(f"Cikti: {path}")
        lines.append("")
    txt_yolu.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    logger.info(f"Master rapor kaydedildi: {txt_yolu.name}")


def _print_terminal_block(baslik: str, emoji: str = "🚀", detay: str = "") -> None:
    print("\n" * 2, end="")
    print(TERMINAL_AYIRICI)
    print(f"{emoji} {baslik}")
    if detay:
        print(detay)
    print(TERMINAL_AYIRICI)
    print("\n" * 2, end="")


def _start_step(adim_no: str, baslik: str, detay: str = "") -> None:
    logger.info(f"Adim {adim_no} basliyor: {baslik}")
    if _PIPELINE_EVENT_HOOK:
        _PIPELINE_EVENT_HOOK("start", adim_no, baslik, "", detay, [])
    _print_terminal_block(f"ADIM {adim_no} | {baslik}", "🔄", detay)


def _finish_step(adim_no: str, baslik: str, durum: str, detay: str = "", outputs: Optional[list] = None) -> None:
    emoji = "✅" if durum.startswith("✅") else "⚠️" if durum.startswith("⚠️") else "⏭️" if durum.startswith("⏭️") else "❌"
    satirlar = [f"Durum: {durum}"]
    if detay:
        satirlar.append(f"Detay: {detay}")
    if outputs:
        satirlar.extend([f"Cikti: {path}" for path in outputs[:4]])
    logger.info(f"Adim {adim_no} tamamlandi: {baslik} -> {durum}")
    if _PIPELINE_EVENT_HOOK:
        _PIPELINE_EVENT_HOOK("finish", adim_no, baslik, durum, detay, outputs or [])
    _print_terminal_block(f"ADIM {adim_no} | {baslik}", emoji, "\n".join(satirlar))


def _select_video():
    media_patterns = ("*.mp4", "*.mp3", "*.wav", "*.mov", "*.mkv", "*.m4v")
    video_files = [path for pattern in media_patterns for path in INPUTS_DIR.glob(pattern)]
    if not video_files:
        logger.error("❌ workspace/00_Inputs klasorunde video/ses bulunamadi!")
        return None
    print("\n📂 Lutfen islenecek medya dosyasini secin:")
    for idx, video in enumerate(video_files, start=1):
        print(f"  [{idx}] {video.name}")
    try:
        return video_files[int(input("👉 Seciminiz: ")) - 1]
    except (ValueError, IndexError):
        logger.error("❌ Gecersiz secim! Otomasyon iptal edildi.")
        return None


def _select_srt_input():
    default_srt = find_subtitle_file("subtitle_tr.srt")
    if default_srt and default_srt.exists():
        logger.info("Ana SRT olarak subtitle_tr.srt otomatik secildi.")
        return default_srt

    srt_files = list_subtitle_files()
    if not srt_files:
        logger.error("❌ 100_Altyazı klasorunde kullanilabilir bir SRT bulunamadi!")
        return None

    print("\n📂 Lutfen kullanilacak ana SRT dosyasini secin:")
    for idx, srt in enumerate(srt_files, start=1):
        print(f"  [{idx}] {srt.name}")

    try:
        return srt_files[int(input("👉 Seciminiz: ")) - 1]
    except (ValueError, IndexError):
        logger.error("❌ Gecersiz secim! Otomasyon iptal edildi.")
        return None


def _resolve_primary_input(requested_steps: set[str], planned_steps: set[str]) -> Optional[Path]:
    subtitle_dependency_needed = "subtitle" in planned_steps
    explicit_subtitle_requested = "subtitle" in requested_steps
    available_srt_files = list_subtitle_files()

    if explicit_subtitle_requested:
        _print_terminal_block("GIRDI SECIMI", "📥", "Secilen moduller altyazi olusturacagi icin kaynak video/ses dosyasi secilecek.")
        secilen = _select_video()
        if secilen:
            _print_terminal_block("GIRDI HAZIR", "✅", f"Secilen kaynak medya: {secilen.name}")
        return secilen

    if available_srt_files:
        _print_terminal_block("GIRDI SECIMI", "📥", "Secilen moduller mevcut bir ana SRT dosyasi uzerinden calisacak.")
        secilen = _select_srt_input()
        if secilen:
            _print_terminal_block("GIRDI HAZIR", "✅", f"Secilen ana SRT: {secilen.name}")
        return secilen

    if subtitle_dependency_needed:
        _print_terminal_block("GIRDI SECIMI", "📥", "Secilen moduller icin once standart altyazi gerektigi icin kaynak video/ses dosyasi secilecek.")
        secilen = _select_video()
        if secilen:
            _print_terminal_block("GIRDI HAZIR", "✅", f"Secilen kaynak medya: {secilen.name}")
        return secilen

    logger.error("❌ Bu secim icin gerekli bir SRT bulunamadi.")
    return None


def _selected_srt_input(path: Optional[Path]) -> Optional[Path]:
    if path and path.exists() and path.suffix.lower() == ".srt":
        return path
    return None


def _clear_outputs_for_full_pipeline() -> None:
    if not OUTPUTS_DIR.exists():
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        return

    silinen_oge_sayisi = 0
    for path in OUTPUTS_DIR.iterdir():
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            silinen_oge_sayisi += 1
        except Exception as exc:
            logger.warning(f"Output temizligi sirasinda silinemeyen oge atlandi: {path.name} | {exc}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Tum pipeline secildigi icin workspace/00_Outputs klasoru sifirlandi. Silinen oge sayisi: {silinen_oge_sayisi}")


def _validate_pipeline_dag() -> None:
    visiting = set()
    visited = set()
    def visit(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            raise DependencyError(f"Bagimlilik dongusu tespit edildi: {node}")
        if node not in PIPELINE_STEP_BY_KEY:
            raise DependencyError(f"Tanimlanmamis pipeline adimi: {node}")
        visiting.add(node)
        for dep in PIPELINE_DEPENDENCIES.get(node, set()):
            visit(dep)
        visiting.remove(node)
        visited.add(node)
    for node in PIPELINE_ORDER:
        visit(node)


def _resolve_dependencies(selected_steps: Optional[set]) -> tuple[set[str], list[str]]:
    requested_steps = _all_pipeline_steps() if selected_steps is None else set(selected_steps)
    resolved = set()
    resolving = set()
    def resolve(node: str) -> None:
        if node not in PIPELINE_STEP_BY_KEY:
            raise DependencyError(f"Gecersiz modul secimi: {node}")
        if node in resolved:
            return
        if node in resolving:
            raise DependencyError(f"Bagimlilik dongusu tespit edildi: {node}")
        resolving.add(node)
        for dependency in PIPELINE_DEPENDENCIES.get(node, set()):
            resolve(dependency)
        resolving.remove(node)
        resolved.add(node)
    for step_key in requested_steps:
        resolve(step_key)
    auto_added = _ordered_steps(resolved - requested_steps)
    return set(_ordered_steps(resolved)), auto_added


def _pipeline_number_map() -> dict[str, str]:
    return {str(step["number"]): step["key"] for step in PIPELINE_STEPS}


def _print_pipeline_selection_menu() -> None:
    _print_terminal_block("COKLU CALISMA SECIMI", "🧩", "Calismasini istedigin adimlari virgulle sec.\nBuradaki numaralar ana menudeki modullerle aynidir.\nBos birakirsan tum pipeline calisir. Ornek: 3,4,5,6,7,8")
    for step in PIPELINE_STEPS:
        entry = MODULE_BY_KEY.get(step["key"])
        group_label = GROUP_TITLE_BY_KEY.get(entry.group, entry.group) if entry else "GRUP"
        group_label = (
            group_label.replace("📝", "")
            .replace("📺", "")
            .replace("📱", "")
            .replace("🔎", "")
            .replace("🧰", "")
            .replace("(", "")
            .replace(")", "")
            .strip()
        )
        print(f"[{step['number']}] {group_label} - {step['label']}")


def _parse_pipeline_selection(raw: str) -> tuple[Optional[set[str]], list[str]]:
    raw = raw.strip()
    if not raw:
        return None, []

    number_to_key = _pipeline_number_map()
    selected = set()
    invalid = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if item in number_to_key:
            selected.add(number_to_key[item])
        else:
            invalid.append(item)
    if invalid or not selected:
        return set(), invalid or [raw]
    return selected, []


def _get_pipeline_selection(preselected_selection: Optional[str] = None) -> tuple[Optional[set[str]], list[str]]:
    while True:
        if preselected_selection is None:
            _print_pipeline_selection_menu()
            raw = input("\n👉 Adim secimi: ").strip()
        else:
            raw = preselected_selection.strip()

        selected, invalid = _parse_pipeline_selection(raw)
        if invalid:
            mesaj = ", ".join(invalid)
            logger.warning(f"❌ Gecersiz adim secimi: {mesaj}. Lutfen listeden gecerli numaralar girin.")
            if preselected_selection is not None:
                return set(), invalid
            continue
        return selected, []


def _setup_selected_llms(
    selected_steps: Optional[set],
    summary_steps: Optional[set] = None,
) -> tuple[Optional[CentralLLM], Optional[CentralLLM], dict[str, CentralLLM]]:
    cache: dict[tuple[str, str], CentralLLM] = {}

    def get_llm(provider: str, model_name: str) -> CentralLLM:
        key = (provider.upper(), model_name)
        if key not in cache:
            cache[key] = CentralLLM(provider=provider, model_name=model_name)
        return cache[key]

    need_main = selected_steps is None or bool(selected_steps & MAIN_LLM_STEPS)
    need_smart = selected_steps is None or bool(selected_steps & SMART_LLM_STEPS)
    llm_ana = None
    llm_cila = None
    step_llm_overrides: dict[str, CentralLLM] = {}

    if (need_main or need_smart) and selected_steps is not None and _prompt_selected_steps_recommended_profile(selected_steps, summary_steps):
        if need_main:
            llm_ana = get_llm(*get_default_llm_config("main"))
        if need_smart:
            llm_cila = get_llm(*get_default_llm_config("smart"))
            step_llm_overrides = _build_selected_step_llm_overrides(selected_steps, get_llm)

        summary_lines = []
        if llm_ana:
            summary_lines.append(f"Main varsayilani: {llm_ana.model_name}")
        if llm_cila:
            summary_lines.append(f"Smart varsayilani: {llm_cila.model_name}")
        display_steps = summary_steps if summary_steps is not None else selected_steps
        for step_key in _ordered_steps(display_steps or set()):
            entry = MODULE_BY_KEY.get(step_key)
            if not entry:
                continue
            roles = []
            if entry.requires_main_llm and llm_ana:
                roles.append(f"main={llm_ana.model_name}")
            if entry.requires_smart_llm:
                active_smart_llm = step_llm_overrides.get(step_key) or llm_cila
                if active_smart_llm:
                    roles.append(f"smart={active_smart_llm.model_name}")
            if roles:
                summary_lines.append(f"{entry.number}: " + " | ".join(roles))
            else:
                summary_lines.append(f"{entry.number}: LLM gerekmez")
        if summary_lines:
            _print_terminal_block("SECILEN OTO PROFIL HAZIR", "✅", "\n".join(summary_lines))
        return llm_ana, llm_cila, step_llm_overrides

    if need_main:
        _print_terminal_block("ANA YAPAY ZEKA SECIMI", "🛠️", "Lokal LLM tavsiye edilir.\nBu model, yaraticilik gerektirmeyen gorevlerde dogru cikti uretimi icin kullanilacaktir.")
        saglayici_ana, model_ana = select_llm("main")
        llm_ana = get_llm(saglayici_ana, model_ana)
        _print_terminal_block("ANA YAPAY ZEKA HAZIR", "✅", f"Bu model, yaraticilik gerektirmeyen gorevlerde dogru cikti uretimi icin kullanilacaktir.\nSecilen yapay zeka modeli: {model_ana}")
    if need_smart:
        _print_terminal_block("YARATICI YAPAY ZEKA SECIMI", "✨", "Online LLM tavsiye edilir.\nBu model, yaratici gorevlerde, ozgun icerik uretmede ve analiz etme gorevleri icin kullanilacaktir. Online ya da gelismis modeller tavsiye edilir.")
        saglayici_cila, model_cila = select_llm("smart")
        llm_cila = get_llm(saglayici_cila, model_cila)
        _print_terminal_block("YARATICI YAPAY ZEKASI HAZIR", "✅", f"Online LLM tavsiye edilir.\nBu model, yaratici gorevlerde, ozgun icerik uretmede ve analiz etme gorevleri icin kullanilacaktir. Online ya da gelismis modeller tavsiye edilir.\nSecilen yapay zeka modeli: {model_cila}")
    return llm_ana, llm_cila, step_llm_overrides


def _setup_full_pipeline_llms(selected_steps: Optional[set]) -> tuple[Optional[CentralLLM], Optional[CentralLLM], dict[str, CentralLLM]]:
    cache: dict[tuple[str, str], CentralLLM] = {}

    def get_llm(provider: str, model_name: str) -> CentralLLM:
        key = (provider.upper(), model_name)
        if key not in cache:
            cache[key] = CentralLLM(provider=provider, model_name=model_name)
        return cache[key]

    llm_ana = get_llm(*FULL_AUTOMATION_MAIN_LLM)
    llm_cila = get_llm("OLLAMA", "deepseek-v3.1:671b-cloud")
    step_llm_overrides: dict[str, CentralLLM] = {}

    for step_key, (_module_number, _provider, model_name) in FULL_AUTOMATION_STEP_LLM_MODELS.items():
        if selected_steps is not None and step_key not in selected_steps:
            continue
        step_llm_overrides[step_key] = get_llm(_provider, model_name)
    if selected_steps is None or "critic" in selected_steps:
        step_llm_overrides["critic_trim"] = get_llm(*FULL_AUTOMATION_TRIM_LLM)

    summary_lines = [
        "Full otomasyon secildigi icin LLM secimi sorulmayacak.",
        "101: Whisper (lokal transcription)",
        f"102: {FULL_AUTOMATION_MAIN_LLM[1]} (lokal ana LLM)",
        f"103: {FULL_AUTOMATION_TRANSLATION_MODEL} (TranslateGemma)",
        f"202 trim raporu: {FULL_AUTOMATION_TRIM_LLM[1]}",
        f"Main LLM varsayilani: {FULL_AUTOMATION_MAIN_LLM[1]}",
    ]
    for step_key, (module_number, _provider, model_name) in FULL_AUTOMATION_STEP_LLM_MODELS.items():
        if selected_steps is not None and step_key not in selected_steps:
            continue
        entry = MODULE_BY_KEY.get(step_key)
        if not entry:
            continue
        roles = []
        if entry.requires_main_llm:
            if entry.requires_smart_llm:
                roles.append(f"main={FULL_AUTOMATION_MAIN_LLM[0]}:{FULL_AUTOMATION_MAIN_LLM[1]}")
            else:
                roles.append(f"main={_provider}:{model_name}")
        if entry.requires_smart_llm:
            roles.append(f"smart={_provider}:{model_name}")
        if roles:
            summary_lines.append(f"{module_number}: " + " | ".join(roles))
        else:
            summary_lines.append(f"{module_number}: LLM gerekmez")

    _print_terminal_block("FULL OTOMASYON OTOMATIK LLM PROFILI", "🤖", "\n".join(summary_lines))
    return llm_ana, llm_cila, step_llm_overrides


def _find_existing_srt(video_yolu: Path, grammar_fixed: bool = False) -> Optional[Path]:
    adaylar = []
    for stem in _candidate_video_stems(video_yolu):
        if grammar_fixed:
            adaylar.extend(
                [
                    subtitle_output_path(f"{stem}_standart_tr_grammar_fixed.srt"),
                    subtitle_output_path(f"{stem}_standard_tr_grammar_fixed.srt"),
                    subtitle_output_path(f"{stem}_raw_grammar_fixed.srt"),
                    subtitle_output_path(f"{stem}_tr.srt"),
                ]
            )
        else:
            adaylar.extend(
                [
                    subtitle_output_path(f"{stem}_standart_tr.srt"),
                    subtitle_output_path(f"{stem}_standard_tr.srt"),
                    subtitle_output_path(f"{stem}_raw_tr.srt"),
                    subtitle_output_path(f"{stem}_raw.srt"),
                ]
            )

    if _is_generic_subtitle_output(video_yolu):
        if grammar_fixed:
            generic_srt = find_subtitle_file("subtitle_tr.srt")
        else:
            generic_srt = find_subtitle_artifact("subtitle_raw_tr.srt")
        if generic_srt:
            adaylar.append(generic_srt)

    gorulen = set()
    for path in adaylar:
        key = str(path).lower()
        if key in gorulen:
            continue
        gorulen.add(key)
        if path.exists():
            return path
    return None


def _find_existing_translation_outputs(srt_path: Optional[Path]) -> list[str]:
    required_outputs = _translation_enabled_output_paths()
    if not required_outputs:
        return []

    for stem in _candidate_video_stems(srt_path):
        adaylar = []
        if _env_bool("TRANSLATION_ENABLE_ENGLISH", True):
            adaylar.append(subtitle_intermediate_output_path(f"{stem}_en.srt"))
        if _env_bool("TRANSLATION_ENABLE_GERMAN", True):
            adaylar.append(subtitle_intermediate_output_path(f"{stem}_de.srt"))
        if adaylar and all(path.exists() for path in adaylar):
            return [str(path) for path in adaylar]

    if _is_generic_subtitle_output(srt_path):
        if all(path.exists() for path in required_outputs):
            return [str(path) for path in required_outputs]

    return []


def _is_auto_added_step(step_key: str, requested_steps: set[str], planned_steps: set[str]) -> bool:
    return step_key in planned_steps and step_key not in requested_steps


def _build_reused_step_entry(status: str, detail: str, outputs: Optional[list] = None) -> dict:
    return {
        "status": status,
        "detail": detail,
        "duration": "",
        "outputs": outputs or [],
    }


def _restore_existing_output_from_step(
    step_entry: Optional[dict],
    *,
    preferred_names: tuple[str, ...] = (),
    suffix: str | None = None,
) -> Optional[Path]:
    if not isinstance(step_entry, dict):
        return None

    outputs = step_entry.get("outputs", [])
    if not isinstance(outputs, list):
        return None

    normalized_preferred = tuple(name.lower() for name in preferred_names)
    existing_paths: list[Path] = []
    prioritized_paths: list[Path] = []

    for raw_output in outputs:
        try:
            path = Path(str(raw_output))
        except Exception:
            continue
        if not path.exists():
            continue
        if suffix and path.suffix.lower() != suffix.lower():
            continue
        if normalized_preferred and path.name.lower() in normalized_preferred:
            prioritized_paths.append(path)
        else:
            existing_paths.append(path)

    if prioritized_paths:
        return prioritized_paths[0]
    if existing_paths:
        return existing_paths[0]
    return None


def _skip_step(rapor: dict, key: str, baslik: str, detail: str, outputs: Optional[list] = None):
    mevcut = rapor["steps"].get(key)
    if mevcut and str(mevcut.get("status", "")).startswith("✅"):
        logger.info(f"{baslik} adimi checkpoint'te tamamlanmis bulundu; tekrar calistirilmiyor.")
        return
    rapor["steps"][key] = {"status": "⏭️ Atlandi", "detail": detail, "outputs": outputs or []}


def _start_pipeline_step(step_key: str, detail: str = "") -> None:
    _start_step(_pipeline_number(step_key), _pipeline_label(step_key), detail)


def _finish_pipeline_step(step_key: str, status: str, detail: str = "", outputs: Optional[list] = None) -> None:
    _finish_step(_pipeline_number(step_key), _pipeline_label(step_key), status, detail, outputs or [])


def _countdown_between_steps(seconds: int, next_step_key: str) -> None:
    if seconds <= 0:
        return
    next_label = _pipeline_label(next_step_key)
    logger.info(f"Sonraki adim icin {seconds} saniyelik bekleme basladi: {next_label}")
    for remaining in range(seconds, 0, -1):
        message = f"⏳ Sonraki adim {remaining} saniye sonra baslayacak: {next_label}"
        print(f"\r{message}   ", end="", flush=True)
        time.sleep(1)
    print("\r" + " " * (len(message) + 3) + "\r", end="", flush=True)


def _result_outputs(result: Optional[dict], *keys: str) -> list[str]:
    if not isinstance(result, dict):
        return []
    output_keys = keys or ("json_path", "txt_path", "language_txt_paths")
    outputs = []
    for key in output_keys:
        value = result.get(key)
        if isinstance(value, (list, tuple)):
            outputs.extend([str(item) for item in value if item])
        elif value:
            outputs.append(str(value))
    return outputs


def _store_pipeline_step_result(rapor: dict, step_key: str, status: str, detail: str = "", outputs: Optional[list] = None, duration: str = "") -> dict:
    rapor["steps"][_pipeline_report_key(step_key)] = {"status": status, "detail": detail, "duration": duration, "outputs": outputs or []}
    return rapor["steps"][_pipeline_report_key(step_key)]


def _skip_pipeline_step(rapor: dict, step_key: str, detail: str, outputs: Optional[list] = None) -> None:
    _skip_step(rapor, _pipeline_report_key(step_key), _pipeline_label(step_key), detail, outputs)

def run_selected_automation(
    video_yolu: Path,
    llm_ana: Optional[CentralLLM],
    llm_cila: Optional[CentralLLM],
    selected_steps: set[str],
    burn_reel_subtitles: bool = False,
    requested_steps: Optional[set] = None,
    planned_steps: Optional[set] = None,
    auto_added_dependencies: Optional[list[str]] = None,
    previous_steps: Optional[dict] = None,
    step_delay_seconds: int = 5,
    step_llm_overrides: Optional[dict[str, CentralLLM]] = None,
) -> dict:
    mod_altyazi = _load_runtime_module("moduller.subtitle_generator")
    mod_gramer = _load_runtime_module("moduller.subtitle_grammar_editor")
    mod_ceviri = _load_runtime_module("moduller.subtitle_translator")
    mod_broll = _load_runtime_module("moduller.broll_prompt_generator")
    mod_carousel = _load_runtime_module("moduller.instagram_carousel_generator")
    import moduller.hook_rewriter
    mod_ig_metadata = _load_runtime_module("moduller.instagram_engagement_planner")
    import moduller.metadata_olusturucu
    mod_muzik = _load_runtime_module("moduller.music_prompt_generator")
    mod_reel = _load_runtime_module("moduller.instagram_reels_generator")
    mod_story = _load_runtime_module("moduller.instagram_story_generator")
    mod_thumbnail = _load_runtime_module("moduller.thumbnail_prompt_generator")
    import moduller.trim_suggester
    mod_video_analiz = _load_runtime_module("moduller.video_analysis_generator")
    mod_youtube_metadata = _load_runtime_module("moduller.youtube_description_generator")
    mod_video_critic = _load_runtime_module("moduller.video_critic")

    _ = burn_reel_subtitles
    requested_steps = requested_steps or set(selected_steps)
    planned_steps = planned_steps or set(selected_steps)
    auto_added_dependencies = auto_added_dependencies or []
    step_llm_overrides = step_llm_overrides or {}

    pipeline_started_at = time.perf_counter()
    rapor = {
        "video": video_yolu.name,
        "models": {
            "main_model": llm_ana.model_name if llm_ana else "",
            "grammar_model": llm_ana.model_name if _step_selected(selected_steps, "grammar") and llm_ana else "",
            "smart_model": llm_cila.model_name if llm_cila else "",
            "trim_model": (
                step_llm_overrides["critic_trim"].model_name
                if _step_selected(selected_steps, "critic") and step_llm_overrides.get("critic_trim")
                else ""
            ),
            "step_model_overrides": {
                _pipeline_number(step_key): llm.model_name
                for step_key, llm in step_llm_overrides.items()
                if step_key in PIPELINE_STEP_BY_KEY
            },
        },
        "critic_issues": [],
        "routing_decisions": _legacy_routing(),
        "execution_context": {
            "requested_steps": _ordered_steps(requested_steps),
            "planned_steps": _ordered_steps(planned_steps),
            "auto_added_dependencies": auto_added_dependencies,
            "checkpoint_path": str(_checkpoint_path(video_yolu.stem)),
        },
        "steps": dict(previous_steps or {}),
    }

    previous_subtitle_step = rapor["steps"].get(_pipeline_report_key("subtitle"))
    previous_grammar_step = rapor["steps"].get(_pipeline_report_key("grammar"))
    srt_standart_yolu: Optional[Path] = _restore_existing_output_from_step(
        previous_subtitle_step,
        preferred_names=("subtitle_raw_tr.srt",),
        suffix=".srt",
    )
    srt_grammar_fixed_yolu: Optional[Path] = _restore_existing_output_from_step(
        previous_grammar_step,
        preferred_names=("subtitle_tr.srt",),
        suffix=".srt",
    )
    critic_data: Optional[dict] = None
    trim_result: Optional[dict] = None
    broll_result: Optional[dict] = None
    reels_result: Optional[dict] = None
    metadata_data: Optional[dict] = None

    global _PIPELINE_EVENT_HOOK
    started_step_count = 0
    step_started_at: Dict[str, float] = {}

    def checkpoint_event(event: str, adim_no: str, baslik: str, durum: str, detay: str, outputs: list) -> None:
        run_state = "failed" if event == "finish" and durum.startswith("❌") else "running"
        _save_checkpoint(video_yolu.stem, rapor, requested_steps, planned_steps, auto_added_dependencies, run_state, adim_no, baslik, durum or detay)

    def finalize(run_state: str, detail: str = "") -> dict:
        global _PIPELINE_EVENT_HOOK
        if detail:
            logger.info(detail)
        toplam_sure = format_elapsed(time.perf_counter() - pipeline_started_at)
        rapor["execution_context"]["total_duration"] = toplam_sure
        logger.info(f"⏱️ Modul 10 toplam sure: {toplam_sure}")
        _PIPELINE_EVENT_HOOK = None
        _save_master_report(video_yolu.stem, rapor)
        _save_checkpoint(video_yolu.stem, rapor, requested_steps, planned_steps, auto_added_dependencies, run_state, current_status=detail)
        return rapor

    def current_routing() -> dict:
        return rapor["routing_decisions"] if rapor["routing_decisions"] else _legacy_routing()

    def attach_step_duration(step_key: str, detail: str = "") -> tuple[str, str]:
        started_at = step_started_at.pop(step_key, None)
        if started_at is None:
            return detail, ""
        elapsed = format_elapsed(time.perf_counter() - started_at)
        logger.info(f"⏱️ Adim {_pipeline_number(step_key)} sure: {elapsed}")
        return detail, elapsed

    def complete(step_key: str, status: str, detail: str = "", outputs: Optional[list] = None) -> None:
        detail, duration = attach_step_duration(step_key, detail)
        _store_pipeline_step_result(rapor, step_key, status, detail, outputs, duration)
        final_detail = f"{detail} | Sure: {duration}" if duration and detail else (f"Sure: {duration}" if duration else detail)
        _finish_pipeline_step(step_key, status, final_detail, outputs)

    def complete_silent(step_key: str, status: str, detail: str = "", outputs: Optional[list] = None) -> None:
        detail, duration = attach_step_duration(step_key, detail)
        _store_pipeline_step_result(rapor, step_key, status, detail, outputs, duration)
        logger.info(f"Adim {_pipeline_number(step_key)} sessiz tamamlandi: {_pipeline_label(step_key)} -> {status}" + (f" | Sure: {duration}" if duration else ""))

    def start(step_key: str, detail: str = "") -> None:
        nonlocal started_step_count
        if started_step_count > 0 and step_delay_seconds > 0:
            _countdown_between_steps(step_delay_seconds, step_key)
        started_step_count += 1
        step_started_at[step_key] = time.perf_counter()
        _start_pipeline_step(step_key, detail)

    def refresh_metadata() -> Optional[dict]:
        nonlocal metadata_data
        if not srt_grammar_fixed_yolu:
            return metadata_data
        metadata_llm = step_llm_overrides.get("description") or llm_cila or llm_ana
        model_name = metadata_llm.model_name if metadata_llm else ""
        moduller.metadata_olusturucu.update_combined_metadata(srt_grammar_fixed_yolu, model_name)
        loaded = load_related_json(srt_grammar_fixed_yolu, "_metadata.json")
        metadata_data = loaded if isinstance(loaded, dict) else metadata_data
        return metadata_data

    _PIPELINE_EVENT_HOOK = checkpoint_event
    _save_checkpoint(video_yolu.stem, rapor, requested_steps, planned_steps, auto_added_dependencies, "running", current_status="Pipeline baslatildi.")

    subtitle_auto_reused = False
    grammar_auto_reused = False
    translation_auto_reused = False
    metadata_data = None
    critic_data = None
    trim_result = None
    broll_result = None

    if _step_selected(selected_steps, "subtitle"):
        auto_added_subtitle = _is_auto_added_step("subtitle", requested_steps, planned_steps)
        secilen_srt = _selected_srt_input(video_yolu)
        mevcut_standart_srt = _find_existing_srt(video_yolu, grammar_fixed=False)
        if auto_added_subtitle and secilen_srt:
            srt_standart_yolu = secilen_srt
            complete_silent(
                "subtitle",
                "✅ Secilen SRT kullanildi",
                "Bagimlilik olarak eklendigi icin secilen ana SRT dosyasi standart altyazi girdisi olarak kullanildi.",
                [str(srt_standart_yolu)],
            )
            subtitle_auto_reused = True
        elif auto_added_subtitle and mevcut_standart_srt:
            srt_standart_yolu = mevcut_standart_srt
            outputs = [str(srt_standart_yolu)]
            shorts_srt = subtitle_intermediate_output_path("subtitle_raw_shorts.srt")
            whisper_en_srt = subtitle_intermediate_output_path("subtitle_raw_en.srt")
            if _is_output_fresh(shorts_srt, srt_standart_yolu):
                outputs.append(str(shorts_srt))
            if _is_output_fresh(whisper_en_srt, srt_standart_yolu):
                outputs.append(str(whisper_en_srt))
            complete_silent(
                "subtitle",
                "✅ Mevcut cikti kullanildi",
                "Bagimlilik olarak eklendigi icin mevcut standart SRT yeniden kullanildi.",
                outputs,
            )
            subtitle_auto_reused = True
        else:
            start("subtitle", "🎬 Standart altyazi dosyasi hazirlaniyor.")
            try:
                srt_standart_yolu = mod_altyazi.run_automatic(video_yolu)
                outputs = [str(srt_standart_yolu)]
                shorts_srt = subtitle_intermediate_output_path("subtitle_raw_shorts.srt")
                whisper_en_srt = subtitle_intermediate_output_path("subtitle_raw_en.srt")
                if _is_output_fresh(shorts_srt, srt_standart_yolu):
                    outputs.append(str(shorts_srt))
                if _is_output_fresh(whisper_en_srt, srt_standart_yolu):
                    outputs.append(str(whisper_en_srt))
                complete("subtitle", "✅ Basarili", "Altyazi dosyalari uretildi.", outputs)
            except Exception as exc:
                complete("subtitle", f"❌ Hata: {exc}")
                return finalize("failed", "Altyazi adimi basarisiz oldu.")
    else:
        srt_standart_yolu = _selected_srt_input(video_yolu) or _find_existing_srt(video_yolu, grammar_fixed=False)
        _skip_pipeline_step(rapor, "subtitle", "Kullanici secimine gore atlandi. Mevcut standart SRT aranacak.", [str(srt_standart_yolu)] if srt_standart_yolu else [])

    if _step_selected(selected_steps, "grammar"):
        secilen_srt = _selected_srt_input(video_yolu)
        existing_grammar_srt = _find_existing_srt(video_yolu, grammar_fixed=True)
        auto_added_grammar = _is_auto_added_step("grammar", requested_steps, planned_steps)
        if auto_added_grammar and secilen_srt and _selected_srt_can_skip_grammar(secilen_srt):
            srt_grammar_fixed_yolu = secilen_srt
            complete_silent(
                "grammar",
                "✅ Secilen SRT grammar-fixed kabul edildi",
                "Bagimlilik olarak eklenen gramer adimi, secilen SRT ham görünmedigi icin yeniden calistirilmadi.",
                [str(srt_grammar_fixed_yolu)],
            )
            grammar_auto_reused = True
        elif auto_added_grammar and existing_grammar_srt and _is_output_fresh(existing_grammar_srt, srt_standart_yolu):
            srt_grammar_fixed_yolu = existing_grammar_srt
            complete_silent(
                "grammar",
                "✅ Mevcut cikti kullanildi",
                "Bagimlilik olarak eklendigi icin mevcut grammar-fixed SRT yeniden kullanildi.",
                [str(srt_grammar_fixed_yolu)],
            )
            grammar_auto_reused = True
        elif not srt_standart_yolu or not llm_ana:
            start("grammar", "🧹 Kullanilacak SRT dosyasi son kez temizleniyor.")
            missing = []
            if not srt_standart_yolu:
                missing.append("Standart SRT")
            if not llm_ana:
                missing.append("ANA LLM")
            missing_text = " ve ".join(missing) if missing else "gerekli girdiler"
            complete("grammar", f"❌ Hata: {missing_text} yok")
            return finalize("failed", f"Gramer adimi icin gerekli girdi eksik: {missing_text}.")
        else:
            start("grammar", "🧹 Kullanilacak SRT dosyasi son kez temizleniyor.")
            try:
                srt_grammar_fixed_yolu = mod_gramer.run_automatic(srt_standart_yolu, llm_ana)
                grammar_outputs = [str(srt_grammar_fixed_yolu)]
                for extra_path in (
                    subtitle_output_path("subtitle_en.srt"),
                    subtitle_output_path("subtitle_shorts.srt"),
                    subtitle_intermediate_output_path("Gramer_Duzenleyici_Raporu.txt"),
                    subtitle_intermediate_output_path("grammar_video_glossary.json"),
                    subtitle_intermediate_output_path("subtitle_raw_tr_glossary_fixed.srt"),
                    subtitle_intermediate_output_path("subtitle_raw_shorts_glossary_fixed.srt"),
                    subtitle_intermediate_output_path("grammar_llm_debug.txt"),
                ):
                    if extra_path.exists():
                        grammar_outputs.append(str(extra_path))
                complete(
                    "grammar",
                    "✅ Basarili",
                    "Gramer duzeltmesi tamamlandi.",
                    grammar_outputs,
                )
            except Exception as exc:
                srt_grammar_fixed_yolu = srt_standart_yolu
                complete("grammar", f"⚠️ Hata ama devam edildi: {exc}")
    else:
        secilen_srt = _selected_srt_input(video_yolu)
        srt_grammar_fixed_yolu = secilen_srt or _find_existing_srt(video_yolu, grammar_fixed=True)
        if not srt_grammar_fixed_yolu and secilen_srt and secilen_srt.name == "subtitle_tr.srt":
            srt_grammar_fixed_yolu = secilen_srt
        _skip_pipeline_step(rapor, "grammar", "Kullanici secimine gore atlandi. Mevcut grammar-fixed SRT varsa kullanilacak.", [str(srt_grammar_fixed_yolu)] if srt_grammar_fixed_yolu else [])

    if not srt_grammar_fixed_yolu and (set(selected_steps) - {"subtitle"}):
        rapor["steps"]["pipeline_stop"] = {"status": "❌ Devam edilemedi", "detail": "Asagidaki moduller icin gerekli SRT bulunamadi. Once altyazi veya gramer adimlarini run.", "outputs": []}
        return finalize("failed", "Pipeline gerekli SRT bulunamadigi icin durduruldu.")

    if _step_selected(selected_steps, "translation"):
        existing_translations = _find_existing_translation_outputs(srt_grammar_fixed_yolu)
        auto_added_translation = _is_auto_added_step("translation", requested_steps, planned_steps)
        if auto_added_translation and len(existing_translations) >= len(_translation_enabled_output_paths()):
            complete_silent(
                "translation",
                "✅ Mevcut cikti kullanildi",
                "Bagimlilik olarak eklendigi icin mevcut ceviri dosyalari yeniden kullanildi.",
                existing_translations,
            )
            translation_auto_reused = True
        elif not srt_grammar_fixed_yolu:
            start("translation", "🌍 Standart altyazi aktif hedef dillere cevriliyor.")
            complete("translation", "❌ Hata: Gerekli SRT yok")
        else:
            start("translation", "🌍 Standart altyazi aktif hedef dillere cevriliyor.")
            try:
                mod_ceviri.run_automatic(srt_grammar_fixed_yolu)
                outputs = [str(path) for path in _translation_enabled_output_paths() if path.exists()]
                outputs.append(str(subtitle_intermediate_output_path("translation_llm_debug.txt")))
                complete(
                    "translation",
                    "✅ Basarili",
                    "Ceviriler olusturuldu.",
                    outputs,
                )
            except Exception as exc:
                complete("translation", f"❌ Hata: {exc}")
    else:
        _skip_pipeline_step(rapor, "translation", "Kullanici secimine gore atlandi.")

    if _step_selected(selected_steps, "description"):
        start("description", "📝 Uc dilde description, baslik ve hashtag paketleri olusturuluyor.")
        description_llm = step_llm_overrides.get("description") or llm_cila
        if not srt_grammar_fixed_yolu or not description_llm:
            complete("description", "❌ Hata: Gerekli SRT veya YARATICI YAPAY ZEKA yok")
        else:
            description_result = None
            description_error = None
            try:
                description_result = mod_youtube_metadata.run_automatic(
                    srt_grammar_fixed_yolu,
                    description_llm,
                )
            except Exception as exc:
                description_error = exc

            outputs = _result_outputs(description_result)
            if description_result:
                detail = "Video metadata paketi uretildi: description."
                if description_error:
                    detail += f" Eksik kisimlar: description hata: {description_error}"
                complete("description", "✅ Basarili", detail, outputs)
                refresh_metadata()
            else:
                complete(
                    "description",
                    "⚠️ Sonuc yok",
                    "Video metadata paketi uretilemedi."
                    + (f" Ayrintilar: description hata: {description_error}" if description_error else ""),
                    outputs,
                )
    else:
        refresh_metadata()
        _skip_pipeline_step(rapor, "description", "Kullanici secimine gore atlandi.")

    if _step_selected(selected_steps, "critic"):
        start("critic", "🎯 Videonun acilisi, akisi ve retention riski analiz ediliyor.")
        critic_llm = step_llm_overrides.get("critic") or llm_cila
        if not srt_grammar_fixed_yolu or not critic_llm or not llm_ana:
            complete("critic", "❌ Hata: Gerekli SRT veya ANA+YARATICI YAPAY ZEKA yok")
        else:
            analysis_bundle = mod_video_analiz.run_automatic(
                srt_grammar_fixed_yolu,
                llm_ana,
                critic_llm,
                trim_llm=step_llm_overrides.get("critic_trim"),
                feedback_data=None,
            )
            if analysis_bundle:
                critic_result = analysis_bundle.get("critic_result")
                trim_result = analysis_bundle.get("trim_result")
                critic_data = analysis_bundle.get("critic_data")
                rapor["routing_decisions"] = analysis_bundle.get("routing_decisions") or _legacy_routing()
                rapor["critic_issues"] = analysis_bundle.get("critic_issues", [])
                complete(
                    "critic",
                    "✅ Basarili",
                    analysis_bundle.get("detail", "Video analiz paketi uretildi."),
                    analysis_bundle.get("outputs", []),
                )
            else:
                rapor["routing_decisions"] = _legacy_routing()
                complete(
                    "critic",
                    "⚠️ Sonuc yok",
                    "Video analiz paketi uretilemedi.",
                    [],
                )
    else:
        critic_data = load_related_json(srt_grammar_fixed_yolu, "_video_critic.json") if srt_grammar_fixed_yolu else None
        rapor["routing_decisions"] = critic_data.get("routing_decisions", _legacy_routing()) if isinstance(critic_data, dict) else _legacy_routing()
        rapor["critic_issues"] = critic_data.get("biggest_issues", []) if isinstance(critic_data, dict) else []
        mevcut_trim = load_related_json(srt_grammar_fixed_yolu, "_trim_suggestions.json") if srt_grammar_fixed_yolu else None
        trim_result = {"data": mevcut_trim} if isinstance(mevcut_trim, dict) else None
        _skip_pipeline_step(rapor, "critic", "Kullanici secimine gore atlandi. Mevcut critic ve trim raporlari varsa kullanilacak.")

    if _step_selected(selected_steps, "broll"):
        start("broll", "🎞️ Retention'i destekleyecek ara goruntu fikirleri cikartiliyor.")
        broll_llm = step_llm_overrides.get("broll") or llm_cila
        if not srt_grammar_fixed_yolu or not broll_llm:
            complete("broll", "❌ Hata: Gerekli SRT veya YARATICI YAPAY ZEKA yok")
        else:
            try:
                broll_result = mod_broll.run_automatic(
                    srt_grammar_fixed_yolu,
                    broll_llm,
                    feedback_data=None,
                    critic_data=critic_data,
                    trim_data=trim_result["data"] if trim_result else None,
                    respect_routing=True,
                )
                outputs = _result_outputs(broll_result)
                fallback_txt = txt_output_path("broll")
                if not outputs and fallback_txt.exists():
                    outputs = [str(fallback_txt)]
                detail = (broll_result or {}).get("detail") or _build_forced_detail(current_routing(), "broll_generator", "B-Roll fikirleri uretildi.")
                complete("broll", "✅ Basarili" if broll_result else "⚠️ Sonuc yok", detail, outputs)
            except Exception as exc:
                complete("broll", f"⚠️ Hata ama devam edildi: {exc}")
    else:
        mevcut_broll = load_related_json(srt_grammar_fixed_yolu, "_B_roll_fikirleri.json") if srt_grammar_fixed_yolu else None
        broll_result = {"data": mevcut_broll} if isinstance(mevcut_broll, list) else None
        _skip_pipeline_step(rapor, "broll", "Kullanici secimine gore atlandi.")

    if _step_selected(selected_steps, "thumbnail_main"):
        start("thumbnail_main", "🖼️ Ana video icin thumbnail promptlari uretiliyor.")
        thumbnail_main_llm = step_llm_overrides.get("thumbnail_main") or llm_cila
        if not srt_grammar_fixed_yolu or not thumbnail_main_llm:
            complete("thumbnail_main", "❌ Hata: Gerekli SRT veya YARATICI YAPAY ZEKA yok")
        else:
            try:
                metadata_data = refresh_metadata()
                result = mod_thumbnail.create_main_video_thumbnails(
                    srt_grammar_fixed_yolu,
                    thumbnail_main_llm,
                    metadata_data=metadata_data,
                    critic_data=critic_data,
                    broll_data=broll_result["data"] if broll_result else None,
                )
                complete("thumbnail_main", "✅ Basarili" if result else "⚠️ Sonuc yok", "Ana video thumbnail promptlari olusturuldu." if result else "Thumbnail promptlari uretilemedi.", _result_outputs(result))
            except Exception as exc:
                complete("thumbnail_main", f"⚠️ Hata ama devam edildi: {exc}")
    else:
        _skip_pipeline_step(rapor, "thumbnail_main", "Kullanici secimine gore atlandi.")

    if _step_selected(selected_steps, "carousel"):
        start("carousel", "📚 YouTube icerigi Instagram carousel formatina donusturuluyor.")
        carousel_llm = step_llm_overrides.get("carousel") or llm_cila
        if not srt_grammar_fixed_yolu or not carousel_llm or not llm_ana:
            complete("carousel", "❌ Hata: Gerekli SRT veya ANA+YARATICI YAPAY ZEKA yok")
        else:
            try:
                result = mod_carousel.run_automatic(
                    srt_grammar_fixed_yolu,
                    carousel_llm,
                    critic_data=critic_data,
                    trim_data=trim_result["data"] if trim_result else None,
                    metadata_data=refresh_metadata(),
                    broll_data=broll_result["data"] if broll_result else None,
                    draft_llm=llm_ana,
                )
                complete("carousel", "✅ Basarili" if result else "⚠️ Sonuc yok", "Instagram carousel fikirleri olusturuldu." if result else "Carousel fikirleri uretilemedi.", _result_outputs(result))
            except Exception as exc:
                complete("carousel", f"⚠️ Hata ama devam edildi: {exc}")
    else:
        _skip_pipeline_step(rapor, "carousel", "Kullanici secimine gore atlandi.")

    if _step_selected(selected_steps, "reels"):
        start("reels", "📱 Reels videolari icin anlamli ve viral olabilecek kesitler belirleniyor.")
        reels_llm = step_llm_overrides.get("reels") or llm_cila
        if not srt_grammar_fixed_yolu or not reels_llm:
            complete("reels", "❌ Hata: Gerekli SRT veya YARATICI YAPAY ZEKA yok")
        else:
            try:
                reels_result = mod_reel.run_automatic(
                    srt_grammar_fixed_yolu,
                    reels_llm,
                    reel_sayisi=8,
                    metadata_data=refresh_metadata(),
                    critic_data=critic_data,
                    trim_data=trim_result["data"] if trim_result else None,
                    broll_data=broll_result["data"] if broll_result else None,
                    draft_llm=llm_ana,
                    ranker_llm=reels_llm,
                    respect_routing=True,
                )
                detail = (reels_result or {}).get("detail") or _build_forced_detail(current_routing(), "reels_shorts", "Instagram Reels fikirleri olusturuldu.")
                complete("reels", "✅ Basarili" if reels_result else "⚠️ Sonuc yok", detail, _result_outputs(reels_result))
            except Exception as exc:
                complete("reels", f"⚠️ Hata ama devam edildi: {exc}")
    else:
        mevcut_reels = mod_reel.load_latest_reels_data()
        reels_result = {"data": mevcut_reels.get("ideas", [])} if isinstance(mevcut_reels, dict) else None
        _skip_pipeline_step(rapor, "reels", "Kullanici secimine gore atlandi.")

    if _step_selected(selected_steps, "story"):
        start("story", "📲 Story akisi konseptleri olusturuluyor.")
        story_llm = step_llm_overrides.get("story") or llm_cila
        if not srt_grammar_fixed_yolu or not story_llm or not llm_ana:
            complete("story", "❌ Hata: Gerekli SRT veya ANA+YARATICI YAPAY ZEKA yok")
        else:
            try:
                metadata_data = refresh_metadata()
                result = mod_story.run_automatic(
                    srt_grammar_fixed_yolu,
                    story_llm,
                    metadata_data=metadata_data,
                    broll_data=broll_result["data"] if broll_result else None,
                    critic_data=critic_data,
                    draft_llm=llm_ana,
                )
                complete("story", "✅ Basarili" if result else "⚠️ Sonuc yok", "Instagram story fikirleri olusturuldu." if result else "Story fikirleri uretilemedi.", _result_outputs(result))
            except Exception as exc:
                complete("story", f"⚠️ Hata ama devam edildi: {exc}")
    else:
        _skip_pipeline_step(rapor, "story", "Kullanici secimine gore atlandi.")

    if _step_selected(selected_steps, "ig_metadata"):
        start("ig_metadata", "🗓️ Instagram icin haftalik paylasim takvimi olusturuluyor.")
        if not srt_grammar_fixed_yolu:
            complete("ig_metadata", "❌ Hata: Gerekli SRT yok")
        else:
            try:
                ig_metadata_llm = step_llm_overrides.get("ig_metadata") or llm_cila
                result = mod_ig_metadata.run_automatic(
                    srt_grammar_fixed_yolu,
                    ig_metadata_llm,
                )
                complete("ig_metadata", "✅ Basarili" if result else "⚠️ Sonuc yok", "Instagram paylasim takvimi olusturuldu." if result else "Instagram paylasim takvimi uretilemedi.", _result_outputs(result))
            except Exception as exc:
                complete("ig_metadata", f"⚠️ Hata ama devam edildi: {exc}")
    else:
        _skip_pipeline_step(rapor, "ig_metadata", "Kullanici secimine gore atlandi.")

    if _step_selected(selected_steps, "music"):
        start("music", "🎵 Videonun akisina uygun background muzik planlari uretiliyor.")
        music_llm = step_llm_overrides.get("music") or llm_ana
        if not srt_grammar_fixed_yolu or not music_llm:
            complete("music", "❌ Hata: Gerekli SRT veya ANA LLM yok")
        else:
            try:
                metadata_data = refresh_metadata()
                result = mod_muzik.run_automatic(
                    srt_grammar_fixed_yolu,
                    music_llm,
                    metadata_data=metadata_data,
                    critic_data=critic_data,
                    broll_data=broll_result["data"] if broll_result else None,
                )
                complete("music", "✅ Basarili" if result else "⚠️ Sonuc yok", "Muzik prompt planlari olusturuldu." if result else "Muzik promptlari uretilemedi.", _result_outputs(result))
            except Exception as exc:
                complete("music", f"⚠️ Hata ama devam edildi: {exc}")
    else:
        _skip_pipeline_step(rapor, "music", "Kullanici secimine gore atlandi.")

    return finalize("completed", "Coklu secim pipeline tamamlandi.")


def run_automatic(video_yolu: Path, llm_ana: CentralLLM, llm_cila: CentralLLM) -> dict:
    tum_adimlar = _all_pipeline_steps()
    _clear_outputs_for_full_pipeline()
    return run_selected_automation(video_yolu, llm_ana, llm_cila, tum_adimlar, requested_steps=tum_adimlar, planned_steps=tum_adimlar, auto_added_dependencies=[], previous_steps=None, step_delay_seconds=5)


def run(preselected_selection: Optional[str] = None):
    global _PIPELINE_EVENT_HOOK
    selected_steps, invalid_numbers = _get_pipeline_selection(preselected_selection)
    if invalid_numbers:
        return
    full_pipeline_requested = selected_steps is None
    requested_steps = _all_pipeline_steps() if selected_steps is None else set(selected_steps)
    try:
        _validate_pipeline_dag()
        planned_steps, auto_added_dependencies = _resolve_dependencies(selected_steps)
    except DependencyError as exc:
        logger.error(f"❌ Bagimlilik yapisi gecersiz: {exc}")
        return

    if full_pipeline_requested:
        _print_terminal_block("OUTPUT TEMIZLIGI", "🧹", "Tum moduller calisacagi icin workspace/00_Outputs altindaki grup klasorleri tamamen sifirlanacak. Sadece workspace/00_Inputs altindaki kaynak video korunur.")
        _clear_outputs_for_full_pipeline()

    secilen_video = _resolve_primary_input(requested_steps, planned_steps)
    if not secilen_video:
        return

    checkpoint = _load_checkpoint(secilen_video.stem)
    resumable_states = {"running", "failed"}
    resume_steps = set()
    previous_steps = {}
    if checkpoint and checkpoint.get("run_state") in resumable_states:
        checkpoint_completed = set(checkpoint.get("completed_steps", [])) & planned_steps
        if checkpoint_completed:
            logger.info(
                "Checkpoint bulundu ancak secilen moduller atlanmayacak; yeniden calistirilacak: "
                + ", ".join(_ordered_steps(checkpoint_completed))
            )
            detay_satirlari = []
            for step_key in _ordered_steps(checkpoint_completed):
                entry = MODULE_BY_KEY.get(step_key)
                if entry:
                    detay_satirlari.append(f"{entry.number}: {entry.title}")
            if detay_satirlari:
                _print_terminal_block(
                    "CHECKPOINT BULUNDU",
                    "♻️",
                    "Asagidaki adimlar daha once tamamlanmis olsa da bu calistirmada yeniden islenecek:\n"
                    + "\n".join(detay_satirlari),
                )

    pending_steps = planned_steps - resume_steps
    reusable_steps = set()
    if _is_auto_added_step("subtitle", requested_steps, planned_steps):
        secilen_srt = _selected_srt_input(secilen_video)
        mevcut_srt = _find_existing_srt(secilen_video, grammar_fixed=False)
        mevcut_grammar = _find_existing_srt(secilen_video, grammar_fixed=True)
        if secilen_srt:
            previous_steps[_pipeline_report_key("subtitle")] = _build_reused_step_entry(
                "✅ Secilen SRT kullanildi",
                "Bagimlilik olarak eklendigi icin secilen ana SRT dosyasi standart altyazi girdisi olarak kullanildi.",
                [str(secilen_srt)],
            )
            reusable_steps.add("subtitle")
        elif mevcut_srt or mevcut_grammar:
            outputs = [str(mevcut_srt)] if mevcut_srt else [str(mevcut_grammar)]
            shorts_srt = subtitle_intermediate_output_path("subtitle_raw_shorts.srt")
            whisper_en_srt = subtitle_intermediate_output_path("subtitle_raw_en.srt")
            if _is_output_fresh(shorts_srt, mevcut_srt or mevcut_grammar):
                outputs.append(str(shorts_srt))
            if _is_output_fresh(whisper_en_srt, mevcut_srt or mevcut_grammar):
                outputs.append(str(whisper_en_srt))
            previous_steps[_pipeline_report_key("subtitle")] = _build_reused_step_entry(
                "✅ Mevcut cikti kullanildi",
                "Bagimlilik olarak eklendigi icin mevcut altyazi ciktisi yeniden kullanildi.",
                outputs,
            )
            reusable_steps.add("subtitle")

    if _is_auto_added_step("grammar", requested_steps, planned_steps):
        secilen_srt = _selected_srt_input(secilen_video)
        mevcut_grammar = _find_existing_srt(secilen_video, grammar_fixed=True)
        if secilen_srt and _selected_srt_can_skip_grammar(secilen_srt):
            previous_steps[_pipeline_report_key("grammar")] = _build_reused_step_entry(
                "✅ Secilen SRT grammar-fixed kabul edildi",
                "Bagimlilik olarak eklenen gramer adimi, secilen SRT ham görünmedigi icin yeniden calistirilmadi.",
                [str(secilen_srt)],
            )
            reusable_steps.add("grammar")
        elif mevcut_grammar:
            previous_steps[_pipeline_report_key("grammar")] = _build_reused_step_entry(
                "✅ Mevcut cikti kullanildi",
                "Bagimlilik olarak eklendigi icin mevcut grammar-fixed SRT yeniden kullanildi.",
                [str(mevcut_grammar)],
            )
            reusable_steps.add("grammar")

    if _is_auto_added_step("translation", requested_steps, planned_steps):
        mevcut_ceviriler = _find_existing_translation_outputs(_find_existing_srt(secilen_video, grammar_fixed=True))
        if len(mevcut_ceviriler) >= 2:
            previous_steps[_pipeline_report_key("translation")] = _build_reused_step_entry(
                "✅ Mevcut cikti kullanildi",
                "Bagimlilik olarak eklendigi icin mevcut ceviri dosyalari yeniden kullanildi.",
                mevcut_ceviriler,
            )
            reusable_steps.add("translation")

    if reusable_steps:
        logger.info(
            "Mevcut ciktidan sessizce yeniden kullanilan bagimliliklar: "
            + ", ".join(_ordered_steps(reusable_steps))
        )
        pending_steps -= reusable_steps

    if auto_added_dependencies:
        logger.info("Secilen moduller icin zorunlu bagimliliklar otomatik eklendi: " + ", ".join(auto_added_dependencies))
    if not pending_steps and previous_steps:
        logger.info("Bu secim icin calistirilacak yeni adim kalmadi. Checkpoint zaten ilerlemeyi kaydetmis.")
        return

    step_llm_overrides: dict[str, CentralLLM] = {}
    try:
        if full_pipeline_requested:
            llm_ana, llm_cila, step_llm_overrides = _setup_full_pipeline_llms(pending_steps)
        else:
            llm_ana, llm_cila, step_llm_overrides = _setup_selected_llms(
                pending_steps,
                summary_steps=requested_steps,
            )
    except Exception as e:
        logger.error(f"❌ LLM baslatilamadi: {e}")
        return

    try:
        rapor = run_selected_automation(secilen_video, llm_ana, llm_cila, pending_steps, requested_steps=requested_steps, planned_steps=planned_steps, auto_added_dependencies=auto_added_dependencies, previous_steps=previous_steps, step_delay_seconds=5, step_llm_overrides=step_llm_overrides)
    except Exception as e:
        _PIPELINE_EVENT_HOOK = None
        logger.error(f"❌ Coklu secim pipeline beklenmeyen sekilde durdu: {e}")
        return

    _print_terminal_block("MASTER PIPELINE SONUC RAPORU", "🎉", "Tum adimlarin nihai durumu asagida listeleniyor.")
    for step in PIPELINE_STEPS:
        bilgi = rapor.get("steps", {}).get(step["report_key"])
        if not bilgi or str(bilgi.get("status", "")).startswith("⏭️"):
            continue
        satir = f"{step['number']}. {step['label']}: {bilgi.get('status', '')}"
        if bilgi.get("duration"):
            satir += f" | Sure: {bilgi.get('duration')}"
        print(satir)
    print(TERMINAL_AYIRICI)
    logger.info("Tum otomasyon sureci tamamlandi. Ciktilar workspace/00_Outputs altindaki grup klasorlerine yazildi.")
