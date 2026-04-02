from pathlib import Path
from typing import Optional

from moduller._module_alias import load_numbered_module
from moduller.hook_rewriter import (
    build_critic_summary as build_hook_critic_summary,
    run_automatic as run_hook_automatic,
)
from moduller.llm_manager import (
    CentralLLM,
    get_module_recommended_llm_config,
    print_module_llm_choice_summary,
    prompt_module_llm_plan,
    select_llm,
)
from moduller.logger import get_logger
from moduller.social_media_utils import select_primary_srt
from moduller.srt_utils import parse_srt_blocks, read_srt_file
from moduller.trim_suggester import (
    build_critic_summary as build_trim_critic_summary,
    run_automatic as run_trim_automatic,
)

_ANALYTICS_FEEDBACK_MODULE = load_numbered_module("402_analitik_geri_bildirim_dongusu.py")
build_feedback_summary = _ANALYTICS_FEEDBACK_MODULE.build_feedback_summary
load_latest_feedback_data = _ANALYTICS_FEEDBACK_MODULE.load_latest_feedback_data

_VIDEO_CRITIC_MODULE = load_numbered_module("202_video_critic.py")
run_critic_automatic = _VIDEO_CRITIC_MODULE.run_automatic

logger = get_logger("video_analysis")

CRITIC_TRANSCRIPT_MAX_CHARS = 24000
TRIM_TRANSCRIPT_MAX_CHARS = 22000
HOOK_INTRO_BLOCK_LIMIT = 12
HOOK_INTRO_MAX_CHARS = 5000


def _select_srt() -> Optional[Path]:
    return select_primary_srt(logger, "Video Analiz Ureticisi")


def _serialize_blocks(girdi_dosyasi: Path) -> list[str]:
    icerik = read_srt_file(girdi_dosyasi)
    bloklar = parse_srt_blocks(icerik)
    return [f"[{b.timing_line}] {b.text_content}" for b in bloklar if b.is_processable]


def _prepare_shared_context(girdi_dosyasi: Path, feedback_data: Optional[dict] = None) -> dict:
    satirlar = _serialize_blocks(girdi_dosyasi)
    serialized = "\n".join(satirlar)
    active_feedback = feedback_data or load_latest_feedback_data()
    return {
        "critic_transcript": serialized[:CRITIC_TRANSCRIPT_MAX_CHARS],
        "trim_transcript": serialized[:TRIM_TRANSCRIPT_MAX_CHARS],
        "intro_transcript": "\n".join(satirlar[:HOOK_INTRO_BLOCK_LIMIT])[:HOOK_INTRO_MAX_CHARS],
        "feedback_data": active_feedback,
        "feedback_summary": build_feedback_summary(active_feedback),
    }


def _result_outputs(result: Optional[dict]) -> list[str]:
    if not isinstance(result, dict):
        return []
    outputs = []
    for key in ("json_path", "txt_path"):
        value = result.get(key)
        if value:
            outputs.append(str(value))
    return outputs


def _component_label(name: str, result: Optional[dict]) -> Optional[str]:
    if not isinstance(result, dict):
        return None
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    if data.get("skipped_by_routing"):
        return f"{name}(skip)"
    return name


def _build_hook_appendix_text(hook_data: dict, hook_model_name: str) -> str:
    lines = [
        "ACILIS VE PAKETLEME EK NOTLARI",
        "-" * 50,
        f"Hook Modeli: {hook_model_name}",
        "",
    ]

    if hook_data.get("skipped_by_routing"):
        lines.extend(
            [
                "HOOK DURUMU",
                "-" * 50,
                hook_data.get("current_hook_problem", ""),
                "",
                "KISA NOTLAR",
                "-" * 50,
            ]
        )
        for item in hook_data.get("editor_notes", []):
            lines.append(f"- {item}")
        return "\n".join(lines).strip()

    lines.extend(
        [
            "MEVCUT ACILIS GOZLEMI",
            "-" * 50,
            hook_data.get("current_hook_problem", ""),
            "",
            "ONERILEN ANA HOOK",
            "-" * 50,
            hook_data.get("recommended_primary_hook", ""),
            "",
            "ALTERNATIF HOOK ACIKLARI",
            "-" * 50,
        ]
    )
    for idx, item in enumerate(hook_data.get("improved_hooks", [])[:3], 1):
        lines.append(f"Alternatif {idx}: {item.get('hook', '')}")
        lines.append(f"Neden ise yarar: {item.get('why_it_works', '')}")
        lines.append("")

    if hook_data.get("editor_notes"):
        lines.extend(
            [
                "PAKETLEME NOTLARI",
                "-" * 50,
            ]
        )
        for item in hook_data.get("editor_notes", [])[:3]:
            lines.append(f"- {item}")

    return "\n".join(lines).strip()


def _append_hook_to_critic_report(
    girdi_dosyasi: Path,
    critic_result: Optional[dict],
    hook_result: Optional[dict],
    hook_model_name: str,
) -> None:
    if not isinstance(critic_result, dict) or not isinstance(hook_result, dict):
        return

    critic_txt = critic_result.get("txt_path")
    hook_data = hook_result.get("data")
    if not critic_txt or not isinstance(hook_data, dict):
        return

    critic_txt_path = Path(critic_txt)
    if not critic_txt_path.exists():
        return

    critic_text = critic_txt_path.read_text(encoding="utf-8")
    hook_text = _build_hook_appendix_text(hook_data, hook_model_name).strip()
    if not hook_text:
        return

    merged_text = critic_text.rstrip() + "\n\n" + ("=" * 60) + "\n" + hook_text + "\n"
    critic_txt_path.write_text(merged_text, encoding="utf-8")


def run_automatic(
    girdi_dosyasi: Path,
    llm_ana: CentralLLM,
    llm_yaratici: CentralLLM,
    trim_llm: Optional[CentralLLM] = None,
    feedback_data: Optional[dict] = None,
) -> Optional[dict]:
    logger.info(f"🔄 OTOMASYON: {girdi_dosyasi.name} icin video analiz paketi uretiliyor...")

    shared = _prepare_shared_context(girdi_dosyasi, feedback_data=feedback_data)
    critic_result = None
    hook_result = None
    trim_result = None
    critic_data = None
    warnings = []

    logger.info("🎯 Video critic analizi baslatiliyor...")
    try:
        critic_result = run_critic_automatic(
            girdi_dosyasi,
            llm_yaratici,
            prepared_transcript=shared["critic_transcript"],
        )
        critic_data = critic_result.get("data") if critic_result else None
    except Exception as exc:
        warnings.append(f"critic hata: {exc}")
        logger.warning(f"Video critic adimi sorun verdi: {exc}")

    hook_critic_summary = build_hook_critic_summary(critic_data)
    trim_critic_summary = build_trim_critic_summary(critic_data)

    logger.info("🪝 Hook analizi baslatiliyor...")
    try:
        hook_result = run_hook_automatic(
            girdi_dosyasi,
            llm_yaratici,
            critic_data=critic_data,
            feedback_data=shared["feedback_data"],
            intro_transcript=shared["intro_transcript"],
            critic_summary=hook_critic_summary,
            feedback_summary=shared["feedback_summary"],
            respect_routing=True,
            save_txt=False,
        )
    except Exception as exc:
        warnings.append(f"hook hata: {exc}")
        logger.warning(f"Hook adimi sorun verdi: {exc}")

    try:
        _append_hook_to_critic_report(
            girdi_dosyasi,
            critic_result,
            hook_result,
            llm_yaratici.model_name,
        )
    except Exception as exc:
        warnings.append(f"hook rapor birlestirme hata: {exc}")
        logger.warning(f"Hook raporu critic raporuna eklenemedi: {exc}")

    logger.info("✂️ Trim analizi baslatiliyor...")
    try:
        active_trim_llm = trim_llm or llm_ana
        trim_result = run_trim_automatic(
            girdi_dosyasi,
            active_trim_llm,
            critic_data=critic_data,
            feedback_data=shared["feedback_data"],
            prepared_transcript=shared["trim_transcript"],
            critic_summary=trim_critic_summary,
            feedback_summary=shared["feedback_summary"],
            respect_routing=True,
        )
    except Exception as exc:
        warnings.append(f"trim hata: {exc}")
        logger.warning(f"Trim adimi sorun verdi: {exc}")

    if not critic_result and not hook_result and not trim_result:
        logger.error("❌ Video analiz uretimi basarisiz oldu.")
        return None

    outputs = _result_outputs(critic_result) + _result_outputs(hook_result) + _result_outputs(trim_result)
    detail_parts = []
    for name, result in (("critic", critic_result), ("hook", hook_result), ("trim", trim_result)):
        label = _component_label(name, result)
        if label:
            detail_parts.append(label)

    detail = "Video analiz paketi uretildi: " + ", ".join(detail_parts) + "."
    if warnings:
        detail += " Eksik kisimlar: " + " | ".join(warnings)

    logger.info("🎉 Video analiz uretimi tamamlandi.")
    return {
        "critic_result": critic_result,
        "hook_result": hook_result,
        "trim_result": trim_result,
        "critic_data": critic_data,
        "routing_decisions": critic_data.get("routing_decisions", {}) if isinstance(critic_data, dict) else {},
        "critic_issues": critic_data.get("biggest_issues", []) if isinstance(critic_data, dict) else [],
        "outputs": outputs,
        "warnings": warnings,
        "detail": detail,
    }


def run() -> None:
    print("\n" + "=" * 60)
    print("VIDEO ANALIZ URETICISI")
    print("=" * 60)

    girdi = _select_srt()
    if not girdi:
        return

    use_recommended = prompt_module_llm_plan("202", needs_main=True, needs_smart=True)
    if use_recommended:
        ana_saglayici, ana_model_adi = get_module_recommended_llm_config("202", "main")
        yaratici_saglayici, yaratici_model_adi = get_module_recommended_llm_config("202", "smart")
        print_module_llm_choice_summary(
            "202",
            {"main": (ana_saglayici, ana_model_adi), "smart": (yaratici_saglayici, yaratici_model_adi)},
        )
    else:
        ana_saglayici, ana_model_adi = select_llm("main")
        yaratici_saglayici, yaratici_model_adi = select_llm("smart")
    llm_ana = CentralLLM(provider=ana_saglayici, model_name=ana_model_adi)
    llm_yaratici = CentralLLM(provider=yaratici_saglayici, model_name=yaratici_model_adi)

    result = run_automatic(girdi, llm_ana, llm_yaratici, feedback_data=load_latest_feedback_data())
    if not result:
        return

    if not result.get("critic_result"):
        logger.warning("Video critic raporu uretilemedi.")
    if not result.get("hook_result"):
        logger.warning("Hook analizi uretilemedi.")
    if not result.get("trim_result"):
        logger.warning("Trim analizi uretilemedi.")

    logger.info("🎉 Video analiz uretimi tamamlandi.")
