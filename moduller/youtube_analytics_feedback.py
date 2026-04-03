import json
import os
import time
from pathlib import Path
from typing import Optional

from moduller.llm_manager import (
    CentralLLM,
    get_default_llm_config,
    get_module_recommended_llm_config,
    print_module_llm_choice_summary,
    prompt_module_llm_plan,
    select_llm,
)
from moduller.logger import get_logger
from moduller.output_paths import grouped_json_output_path, json_output_path, txt_output_path
from moduller.youtube_analytics_api import (
    fetch_authenticated_channel_profile,
    fetch_channel_analytics_snapshot,
    fetch_retention_points_via_api,
    oauth_setup_summary,
    youtube_analytics_available,
)
from moduller.youtube_analytics_llm import (
    build_channel_action_prompt,
    build_channel_analysis_prompt,
    build_channel_combined_prompt,
    build_video_action_prompt,
    build_video_analysis_prompt,
    build_video_combined_prompt,
    call_analytics_llm_json,
    channel_action_fallback,
    channel_analysis_fallback,
    video_action_fallback,
    video_analysis_fallback,
)
from moduller.youtube_analytics_reporting import (
    build_action_plan_txt,
    build_channel_txt_report,
    build_channel_external_prompt,
    build_feedback_summary,
    build_legacy_feedback_report_text,
    build_retention_analysis,
    build_video_external_prompt,
    build_video_txt_report,
    format_number,
    format_percent,
    summarize_channel_snapshot,
    summarize_selected_video,
    video_views,
)

logger = get_logger("feedback")

FEEDBACK_JSON_PATH = json_output_path("analytics_feedback")
FEEDBACK_TXT_PATH = txt_output_path("analytics_feedback")
CHANNEL_JSON_PATH = json_output_path("analytics_channel_report")
CHANNEL_TXT_PATH = txt_output_path("analytics_channel_analysis")
VIDEO_TXT_PATH = txt_output_path("analytics_video_analysis")
ACTION_PLAN_TXT_PATH = txt_output_path("analytics_action_plan")
CHANNEL_PROMPT_TXT_PATH = txt_output_path("analytics_channel_prompt")
VIDEO_PROMPT_TXT_PATH = txt_output_path("analytics_video_prompt")
SNAPSHOT_CACHE_PATH = grouped_json_output_path("research", "YouTube_Analytics_Snapshot_Cache.json")
RETENTION_CACHE_PATH = grouped_json_output_path("research", "YouTube_Analytics_Retention_Cache.json")


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(str(os.getenv(name, str(default))).strip()))
    except Exception:
        return max(0, default)


ANALYTICS_LLM_MODE = str(os.getenv("ANALYTICS_LLM_MODE", "fast") or "fast").strip().lower()
ANALYTICS_FAST_FALLBACK_TO_DEEP = str(
    os.getenv("ANALYTICS_FAST_FALLBACK_TO_DEEP", "true") or "true"
).strip().lower() not in {"0", "false", "no", "off"}
SNAPSHOT_CACHE_TTL_SECONDS = _env_int("ANALYTICS_SNAPSHOT_CACHE_TTL_SECONDS", 1800)
RETENTION_CACHE_TTL_SECONDS = _env_int("ANALYTICS_RETENTION_CACHE_TTL_SECONDS", 21600)


def load_latest_feedback_data() -> Optional[dict]:
    if not FEEDBACK_JSON_PATH.exists():
        return None
    try:
        return json.loads(FEEDBACK_JSON_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Analytics feedback raporu okunamadi: {exc}")
        return None


def _save_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception as exc:
        logger.warning(f"Analytics cache okunamadi ({path.name}): {exc}")
        return {}


def _save_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _now_ts() -> float:
    return float(time.time())


def _snapshot_cache_key(channel_days: int, recent_video_limit: int, per_video_days: int) -> str:
    return f"{channel_days}:{recent_video_limit}:{per_video_days}"


def _get_snapshot_with_cache(
    channel_days: int = 28,
    recent_video_limit: int = 10,
    per_video_days: int = 90,
) -> tuple[dict, dict]:
    cache_key = _snapshot_cache_key(channel_days, recent_video_limit, per_video_days)
    cache = _load_cache(SNAPSHOT_CACHE_PATH)
    entries = cache.get("entries", {}) if isinstance(cache.get("entries"), dict) else {}
    cached_entry = entries.get(cache_key) if isinstance(entries, dict) else None
    now_ts = _now_ts()
    if isinstance(cached_entry, dict):
        age_seconds = max(0.0, now_ts - float(cached_entry.get("saved_at_ts") or 0.0))
        if SNAPSHOT_CACHE_TTL_SECONDS <= 0 or age_seconds <= SNAPSHOT_CACHE_TTL_SECONDS:
            snapshot = cached_entry.get("snapshot")
            if isinstance(snapshot, dict):
                return snapshot, {
                    "used_cache": True,
                    "cache_key": cache_key,
                    "age_seconds": round(age_seconds, 2),
                    "ttl_seconds": SNAPSHOT_CACHE_TTL_SECONDS,
                }

    snapshot = fetch_channel_analytics_snapshot(
        channel_days=channel_days,
        recent_video_limit=recent_video_limit,
        per_video_days=per_video_days,
    )
    entries[cache_key] = {
        "saved_at_ts": now_ts,
        "snapshot": snapshot,
    }
    cache["entries"] = entries
    _save_cache(SNAPSHOT_CACHE_PATH, cache)
    return snapshot, {
        "used_cache": False,
        "cache_key": cache_key,
        "age_seconds": 0.0,
        "ttl_seconds": SNAPSHOT_CACHE_TTL_SECONDS,
    }


def _get_retention_with_cache(context: dict, since_days: int = 365) -> tuple[dict, dict]:
    video_id = str(context.get("video_id") or "").strip()
    if not video_id:
        return {}, {"used_cache": False, "video_id": "", "ttl_seconds": RETENTION_CACHE_TTL_SECONDS}

    cache = _load_cache(RETENTION_CACHE_PATH)
    entries = cache.get("entries", {}) if isinstance(cache.get("entries"), dict) else {}
    cached_entry = entries.get(video_id) if isinstance(entries, dict) else None
    now_ts = _now_ts()
    if isinstance(cached_entry, dict):
        age_seconds = max(0.0, now_ts - float(cached_entry.get("saved_at_ts") or 0.0))
        if RETENTION_CACHE_TTL_SECONDS <= 0 or age_seconds <= RETENTION_CACHE_TTL_SECONDS:
            retention_analysis = cached_entry.get("retention_analysis")
            if isinstance(retention_analysis, dict):
                return retention_analysis, {
                    "used_cache": True,
                    "video_id": video_id,
                    "age_seconds": round(age_seconds, 2),
                    "ttl_seconds": RETENTION_CACHE_TTL_SECONDS,
                }

    retention_points, _ = fetch_retention_points_via_api(context=context, since_days=since_days)
    retention_analysis = build_retention_analysis(retention_points, context=context)
    entries[video_id] = {
        "saved_at_ts": now_ts,
        "retention_analysis": retention_analysis,
    }
    cache["entries"] = entries
    _save_cache(RETENTION_CACHE_PATH, cache)
    return retention_analysis, {
        "used_cache": False,
        "video_id": video_id,
        "age_seconds": 0.0,
        "ttl_seconds": RETENTION_CACHE_TTL_SECONDS,
    }


def _ensure_analytics_ready() -> None:
    if not youtube_analytics_available():
        raise RuntimeError(
            "YouTube Analytics OAuth kurulumu bulunamadi.\n"
            f"{oauth_setup_summary()}"
        )


def _find_video_by_id(snapshot: dict, video_id: str) -> Optional[dict]:
    for video in snapshot.get("recent_videos", []) or []:
        if str(video.get("video_id") or "").strip() == str(video_id or "").strip():
            return video
    return None


def _prepare_channel_context(snapshot: Optional[dict] = None) -> dict:
    snapshot_meta = {"used_cache": False, "cache_key": "", "age_seconds": 0.0, "ttl_seconds": SNAPSHOT_CACHE_TTL_SECONDS}
    if snapshot is None:
        snapshot, snapshot_meta = _get_snapshot_with_cache(channel_days=28, recent_video_limit=10, per_video_days=90)
    channel_summary = summarize_channel_snapshot(snapshot)
    return {
        "snapshot": snapshot,
        "channel_summary": channel_summary,
        "snapshot_cache": snapshot_meta,
    }


def _build_scoped_channel_summary(channel_summary: dict, format_key: str) -> Optional[dict]:
    breakdown = channel_summary.get("content_format_breakdown", {}) or {}
    target = breakdown.get(format_key, {}) or {}
    if not target or int(target.get("video_count") or 0) <= 0:
        return None

    other_key = "long_form" if format_key == "shorts" else "shorts"
    other = breakdown.get(other_key, {}) or {}
    recent_videos = [
        video for video in (channel_summary.get("recent_videos", []) or [])
        if str(video.get("format_type") or "") == format_key
    ]
    label = str(target.get("label") or format_key).strip()
    return {
        "analysis_scope": format_key,
        "analysis_scope_label": label,
        "channel": channel_summary.get("channel", {}) or {},
        "overall_window_metrics": channel_summary.get("window_metrics", {}) or {},
        "overall_recent_video_benchmarks": channel_summary.get("recent_video_benchmarks", {}) or {},
        "format_summary": target,
        "other_format_summary": other,
        "title_keywords": target.get("title_keywords", []) or [],
        "recent_videos": recent_videos[:10],
    }


def _run_channel_format_bundles(
    analysis_llm: Optional[CentralLLM],
    action_llm: Optional[CentralLLM],
    channel_summary: dict,
) -> dict:
    results: dict[str, dict] = {}
    for format_key in ("shorts", "long_form"):
        scoped_summary = _build_scoped_channel_summary(channel_summary, format_key)
        if scoped_summary is None:
            continue
        scoped_llm_analysis, scoped_action_plan, scoped_generation_mode = _run_channel_llm_bundle(
            analysis_llm,
            action_llm,
            scoped_summary,
        )
        results[format_key] = {
            "summary": (channel_summary.get("content_format_breakdown", {}) or {}).get(format_key, {}) or {},
            "scoped_channel_summary": scoped_summary,
            "llm_analysis": scoped_llm_analysis,
            "critic_and_action_plan": scoped_action_plan,
            "generation_mode": scoped_generation_mode,
        }
    return results


def _prepare_video_context(video_id: Optional[str] = None, snapshot: Optional[dict] = None) -> dict:
    channel_context = _prepare_channel_context(snapshot=snapshot)
    resolved_snapshot = channel_context["snapshot"]
    selected_video = _find_video_by_id(resolved_snapshot, video_id) if video_id else None
    if selected_video is None:
        selected_video = (resolved_snapshot.get("recent_videos", []) or [None])[0]
    if not selected_video:
        raise RuntimeError("Analiz icin son videolar listelenemedi.")

    retention_context = {
        "video_id": selected_video.get("video_id", ""),
        "video_title": selected_video.get("title", ""),
        "duration_seconds": selected_video.get("duration_seconds", 0),
    }
    try:
        retention_analysis, retention_cache = _get_retention_with_cache(retention_context, since_days=365)
    except Exception as exc:
        logger.warning(f"Secilen video retention verisi alinamadi: {exc}")
        retention_analysis = {}
        retention_cache = {
            "used_cache": False,
            "video_id": retention_context.get("video_id", ""),
            "ttl_seconds": RETENTION_CACHE_TTL_SECONDS,
            "error": str(exc),
        }

    video_summary = summarize_selected_video(resolved_snapshot, selected_video, retention_analysis)
    return {
        "snapshot": resolved_snapshot,
        "selected_video": selected_video,
        "retention_analysis": retention_analysis,
        "video_summary": video_summary,
        "snapshot_cache": channel_context["snapshot_cache"],
        "retention_cache": retention_cache,
    }


def _split_combined_payload(payload: Optional[dict]) -> tuple[Optional[dict], Optional[dict]]:
    if not isinstance(payload, dict):
        return None, None
    llm_analysis = payload.get("llm_analysis")
    action_plan = payload.get("critic_and_action_plan")
    return (
        llm_analysis if isinstance(llm_analysis, dict) else None,
        action_plan if isinstance(action_plan, dict) else None,
    )


def _same_llm(a: Optional[CentralLLM], b: Optional[CentralLLM]) -> bool:
    if a is None or b is None:
        return False
    return a.provider == b.provider and a.model_name == b.model_name


def _run_channel_llm_bundle(
    analysis_llm: Optional[CentralLLM],
    action_llm: Optional[CentralLLM],
    channel_summary: dict,
) -> tuple[dict, dict, str]:
    mode = ANALYTICS_LLM_MODE if ANALYTICS_LLM_MODE in {"fast", "deep"} else "fast"
    llm_analysis: Optional[dict] = None
    action_plan: Optional[dict] = None
    if mode == "fast" and _same_llm(analysis_llm, action_llm):
        combined = call_analytics_llm_json(analysis_llm, build_channel_combined_prompt(channel_summary, mode=mode))
        llm_analysis, action_plan = _split_combined_payload(combined)
        if llm_analysis and action_plan:
            return llm_analysis, action_plan, "fast_single_pass"
        if not ANALYTICS_FAST_FALLBACK_TO_DEEP:
            if llm_analysis is None:
                llm_analysis = channel_analysis_fallback(channel_summary)
            if action_plan is None:
                action_plan = channel_action_fallback(channel_summary, llm_analysis)
            return llm_analysis, action_plan, "fast_single_pass_fallback"

    if llm_analysis is None:
        llm_analysis = call_analytics_llm_json(analysis_llm, build_channel_analysis_prompt(channel_summary))
    if llm_analysis is None:
        llm_analysis = channel_analysis_fallback(channel_summary)
    if action_plan is None:
        action_plan = call_analytics_llm_json(action_llm, build_channel_action_prompt(channel_summary, llm_analysis))
    if action_plan is None:
        action_plan = channel_action_fallback(channel_summary, llm_analysis)
    return llm_analysis, action_plan, "deep_two_pass_hybrid" if not _same_llm(analysis_llm, action_llm) else "deep_two_pass"


def _run_video_llm_bundle(
    analysis_llm: Optional[CentralLLM],
    action_llm: Optional[CentralLLM],
    video_summary: dict,
) -> tuple[dict, dict, str]:
    mode = ANALYTICS_LLM_MODE if ANALYTICS_LLM_MODE in {"fast", "deep"} else "fast"
    llm_analysis: Optional[dict] = None
    action_plan: Optional[dict] = None
    if mode == "fast" and _same_llm(analysis_llm, action_llm):
        combined = call_analytics_llm_json(analysis_llm, build_video_combined_prompt(video_summary, mode=mode))
        llm_analysis, action_plan = _split_combined_payload(combined)
        if llm_analysis and action_plan:
            return llm_analysis, action_plan, "fast_single_pass"
        if not ANALYTICS_FAST_FALLBACK_TO_DEEP:
            if llm_analysis is None:
                llm_analysis = video_analysis_fallback(video_summary)
            if action_plan is None:
                action_plan = video_action_fallback(video_summary, llm_analysis)
            return llm_analysis, action_plan, "fast_single_pass_fallback"

    if llm_analysis is None:
        llm_analysis = call_analytics_llm_json(analysis_llm, build_video_analysis_prompt(video_summary))
    if llm_analysis is None:
        llm_analysis = video_analysis_fallback(video_summary)
    if action_plan is None:
        action_plan = call_analytics_llm_json(action_llm, build_video_action_prompt(video_summary, llm_analysis))
    if action_plan is None:
        action_plan = video_action_fallback(video_summary, llm_analysis)
    return llm_analysis, action_plan, "deep_two_pass_hybrid" if not _same_llm(analysis_llm, action_llm) else "deep_two_pass"


def _save_channel_reports(payload: dict) -> dict:
    json_path = _save_json(CHANNEL_JSON_PATH, payload)
    CHANNEL_TXT_PATH.write_text(build_channel_txt_report(payload), encoding="utf-8")
    ACTION_PLAN_TXT_PATH.write_text(build_action_plan_txt(payload), encoding="utf-8")
    CHANNEL_PROMPT_TXT_PATH.write_text(build_channel_external_prompt(payload), encoding="utf-8")
    return {
        "json_path": json_path,
        "channel_txt_path": CHANNEL_TXT_PATH,
        "action_txt_path": ACTION_PLAN_TXT_PATH,
        "prompt_txt_path": CHANNEL_PROMPT_TXT_PATH,
    }


def _save_video_reports(payload: dict) -> dict:
    legacy_payload = dict(payload.get("retention_analysis", {}) or {})
    legacy_payload.update(
        {
            "analysis_type": "selected_video",
            "channel": payload.get("channel", {}),
            "video_summary": payload.get("video_summary", {}),
            "retention_analysis": payload.get("retention_analysis", {}),
            "llm_analysis": payload.get("llm_analysis", {}),
            "critic_and_action_plan": payload.get("critic_and_action_plan", {}),
        }
    )
    json_path = _save_json(FEEDBACK_JSON_PATH, legacy_payload)
    FEEDBACK_TXT_PATH.write_text(build_legacy_feedback_report_text(payload), encoding="utf-8")
    VIDEO_TXT_PATH.write_text(build_video_txt_report(payload), encoding="utf-8")
    ACTION_PLAN_TXT_PATH.write_text(build_action_plan_txt(payload), encoding="utf-8")
    VIDEO_PROMPT_TXT_PATH.write_text(build_video_external_prompt(payload), encoding="utf-8")
    return {
        "json_path": json_path,
        "feedback_txt_path": FEEDBACK_TXT_PATH,
        "video_txt_path": VIDEO_TXT_PATH,
        "action_txt_path": ACTION_PLAN_TXT_PATH,
        "prompt_txt_path": VIDEO_PROMPT_TXT_PATH,
    }


def run_channel_automatic(
    main_llm: Optional[CentralLLM] = None,
    smart_llm: Optional[CentralLLM] = None,
    llm_info: Optional[dict] = None,
) -> Optional[dict]:
    _ensure_analytics_ready()
    logger.info("📊 Kanal analytics snapshot'i aliniyor...")
    context = _prepare_channel_context()
    snapshot = context["snapshot"]
    channel_summary = context["channel_summary"]

    analysis_llm = main_llm
    action_llm = smart_llm or main_llm
    active_llm_info = llm_info or {"enabled": False, "main_provider": "", "main_model_name": "", "smart_provider": "", "smart_model_name": ""}
    if analysis_llm is None and action_llm is None and not llm_info:
        main_provider, main_model_name = get_default_llm_config("main")
        smart_provider, smart_model_name = get_default_llm_config("smart")
        analysis_llm = CentralLLM(provider=main_provider, model_name=main_model_name)
        action_llm = CentralLLM(provider=smart_provider, model_name=smart_model_name)
        active_llm_info = {
            "enabled": True,
            "main_provider": main_provider,
            "main_model_name": main_model_name,
            "smart_provider": smart_provider,
            "smart_model_name": smart_model_name,
        }

    llm_analysis, critic_and_action_plan, generation_mode = _run_channel_llm_bundle(analysis_llm, action_llm, channel_summary)
    format_specific_analysis = _run_channel_format_bundles(analysis_llm, action_llm, channel_summary)

    payload = {
        "analysis_type": "channel",
        "generation_mode": generation_mode,
        "llm_info": active_llm_info,
        "channel": snapshot.get("channel", {}),
        "raw_snapshot": snapshot,
        "channel_summary": channel_summary,
        "llm_analysis": llm_analysis,
        "critic_and_action_plan": critic_and_action_plan,
        "format_specific_analysis": format_specific_analysis,
        "snapshot_cache": context["snapshot_cache"],
    }
    payload.update(_save_channel_reports(payload))
    return payload


def run_video_automatic(
    video_id: Optional[str] = None,
    snapshot: Optional[dict] = None,
    main_llm: Optional[CentralLLM] = None,
    smart_llm: Optional[CentralLLM] = None,
    llm_info: Optional[dict] = None,
) -> Optional[dict]:
    _ensure_analytics_ready()
    if snapshot is None:
        logger.info("📺 Son 10 video snapshot'i aliniyor...")
    context = _prepare_video_context(video_id=video_id, snapshot=snapshot)
    snapshot = context["snapshot"]
    selected_video = context["selected_video"]
    retention_analysis = context["retention_analysis"]
    video_summary = context["video_summary"]

    analysis_llm = main_llm
    action_llm = smart_llm or main_llm
    active_llm_info = llm_info or {"enabled": False, "main_provider": "", "main_model_name": "", "smart_provider": "", "smart_model_name": ""}
    if analysis_llm is None and action_llm is None and not llm_info:
        main_provider, main_model_name = get_default_llm_config("main")
        smart_provider, smart_model_name = get_default_llm_config("smart")
        analysis_llm = CentralLLM(provider=main_provider, model_name=main_model_name)
        action_llm = CentralLLM(provider=smart_provider, model_name=smart_model_name)
        active_llm_info = {
            "enabled": True,
            "main_provider": main_provider,
            "main_model_name": main_model_name,
            "smart_provider": smart_provider,
            "smart_model_name": smart_model_name,
        }

    llm_analysis, critic_and_action_plan, generation_mode = _run_video_llm_bundle(analysis_llm, action_llm, video_summary)

    payload = {
        "analysis_type": "selected_video",
        "generation_mode": generation_mode,
        "llm_info": active_llm_info,
        "channel": snapshot.get("channel", {}),
        "raw_snapshot": snapshot,
        "selected_video": selected_video,
        "video_summary": video_summary,
        "retention_analysis": retention_analysis,
        "llm_analysis": llm_analysis,
        "critic_and_action_plan": critic_and_action_plan,
        "snapshot_cache": context["snapshot_cache"],
        "retention_cache": context["retention_cache"],
    }
    payload.update(_save_video_reports(payload))
    return payload


def run_automatic(mode: str = "channel", video_id: Optional[str] = None) -> Optional[dict]:
    main_provider, main_model_name = get_default_llm_config("main")
    smart_provider, smart_model_name = get_default_llm_config("smart")
    main_llm = CentralLLM(provider=main_provider, model_name=main_model_name)
    smart_llm = CentralLLM(provider=smart_provider, model_name=smart_model_name)
    llm_info = {
        "enabled": True,
        "main_provider": main_provider,
        "main_model_name": main_model_name,
        "smart_provider": smart_provider,
        "smart_model_name": smart_model_name,
    }
    if str(mode).strip().lower() in {"video", "selected_video", "last_video"}:
        return run_video_automatic(video_id=video_id, main_llm=main_llm, smart_llm=smart_llm, llm_info=llm_info)
    return run_channel_automatic(main_llm=main_llm, smart_llm=smart_llm, llm_info=llm_info)


def _select_video_from_snapshot(snapshot: dict) -> Optional[str]:
    videos = list(snapshot.get("recent_videos", []) or [])
    if not videos:
        logger.error("Kanalin son videolari listelenemedi.")
        return None

    print("\n" + "-" * 72)
    print("KANALIN SON 10 VIDEOSU")
    print("-" * 72)
    for index, video in enumerate(videos[:10], start=1):
        print(
            f"[{index}] {video.get('title', '')} | "
            f"{video.get('duration_label', '')} | "
            f"{format_number(video_views(video))} izlenme"
        )
    print("-" * 72)

    choice = input("👉 Analiz etmek istediginiz videoyu secin: ").strip()
    try:
        return videos[int(choice) - 1].get("video_id", "")
    except Exception:
        logger.error("Gecersiz video secimi.")
        return None


def _print_channel_result(result: dict) -> None:
    summary = result.get("channel_summary", {}) or {}
    print("\n" + "=" * 72)
    print("YOUTUBE ANALYTICS | KANAL ANALIZI TAMAMLANDI")
    print("=" * 72)
    print(f"Kanal: {(summary.get('channel', {}) or {}).get('channel_title', '')}")
    print(f"Son pencere goruntulenme: {format_number(summary.get('window_metrics', {}).get('views'))}")
    print(f"Net abone: {format_number(summary.get('window_metrics', {}).get('net_subscribers'))}")
    print(f"Ortalama izlenme yuzdesi: {format_percent(summary.get('window_metrics', {}).get('average_view_percentage'))}")
    format_breakdown = summary.get("content_format_breakdown", {}) or {}
    shorts_summary = format_breakdown.get("shorts", {}) or {}
    long_form_summary = format_breakdown.get("long_form", {}) or {}
    print(
        f"Format kirilimi: Shorts {format_number(shorts_summary.get('video_count'))} | "
        f"Uzun Video {format_number(long_form_summary.get('video_count'))}"
    )
    print(f"Rapor: {result.get('channel_txt_path')}")
    print(f"Aksiyon Plani: {result.get('action_txt_path')}")
    print(f"ChatGPT/Gemini Promptu: {result.get('prompt_txt_path')}")
    print(f"JSON: {result.get('json_path')}")
    print("=" * 72)


def _print_video_result(result: dict) -> None:
    selected_video = (result.get("video_summary", {}) or {}).get("selected_video", {}) or {}
    retention = result.get("retention_analysis", {}) or {}
    print("\n" + "=" * 72)
    print("YOUTUBE ANALYTICS | SECILEN VIDEO ANALIZI TAMAMLANDI")
    print("=" * 72)
    print(f"Video: {selected_video.get('title', '')}")
    print(f"Goruntulenme: {format_number(selected_video.get('views'))}")
    print(f"Ortalama izlenme yuzdesi: {format_percent(selected_video.get('average_view_percentage'))}")
    if retention:
        print(f"Ilk 30 saniye kaybi: {format_percent(retention.get('first_30s_drop_pct'))}")
    print(f"Rapor: {result.get('video_txt_path')}")
    print(f"Aksiyon Plani: {result.get('action_txt_path')}")
    print(f"ChatGPT/Gemini Promptu: {result.get('prompt_txt_path')}")
    print(f"JSON: {result.get('json_path')}")
    print("=" * 72)


def run() -> None:
    print("\n" + "=" * 72)
    print("YOUTUBE ANALYTICS ANALIZI")
    print("=" * 72)
    print("[1] Kanali analiz et")
    print("[2] Kanaldaki son 10 videodan birini sec ve analiz et")
    print("=" * 72)

    choice = input("👉 Seciminiz (1 veya 2): ").strip()
    if choice not in {"1", "2"}:
        logger.error("Gecersiz secim.")
        return

    try:
        channel = fetch_authenticated_channel_profile()
        print("\n" + "-" * 72)
        print(f"✅ OAuth ile kanala erisim saglandi: {channel.get('channel_title', '')}")
        print("-" * 72)
    except Exception as exc:
        logger.error(f"YouTube Analytics OAuth veya kanal erisimi basarisiz: {exc}")
        print(oauth_setup_summary())
        return

    use_recommended = prompt_module_llm_plan("402", needs_main=True, needs_smart=True)
    if use_recommended:
        provider_main, model_name_main = get_module_recommended_llm_config("402", "main")
        provider_smart, model_name_smart = get_module_recommended_llm_config("402", "smart")
        print_module_llm_choice_summary("402", {"main": (provider_main, model_name_main), "smart": (provider_smart, model_name_smart)})
    else:
        provider_main, model_name_main = select_llm("main")
        provider_smart, model_name_smart = select_llm("smart")
    main_llm = CentralLLM(provider=provider_main, model_name=model_name_main)
    smart_llm = CentralLLM(provider=provider_smart, model_name=model_name_smart)
    llm_info = {
        "enabled": True,
        "main_provider": provider_main,
        "main_model_name": model_name_main,
        "smart_provider": provider_smart,
        "smart_model_name": model_name_smart,
    }

    if choice == "1":
        result = run_channel_automatic(main_llm=main_llm, smart_llm=smart_llm, llm_info=llm_info)
        if not result:
            logger.error("Kanal analytics analizi uretilemedi.")
            return
        _print_channel_result(result)
        return

    snapshot, _ = _get_snapshot_with_cache(channel_days=28, recent_video_limit=10, per_video_days=90)
    selected_video_id = _select_video_from_snapshot(snapshot)
    if not selected_video_id:
        return
    result = run_video_automatic(video_id=selected_video_id, snapshot=snapshot, main_llm=main_llm, smart_llm=smart_llm, llm_info=llm_info)
    if not result:
        logger.error("Secilen video analytics analizi uretilemedi.")
        return
    _print_video_result(result)
