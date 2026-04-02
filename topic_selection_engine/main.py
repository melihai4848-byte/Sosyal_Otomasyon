from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moduller.output_paths import output_group_dir
from moduller.llm_manager import (
    get_module_recommended_llm_config,
    print_module_llm_choice_summary,
    prompt_module_llm_plan,
    select_llm,
)
from moduller.trend_cache_utils import save_trend_reports
from topic_selection_engine.channel_profile import build_dynamic_channel_profile
from topic_selection_engine.config import load_settings
from topic_selection_engine.ingest import collect_mock_signals, load_signals_from_paths
from topic_selection_engine.live_sources import collect_live_signals
from topic_selection_engine.llm import IdeaGenerator
from topic_selection_engine.models import RawSignal, TopicCluster
from topic_selection_engine.outputs import export_outputs
from topic_selection_engine.processing import (
    build_topic_clusters,
    clean_signals,
    deduplicate_signals,
    filter_relevant_signals,
)
from topic_selection_engine.scoring import score_topic_clusters
from moduller.logger import get_logger
from moduller.retry_utils import retry_with_backoff

logger = get_logger("topic")
MIN_TOPIC_IDEA_COUNT = 5


def _collect_source_mock_signals(source_name: str, since_days: int, limit: int) -> List[RawSignal]:
    return [
        signal for signal in collect_mock_signals(since_days=since_days, limit=limit)
        if signal.source == source_name
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Topic Selection Engine MVP")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--since-days", type=int, default=14, help="Collect signals from the last N days")
    parser.add_argument("--limit", type=int, default=200, help="Maximum raw signals to ingest")
    parser.add_argument(
        "--output-dir",
        default=str(output_group_dir("research")),
        help="Output directory for JSON cache and TXT report",
    )
    parser.add_argument("--youtube-file", action="append", default=[], help="Optional local YouTube JSON/JSONL/CSV export")
    parser.add_argument("--reddit-file", action="append", default=[], help="Optional local Reddit JSON/JSONL/CSV export")
    parser.add_argument("--channel-file", action="append", default=[], help="Optional local channel JSON/JSONL/CSV export")
    return parser


def _model_copy(instance, update: dict):
    if hasattr(instance, "model_copy"):
        return instance.model_copy(update=update)
    return instance.copy(update=update)


def collect_signals(
    settings,
    since_days: int,
    limit: int,
    include_live: bool = True,
    preloaded_signals: List[RawSignal] | None = None,
):
    raw_signals: List[RawSignal] = list(preloaded_signals or [])
    mock_enabled_sources: list[str] = []
    live_bundle = {
        "signals": [],
        "top_keywords": [],
        "rising_queries": [],
        "viral_topics": [],
        "sources_used": [],
        "notes": [],
        "fetched_at": "",
    }

    for source_name, source_settings in settings.sources.items():
        if not source_settings.enabled:
            continue

        local_signals = load_signals_from_paths(source_name, source_settings, limit=max(limit - len(raw_signals), 0))
        raw_signals.extend(local_signals)

        if source_settings.use_mock:
            if include_live:
                mock_enabled_sources.append(source_name)
            elif len(raw_signals) < limit:
                raw_signals.extend(_collect_source_mock_signals(source_name, since_days=since_days, limit=limit))

    if include_live and len(raw_signals) < limit:
        live_bundle = retry_with_backoff(
            lambda: collect_live_signals(
                settings=settings,
                since_days=since_days,
                limit=max(limit - len(raw_signals), 0),
            ),
            "YouTube konu bulucu canli veri toplama",
            logger,
            max_attempts=4,
            base_delay_seconds=15,
            max_delay_seconds=60,
        )
        raw_signals.extend(live_bundle.get("signals", []))

    cleaned = clean_signals(raw_signals)
    relevant = filter_relevant_signals(cleaned, settings.niche_keywords)
    deduped = deduplicate_signals(relevant)

    if include_live and len(deduped) < MIN_TOPIC_IDEA_COUNT and mock_enabled_sources:
        live_bundle.setdefault("notes", []).append(
            "Canli ve yerel kaynaklar yeterli sinyal uretmedigi icin mock fallback devreye girdi."
        )
        for source_name in mock_enabled_sources:
            raw_signals.extend(_collect_source_mock_signals(source_name, since_days=since_days, limit=limit))
        cleaned = clean_signals(raw_signals)
        relevant = filter_relevant_signals(cleaned, settings.niche_keywords)
        deduped = deduplicate_signals(relevant)

    return deduped[:limit], live_bundle


def _select_candidate_clusters(scored_clusters: List[TopicCluster], settings) -> List[TopicCluster]:
    if not scored_clusters:
        return []

    min_target = max(MIN_TOPIC_IDEA_COUNT, settings.top_n)
    hard_cap = min(len(scored_clusters), max(settings.top_n * 2, MIN_TOPIC_IDEA_COUNT + 3))
    candidates: List[TopicCluster] = []

    for cluster in scored_clusters:
        if cluster.final_score < settings.scoring_threshold and len(candidates) >= min_target:
            break
        if candidates and len(candidates) >= min_target:
            previous_score = candidates[-1].final_score
            if previous_score - cluster.final_score >= 12.0:
                break
        candidates.append(cluster)
        if len(candidates) >= hard_cap:
            break

    if len(candidates) < min_target:
        seen_topics = {cluster.canonical_topic for cluster in candidates}
        for cluster in scored_clusters:
            if cluster.canonical_topic in seen_topics:
                continue
            candidates.append(cluster)
            seen_topics.add(cluster.canonical_topic)
            if len(candidates) >= min_target:
                break

    return candidates


def run_engine(
    config_path: str | None,
    since_days: int,
    limit: int,
    output_dir: str,
    youtube_files: List[str] | None = None,
    reddit_files: List[str] | None = None,
    channel_files: List[str] | None = None,
    include_live: bool = True,
    llm_provider_override: str | None = None,
    llm_model_override: str | None = None,
):
    settings = load_settings(config_path)

    youtube_files = youtube_files or []
    reddit_files = reddit_files or []
    channel_files = channel_files or []

    source_updates = {}
    for source_name, files in [("youtube", youtube_files), ("reddit", reddit_files), ("channel", channel_files)]:
        if not files or source_name not in settings.sources:
            continue
        source_settings = settings.sources[source_name]
        update = {"input_paths": list(source_settings.input_paths) + files, "use_mock": source_settings.use_mock}
        source_updates[source_name] = _model_copy(source_settings, update)

    settings = _model_copy(
        settings,
        {
            "output_dir": output_dir,
            "sources": {**settings.sources, **source_updates},
        },
    )

    if llm_provider_override or llm_model_override:
        settings = _model_copy(
            settings,
            {
                "llm": _model_copy(
                    settings.llm,
                    {
                        "provider": llm_provider_override or settings.llm.provider,
                        "model_name": llm_model_override or settings.llm.model_name,
                    },
                )
            },
        )

    channel_profile = build_dynamic_channel_profile(settings)
    dynamic_source_updates = dict(settings.sources)
    preloaded_channel_signals = list(channel_profile.get("recent_video_signals", []))
    if preloaded_channel_signals and "channel" in dynamic_source_updates:
        dynamic_source_updates["channel"] = _model_copy(
            dynamic_source_updates["channel"],
            {"use_mock": False},
        )

    dynamic_settings_payload = {
        "output_dir": output_dir,
        "sources": dynamic_source_updates,
    }
    if channel_profile.get("inferred_niche"):
        dynamic_settings_payload.update(
            {
                "niche": channel_profile.get("inferred_niche", settings.niche),
                "target_audiences": channel_profile.get("target_audiences", settings.target_audiences),
                "niche_keywords": channel_profile.get("niche_keywords", settings.niche_keywords),
                "channel_strength_topics": channel_profile.get("channel_strength_topics", settings.channel_strength_topics),
                "easy_topic_keywords": channel_profile.get("easy_topic_keywords", settings.easy_topic_keywords),
                "monetizable_keywords": channel_profile.get("monetizable_keywords", settings.monetizable_keywords),
                "compare_keywords": channel_profile.get("compare_keywords", settings.compare_keywords),
            }
        )
    settings = _model_copy(settings, dynamic_settings_payload)

    signals, live_bundle = collect_signals(
        settings=settings,
        since_days=since_days,
        limit=limit,
        include_live=include_live,
        preloaded_signals=preloaded_channel_signals,
    )
    save_trend_reports(live_bundle)
    channel_signals = [signal for signal in signals if signal.source == "channel"]

    clusters: List[TopicCluster] = build_topic_clusters(
        signals=signals,
        freshness_half_life_days=settings.freshness_half_life_days,
    )
    scored_clusters = score_topic_clusters(clusters, settings, channel_signals)
    candidate_clusters = _select_candidate_clusters(scored_clusters, settings)

    generator = IdeaGenerator(settings, channel_profile=channel_profile, live_trend_data=live_bundle)
    ideas = generator.generate_ranked_ideas(candidate_clusters)
    selected_cluster_map = {cluster.canonical_topic: cluster for cluster in candidate_clusters}
    filtered_clusters = [
        selected_cluster_map[idea.canonical_topic]
        for idea in ideas
        if idea.canonical_topic in selected_cluster_map
    ]
    json_path, md_path = export_outputs(
        output_dir=settings.output_dir,
        ideas=ideas,
        clusters=filtered_clusters,
        channel_profile=channel_profile,
        llm_info=generator.llm_info,
    )

    return {
        "signals": signals,
        "clusters": filtered_clusters,
        "ideas": ideas,
        "json_path": json_path,
        "md_path": md_path,
        "live_trend_data": live_bundle,
        "channel_profile": channel_profile,
        "llm_info": generator.llm_info,
    }


def run_from_menu():
    print("\n" + "=" * 60)
    print("YOUTUBE TRENDS KONU FIKIRLERI | Sıradaki video ne olmalı?")
    print("=" * 60)
    since_days = 14
    limit = 200
    output_dir = str(output_group_dir("research"))
    print("Varsayilan ayarlar kullaniliyor: son 14 gun, maksimum 200 sinyal, cikti klasoru 00_Outputs/400_Arastirma_Sonuclari.")
    use_recommended = prompt_module_llm_plan("401", needs_smart=True)
    if use_recommended:
        saglayici, model_adi = get_module_recommended_llm_config("401", "smart")
        print_module_llm_choice_summary("401", {"smart": (saglayici, model_adi)})
    else:
        saglayici, model_adi = select_llm("smart")

    try:
        result = run_engine(
            config_path=None,
            since_days=since_days,
            limit=limit,
            output_dir=output_dir,
            youtube_files=[],
            reddit_files=[],
            channel_files=[],
            include_live=True,
            llm_provider_override=saglayici,
            llm_model_override=model_adi,
        )
    except Exception as exc:
        print(f"❌ Topic Selection Engine hata verdi: {exc}")
        return

    print("\n" + "=" * 60)
    print("YOUTUBE TRENDS KONU FIKIRLERI SONUCU")
    print("=" * 60)
    channel_profile = result.get("channel_profile", {})
    llm_info = result.get("llm_info", {})
    for status_line in channel_profile.get("status_messages", []):
        print(status_line)
    if llm_info.get("enabled"):
        print(f"Konu seciminde YARATICI YAPAY ZEKA kullanildi: {llm_info.get('provider')} / {llm_info.get('model_name')}")
    if channel_profile.get("channel_name"):
        print(f"Kanal Adi: {channel_profile.get('channel_name')}")
    if channel_profile.get("inferred_niche"):
        print(f"Kanal Nis Profili: {channel_profile.get('inferred_niche')}")
    if channel_profile.get("notes"):
        print("\nKanal Profil Notlari:")
        for note in channel_profile.get("notes", [])[:5]:
            print(f"- {note}")
    for idea in result["ideas"][:5]:
        print(f"#{idea.rank} | {idea.title} | Skor: {idea.final_score:.2f}")
    if result.get("live_trend_data", {}).get("sources_used"):
        print(f"\nCanli kaynaklar: {', '.join(result['live_trend_data']['sources_used'])}")
        print(
            f"Canli sinyal sayisi: {len(result.get('live_trend_data', {}).get('signals', []))} | "
            f"Final temiz sinyal sayisi: {len(result.get('signals', []))}"
        )
    elif result.get("live_trend_data", {}).get("notes"):
        print("\n⚠️ Canli trend kaynaklari sinyal uretmedi.")
        for note in result["live_trend_data"].get("notes", [])[:5]:
            print(f"- {note}")
    print(f"\nJSON Cache: {result['json_path']}")
    print(f"TXT:        {result['md_path']}")
    print("=" * 60)


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_engine(
        config_path=args.config,
        since_days=args.since_days,
        limit=args.limit,
        output_dir=args.output_dir,
        youtube_files=args.youtube_file,
        reddit_files=args.reddit_file,
        channel_files=args.channel_file,
        include_live=True,
    )
    print(f"Created JSON cache: {result['json_path']}")
    print(f"Created TXT report: {result['md_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
