from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

from moduller.output_paths import json_output_path, txt_output_path
from topic_selection_engine.models import RankedIdea, TopicCluster, model_dump_compat


def _serialize_channel_profile(channel_profile: Optional[dict]) -> dict:
    profile = dict(channel_profile or {})
    recent_signals = profile.pop("recent_video_signals", [])
    if recent_signals:
        profile["recent_video_signals"] = [model_dump_compat(item) for item in recent_signals]
    return profile


def _text_report(
    ideas: Iterable[RankedIdea],
    clusters: Iterable[TopicCluster],
    channel_profile: Optional[dict] = None,
    llm_info: Optional[dict] = None,
) -> str:
    cluster_map = {cluster.canonical_topic: cluster for cluster in clusters}
    ideas = list(ideas)
    clusters = list(clusters)
    channel_profile = channel_profile or {}
    llm_info = llm_info or {}
    avg_score = sum(item.final_score for item in ideas) / len(ideas) if ideas else 0.0
    lines: List[str] = [
        "=== YOUTUBE TRENDS KONU FIKIRLERI RAPORU ===",
        "",
        "OZET",
        "",
    ]
    if channel_profile.get("channel_name"):
        lines.extend(
            [
                f"- Kanal adi: {channel_profile.get('channel_name')}",
                f"- Kanal nis profili: {channel_profile.get('inferred_niche', '')}",
                f"- Niche keywordleri: {', '.join(channel_profile.get('niche_keywords', [])) or 'Yok'}",
                f"- Hedef kitle tahmini: {', '.join(channel_profile.get('target_audiences', [])) or 'Yok'}",
                "",
            ]
        )
    if llm_info.get("enabled"):
        lines.extend(
            [
                f"- Yaratıcı yapay zeka: {llm_info.get('provider')} / {llm_info.get('model_name')}",
                "",
            ]
        )
    lines.extend(
        [
        f"- Siralanan fikir sayisi: {len(ideas)}",
        f"- Kume sayisi: {len(clusters)}",
        f"- Ortalama nihai skor: {avg_score:.2f}",
        "",
        "SIRALANMIS FIKIRLER",
        "",
        ]
    )

    for idea in ideas:
        cluster = cluster_map.get(idea.canonical_topic)
        lines.append(f"#{idea.rank} - {idea.title}")
        lines.append("-" * 60)
        lines.append(f"Konu: {idea.canonical_topic}")
        lines.append(f"Nihai skor: {idea.final_score:.2f}")
        lines.append(f"Hedef kitle: {idea.target_audience}")
        lines.append(f"Format: {idea.format}")
        lines.append(f"Neden simdi: {' | '.join(idea.why_now)}")
        lines.append(f"Hooklar: {' | '.join(idea.hooks)}")
        lines.append(f"Aci: {idea.angle}")
        lines.append(f"Riskler: {' | '.join(idea.risks)}")
        lines.append(f"Sonraki aksiyon: {idea.next_action}")
        lines.append(f"Mantiksal ozet: {idea.reasoning_summary}")
        if cluster is not None:
            lines.append(f"Kaynak sayisi: {cluster.source_count}")
            lines.append(f"Dis kaynak sinyali: {cluster.external_source_count}")
            lines.append(f"Kanal ici tekrar sinyali: {cluster.channel_source_count}")
            lines.append(f"Yeni sinyal sayisi: {cluster.recent_signal_count}")
            lines.append(f"Dis yeni sinyal sayisi: {cluster.external_recent_signal_count}")
            lines.append(f"Trend hizi: {cluster.trend_velocity:.2f}")
            lines.append(f"Dis trend hizi: {cluster.external_trend_velocity:.2f}")
            lines.append(f"Ortalama etkilesim skoru: {cluster.avg_engagement_score:.2f}")
            lines.append(f"Dis etkilesim skoru: {cluster.external_avg_engagement_score:.2f}")
            lines.append(f"Kitleler: {', '.join(cluster.audiences)}")
            lines.append(f"Duygu tetikleyicileri: {', '.join(cluster.emotion_triggers)}")
            lines.append(f"Anahtar kelimeler: {', '.join(cluster.keywords)}")
            lines.append(f"Trend skor notu: {cluster.score_reasoning.get('trend', '')}")
        lines.append("")

    lines.extend([
        "KUME OZETI",
        "",
    ])
    for cluster in clusters:
        lines.append(
            f"- {cluster.canonical_topic} -> skor {cluster.final_score:.2f}, tazelik {cluster.freshness_score:.2f}, "
            f"kaynak {cluster.source_count}, yeni {cluster.recent_signal_count}, hiz {cluster.trend_velocity:.2f}"
        )

    lines.append("")
    return "\n".join(lines)


def export_outputs(
    output_dir: str | Path,
    ideas: List[RankedIdea],
    clusters: List[TopicCluster],
    channel_profile: Optional[dict] = None,
    llm_info: Optional[dict] = None,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = json_output_path("topic_selector")
    md_path = txt_output_path("topic_selector")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "channel_profile": _serialize_channel_profile(channel_profile),
                "llm_info": dict(llm_info or {}),
                "ideas": [model_dump_compat(item) for item in ideas],
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    md_path.write_text(
        _text_report(ideas, clusters, channel_profile=channel_profile, llm_info=llm_info),
        encoding="utf-8",
    )
    return json_path, md_path
