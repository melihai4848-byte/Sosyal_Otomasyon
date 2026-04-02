from __future__ import annotations

from typing import Dict, Iterable, List

from topic_selection_engine.config import EngineSettings
from topic_selection_engine.models import RawSignal, TopicCluster


PRODUCTION_EASE_MAP: Dict[str, float] = {
    "almanya_yasam_maliyeti": 84.0,
    "almanya_kira_ve_ev": 72.0,
    "almanya_burokrasi": 80.0,
    "almanya_is_ve_almanca": 78.0,
    "almanya_ogrenci_hayati": 76.0,
    "almanya_genel_yasam": 68.0,
}

MONETIZATION_MAP: Dict[str, float] = {
    "almanya_yasam_maliyeti": 82.0,
    "almanya_kira_ve_ev": 74.0,
    "almanya_burokrasi": 58.0,
    "almanya_is_ve_almanca": 88.0,
    "almanya_ogrenci_hayati": 61.0,
    "almanya_genel_yasam": 50.0,
}


def _clamp(score: float) -> float:
    return round(max(0.0, min(100.0, score)), 2)


def _channel_success_topics(channel_signals: Iterable[RawSignal]) -> List[str]:
    successful = []
    for signal in channel_signals:
        views = float(signal.metadata.get("views", 0))
        ctr = float(signal.metadata.get("ctr", 0))
        if views >= 60000 or ctr >= 7.0:
            successful.append(signal.title.lower())
    return successful


def _recent_channel_titles(channel_signals: Iterable[RawSignal]) -> List[str]:
    return [signal.title.lower() for signal in channel_signals]


def _external_source_diversity(cluster: TopicCluster) -> int:
    return sum(1 for source, count in cluster.source_breakdown.items() if source != "channel" and count > 0)


def _trend_score(cluster: TopicCluster) -> float:
    source_diversity = _external_source_diversity(cluster)
    if cluster.external_source_count <= 0:
        return _clamp(6.0 + min(cluster.channel_source_count * 2.0, 10.0))

    base = (
        cluster.external_source_count * 13.0
        + source_diversity * 14.0
        + cluster.external_freshness_score * 0.24
        + cluster.external_trend_velocity * 0.34
        + min(cluster.external_avg_engagement_score / 2500.0, 18.0)
    )
    base -= min(cluster.channel_source_count * 1.5, 9.0)
    return _clamp(base)


def _audience_fit_score(cluster: TopicCluster, settings: EngineSettings) -> float:
    target_blob = " ".join(settings.target_audiences).lower()
    matches = 0
    for audience in cluster.audiences:
        if audience.replace("_", " ")[:6] in target_blob:
            matches += 1
    pain_bonus = min(len(cluster.pain_points) * 8.0, 24.0)
    return _clamp(45.0 + matches * 14.0 + pain_bonus)


def _channel_fit_score(cluster: TopicCluster, channel_signals: Iterable[RawSignal]) -> float:
    successful_titles = _channel_success_topics(channel_signals)
    keyword_hits = sum(1 for title in successful_titles if any(keyword in title for keyword in cluster.keywords[:4]))
    canonical_tokens = cluster.canonical_topic.replace("_", " ").split()
    canonical_focus = canonical_tokens[1] if len(canonical_tokens) > 1 else canonical_tokens[0]
    canonical_hits = sum(1 for title in successful_titles if canonical_focus in title)
    return _clamp(35.0 + keyword_hits * 14.0 + canonical_hits * 12.0)


def _novelty_score(cluster: TopicCluster, channel_signals: Iterable[RawSignal]) -> float:
    recent_titles = _recent_channel_titles(channel_signals)
    overlap = sum(1 for title in recent_titles if any(keyword in title for keyword in cluster.keywords[:4]))
    canonical_tokens = cluster.canonical_topic.replace("_", " ").split()
    canonical_focus = canonical_tokens[1] if len(canonical_tokens) > 1 else canonical_tokens[0]
    canonical_hits = sum(1 for title in recent_titles if canonical_focus in title)

    if cluster.external_source_count <= 0:
        return _clamp(12.0 - overlap * 4.0)

    base = 92.0 - overlap * 18.0 - canonical_hits * 10.0 - cluster.channel_source_count * 4.0
    if cluster.channel_source_count >= cluster.external_source_count:
        base -= 10.0
    return _clamp(base)


def _production_ease_score(cluster: TopicCluster) -> float:
    return _clamp(PRODUCTION_EASE_MAP.get(cluster.canonical_topic, 65.0))


def _hook_potential_score(cluster: TopicCluster) -> float:
    emotion_bonus = len(cluster.emotion_triggers) * 12.0
    question_bonus = len(cluster.questions) * 6.0
    comparison_bonus = 16.0 if "comparison" in cluster.emotion_triggers else 0.0
    fear_bonus = 14.0 if "fear" in cluster.emotion_triggers else 0.0
    return _clamp(35.0 + emotion_bonus + question_bonus + comparison_bonus + fear_bonus)


def _monetization_score(cluster: TopicCluster) -> float:
    return _clamp(MONETIZATION_MAP.get(cluster.canonical_topic, 55.0))


def _score_reasoning(
    cluster: TopicCluster,
    trend_score: float,
    audience_fit_score: float,
    channel_fit_score: float,
    novelty_score: float,
    production_ease_score: float,
    hook_potential_score: float,
    monetization_score: float,
) -> Dict[str, str]:
    return {
        "trend": (
            f"Dis kaynaklardan gelen {cluster.external_source_count} sinyal, "
            f"{cluster.external_recent_signal_count} yeni sinyal ve "
            f"{cluster.external_trend_velocity:.2f} hiz degeri trend skorunu {trend_score:.2f} seviyesine cekti. "
            f"Kanal ici tekrar sinyali: {cluster.channel_source_count}."
        ),
        "audience_fit": (
            f"Hedef kitle eslesmeleri {', '.join(cluster.audiences) if cluster.audiences else 'yok'}; "
            f"acili nokta sayisi {len(cluster.pain_points)}."
        ),
        "channel_fit": (
            f"Kanal gecmisindeki benzer basliklar ve anahtar kelime uyumu channel fit skorunu {channel_fit_score:.2f} yapti."
        ),
        "novelty": (
            f"Son kanal basliklariyla anahtar kelime ve tema overlap seviyesi dikkate alinarak novelty {novelty_score:.2f} hesaplandi."
        ),
        "production_ease": (
            f"Konu tipi ve deneyim bazli uretim kolayligi {production_ease_score:.2f} olarak degerlendirildi."
        ),
        "hook_potential": (
            f"Duygu tetikleyicileri {', '.join(cluster.emotion_triggers)} ve soru sayisi hook potansiyelini {hook_potential_score:.2f} yapti."
        ),
        "monetization": (
            f"Kariyer/maas/karar temasi yakinligi monetization skorunu {monetization_score:.2f} seviyesine getirdi."
        ),
    }


def score_topic_clusters(
    clusters: List[TopicCluster],
    settings: EngineSettings,
    channel_signals: Iterable[RawSignal],
) -> List[TopicCluster]:
    weights = settings.ranking_weights
    scored_clusters: List[TopicCluster] = []

    for cluster in clusters:
        trend_score = _trend_score(cluster)
        audience_fit_score = _audience_fit_score(cluster, settings)
        channel_fit_score = _channel_fit_score(cluster, channel_signals)
        novelty_score = _novelty_score(cluster, channel_signals)
        production_ease_score = _production_ease_score(cluster)
        hook_potential_score = _hook_potential_score(cluster)
        monetization_score = _monetization_score(cluster)

        final_score = (
            weights.trend * trend_score
            + weights.audience_fit * audience_fit_score
            + weights.channel_fit * channel_fit_score
            + weights.novelty * novelty_score
            + weights.production_ease * production_ease_score
            + weights.hook_potential * hook_potential_score
            + weights.monetization * monetization_score
        )

        update_payload = {
            "trend_score": _clamp(trend_score),
            "audience_fit_score": _clamp(audience_fit_score),
            "channel_fit_score": _clamp(channel_fit_score),
            "novelty_score": _clamp(novelty_score),
            "production_ease_score": _clamp(production_ease_score),
            "hook_potential_score": _clamp(hook_potential_score),
            "monetization_score": _clamp(monetization_score),
            "final_score": _clamp(final_score),
            "score_reasoning": _score_reasoning(
                cluster,
                trend_score,
                audience_fit_score,
                channel_fit_score,
                novelty_score,
                production_ease_score,
                hook_potential_score,
                monetization_score,
            ),
        }

        if hasattr(cluster, "model_copy"):
            scored_clusters.append(cluster.model_copy(update=update_payload))
        else:
            scored_clusters.append(cluster.copy(update=update_payload))

    return sorted(scored_clusters, key=lambda item: item.final_score, reverse=True)
