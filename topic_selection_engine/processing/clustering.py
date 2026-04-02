from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Iterable, List

from topic_selection_engine.models import RawSignal, TopicCluster
from topic_selection_engine.processing.extraction import extract_signal_features


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        normalized = (item or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _freshness_score(published_times: List[datetime], half_life_days: int) -> float:
    if not published_times:
        return 0.0

    now = datetime.now(timezone.utc)
    scores = []
    for item in published_times:
        delta_days = max((now - item).days, 0)
        score = max(0.0, 100.0 - (delta_days / max(half_life_days, 1)) * 50.0)
        scores.append(score)
    return round(sum(scores) / len(scores), 2)


def _engagement_score(metadata: Dict[str, object]) -> float:
    numeric_keys = ["views", "comments", "comment_count", "comments_count", "likes", "upvotes", "ctr", "avg_view_duration", "awards"]
    total = 0.0
    for key in numeric_keys:
        value = metadata.get(key, 0)
        try:
            total += float(value)
        except Exception:
            continue
    return total


def _trend_velocity(published_times: List[datetime], half_life_days: int) -> tuple[int, float]:
    if not published_times:
        return 0, 0.0

    now = datetime.now(timezone.utc)
    recent_count = 0
    for item in published_times:
        delta_days = max((now - item).days, 0)
        if delta_days <= half_life_days:
            recent_count += 1

    velocity = min(100.0, recent_count * 20.0)
    return recent_count, round(velocity, 2)


def build_topic_clusters(signals: List[RawSignal], freshness_half_life_days: int) -> List[TopicCluster]:
    grouped: Dict[str, Dict[str, object]] = defaultdict(lambda: {
        "aliases": [],
        "audiences": [],
        "emotion_triggers": [],
        "keywords": [],
        "questions": [],
        "pain_points": [],
        "source_breakdown": defaultdict(int),
        "supporting_signal_ids": [],
        "published_times": [],
        "external_published_times": [],
        "engagement_scores": [],
        "external_engagement_scores": [],
    })

    for signal in signals:
        features = extract_signal_features(signal)
        for canonical_topic in features["canonical_topics"]:
            group = grouped[canonical_topic]
            group["aliases"].extend(features["aliases"])
            group["audiences"].extend(features["audiences"])
            group["emotion_triggers"].extend(features["emotion_triggers"])
            group["keywords"].extend(features["keywords"])
            group["questions"].extend(features["questions"])
            group["pain_points"].extend(features["pain_points"])
            group["source_breakdown"][signal.source] += 1
            group["supporting_signal_ids"].append(signal.source_id)
            group["published_times"].append(signal.published_at)
            engagement_score = _engagement_score(signal.metadata)
            group["engagement_scores"].append(engagement_score)
            if signal.source != "channel":
                group["external_published_times"].append(signal.published_at)
                group["external_engagement_scores"].append(engagement_score)

    clusters: List[TopicCluster] = []
    for canonical_topic, data in grouped.items():
        source_breakdown = dict(data["source_breakdown"])
        recent_signal_count, trend_velocity = _trend_velocity(data["published_times"], freshness_half_life_days)
        external_recent_signal_count, external_trend_velocity = _trend_velocity(
            data["external_published_times"],
            freshness_half_life_days,
        )
        engagement_scores = data["engagement_scores"] or [0.0]
        external_engagement_scores = data["external_engagement_scores"] or [0.0]
        channel_source_count = int(source_breakdown.get("channel", 0))
        external_source_count = max(sum(source_breakdown.values()) - channel_source_count, 0)
        clusters.append(
            TopicCluster(
                canonical_topic=canonical_topic,
                aliases=_dedupe_keep_order(data["aliases"])[:8],
                source_count=sum(source_breakdown.values()),
                source_breakdown=source_breakdown,
                channel_source_count=channel_source_count,
                external_source_count=external_source_count,
                recent_signal_count=recent_signal_count,
                external_recent_signal_count=external_recent_signal_count,
                avg_engagement_score=round(sum(engagement_scores) / len(engagement_scores), 2),
                external_avg_engagement_score=round(sum(external_engagement_scores) / len(external_engagement_scores), 2),
                trend_velocity=trend_velocity,
                external_trend_velocity=external_trend_velocity,
                audiences=_dedupe_keep_order(data["audiences"])[:6],
                emotion_triggers=_dedupe_keep_order(data["emotion_triggers"])[:6],
                freshness_score=_freshness_score(data["published_times"], freshness_half_life_days),
                external_freshness_score=_freshness_score(data["external_published_times"], freshness_half_life_days),
                keywords=_dedupe_keep_order(data["keywords"])[:10],
                questions=_dedupe_keep_order(data["questions"])[:5],
                pain_points=_dedupe_keep_order(data["pain_points"])[:8],
                supporting_signal_ids=_dedupe_keep_order(data["supporting_signal_ids"]),
            )
        )

    return clusters
