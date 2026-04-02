from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


class RawSignal(BaseModel):
    source: Literal["youtube", "reddit", "channel", "google_trends", "tiktok"]
    source_id: str
    title: str
    text: str
    url: str
    published_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TopicCluster(BaseModel):
    canonical_topic: str
    aliases: List[str] = Field(default_factory=list)
    source_count: int = 0
    source_breakdown: Dict[str, int] = Field(default_factory=dict)
    channel_source_count: int = 0
    external_source_count: int = 0
    recent_signal_count: int = 0
    external_recent_signal_count: int = 0
    avg_engagement_score: float = 0.0
    external_avg_engagement_score: float = 0.0
    trend_velocity: float = 0.0
    external_trend_velocity: float = 0.0
    audiences: List[str] = Field(default_factory=list)
    emotion_triggers: List[str] = Field(default_factory=list)
    freshness_score: float = 0.0
    external_freshness_score: float = 0.0
    keywords: List[str] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list)
    pain_points: List[str] = Field(default_factory=list)
    supporting_signal_ids: List[str] = Field(default_factory=list)
    trend_score: float = 0.0
    audience_fit_score: float = 0.0
    channel_fit_score: float = 0.0
    novelty_score: float = 0.0
    production_ease_score: float = 0.0
    hook_potential_score: float = 0.0
    monetization_score: float = 0.0
    final_score: float = 0.0
    score_reasoning: Dict[str, str] = Field(default_factory=dict)


class RankedIdea(BaseModel):
    rank: int
    canonical_topic: str
    final_score: float
    title: str
    target_audience: str
    why_now: List[str] = Field(default_factory=list)
    hooks: List[str] = Field(default_factory=list)
    angle: str
    risks: List[str] = Field(default_factory=list)
    next_action: str
    format: Literal["short", "long"]
    source_count: int
    scores: Dict[str, float] = Field(default_factory=dict)
    supporting_keywords: List[str] = Field(default_factory=list)
    audiences: List[str] = Field(default_factory=list)
    emotion_triggers: List[str] = Field(default_factory=list)
    reasoning_summary: str = ""


def model_dump_compat(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
