from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml
from pydantic import BaseModel, Field


class SourceSettings(BaseModel):
    enabled: bool = True
    use_mock: bool = True
    input_paths: List[str] = Field(default_factory=list)
    title_field: str = "title"
    text_fields: List[str] = Field(default_factory=lambda: ["text", "body", "comment", "comments"])
    date_field: str = "published_at"
    url_field: str = "url"
    id_field: str = "id"
    metadata_fields: List[str] = Field(default_factory=list)


class LLMSettings(BaseModel):
    enabled: bool = True
    provider: str = "AUTO"
    model_name: str = ""
    max_ideas: int = 10


class RankingWeights(BaseModel):
    trend: float = 0.25
    audience_fit: float = 0.20
    channel_fit: float = 0.20
    novelty: float = 0.10
    production_ease: float = 0.10
    hook_potential: float = 0.10
    monetization: float = 0.05


class EngineSettings(BaseModel):
    niche: str
    target_audiences: List[str] = Field(default_factory=list)
    output_dir: str = "./out"
    sources: Dict[str, SourceSettings] = Field(default_factory=dict)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    ranking_weights: RankingWeights = Field(default_factory=RankingWeights)
    scoring_threshold: float = 40.0
    freshness_half_life_days: int = 14
    top_n: int = 10
    niche_keywords: List[str] = Field(default_factory=list)
    channel_strength_topics: List[str] = Field(default_factory=list)
    easy_topic_keywords: List[str] = Field(default_factory=list)
    monetizable_keywords: List[str] = Field(default_factory=list)
    compare_keywords: List[str] = Field(default_factory=list)


def load_settings(config_path: str | Path | None = None) -> EngineSettings:
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "default.yaml"
    else:
        config_path = Path(config_path)

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return EngineSettings(**raw)
