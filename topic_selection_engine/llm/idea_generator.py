from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from moduller.logger import get_logger
from topic_selection_engine.config import EngineSettings
from topic_selection_engine.llm.helpers import build_topic_llm, call_topic_llm_json
from topic_selection_engine.models import RankedIdea, TopicCluster

logger = get_logger("topic_idea_generator")
MIN_TOPIC_IDEA_COUNT = 5

FORMAT_HINTS = {
    "almanya_yasam_maliyeti": "long",
    "almanya_kira_ve_ev": "long",
    "almanya_burokrasi": "long",
    "almanya_is_ve_almanca": "long",
    "almanya_ogrenci_hayati": "short",
    "almanya_genel_yasam": "long",
}


class IdeaGenerator:
    def __init__(
        self,
        settings: EngineSettings,
        channel_profile: Optional[dict] = None,
        live_trend_data: Optional[dict] = None,
    ):
        self.settings = settings
        self.channel_profile = channel_profile or {}
        self.live_trend_data = live_trend_data or {}
        self._llm = None
        self._llm_info = {"enabled": False, "provider": "", "model_name": ""}

    @property
    def llm_info(self) -> dict:
        return dict(self._llm_info)

    def _get_llm(self):
        if self._llm is not None:
            return self._llm

        llm, llm_info = build_topic_llm(self.settings)
        self._llm = llm
        self._llm_info = llm_info
        return self._llm

    def _normalize_list(self, values, limit: int, fallback: Optional[List[str]] = None) -> List[str]:
        items = [str(item).strip() for item in (values or []) if str(item).strip()]
        items = list(dict.fromkeys(items))
        if items:
            return items[:limit]
        return list(fallback or [])[:limit]

    def _recent_channel_titles(self) -> List[str]:
        return [str(item).strip() for item in self.channel_profile.get("recent_video_titles", []) if str(item).strip()]

    def _normalize_title_for_compare(self, title: str) -> str:
        normalized = re.sub(r"[^a-z0-9çğıöşü]+", " ", str(title or "").lower())
        return re.sub(r"\s+", " ", normalized).strip()

    def _title_token_overlap(self, title_a: str, title_b: str) -> float:
        tokens_a = set(self._normalize_title_for_compare(title_a).split())
        tokens_b = set(self._normalize_title_for_compare(title_b).split())
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / max(min(len(tokens_a), len(tokens_b)), 1)

    def _is_too_similar_to_recent_titles(self, title: str) -> bool:
        candidate = self._normalize_title_for_compare(title)
        if not candidate:
            return False

        for recent_title in self._recent_channel_titles():
            normalized_recent = self._normalize_title_for_compare(recent_title)
            if not normalized_recent:
                continue
            if candidate == normalized_recent:
                return True
            if len(candidate.split()) >= 5 and (candidate in normalized_recent or normalized_recent in candidate):
                return True
            if self._title_token_overlap(candidate, normalized_recent) >= 0.78:
                return True
        return False

    def _distinct_title_template(self, cluster: TopicCluster) -> str:
        templates = {
            "almanya_yasam_maliyeti": "Almanya'da maas yuksek ama neden para yine yetmiyor?",
            "almanya_kira_ve_ev": "Almanya'da ev bulmak neden daha da zorlasti?",
            "almanya_burokrasi": "Almanya'da oturum ve evrak isleri neden bu kadar yavasliyor?",
            "almanya_is_ve_almanca": "Almanya'da is bulmakta asil sorun Almanca mi sistem mi?",
            "almanya_ogrenci_hayati": "Almanya'da ogrenci olarak ilk hangi barinma secenegi mantikli?",
            "almanya_genel_yasam": "Almanya'ya gelmeden once hangi gercekleri bilmek gerekiyor?",
        }
        title = templates.get(cluster.canonical_topic)
        if title:
            return title

        topic_label = cluster.canonical_topic.replace("topic_", "").replace("_", " ").strip()
        topic_label = topic_label.title() if topic_label else "Bu Konu"
        return f"{topic_label}: bugun bu konuda gercekte ne degisti?"

    def _finalize_title(self, title: str, cluster: TopicCluster, fallback_title: str) -> str:
        candidate = str(title or "").strip() or fallback_title
        if not self._is_too_similar_to_recent_titles(candidate):
            return candidate

        distinct_fallback = self._distinct_title_template(cluster)
        if not self._is_too_similar_to_recent_titles(distinct_fallback):
            return distinct_fallback
        return fallback_title

    def _coerce_format(self, value: object, cluster: TopicCluster) -> str:
        output_format = str(value or "").strip().lower()
        if output_format not in {"short", "long"}:
            output_format = FORMAT_HINTS.get(cluster.canonical_topic, "long")
        return output_format

    def _channel_context(self) -> dict:
        return {
            "channel_name": self.channel_profile.get("channel_name", ""),
            "inferred_niche": self.channel_profile.get("inferred_niche", ""),
            "channel_description": self.channel_profile.get("channel_description", ""),
            "niche_keywords": self.channel_profile.get("niche_keywords", [])[:12],
            "target_audiences": self.channel_profile.get("target_audiences", [])[:6],
            "recent_video_titles": self.channel_profile.get("recent_video_titles", [])[:10],
            "live_trend_keywords": list(self.live_trend_data.get("top_keywords", []))[:12],
            "live_trend_topics": list(self.live_trend_data.get("viral_topics", []))[:10],
        }

    def _build_prompt(self, cluster: TopicCluster) -> str:
        cluster_payload = {
            "canonical_topic": cluster.canonical_topic,
            "aliases": cluster.aliases,
            "source_count": cluster.source_count,
            "external_source_count": cluster.external_source_count,
            "channel_source_count": cluster.channel_source_count,
            "audiences": cluster.audiences,
            "emotion_triggers": cluster.emotion_triggers,
            "freshness_score": cluster.freshness_score,
            "external_freshness_score": cluster.external_freshness_score,
            "keywords": cluster.keywords,
            "questions": cluster.questions,
            "pain_points": cluster.pain_points,
            "scores": {
                "trend_score": cluster.trend_score,
                "audience_fit_score": cluster.audience_fit_score,
                "channel_fit_score": cluster.channel_fit_score,
                "novelty_score": cluster.novelty_score,
                "production_ease_score": cluster.production_ease_score,
                "hook_potential_score": cluster.hook_potential_score,
                "monetization_score": cluster.monetization_score,
                "final_score": cluster.final_score,
            },
            "score_reasoning": cluster.score_reasoning,
        }
        return f"""
Sen bir YouTube stratejisti ve kanal büyütme editörüsün.

Görevin:
Zaten puanlanmış bir topic cluster'ı bu kanal için uygulanabilir tek bir video fikrine dönüştür.

KRİTİK KURALLAR:
- Sadece verilen cluster'lardan konuş.
- Kanalın gerçek içeriğiyle alakasız açılar önermeyin.
- Başlık, hook ve açı Türkçe olsun.
- recent_video_titles alanındaki başlıklara çok benzeyen, onları tekrar paketleyen veya yarı kopya olan title üretme.
- Sadece geçerli JSON döndür.

JSON ŞEMASI:
{{
  "title": "",
  "target_audience": "",
  "why_now": ["", ""],
  "hooks": ["", "", ""],
  "angle": "",
  "risks": ["", ""],
  "next_action": "",
  "format": "long",
  "reasoning_summary": ""
}}

KANAL BAĞLAMI:
{json.dumps(self._channel_context(), ensure_ascii=False)}

TOPIC CLUSTER:
{json.dumps(cluster_payload, ensure_ascii=False)}
""".strip()

    def _build_ranked_prompt(self, clusters: List[TopicCluster]) -> str:
        candidate_payload = []
        for cluster in clusters:
            candidate_payload.append(
                {
                    "canonical_topic": cluster.canonical_topic,
                    "current_score": cluster.final_score,
                    "source_count": cluster.source_count,
                    "external_source_count": cluster.external_source_count,
                    "channel_source_count": cluster.channel_source_count,
                    "recent_signal_count": cluster.recent_signal_count,
                    "external_recent_signal_count": cluster.external_recent_signal_count,
                    "source_breakdown": cluster.source_breakdown,
                    "audiences": cluster.audiences,
                    "emotion_triggers": cluster.emotion_triggers,
                    "keywords": cluster.keywords[:8],
                    "questions": cluster.questions[:4],
                    "pain_points": cluster.pain_points[:5],
                    "reasoning": {
                        "trend": cluster.score_reasoning.get("trend", ""),
                        "channel_fit": cluster.score_reasoning.get("channel_fit", ""),
                        "novelty": cluster.score_reasoning.get("novelty", ""),
                    },
                }
            )

        return f"""
Sen kıdemli bir YouTube kanal stratejisti, trend yorumcusu ve paketleme editörüsün.

Görevin:
Bir kanal profili ve önceden puanlanmış candidate topic cluster listesi verilecek.
Bu adaylar arasından kanala gerçekten uyan, trend ile kanal kimliğini birleştiren en güçlü video konularını seç ve sırala.

KRİTİK KURALLAR:
- Sadece verilen candidate cluster'lar arasından seçim yap.
- canonical_topic alanı mutlaka listedeki adaylardan biri olsun.
- Kanalla alakasız ama genel olarak popüler görünen adayları ele.
- Aynı mikro-açının tekrarlarına izin verme.
- recent_video_titles ile çok benzer veya neredeyse aynı başlığı tekrar önerme.
- Tüm metin alanları Türkçe olsun.
- Tamamen uydurma bilgi yazma.
- Sadece geçerli JSON döndür.
- En fazla {min(self.settings.top_n, self.settings.llm.max_ideas)} fikir döndür.

JSON ŞEMASI:
{{
  "ranked_ideas": [
    {{
      "canonical_topic": "",
      "title": "",
      "target_audience": "",
      "why_now": ["", ""],
      "hooks": ["", "", ""],
      "angle": "",
      "risks": ["", ""],
      "next_action": "",
      "format": "short",
      "reasoning_summary": ""
    }}
  ]
}}

KANAL BAĞLAMI:
{json.dumps(self._channel_context(), ensure_ascii=False)}

ADAY CLUSTER'LAR:
{json.dumps(candidate_payload, ensure_ascii=False)}
""".strip()

    def _fallback_idea(self, cluster: TopicCluster, rank: int) -> RankedIdea:
        audience = (
            cluster.audiences[0].replace("_", " ")
            if cluster.audiences
            else (
                self.settings.target_audiences[0].replace("_", " ")
                if self.settings.target_audiences
                else "general audience"
            )
        )
        hook_keywords = cluster.keywords[:3] or [cluster.canonical_topic.replace("_", " ")]
        title_map = {
            "almanya_yasam_maliyeti": "Almanya'da 2026'da gecinmek icin kac Euro gerekiyor?",
            "almanya_kira_ve_ev": "Almanya'da ev bulmak neden zorlasti ve nasil avantaj saglanir?",
            "almanya_burokrasi": "Auslanderbehorde beklerken ne yapmali? Gercek adimlar",
            "almanya_is_ve_almanca": "Almanya'da is bulmak icin Almanca seviyesi gercekte ne kadar onemli?",
            "almanya_ogrenci_hayati": "Almanya'da ogrenci icin yurt mu WG mi daha mantikli?",
            "almanya_genel_yasam": "Almanya'ya gelmeden once bilmeniz gereken gizli gercekler",
        }

        trend_signal_count = cluster.external_source_count or cluster.source_count
        trend_freshness = cluster.external_freshness_score or cluster.freshness_score
        why_now = [
            f"Bu konu dis kaynaklarda {trend_signal_count} farkli sinyalde tekrar etti.",
            f"Tazelik skoru {trend_freshness} oldugu icin ilgi hala sicak.",
        ]
        if cluster.pain_points:
            why_now.append(f"Izleyici acisi net: {cluster.pain_points[0]}")

        secondary_hook = (
            f"Almanya hayalindeki en buyuk risk aslinda {hook_keywords[-1]} olabilir."
            if cluster.canonical_topic.startswith("almanya_")
            else f"Bu konuda insanlarin en kritik yanilgisi aslinda {hook_keywords[-1]} olabilir."
        )
        hooks = [
            f"{hook_keywords[0].title()} konusunda insanlar yanlis seye mi odaklaniyor?",
            secondary_hook,
            "Bu karari vermeden once herkesin kacirdigi ayrinti ne?",
        ]

        title = title_map.get(cluster.canonical_topic, "")
        if not title:
            topic_label = cluster.canonical_topic.replace("topic_", "").replace("_", " ").strip()
            topic_label = topic_label.title() if topic_label else "Sonraki Video"
            title = f"{topic_label}: izleyicinin su an en cok merak ettigi sey ne?"

        return RankedIdea(
            rank=rank,
            canonical_topic=cluster.canonical_topic,
            final_score=cluster.final_score,
            title=title,
            target_audience=audience,
            why_now=why_now,
            hooks=hooks[:3],
            angle="Gercek hayattaki karar baskisini, rakam + deneyim + karsilastirma ile anlat.",
            risks=[
                "Veri hizla eskiyebilir; yayin tarihini acikca belirtmek gerekir.",
                "Kisisel deneyim ile genellemeyi dikkatli ayirmak gerekir.",
            ],
            next_action=f"Bu konu icin once {hook_keywords[0]} odakli veri ve ornekler topla, sonra thumbnail acisini test et.",
            format=FORMAT_HINTS.get(cluster.canonical_topic, "long"),
            source_count=cluster.source_count,
            scores={
                "trend_score": cluster.trend_score,
                "audience_fit_score": cluster.audience_fit_score,
                "channel_fit_score": cluster.channel_fit_score,
                "novelty_score": cluster.novelty_score,
                "production_ease_score": cluster.production_ease_score,
                "hook_potential_score": cluster.hook_potential_score,
                "monetization_score": cluster.monetization_score,
            },
            supporting_keywords=cluster.keywords,
            audiences=cluster.audiences,
            emotion_triggers=cluster.emotion_triggers,
            reasoning_summary=(
                f"Trend {cluster.trend_score:.2f}, audience fit {cluster.audience_fit_score:.2f}, "
                f"hook potential {cluster.hook_potential_score:.2f}. "
                f"Ana neden: {cluster.score_reasoning.get('trend', '')}"
            ).strip(),
        )

    def _llm_idea(self, cluster: TopicCluster, rank: int) -> Optional[RankedIdea]:
        llm = self._get_llm()
        if llm is None:
            return None

        payload = call_topic_llm_json(
            llm,
            self._build_prompt(cluster),
            profile="creative_ranker",
            logger_override=logger,
            retries=2,
        )
        if not isinstance(payload, dict):
            return None

        fallback = self._fallback_idea(cluster, rank)
        return RankedIdea(
            rank=rank,
            canonical_topic=cluster.canonical_topic,
            final_score=cluster.final_score,
            title=self._finalize_title(str(payload.get("title", "")).strip(), cluster, fallback.title),
            target_audience=str(payload.get("target_audience", "")).strip() or fallback.target_audience,
            why_now=self._normalize_list(payload.get("why_now", []), 3, fallback=fallback.why_now),
            hooks=self._normalize_list(payload.get("hooks", []), 3, fallback=fallback.hooks),
            angle=str(payload.get("angle", "")).strip() or fallback.angle,
            risks=self._normalize_list(payload.get("risks", []), 3, fallback=fallback.risks),
            next_action=str(payload.get("next_action", "")).strip() or fallback.next_action,
            format=self._coerce_format(payload.get("format", "long"), cluster),
            source_count=cluster.source_count,
            scores={
                "trend_score": cluster.trend_score,
                "audience_fit_score": cluster.audience_fit_score,
                "channel_fit_score": cluster.channel_fit_score,
                "novelty_score": cluster.novelty_score,
                "production_ease_score": cluster.production_ease_score,
                "hook_potential_score": cluster.hook_potential_score,
                "monetization_score": cluster.monetization_score,
            },
            supporting_keywords=cluster.keywords,
            audiences=cluster.audiences,
            emotion_triggers=cluster.emotion_triggers,
            reasoning_summary=str(payload.get("reasoning_summary", "")).strip() or fallback.reasoning_summary,
        )

    def _llm_ranked_ideas(self, clusters: List[TopicCluster]) -> List[RankedIdea]:
        llm = self._get_llm()
        if llm is None or not clusters:
            return []

        payload = call_topic_llm_json(
            llm,
            self._build_ranked_prompt(clusters),
            profile="creative_ranker",
            logger_override=logger,
            retries=2,
        )
        if not isinstance(payload, dict):
            return []

        items = payload.get("ranked_ideas", [])
        if not isinstance(items, list):
            return []

        cluster_map: Dict[str, TopicCluster] = {cluster.canonical_topic: cluster for cluster in clusters}
        ideas: List[RankedIdea] = []
        seen_topics = set()
        max_ideas = min(self.settings.top_n, self.settings.llm.max_ideas)

        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue

            canonical_topic = str(item.get("canonical_topic", "")).strip()
            cluster = cluster_map.get(canonical_topic)
            if not canonical_topic or cluster is None or canonical_topic in seen_topics:
                continue

            seen_topics.add(canonical_topic)
            fallback = self._fallback_idea(cluster, idx)
            ideas.append(
                RankedIdea(
                    rank=len(ideas) + 1,
                    canonical_topic=canonical_topic,
                    final_score=cluster.final_score,
                    title=self._finalize_title(str(item.get("title", "")).strip(), cluster, fallback.title),
                    target_audience=str(item.get("target_audience", "")).strip() or fallback.target_audience,
                    why_now=self._normalize_list(item.get("why_now", []), 3, fallback=fallback.why_now),
                    hooks=self._normalize_list(item.get("hooks", []), 3, fallback=fallback.hooks),
                    angle=str(item.get("angle", "")).strip() or fallback.angle,
                    risks=self._normalize_list(item.get("risks", []), 3, fallback=fallback.risks),
                    next_action=str(item.get("next_action", "")).strip() or fallback.next_action,
                    format=self._coerce_format(item.get("format", "long"), cluster),
                    source_count=cluster.source_count,
                    scores={
                        "trend_score": cluster.trend_score,
                        "audience_fit_score": cluster.audience_fit_score,
                        "channel_fit_score": cluster.channel_fit_score,
                        "novelty_score": cluster.novelty_score,
                        "production_ease_score": cluster.production_ease_score,
                        "hook_potential_score": cluster.hook_potential_score,
                        "monetization_score": cluster.monetization_score,
                    },
                    supporting_keywords=cluster.keywords,
                    audiences=cluster.audiences,
                    emotion_triggers=cluster.emotion_triggers,
                    reasoning_summary=str(item.get("reasoning_summary", "")).strip() or fallback.reasoning_summary,
                )
            )
            if len(ideas) >= max_ideas:
                break

        return ideas

    def _fallback_ranked_ideas(
        self,
        clusters: List[TopicCluster],
        *,
        start_rank: int = 1,
        used_topics: Optional[set[str]] = None,
        limit: Optional[int] = None,
    ) -> List[RankedIdea]:
        ideas: List[RankedIdea] = []
        seen_topics = set(used_topics or set())
        max_items = int(limit or len(clusters) or 0)
        for cluster in clusters:
            if cluster.canonical_topic in seen_topics:
                continue
            rank = start_rank + len(ideas)
            ideas.append(self._fallback_idea(cluster, rank))
            seen_topics.add(cluster.canonical_topic)
            if len(ideas) >= max_items:
                break
        return ideas

    def generate_ranked_ideas(self, clusters: List[TopicCluster]) -> List[RankedIdea]:
        max_ideas = min(self.settings.top_n, self.settings.llm.max_ideas)
        target_idea_count = min(len(clusters), max(max_ideas, MIN_TOPIC_IDEA_COUNT))
        ranked_ideas = self._llm_ranked_ideas(clusters[: max(max_ideas * 2, 12)])
        if ranked_ideas:
            if len(ranked_ideas) < target_idea_count:
                used_topics = {idea.canonical_topic for idea in ranked_ideas}
                ranked_ideas.extend(
                    self._fallback_ranked_ideas(
                        clusters,
                        start_rank=len(ranked_ideas) + 1,
                        used_topics=used_topics,
                        limit=target_idea_count - len(ranked_ideas),
                    )
                )
            return ranked_ideas[:target_idea_count]

        return self._fallback_ranked_ideas(clusters[:target_idea_count], start_rank=1, limit=target_idea_count)
