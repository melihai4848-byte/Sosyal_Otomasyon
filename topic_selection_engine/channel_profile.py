from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

from moduller.logger import get_logger
from moduller.output_paths import grouped_json_output_path
from topic_selection_engine.llm.helpers import build_topic_llm, call_topic_llm_json, resolve_topic_llm_config
from topic_selection_engine.models import RawSignal

logger = get_logger("topic_channel_profile")
CHANNEL_PROFILE_CACHE_PATH = grouped_json_output_path("research", "Topic_Channel_Profile_Cache.json")
CHANNEL_PROFILE_CACHE_TTL_SECONDS = max(0, int(os.getenv("TOPIC_CHANNEL_PROFILE_CACHE_TTL_SECONDS", "43200") or "43200"))

TOKEN_STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "your", "into", "about",
    "video", "videos", "youtube", "kanal", "channel", "official", "welcome", "new",
    "how", "what", "why", "bir", "ve", "ile", "icin", "ama", "gibi", "daha", "cok",
    "son", "guncel", "guide", "tips", "tip", "tr", "de", "en", "www", "com",
    "neden", "nasil", "geri", "mi", "midir", "hangi",
}

AUDIENCE_HINTS = {
    "beginners": ["beginner", "newbie", "sifirdan", "yeni baslayan", "ilk kez"],
    "students": ["student", "ogrenci", "uni", "universite", "campus", "exam"],
    "professionals": ["career", "professional", "is hayati", "kariyer", "job", "work"],
    "entrepreneurs": ["startup", "founder", "girisim", "business", "sirket"],
    "creators": ["youtube", "content creator", "icerik uretici", "filmmaking", "editing"],
    "families": ["family", "aile", "cocuk", "parent", "ebeveyn"],
    "travelers": ["travel", "seyahat", "gez", "trip", "tatil"],
    "buyers": ["review", "inceleme", "comparison", "karsilastirma", "best", "price"],
    "immigrants": ["expat", "immigrant", "goc", "abroad", "move country"],
    "general_audience": [],
}

MONETIZABLE_HINTS = {
    "job", "career", "salary", "income", "maas", "work", "business", "startup",
    "freelance", "tool", "software", "course", "education", "marketing", "investment",
}
EASY_TOPIC_HINTS = {
    "guide", "tips", "tool", "setup", "workflow", "checklist", "mistakes",
    "rehber", "ipuclari", "liste", "kurulum", "neden", "nasil",
}
COMPARE_HINTS = ["vs", "compare", "comparison", "fark", "karsilastirma", "hangisi", "mi yoksa"]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

    text = str(value or "").strip()
    if not text:
        return _now_utc()

    for parser in (
        lambda v: datetime.fromisoformat(v.replace("Z", "+00:00")),
        lambda v: datetime.fromtimestamp(float(v), tz=timezone.utc),
    ):
        try:
            parsed = parser(text)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return _now_utc()


def _safe_int(value: Any) -> int:
    try:
        cleaned = str(value or "").replace(".", "").replace(",", "").strip()
        return int(float(cleaned))
    except Exception:
        return 0


def _dedupe_keep_order(items: List[str], max_items: int = 20) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        cleaned = str(item or "").strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if len(result) >= max_items:
            break
    return result


def _sanitize_text(text: str) -> str:
    cleaned = re.sub(r"http\S+", " ", str(text or ""), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_keywords(texts: List[str], max_items: int = 12) -> List[str]:
    counts: Dict[str, int] = {}
    for text in texts:
        for token in re.findall(r"[A-Za-zÀ-ÿ0-9çğıöşüÇĞİÖŞÜ#\-/]{3,}", str(text or "")):
            lowered = token.lower().strip("#").strip("-/ ")
            if not lowered or lowered in TOKEN_STOPWORDS:
                continue
            counts[lowered] = counts.get(lowered, 0) + 1

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ranked[:max_items]]


def _normalize_keyword_list(items: List[str], max_items: int = 12) -> List[str]:
    normalized: List[str] = []
    for item in items:
        cleaned = re.sub(r"[^A-Za-zÀ-ÿ0-9çğıöşüÇĞİÖŞÜ#\-/ ]+", " ", str(item or ""))
        cleaned = re.sub(r"\s+", " ", cleaned).strip().strip("#").lower()
        if len(cleaned) < 3 or cleaned in TOKEN_STOPWORDS:
            continue
        normalized.append(cleaned)
    return _dedupe_keep_order(normalized, max_items=max_items)


def _extract_audiences(text_blob: str) -> List[str]:
    lower = str(text_blob or "").lower()
    found = [
        audience
        for audience, patterns in AUDIENCE_HINTS.items()
        if patterns and any(pattern in lower for pattern in patterns)
    ]
    return found or ["general_audience"]


def _extract_selector_from_url(url: str) -> dict:
    text = str(url or "").strip()
    if not text:
        return {}

    patterns = [
        ("id", r"/channel/([A-Za-z0-9_-]+)"),
        ("handle", r"/@([A-Za-z0-9._-]+)"),
        ("username", r"/user/([A-Za-z0-9._-]+)"),
        ("query", r"/c/([A-Za-z0-9._-]+)"),
    ]
    for key, pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return {key: match.group(1).strip()}
    return {"query": text}


def _channel_selectors_from_env() -> dict:
    selectors = {
        "id": os.getenv("YOUTUBE_CHANNEL_ID", "").strip(),
        "handle": os.getenv("YOUTUBE_CHANNEL_HANDLE", "").strip().lstrip("@"),
        "username": os.getenv("YOUTUBE_CHANNEL_USERNAME", "").strip(),
        "query": "",
    }
    url_data = _extract_selector_from_url(os.getenv("YOUTUBE_CHANNEL_URL", "").strip())
    for key, value in url_data.items():
        if value and not selectors.get(key):
            selectors[key] = value
    return selectors


def _google_api_error_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
        error = payload.get("error", {})
        message = str(error.get("message", "")).strip()
        errors = error.get("errors", [])
        reasons = ", ".join(
            str(item.get("reason", "")).strip()
            for item in errors
            if str(item.get("reason", "")).strip()
        )
        detail = " | ".join(part for part in [message, reasons] if part)
        return detail or response.text
    except Exception:
        return response.text


def _request_json(endpoint: str, params: dict) -> dict:
    response = requests.get(endpoint, params=params, timeout=20)
    if response.status_code >= 400:
        detail = _google_api_error_detail(response)
        raise RuntimeError(f"HTTP {response.status_code} | {detail}")
    return response.json()


def _channels_request(api_key: str, params: dict) -> dict:
    return _request_json(
        "https://www.googleapis.com/youtube/v3/channels",
        {
            "part": "snippet,statistics,contentDetails,brandingSettings",
            "key": api_key,
            **params,
        },
    )


def _search_channel(api_key: str, query: str) -> Optional[str]:
    if not query:
        return None
    payload = _request_json(
        "https://www.googleapis.com/youtube/v3/search",
        {
            "part": "snippet",
            "type": "channel",
            "q": query,
            "maxResults": 5,
            "key": api_key,
        },
    )
    for item in payload.get("items", []):
        channel_id = str(item.get("id", {}).get("channelId", "")).strip()
        if channel_id:
            return channel_id
    return None


def _resolve_channel_payload(api_key: str) -> Tuple[Optional[dict], List[str]]:
    notes: List[str] = []
    selectors = _channel_selectors_from_env()

    if not api_key:
        return None, ["YOUTUBE_API_KEY tanimli degil; kanal profili otomatik cikarilamadi."]

    if not any(selectors.values()):
        return None, [
            "API key tek basina hedef kanali belirlemez. "
            "En pratik yontem olarak YOUTUBE_CHANNEL_URL tanimla. "
            "Istersen alternatif olarak YOUTUBE_CHANNEL_ID, YOUTUBE_CHANNEL_HANDLE veya YOUTUBE_CHANNEL_USERNAME da kullanabilirsin."
        ]

    request_attempts = []
    if selectors.get("id"):
        request_attempts.append(("id", {"id": selectors["id"]}))
    if selectors.get("handle"):
        request_attempts.append(("handle", {"forHandle": selectors["handle"]}))
    if selectors.get("username"):
        request_attempts.append(("username", {"forUsername": selectors["username"]}))

    for label, params in request_attempts:
        try:
            payload = _channels_request(api_key, params)
            items = payload.get("items", [])
            if items:
                return items[0], notes
        except Exception as exc:
            notes.append(f"Kanal cozumu basarisiz oldu ({label}): {exc}")

    query = selectors.get("query") or selectors.get("handle") or selectors.get("username")
    try:
        channel_id = _search_channel(api_key, query or "")
        if channel_id:
            payload = _channels_request(api_key, {"id": channel_id})
            items = payload.get("items", [])
            if items:
                return items[0], notes
    except Exception as exc:
        notes.append(f"Kanal arama fallback'i basarisiz oldu: {exc}")

    notes.append("Hedef kanal bulunamadi; dinamik kanal profili olusturulamadi.")
    return None, notes


def _fetch_recent_channel_videos(api_key: str, channel_payload: dict, limit: int = 12) -> Tuple[List[dict], List[RawSignal], List[str]]:
    notes: List[str] = []
    uploads_playlist_id = (
        channel_payload.get("contentDetails", {})
        .get("relatedPlaylists", {})
        .get("uploads", "")
    )
    if not uploads_playlist_id:
        return [], [], ["Kanal uploads playlist bilgisi bulunamadi."]

    try:
        playlist_payload = _request_json(
            "https://www.googleapis.com/youtube/v3/playlistItems",
            {
                "part": "snippet,contentDetails",
                "playlistId": uploads_playlist_id,
                "maxResults": limit,
                "key": api_key,
            },
        )
    except Exception as exc:
        return [], [], [f"Son videolar cekilemedi: {exc}"]

    ordered_video_ids = []
    for item in playlist_payload.get("items", []):
        video_id = str(item.get("contentDetails", {}).get("videoId", "")).strip()
        if video_id:
            ordered_video_ids.append(video_id)
    if not ordered_video_ids:
        return [], [], ["Kanal icin islenebilir son video bulunamadi."]

    try:
        videos_payload = _request_json(
            "https://www.googleapis.com/youtube/v3/videos",
            {
                "part": "snippet,statistics",
                "id": ",".join(ordered_video_ids),
                "key": api_key,
            },
        )
    except Exception as exc:
        return [], [], [f"Video detaylari cekilemedi: {exc}"]

    video_map = {str(item.get("id", "")).strip(): item for item in videos_payload.get("items", [])}
    ordered_videos = [video_map[video_id] for video_id in ordered_video_ids if video_id in video_map]

    signals: List[RawSignal] = []
    channel_title = str(channel_payload.get("snippet", {}).get("title", "")).strip()
    for item in ordered_videos:
        snippet = item.get("snippet", {})
        statistics = item.get("statistics", {})
        video_id = str(item.get("id", "")).strip()
        title = _sanitize_text(snippet.get("title", ""))
        description = _sanitize_text(snippet.get("description", ""))
        if not title and not description:
            continue
        signals.append(
            RawSignal(
                source="channel",
                source_id=f"channel-{video_id}",
                title=title or description[:120],
                text=description or title,
                url=f"https://www.youtube.com/watch?v={video_id}",
                published_at=_parse_datetime(snippet.get("publishedAt")),
                metadata={
                    "views": _safe_int(statistics.get("viewCount")),
                    "likes": _safe_int(statistics.get("likeCount")),
                    "comments": _safe_int(statistics.get("commentCount")),
                    "channel_title": channel_title,
                },
            )
        )
    return ordered_videos, signals, notes


def _top_keywords_from_videos(video_signals: List[RawSignal], max_items: int = 10) -> List[str]:
    ranked = sorted(video_signals, key=lambda item: -_safe_int(item.metadata.get("views", 0)))
    texts = [f"{signal.title} {signal.text}" for signal in ranked[:8]]
    return _extract_keywords(texts, max_items=max_items)


def _infer_niche(channel_name: str, description: str, video_signals: List[RawSignal], keywords: List[str], audiences: List[str]) -> str:
    focus = ", ".join(keywords[:5]) or "tekrar eden konu basliklari"
    recent_examples = ", ".join(signal.title for signal in video_signals[:3] if signal.title)
    audience_text = ", ".join(audiences[:3]).replace("_", " ")
    summary = f"{channel_name} kanali agirlikli olarak {focus} konularina odaklaniyor."
    if audience_text:
        summary += f" Olasi hedef kitle: {audience_text}."
    if recent_examples:
        summary += f" Son videolarda öne cikan basliklar: {recent_examples}."
    if description:
        summary += f" Kanal aciklamasi da bu odaği destekliyor."
    return summary.strip()


def _build_channel_profile_prompt(
    channel_name: str,
    channel_description: str,
    recent_video_signals: List[RawSignal],
) -> str:
    recent_videos = []
    for signal in recent_video_signals[:12]:
        recent_videos.append(
            {
                "title": signal.title,
                "description_excerpt": signal.text[:280],
                "views": _safe_int(signal.metadata.get("views", 0)),
            }
        )

    payload = {
        "channel_name": channel_name,
        "channel_description": channel_description,
        "recent_videos": recent_videos,
    }

    return f"""
Sen kıdemli bir YouTube kanal stratejisti ve içerik pozisyonlama uzmanısın.

Görevin:
Aşağıdaki kanal verilerini analiz et ve kanalın gerçek nişini, güçlü içerik alanlarını ve izleyici odağını temiz şekilde çıkar.

KRİTİK KURALLAR:
- Sadece verilen kanal açıklaması ve son videolardan çıkarım yap.
- Uydurma bilgi ekleme.
- Gereksiz stopword veya anlamsız token üretme. Örneğin "dan", "neden", "geri", "gibi" gibi kelimeleri keyword yapma.
- niche_summary_tr alanı doğal, kısa ve net Türkçe olsun.
- Tüm keyword listeleri kısa, kullanılabilir ve tekrar etmeyen ifadelerden oluşsun.
- target_audiences alanında snake_case kullanabilirsin.
- Sadece geçerli JSON döndür.

JSON ŞEMASI:
{{
  "niche_summary_tr": "",
  "target_audiences": ["", ""],
  "niche_keywords": ["", "", ""],
  "channel_strength_topics": ["", "", ""],
  "easy_topic_keywords": ["", "", ""],
  "monetizable_keywords": ["", "", ""],
  "compare_keywords": ["", "", ""]
}}

KANAL VERİSİ:
{json.dumps(payload, ensure_ascii=False)}
""".strip()


def _channel_profile_cache_key(settings, channel_name: str, channel_description: str, recent_video_signals: List[RawSignal]) -> str:
    provider, model_name = resolve_topic_llm_config(settings)
    payload = {
        "provider": provider or "",
        "model_name": model_name or "",
        "channel_name": channel_name,
        "channel_description": channel_description[:400],
        "recent_videos": [
            {
                "title": signal.title,
                "views": _safe_int(signal.metadata.get("views", 0)),
            }
            for signal in recent_video_signals[:8]
        ],
    }
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _load_channel_profile_llm_cache(cache_key: str) -> Optional[dict]:
    if CHANNEL_PROFILE_CACHE_TTL_SECONDS <= 0 or not CHANNEL_PROFILE_CACHE_PATH.exists():
        return None
    try:
        payload = json.loads(CHANNEL_PROFILE_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Kanal profil cache okunamadi: {exc}")
        return None

    entries = payload.get("entries", {})
    if not isinstance(entries, dict):
        return None
    entry = entries.get(cache_key)
    if not isinstance(entry, dict):
        return None

    cached_at = _parse_datetime(entry.get("cached_at"))
    age_seconds = max((_now_utc() - cached_at).total_seconds(), 0.0)
    if age_seconds > CHANNEL_PROFILE_CACHE_TTL_SECONDS:
        return None

    profile = entry.get("profile")
    if not isinstance(profile, dict):
        return None
    return profile


def _save_channel_profile_llm_cache(cache_key: str, profile: dict) -> None:
    if CHANNEL_PROFILE_CACHE_TTL_SECONDS <= 0 or not isinstance(profile, dict):
        return

    payload = {"entries": {}}
    if CHANNEL_PROFILE_CACHE_PATH.exists():
        try:
            loaded = json.loads(CHANNEL_PROFILE_CACHE_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except Exception:
            payload = {"entries": {}}

    entries = payload.get("entries", {})
    if not isinstance(entries, dict):
        entries = {}

    entries[cache_key] = {
        "cached_at": _now_utc().isoformat(),
        "profile": profile,
    }

    # Tek dosyada sınırlı sayıda cache tut.
    ordered_keys = sorted(
        entries.keys(),
        key=lambda key: _parse_datetime((entries.get(key) or {}).get("cached_at")).timestamp(),
        reverse=True,
    )
    payload["entries"] = {key: entries[key] for key in ordered_keys[:8]}
    CHANNEL_PROFILE_CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _infer_channel_profile_with_llm(settings, channel_name: str, channel_description: str, recent_video_signals: List[RawSignal]) -> Optional[dict]:
    llm, llm_info = build_topic_llm(settings)
    if llm is None:
        return None

    payload = call_topic_llm_json(
        llm,
        _build_channel_profile_prompt(channel_name, channel_description, recent_video_signals),
        profile="analytic_json",
        logger_override=logger,
        retries=2,
    )
    if not isinstance(payload, dict):
        return None

    return {
        "llm_info": llm_info,
        "niche_summary_tr": _sanitize_text(payload.get("niche_summary_tr", "")),
        "target_audiences": _dedupe_keep_order(
            [str(item).strip().lower() for item in payload.get("target_audiences", []) if str(item).strip()],
            max_items=6,
        ),
        "niche_keywords": _normalize_keyword_list(payload.get("niche_keywords", []), max_items=12),
        "channel_strength_topics": _normalize_keyword_list(payload.get("channel_strength_topics", []), max_items=8),
        "easy_topic_keywords": _normalize_keyword_list(payload.get("easy_topic_keywords", []), max_items=6),
        "monetizable_keywords": _normalize_keyword_list(payload.get("monetizable_keywords", []), max_items=6),
        "compare_keywords": _normalize_keyword_list(payload.get("compare_keywords", []), max_items=6),
    }


def build_dynamic_channel_profile(settings, recent_video_limit: int = 12) -> dict:
    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    channel_payload, notes = _resolve_channel_payload(api_key)
    if not channel_payload:
        return {
            "channel_name": "",
            "channel_id": "",
            "channel_description": "",
            "inferred_niche": "",
            "target_audiences": [],
            "niche_keywords": [],
            "channel_strength_topics": [],
            "easy_topic_keywords": [],
            "monetizable_keywords": [],
            "compare_keywords": [],
            "recent_video_titles": [],
            "recent_video_signals": [],
            "llm_info": {"enabled": False, "provider": "", "model_name": ""},
            "status_messages": [],
            "notes": notes,
        }

    snippet = channel_payload.get("snippet", {})
    branding = channel_payload.get("brandingSettings", {}).get("channel", {})
    channel_name = _sanitize_text(snippet.get("title", ""))
    channel_description = _sanitize_text(branding.get("description") or snippet.get("description", ""))

    recent_videos, recent_video_signals, video_notes = _fetch_recent_channel_videos(
        api_key,
        channel_payload,
        limit=recent_video_limit,
    )
    notes.extend(video_notes)

    recent_titles = [signal.title for signal in recent_video_signals if signal.title]
    text_blob_parts = [channel_name, channel_description] + [f"{signal.title} {signal.text}" for signal in recent_video_signals]
    text_blob = " ".join(part for part in text_blob_parts if part)
    niche_keywords = _dedupe_keep_order(
        _extract_keywords([text_blob] + recent_titles, max_items=12),
        max_items=12,
    )
    target_audiences = _extract_audiences(text_blob)
    channel_strength_topics = _top_keywords_from_videos(recent_video_signals, max_items=8)
    easy_topic_keywords = [keyword for keyword in niche_keywords if keyword in EASY_TOPIC_HINTS][:6] or niche_keywords[:6]
    monetizable_keywords = [keyword for keyword in niche_keywords if keyword in MONETIZABLE_HINTS][:6]
    compare_keywords = [token for token in COMPARE_HINTS if token in text_blob.lower()]
    inferred_niche = _infer_niche(channel_name, channel_description, recent_video_signals, niche_keywords, target_audiences)
    llm_info = {"enabled": False, "provider": "", "model_name": ""}

    cache_key = _channel_profile_cache_key(settings, channel_name, channel_description, recent_video_signals)
    llm_profile = _load_channel_profile_llm_cache(cache_key)
    if llm_profile:
        notes.append("Kanal nishi icin LLM cache kullanildi.")
    else:
        llm_profile = _infer_channel_profile_with_llm(settings, channel_name, channel_description, recent_video_signals)
        if llm_profile:
            _save_channel_profile_llm_cache(cache_key, llm_profile)
    if llm_profile:
        llm_info = dict(llm_profile.get("llm_info", llm_info))
        inferred_niche = llm_profile.get("niche_summary_tr") or inferred_niche
        target_audiences = _dedupe_keep_order(
            list(llm_profile.get("target_audiences", [])) + target_audiences,
            max_items=6,
        )
        niche_keywords = _dedupe_keep_order(
            list(llm_profile.get("niche_keywords", [])) + niche_keywords,
            max_items=12,
        )
        channel_strength_topics = _dedupe_keep_order(
            list(llm_profile.get("channel_strength_topics", [])) + channel_strength_topics,
            max_items=8,
        )
        easy_topic_keywords = _dedupe_keep_order(
            list(llm_profile.get("easy_topic_keywords", [])) + easy_topic_keywords,
            max_items=6,
        )
        monetizable_keywords = _dedupe_keep_order(
            list(llm_profile.get("monetizable_keywords", [])) + monetizable_keywords,
            max_items=6,
        )
        compare_keywords = _dedupe_keep_order(
            list(llm_profile.get("compare_keywords", [])) + compare_keywords,
            max_items=6,
        )
    else:
        notes.append("Kanal nishi icin LLM yorumu alinamadi; heuristic fallback kullanildi.")

    niche_keywords = _normalize_keyword_list(niche_keywords, max_items=12) or niche_keywords[:12]
    channel_strength_topics = _normalize_keyword_list(channel_strength_topics, max_items=8) or niche_keywords[:8]
    easy_topic_keywords = _normalize_keyword_list(easy_topic_keywords, max_items=6) or niche_keywords[:6]
    monetizable_keywords = _normalize_keyword_list(monetizable_keywords, max_items=6)
    compare_keywords = _normalize_keyword_list(compare_keywords, max_items=6)

    status_messages = [
        "YOUTUBE_CHANNEL_URL cozuldu.",
        "YOUTUBE_API_KEY ile kanala erisim saglandi.",
        f"Kanal bulundu: {channel_name or 'Bilinmeyen kanal'}",
    ]
    if llm_info.get("enabled"):
        status_messages.append(
            f"YARATICI YAPAY ZEKA kanal nishini yorumladi: {llm_info.get('provider')} / {llm_info.get('model_name')}"
        )
    if recent_video_signals:
        status_messages.append(f"Son {len(recent_video_signals)} video analiz icin cekildi.")

    return {
        "channel_name": channel_name,
        "channel_id": str(channel_payload.get("id", "")).strip(),
        "channel_description": channel_description,
        "inferred_niche": inferred_niche,
        "target_audiences": target_audiences,
        "niche_keywords": niche_keywords,
        "channel_strength_topics": channel_strength_topics or niche_keywords[:6],
        "easy_topic_keywords": easy_topic_keywords,
        "monetizable_keywords": monetizable_keywords or niche_keywords[:4],
        "compare_keywords": compare_keywords or COMPARE_HINTS[:4],
        "recent_video_titles": recent_titles[:recent_video_limit],
        "recent_video_signals": recent_video_signals,
        "llm_info": llm_info,
        "status_messages": status_messages,
        "notes": _dedupe_keep_order(notes, max_items=10),
    }
