from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
from xml.etree import ElementTree

import requests

from moduller.logger import get_logger
from topic_selection_engine.models import RawSignal

logger = get_logger("TopicSelector_LiveSources")

COMMON_STOPWORDS = {
    "ve", "ile", "icin", "ama", "çok", "daha", "gibi", "olan", "olarak", "neden",
    "nasil", "what", "when", "where", "that", "this", "from", "your", "about", "into",
    "almanya", "germany", "turkiye", "turkish", "video", "youtube", "tiktok",
}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

    text = str(value or "").strip()
    if not text:
        return _now_utc()

    parsers = (
        lambda v: datetime.fromisoformat(v.replace("Z", "+00:00")),
        lambda v: datetime.strptime(v, "%a, %d %b %Y %H:%M:%S %z"),
        lambda v: datetime.fromtimestamp(float(v), tz=timezone.utc),
    )

    for parser in parsers:
        try:
            parsed = parser(text)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except Exception:
            continue

    return _now_utc()


def _safe_int(value: Any) -> int:
    try:
        cleaned = str(value).replace(".", "").replace(",", "").strip()
        return int(float(cleaned))
    except Exception:
        return 0


def _dedupe_keep_order(items: List[str], max_items: int = 30) -> List[str]:
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


def _keywords_from_texts(texts: List[str], seed_keywords: List[str], max_items: int = 20) -> List[str]:
    base = list(seed_keywords)
    counts: Dict[str, int] = {}
    for text in texts:
        for token in re.findall(r"[A-Za-zÀ-ÿ0-9çğıöşüÇĞİÖŞÜ#\-/]{3,}", str(text or "")):
            lowered = token.lower().strip("#")
            if lowered in COMMON_STOPWORDS:
                continue
            counts[lowered] = counts.get(lowered, 0) + 1

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    base.extend(token for token, _ in ranked[:max_items])
    return _dedupe_keep_order(base, max_items=max_items)


def _youtube_queries(niche_keywords: List[str]) -> List[str]:
    env_queries = os.getenv("YOUTUBE_TREND_SEARCH_QUERIES", "").strip()
    if env_queries:
        return _dedupe_keep_order([item.strip() for item in env_queries.split(",")], max_items=6)

    queries = []
    for keyword in niche_keywords[:6]:
        keyword = str(keyword).strip()
        if keyword:
            queries.append(keyword)
    return _dedupe_keep_order(queries or ["almanya", "germany", "goc"], max_items=6)


def _load_records_from_path(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("items", "data", "results", "rows"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [payload]
        return []

    if suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if isinstance(item, dict):
                rows.append(item)
        return rows

    if suffix == ".csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]

    return []


def _google_api_error_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
        error = payload.get("error", {})
        message = error.get("message", "")
        errors = error.get("errors", [])
        reasons = ", ".join(item.get("reason", "") for item in errors if item.get("reason"))
        detail = " | ".join(part for part in [message, reasons] if part)
        return detail or response.text
    except Exception:
        return response.text


def _fetch_youtube_api_signals(niche_keywords: List[str], since_days: int, limit: int) -> Tuple[List[RawSignal], Dict[str, Any]]:
    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not api_key or limit <= 0:
        return [], {
            "source": "youtube_api",
            "keywords": [],
            "topics": [],
            "notes": ["YOUTUBE_API_KEY tanimli degil; YouTube canli veri atlandi."],
        }

    session = requests.Session()
    region_code = os.getenv("YOUTUBE_REGION_CODE", "DE").strip() or "DE"
    language = os.getenv("YOUTUBE_RELEVANCE_LANGUAGE", "tr").strip() or "tr"
    published_after = (_now_utc() - timedelta(days=since_days)).isoformat().replace("+00:00", "Z")

    signals: List[RawSignal] = []
    topic_titles: List[str] = []
    notes: List[str] = []

    for query in _youtube_queries(niche_keywords):
        if len(signals) >= limit:
            break

        try:
            search_resp = session.get(
                "https://www.googleapis.com/youtube/v3/search",
                params={
                    "part": "snippet",
                    "type": "video",
                    "q": query,
                    "maxResults": min(10, limit - len(signals)),
                    "order": "viewCount",
                    "publishedAfter": published_after,
                    "regionCode": region_code,
                    "relevanceLanguage": language,
                    "key": api_key,
                },
                timeout=20,
            )
            if search_resp.status_code != 200:
                raise RuntimeError(_google_api_error_detail(search_resp))
            search_payload = search_resp.json()
        except Exception as exc:
            notes.append(f"YouTube arama istegi basarisiz oldu ({query}): {exc}")
            continue

        video_ids = [
            item.get("id", {}).get("videoId")
            for item in search_payload.get("items", [])
            if item.get("id", {}).get("videoId")
        ]
        if not video_ids:
            continue

        try:
            videos_resp = session.get(
                "https://www.googleapis.com/youtube/v3/videos",
                params={
                    "part": "snippet,statistics",
                    "id": ",".join(video_ids),
                    "key": api_key,
                },
                timeout=20,
            )
            if videos_resp.status_code != 200:
                raise RuntimeError(_google_api_error_detail(videos_resp))
            videos_payload = videos_resp.json()
        except Exception as exc:
            notes.append(f"YouTube video detaylari alinamadi ({query}): {exc}")
            continue

        for item in videos_payload.get("items", []):
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            video_id = item.get("id", "")
            title = str(snippet.get("title", "")).strip()
            description = str(snippet.get("description", "")).strip()
            if not title and not description:
                continue

            topic_titles.append(title)
            signals.append(
                RawSignal(
                    source="youtube",
                    source_id=f"live-yt-{video_id}",
                    title=title or query,
                    text=description or title,
                    url=f"https://www.youtube.com/watch?v={video_id}",
                    published_at=_parse_datetime(snippet.get("publishedAt")),
                    metadata={
                        "views": _safe_int(stats.get("viewCount")),
                        "likes": _safe_int(stats.get("likeCount")),
                        "comments": _safe_int(stats.get("commentCount")),
                        "query": query,
                        "live_trend_source": "youtube_api",
                    },
                )
            )
            if len(signals) >= limit:
                break

    return signals[:limit], {
        "source": "youtube_api",
        "keywords": _keywords_from_texts(topic_titles, _youtube_queries(niche_keywords), max_items=12),
        "topics": _dedupe_keep_order(topic_titles, max_items=12),
        "notes": notes,
    }


def _fetch_google_trends_signals(niche_keywords: List[str], since_days: int, limit: int) -> Tuple[List[RawSignal], Dict[str, Any]]:
    if limit <= 0:
        return [], {"source": "google_trends", "keywords": [], "topics": [], "notes": []}

    geo = os.getenv("GOOGLE_TRENDS_GEO", "DE").strip() or "DE"
    url = f"https://trends.google.com/trending/rss?geo={geo}"

    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        root = ElementTree.fromstring(resp.content)
    except Exception as exc:
        return [], {
            "source": "google_trends",
            "keywords": [],
            "topics": [],
            "notes": [f"Google Trends RSS alinamadi: {exc}"],
        }

    cutoff = _now_utc() - timedelta(days=since_days)
    keyword_blob = " ".join(niche_keywords).lower()

    signals: List[RawSignal] = []
    fallback_signals: List[RawSignal] = []
    topic_titles: List[str] = []
    fallback_titles: List[str] = []
    raw_items = root.findall(".//item")

    for index, item in enumerate(raw_items, start=1):
        title = (item.findtext("title") or "").strip()
        description = (item.findtext("description") or "").strip()
        pub_date = _parse_datetime(item.findtext("pubDate"))
        if pub_date < cutoff:
            continue

        blob = f"{title} {description}".lower()
        candidate = RawSignal(
            source="google_trends",
            source_id=f"google-trends-{index}",
            title=title,
            text=description or title,
            url="https://trends.google.com/trends/trendingsearches/daily",
            published_at=pub_date,
            metadata={
                "live_trend_source": "google_trends_rss",
                "geo": geo,
            },
        )
        if niche_keywords and keyword_blob and not any(keyword.lower() in blob for keyword in niche_keywords):
            fallback_titles.append(title)
            fallback_signals.append(candidate)
            continue

        topic_titles.append(title)
        signals.append(candidate)
        if len(signals) >= limit:
            break

    notes = []
    if not signals and fallback_signals:
        signals = fallback_signals[:limit]
        topic_titles = fallback_titles[:limit]
        notes.append("Google Trends RSS bulundu; nise birebir eslesme cikmadi ama genel trendlerden fallback kullanildi.")

    return signals, {
        "source": "google_trends",
        "keywords": _keywords_from_texts(topic_titles, niche_keywords, max_items=12),
        "topics": _dedupe_keep_order(topic_titles, max_items=12),
        "notes": notes if signals else ["Google Trends RSS bulundu ama nişe uyan kayit cikmadi."],
    }


def _fetch_tiktok_signals(niche_keywords: List[str], since_days: int, limit: int) -> Tuple[List[RawSignal], Dict[str, Any]]:
    if limit <= 0:
        return [], {"source": "tiktok", "keywords": [], "topics": [], "notes": []}

    notes: List[str] = []
    records: List[Dict[str, Any]] = []

    export_path = os.getenv("TIKTOK_TREND_EXPORT_PATH", "").strip()
    if export_path:
        try:
            records.extend(_load_records_from_path(Path(export_path)))
        except Exception as exc:
            notes.append(f"TikTok export dosyasi okunamadi: {exc}")

    api_url = os.getenv("TIKTOK_RESEARCH_API_URL", "").strip()
    api_token = os.getenv("TIKTOK_RESEARCH_API_TOKEN", "").strip()
    if api_url:
        try:
            headers = {"Content-Type": "application/json"}
            if api_token:
                headers["Authorization"] = f"Bearer {api_token}"
            payload = {
                "keywords": niche_keywords[:6],
                "since_days": since_days,
                "limit": limit,
            }
            resp = requests.post(api_url, headers=headers, json=payload, timeout=25)
            resp.raise_for_status()
            api_payload = resp.json()
            if isinstance(api_payload, dict):
                for key in ("items", "data", "results", "rows"):
                    value = api_payload.get(key)
                    if isinstance(value, list):
                        records.extend(item for item in value if isinstance(item, dict))
                        break
            elif isinstance(api_payload, list):
                records.extend(item for item in api_payload if isinstance(item, dict))
        except Exception as exc:
            notes.append(f"TikTok API istegi basarisiz oldu: {exc}")

    if not records:
        if not notes:
            notes.append("TikTok icin API veya export kaynagi tanimli degil; bu kaynak atlandi.")
        return [], {
            "source": "tiktok",
            "keywords": [],
            "topics": [],
            "notes": notes,
        }

    cutoff = _now_utc() - timedelta(days=since_days)
    signals: List[RawSignal] = []
    topic_titles: List[str] = []

    for index, record in enumerate(records, start=1):
        title = str(record.get("title") or record.get("keyword") or record.get("topic") or "").strip()
        description = str(record.get("description") or record.get("text") or record.get("caption") or "").strip()
        published_at = _parse_datetime(
            record.get("published_at")
            or record.get("created_at")
            or record.get("date")
            or record.get("timestamp")
        )
        if published_at < cutoff:
            continue

        blob = f"{title} {description}".lower()
        if niche_keywords and not any(keyword.lower() in blob for keyword in niche_keywords):
            continue

        source_id = str(record.get("id") or record.get("aweme_id") or record.get("item_id") or index)
        url = str(record.get("url") or record.get("share_url") or record.get("video_url") or "").strip()
        topic_titles.append(title or description[:80])
        signals.append(
            RawSignal(
                source="tiktok",
                source_id=f"live-tt-{source_id}",
                title=title or description[:80] or f"TikTok Trend {index}",
                text=description or title,
                url=url,
                published_at=published_at,
                metadata={
                    "views": _safe_int(record.get("views") or record.get("play_count")),
                    "likes": _safe_int(record.get("likes") or record.get("digg_count")),
                    "comments": _safe_int(record.get("comments") or record.get("comment_count")),
                    "sound": str(record.get("sound") or record.get("sound_title") or "").strip(),
                    "live_trend_source": "tiktok_api_or_export",
                },
            )
        )
        if len(signals) >= limit:
            break

    return signals, {
        "source": "tiktok",
        "keywords": _keywords_from_texts(topic_titles, niche_keywords, max_items=12),
        "topics": _dedupe_keep_order(topic_titles, max_items=12),
        "notes": notes,
    }


def collect_live_signals(settings, since_days: int, limit: int) -> Dict[str, Any]:
    if limit <= 0:
        return {
            "signals": [],
            "top_keywords": [],
            "rising_queries": [],
            "viral_topics": [],
            "sources_used": [],
            "notes": [],
            "fetched_at": _now_utc().isoformat(),
        }

    niche_keywords = list(getattr(settings, "niche_keywords", []) or [])
    source_limit = max(5, limit // 3) if limit >= 3 else limit

    youtube_signals, youtube_meta = _fetch_youtube_api_signals(niche_keywords, since_days, source_limit)
    google_signals, google_meta = _fetch_google_trends_signals(niche_keywords, since_days, source_limit)
    tiktok_signals, tiktok_meta = _fetch_tiktok_signals(niche_keywords, since_days, source_limit)

    signals: List[RawSignal] = []
    seen_ids = set()
    for signal in youtube_signals + google_signals + tiktok_signals:
        key = (signal.source, signal.source_id)
        if key in seen_ids:
            continue
        seen_ids.add(key)
        signals.append(signal)
        if len(signals) >= limit:
            break

    keywords = _dedupe_keep_order(
        youtube_meta.get("keywords", [])
        + google_meta.get("keywords", [])
        + tiktok_meta.get("keywords", []),
        max_items=18,
    )
    topics = _dedupe_keep_order(
        youtube_meta.get("topics", [])
        + google_meta.get("topics", [])
        + tiktok_meta.get("topics", []),
        max_items=18,
    )

    sources_used = []
    if youtube_signals:
        sources_used.append("YouTube Data API")
    if google_signals:
        sources_used.append("Google Trends")
    if tiktok_signals:
        sources_used.append("TikTok")

    notes = _dedupe_keep_order(
        youtube_meta.get("notes", []) + google_meta.get("notes", []) + tiktok_meta.get("notes", []),
        max_items=12,
    )

    if not signals:
        nedenler = " | ".join(notes) or "Kaynaklardan veri donmedi."
        logger.warning(
            "Canli trend kaynaklarindan 0 sinyal toplandi. "
            f"Olası nedenler: {nedenler} "
            "Cozum: YOUTUBE_API_KEY degerini, niche keyword filtrelerini, GOOGLE_TRENDS_GEO ayarini ve varsa TikTok API/export baglantilarini kontrol edin."
        )
    else:
        kaynaklar = ", ".join(sources_used) or "bilinmeyen kaynak"
        logger.info(f"Canli trend kaynaklarindan {len(signals)} sinyal toplandi. Aktif kaynaklar: {kaynaklar}")
    return {
        "signals": signals,
        "top_keywords": keywords,
        "rising_queries": keywords[:10],
        "viral_topics": topics,
        "sources_used": sources_used,
        "notes": notes,
        "fetched_at": _now_utc().isoformat(),
    }


def extract_live_keywords(bundle: Dict[str, Any], max_items: int = 12) -> List[str]:
    if not isinstance(bundle, dict):
        return []
    keywords = list(bundle.get("top_keywords", [])) + list(bundle.get("rising_queries", []))
    return _dedupe_keep_order(keywords, max_items=max_items)


def summarize_live_bundle(bundle: Dict[str, Any], max_keywords: int = 10, max_topics: int = 6) -> str:
    if not isinstance(bundle, dict):
        return "Canli trend verisi yok."

    keywords = extract_live_keywords(bundle, max_items=max_keywords)
    topics = _dedupe_keep_order(list(bundle.get("viral_topics", [])), max_items=max_topics)
    sources = ", ".join(bundle.get("sources_used", [])) or "Yok"
    notes = " | ".join(bundle.get("notes", [])) or "Ek not yok."

    return (
        f"Kaynaklar: {sources}\n"
        f"Trend anahtar kelimeler: {', '.join(keywords) if keywords else 'Yok'}\n"
        f"Yukselen konular: {' | '.join(topics) if topics else 'Yok'}\n"
        f"Notlar: {notes}"
    ).strip()

