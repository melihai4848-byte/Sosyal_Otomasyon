import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import requests

from moduller.logger import get_logger
from moduller.retry_utils import retry_with_backoff
from moduller.srt_utils import parse_srt_blocks, read_srt_file

logger = get_logger("YouTubeAnalyticsAPI")

SCOPES = [
    "https://www.googleapis.com/auth/yt-analytics.readonly",
    "https://www.googleapis.com/auth/youtube.readonly",
]

YOUTUBE_DATA_API_BASE = "https://www.googleapis.com/youtube/v3"
YOUTUBE_ANALYTICS_API_BASE = "https://youtubeanalytics.googleapis.com/v2"


def _oauth_paths() -> tuple[Path, Path]:
    client_secret = Path(
        os.getenv("YOUTUBE_OAUTH_CLIENT_SECRET_FILE", "00_Inputs/oauth/google_client_secret.json")
    )
    token_file = Path(
        os.getenv("YOUTUBE_OAUTH_TOKEN_FILE", "00_Inputs/oauth/youtube_analytics_token.json")
    )
    return client_secret, token_file


def _oauth_env_config() -> dict:
    return {
        "client_id": os.getenv("YOUTUBE_OAUTH_CLIENT_ID", "").strip(),
        "client_secret": os.getenv("YOUTUBE_OAUTH_CLIENT_SECRET", "").strip(),
        "project_id": os.getenv("YOUTUBE_OAUTH_PROJECT_ID", "").strip(),
    }


def _oauth_env_ready() -> bool:
    config = _oauth_env_config()
    return bool(config["client_id"] and config["client_secret"])


def _oauth_client_config_from_env() -> dict:
    config = _oauth_env_config()
    if not (config["client_id"] and config["client_secret"]):
        raise RuntimeError(
            "Env tabanli OAuth icin YOUTUBE_OAUTH_CLIENT_ID ve YOUTUBE_OAUTH_CLIENT_SECRET tanimli olmali."
        )
    return {
        "installed": {
            "client_id": config["client_id"],
            "project_id": config["project_id"] or "youtube-analytics-local",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": config["client_secret"],
            "redirect_uris": [
                "http://localhost",
                "http://127.0.0.1",
            ],
        }
    }


def youtube_analytics_available() -> bool:
    client_secret, _ = _oauth_paths()
    return client_secret.exists() or _oauth_env_ready()


def _load_credentials():
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as exc:
        raise RuntimeError(
            "YouTube Analytics API icin google-auth, google-auth-oauthlib ve google-api-python-client kutuphaneleri gerekli."
        ) from exc

    client_secret, token_file = _oauth_paths()
    use_env_oauth = _oauth_env_ready()
    if not client_secret.exists() and not use_env_oauth:
        raise RuntimeError(
            f"OAuth client secret dosyasi bulunamadi: {client_secret}. "
            "Cozum: ya bu dosyayi ekle ya da .env icine YOUTUBE_OAUTH_CLIENT_ID ve YOUTUBE_OAUTH_CLIENT_SECRET yaz."
        )

    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())

    if not creds or not creds.valid:
        if client_secret.exists():
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secret), SCOPES)
        else:
            flow = InstalledAppFlow.from_client_config(_oauth_client_config_from_env(), SCOPES)
        creds = flow.run_local_server(port=0)
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(creds.to_json(), encoding="utf-8")
        logger.info(f"YouTube Analytics OAuth token kaydedildi: {token_file}")

    return creds


def _resolve_video_id(transcript_path: Optional[Path] = None, context: Optional[dict] = None) -> str:
    adaylar = [
        os.getenv("YOUTUBE_ANALYTICS_VIDEO_ID", "").strip(),
        os.getenv("YOUTUBE_VIDEO_ID", "").strip(),
    ]

    if isinstance(context, dict):
        adaylar.extend(
            [
                str(context.get("video_id") or "").strip(),
                str(context.get("youtube_video_id") or "").strip(),
                str(context.get("id") or "").strip(),
            ]
        )

    if transcript_path:
        stem = transcript_path.stem
        adaylar.append(stem.split("_")[0].strip())

    for item in adaylar:
        if item and len(item) >= 11:
            return item
    return ""


def _timestamp_to_seconds(value: str) -> float:
    text = str(value or "").strip().replace(",", ".")
    if not text:
        return 0.0
    parts = text.split(":")
    if len(parts) == 3:
        hour, minute, second = parts
        return int(hour) * 3600 + int(minute) * 60 + float(second)
    if len(parts) == 2:
        minute, second = parts
        return int(minute) * 60 + float(second)
    return float(text)


def _duration_from_transcript(transcript_path: Optional[Path]) -> float:
    if not transcript_path or not transcript_path.exists():
        return 0.0
    try:
        blocks = parse_srt_blocks(read_srt_file(transcript_path))
    except Exception:
        return 0.0

    for block in reversed(blocks):
        if not getattr(block, "is_processable", False):
            continue
        timing = str(getattr(block, "timing_line", "") or "")
        if "-->" not in timing:
            continue
        end_raw = timing.split("-->", 1)[1].strip()
        try:
            return _timestamp_to_seconds(end_raw)
        except Exception:
            continue
    return 0.0


def _duration_from_context(context: Optional[dict]) -> float:
    if not isinstance(context, dict):
        return 0.0
    for key in ("duration_seconds", "video_duration_seconds", "duration_sec", "duration"):
        value = context.get(key)
        try:
            if value is not None:
                return float(value)
        except Exception:
            continue
    return 0.0


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value or "0").replace(",", ".")))
    except Exception:
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value or "0").replace(",", "."))
    except Exception:
        return default


def _parse_iso8601_duration(duration_text: str) -> int:
    text = str(duration_text or "").strip().upper()
    if not text:
        return 0
    match = re.fullmatch(
        r"P(?:\d+Y)?(?:\d+M)?(?:\d+D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?",
        text,
    )
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def _format_duration_label(duration_seconds: int) -> str:
    total_seconds = max(0, int(duration_seconds))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def _is_currently_public_video(item: dict) -> bool:
    status = item.get("status", {}) or {}
    privacy_status = str(status.get("privacyStatus") or "").strip().lower()
    upload_status = str(status.get("uploadStatus") or "").strip().lower()
    publish_at_raw = str(status.get("publishAt") or (item.get("snippet", {}) or {}).get("publishedAt") or "").strip()

    if privacy_status != "public":
        return False
    if upload_status and upload_status not in {"processed", "uploaded"}:
        return False
    if publish_at_raw:
        try:
            publish_at = datetime.fromisoformat(publish_at_raw.replace("Z", "+00:00"))
            if publish_at > datetime.now().astimezone(publish_at.tzinfo):
                return False
        except Exception:
            pass
    return True


def _ensure_valid_token(creds) -> None:
    try:
        from google.auth.transport.requests import Request
    except ImportError:
        return
    if getattr(creds, "expired", False) and getattr(creds, "refresh_token", None):
        creds.refresh(Request())


def _http_error_from_response(prefix: str, response: requests.Response) -> requests.HTTPError:
    try:
        payload = response.json()
    except Exception:
        payload = response.text
    error = requests.HTTPError(f"{prefix}: {payload}")
    error.response = response
    return error


def _authorized_json_get(path: str, params: dict, creds) -> dict:
    _ensure_valid_token(creds)
    url = f"{YOUTUBE_DATA_API_BASE}/{path.lstrip('/')}"

    def _request() -> requests.Response:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {creds.token}"},
            params=params,
            timeout=30,
        )
        if response.status_code == 200:
            return response
        raise _http_error_from_response("YouTube Data API hatasi", response)

    response = retry_with_backoff(
        _request,
        f"YouTube Data API istegi ({path})",
        logger,
        max_attempts=4,
        base_delay_seconds=5,
        max_delay_seconds=20,
    )
    return response.json()


def _parse_report_rows(payload: dict) -> list[dict]:
    headers = payload.get("columnHeaders", []) or []
    rows = payload.get("rows", []) or []
    parsed_rows = []

    for row in rows:
        item = {}
        for header, value in zip(headers, row):
            name = str(header.get("name") or "").strip()
            data_type = str(header.get("dataType") or "").strip().upper()
            if data_type == "INTEGER":
                item[name] = _as_int(value)
            elif data_type in {"FLOAT", "DOUBLE"}:
                item[name] = _as_float(value)
            else:
                item[name] = value
        parsed_rows.append(item)
    return parsed_rows


def _fetch_analytics_rows(
    creds,
    metrics: str,
    start_date: str,
    end_date: str,
    dimensions: Optional[str] = None,
    filters: Optional[str] = None,
    sort: Optional[str] = None,
    max_results: Optional[int] = None,
) -> list[dict]:
    _ensure_valid_token(creds)
    params = {
        "ids": "channel==MINE",
        "startDate": start_date,
        "endDate": end_date,
        "metrics": metrics,
    }
    if dimensions:
        params["dimensions"] = dimensions
    if filters:
        params["filters"] = filters
    if sort:
        params["sort"] = sort
    if max_results:
        params["maxResults"] = max_results

    def _request() -> requests.Response:
        response = requests.get(
            f"{YOUTUBE_ANALYTICS_API_BASE}/reports",
            headers={"Authorization": f"Bearer {creds.token}"},
            params=params,
            timeout=30,
        )
        if response.status_code == 200:
            return response
        raise _http_error_from_response("YouTube Analytics API hatasi", response)

    response = retry_with_backoff(
        _request,
        "YouTube Analytics raporu",
        logger,
        max_attempts=4,
        base_delay_seconds=10,
        max_delay_seconds=40,
    )
    return _parse_report_rows(response.json())


def _date_range(days: int) -> tuple[str, str]:
    end_date = date.today()
    start_date = end_date - timedelta(days=max(1, int(days)))
    return start_date.isoformat(), end_date.isoformat()


def fetch_authenticated_channel_profile(creds=None) -> dict:
    creds = creds or _load_credentials()
    payload = _authorized_json_get(
        "channels",
        {
            "part": "snippet,statistics,contentDetails,brandingSettings",
            "mine": "true",
        },
        creds,
    )
    items = payload.get("items", []) or []
    if not items:
        raise RuntimeError("OAuth ile bagli YouTube kanali bulunamadi.")

    item = items[0]
    snippet = item.get("snippet", {}) or {}
    statistics = item.get("statistics", {}) or {}
    content_details = item.get("contentDetails", {}) or {}
    branding = item.get("brandingSettings", {}) or {}

    uploads_playlist_id = (
        content_details.get("relatedPlaylists", {}) or {}
    ).get("uploads", "")

    return {
        "channel_id": str(item.get("id") or "").strip(),
        "channel_title": str(snippet.get("title") or "").strip(),
        "description": str(snippet.get("description") or "").strip(),
        "published_at": str(snippet.get("publishedAt") or "").strip(),
        "custom_url": str(snippet.get("customUrl") or "").strip(),
        "country": str(snippet.get("country") or "").strip(),
        "subscriber_count": _as_int(statistics.get("subscriberCount")),
        "video_count": _as_int(statistics.get("videoCount")),
        "view_count": _as_int(statistics.get("viewCount")),
        "uploads_playlist_id": uploads_playlist_id,
        "keywords": str((branding.get("channel", {}) or {}).get("keywords") or "").strip(),
    }


def fetch_recent_channel_videos(
    limit: int = 10,
    creds=None,
    uploads_playlist_id: Optional[str] = None,
    min_duration_seconds: int = 300,
    max_scan_items: int = 120,
) -> list[dict]:
    creds = creds or _load_credentials()
    if not uploads_playlist_id:
        uploads_playlist_id = fetch_authenticated_channel_profile(creds).get("uploads_playlist_id")
    if not uploads_playlist_id:
        raise RuntimeError("Kanal uploads playlist kimligi bulunamadi.")
    filtered_videos = []
    next_page_token = ""
    scanned_items = 0
    global_index = 0

    while len(filtered_videos) < limit and scanned_items < max_scan_items:
        params = {
            "part": "snippet,contentDetails",
            "playlistId": uploads_playlist_id,
            "maxResults": min(50, max(10, limit * 3)),
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        playlist_payload = _authorized_json_get("playlistItems", params, creds)
        playlist_items = playlist_payload.get("items", []) or []
        if not playlist_items:
            break

        video_ids = []
        order_map = {}
        for item in playlist_items:
            if scanned_items >= max_scan_items:
                break
            video_id = str((item.get("contentDetails", {}) or {}).get("videoId") or "").strip()
            if not video_id:
                continue
            video_ids.append(video_id)
            order_map[video_id] = global_index
            global_index += 1
            scanned_items += 1

        if not video_ids:
            next_page_token = str(playlist_payload.get("nextPageToken") or "").strip()
            if not next_page_token:
                break
            continue

        videos_payload = _authorized_json_get(
            "videos",
            {
                "part": "snippet,contentDetails,statistics,status",
                "id": ",".join(video_ids),
                "maxResults": len(video_ids),
            },
            creds,
        )

        batch_videos = []
        for item in videos_payload.get("items", []) or []:
            if not _is_currently_public_video(item):
                continue
            video_id = str(item.get("id") or "").strip()
            snippet = item.get("snippet", {}) or {}
            stats = item.get("statistics", {}) or {}
            content_details = item.get("contentDetails", {}) or {}
            duration_seconds = _parse_iso8601_duration(content_details.get("duration", ""))
            if min_duration_seconds and duration_seconds < int(min_duration_seconds):
                continue

            thumbnails = snippet.get("thumbnails", {}) or {}
            thumbnail_url = (
                (thumbnails.get("maxres") or {}).get("url")
                or (thumbnails.get("high") or {}).get("url")
                or (thumbnails.get("medium") or {}).get("url")
                or (thumbnails.get("default") or {}).get("url")
                or ""
            )

            batch_videos.append(
                {
                    "video_id": video_id,
                    "title": str(snippet.get("title") or "").strip(),
                    "description": str(snippet.get("description") or "").strip(),
                    "published_at": str(snippet.get("publishedAt") or "").strip(),
                    "channel_title": str(snippet.get("channelTitle") or "").strip(),
                    "thumbnail_url": thumbnail_url,
                    "duration_seconds": duration_seconds,
                    "duration_label": _format_duration_label(duration_seconds),
                    "view_count": _as_int(stats.get("viewCount")),
                    "like_count": _as_int(stats.get("likeCount")),
                    "comment_count": _as_int(stats.get("commentCount")),
                }
            )

        batch_videos.sort(key=lambda item: order_map.get(item["video_id"], 999999))
        filtered_videos.extend(batch_videos)

        next_page_token = str(playlist_payload.get("nextPageToken") or "").strip()
        if not next_page_token:
            break

    filtered_videos.sort(key=lambda item: item.get("published_at", ""), reverse=True)
    return filtered_videos[:limit]


def _first_report_row(metric_variants: list[str], days: int, filters: Optional[str] = None, creds=None) -> dict:
    creds = creds or _load_credentials()
    start_date, end_date = _date_range(days)
    last_error: Optional[Exception] = None

    for metrics in metric_variants:
        try:
            rows = _fetch_analytics_rows(
                creds,
                metrics=metrics,
                start_date=start_date,
                end_date=end_date,
                filters=filters,
            )
            if rows:
                row = rows[0]
                row["_metrics"] = metrics
                return row
        except Exception as exc:
            last_error = exc
            logger.warning(f"Analytics metrik varyanti okunamadi ({metrics}): {exc}")

    if last_error:
        raise last_error
    return {}


def fetch_channel_summary_metrics(days: int = 28, creds=None) -> dict:
    row = _first_report_row(
        [
            "views,estimatedMinutesWatched,averageViewDuration,averageViewPercentage,subscribersGained,subscribersLost",
            "views,estimatedMinutesWatched,averageViewDuration,subscribersGained,subscribersLost",
            "views,estimatedMinutesWatched,subscribersGained,subscribersLost",
        ],
        days=days,
        creds=creds,
    )
    return {
        "window_days": int(days),
        "views": _as_int(row.get("views")),
        "estimated_minutes_watched": _as_int(row.get("estimatedMinutesWatched")),
        "average_view_duration_seconds": _as_float(row.get("averageViewDuration")),
        "average_view_percentage": _as_float(row.get("averageViewPercentage")),
        "subscribers_gained": _as_int(row.get("subscribersGained")),
        "subscribers_lost": _as_int(row.get("subscribersLost")),
        "net_subscribers": _as_int(row.get("subscribersGained")) - _as_int(row.get("subscribersLost")),
        "metric_variant": row.get("_metrics", ""),
    }


def fetch_channel_daily_metrics(days: int = 28, creds=None) -> list[dict]:
    creds = creds or _load_credentials()
    start_date, end_date = _date_range(days)
    rows = _fetch_analytics_rows(
        creds,
        metrics="views,estimatedMinutesWatched,subscribersGained,subscribersLost",
        start_date=start_date,
        end_date=end_date,
        dimensions="day",
        sort="day",
        max_results=max(31, int(days) + 5),
    )
    daily = []
    for row in rows:
        subscribers_gained = _as_int(row.get("subscribersGained"))
        subscribers_lost = _as_int(row.get("subscribersLost"))
        daily.append(
            {
                "day": str(row.get("day") or ""),
                "views": _as_int(row.get("views")),
                "estimated_minutes_watched": _as_int(row.get("estimatedMinutesWatched")),
                "subscribers_gained": subscribers_gained,
                "subscribers_lost": subscribers_lost,
                "net_subscribers": subscribers_gained - subscribers_lost,
            }
        )
    return daily


def fetch_video_summary_metrics(
    video_id: str,
    since_days: int = 90,
    published_at: Optional[str] = None,
    creds=None,
) -> dict:
    creds = creds or _load_credentials()
    video_id = str(video_id or "").strip()
    if not video_id:
        return {}

    today = date.today()
    if published_at:
        try:
            published_date = datetime.fromisoformat(str(published_at).replace("Z", "+00:00")).date()
            start_date = max(published_date, today - timedelta(days=max(1, int(since_days))))
        except Exception:
            start_date = today - timedelta(days=max(1, int(since_days)))
    else:
        start_date = today - timedelta(days=max(1, int(since_days)))

    row = _first_report_row(
        [
            "views,estimatedMinutesWatched,averageViewDuration,averageViewPercentage,subscribersGained,subscribersLost",
            "views,estimatedMinutesWatched,averageViewDuration,subscribersGained,subscribersLost",
            "views,estimatedMinutesWatched,subscribersGained,subscribersLost",
        ],
        days=max(1, (today - start_date).days + 1),
        filters=f"video=={video_id}",
        creds=creds,
    )
    return {
        "video_id": video_id,
        "window_start": start_date.isoformat(),
        "window_end": today.isoformat(),
        "views": _as_int(row.get("views")),
        "estimated_minutes_watched": _as_int(row.get("estimatedMinutesWatched")),
        "average_view_duration_seconds": _as_float(row.get("averageViewDuration")),
        "average_view_percentage": _as_float(row.get("averageViewPercentage")),
        "subscribers_gained": _as_int(row.get("subscribersGained")),
        "subscribers_lost": _as_int(row.get("subscribersLost")),
        "net_subscribers": _as_int(row.get("subscribersGained")) - _as_int(row.get("subscribersLost")),
        "metric_variant": row.get("_metrics", ""),
    }


def fetch_channel_analytics_snapshot(
    channel_days: int = 28,
    recent_video_limit: int = 10,
    per_video_days: int = 90,
) -> dict:
    creds = _load_credentials()
    channel = fetch_authenticated_channel_profile(creds=creds)
    recent_videos = fetch_recent_channel_videos(
        limit=recent_video_limit,
        creds=creds,
        uploads_playlist_id=channel.get("uploads_playlist_id"),
    )

    enriched_videos = []
    for video in recent_videos:
        analytics = fetch_video_summary_metrics(
            video.get("video_id", ""),
            since_days=per_video_days,
            published_at=video.get("published_at"),
            creds=creds,
        )
        item = dict(video)
        item.update(
            {
                "analytics_views": _as_int(analytics.get("views")),
                "analytics_estimated_minutes_watched": _as_int(analytics.get("estimated_minutes_watched")),
                "analytics_average_view_duration_seconds": _as_float(analytics.get("average_view_duration_seconds")),
                "analytics_average_view_percentage": _as_float(analytics.get("average_view_percentage")),
                "analytics_subscribers_gained": _as_int(analytics.get("subscribers_gained")),
                "analytics_subscribers_lost": _as_int(analytics.get("subscribers_lost")),
                "analytics_net_subscribers": _as_int(analytics.get("net_subscribers")),
                "analytics_window_start": analytics.get("window_start", ""),
                "analytics_window_end": analytics.get("window_end", ""),
            }
        )
        enriched_videos.append(item)

    return {
        "channel": channel,
        "channel_summary": fetch_channel_summary_metrics(days=channel_days, creds=creds),
        "channel_daily": fetch_channel_daily_metrics(days=channel_days, creds=creds),
        "recent_videos": enriched_videos,
        "snapshot_window_days": int(channel_days),
        "per_video_window_days": int(per_video_days),
    }


def fetch_retention_points_via_api(
    transcript_path: Optional[Path] = None,
    context: Optional[dict] = None,
    since_days: int = 90,
    creds=None,
) -> tuple[list[dict], str]:
    video_id = _resolve_video_id(transcript_path=transcript_path, context=context)
    if not video_id:
        raise RuntimeError(
            "YouTube Analytics API kullanilacaksa YOUTUBE_ANALYTICS_VIDEO_ID veya YOUTUBE_VIDEO_ID tanimli olmali."
        )

    creds = creds or _load_credentials()
    start_date, end_date = _date_range(since_days)

    def _request_report() -> requests.Response:
        _ensure_valid_token(creds)
        response = requests.get(
            f"{YOUTUBE_ANALYTICS_API_BASE}/reports",
            headers={"Authorization": f"Bearer {creds.token}"},
            params={
                "ids": "channel==MINE",
                "startDate": start_date,
                "endDate": end_date,
                "metrics": "audienceWatchRatio",
                "dimensions": "elapsedVideoTimeRatio",
                "filters": f"video=={video_id}",
                "sort": "elapsedVideoTimeRatio",
                "maxResults": 200,
            },
            timeout=30,
        )
        if response.status_code == 200:
            return response
        raise _http_error_from_response("YouTube Analytics retention hatasi", response)

    response = retry_with_backoff(
        _request_report,
        "YouTube Analytics retention raporu",
        logger,
        max_attempts=4,
        base_delay_seconds=15,
        max_delay_seconds=60,
    )

    payload = response.json()
    rows = payload.get("rows", []) or []
    duration_seconds = _duration_from_context(context) or _duration_from_transcript(transcript_path) or 100.0
    points = []
    for ratio, retention in rows:
        try:
            points.append(
                {
                    "second": round(float(ratio) * duration_seconds),
                    "retention_pct": round(float(retention) * 100 if float(retention) <= 1.0 else float(retention), 2),
                }
            )
        except Exception:
            continue

    if not points:
        raise RuntimeError(
            "YouTube Analytics API veri dondurdu ancak retention noktasi cikmadi. "
            "Video ID'nin dogru oldugunu ve videonun analytics verisinin olustugunu kontrol et."
        )

    return points, video_id


def oauth_setup_summary() -> str:
    client_secret, token_file = _oauth_paths()
    env_ready = _oauth_env_ready()
    return (
        f"OAuth client secret: {client_secret}\n"
        f"Env OAuth hazir: {'evet' if env_ready else 'hayir'}\n"
        f"OAuth token: {token_file}\n"
        "Bu modul kendi kanalina ait YouTube Analytics verisini okumak icin Desktop OAuth kullanir.\n"
        "Alternatif env alanlari: YOUTUBE_OAUTH_CLIENT_ID, YOUTUBE_OAUTH_CLIENT_SECRET, opsiyonel YOUTUBE_OAUTH_PROJECT_ID"
    )
