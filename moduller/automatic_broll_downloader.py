from __future__ import annotations

import concurrent.futures
import json
import math
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from moduller._module_alias import load_numbered_module
from moduller.config import reload_project_env
from moduller.logger import get_logger
from moduller.output_paths import glob_outputs, grouped_json_output_path, grouped_output_path
from moduller.retry_utils import retry_with_backoff
from moduller.video_edit_utils import timestamp_to_seconds

_BROLL_MODULE = load_numbered_module("203_broll_onerici.py")
build_stock_search_query = _BROLL_MODULE.build_stock_search_query

logger = get_logger("automatic_broll_downloader")

PEXELS_SEARCH_URL = "https://api.pexels.com/v1/videos/search"
PIXABAY_SEARCH_URL = "https://pixabay.com/api/videos/"
FREEPIK_VIDEO_SEARCH_URL = "https://api.freepik.com/v1/videos"
FREEPIK_VIDEO_DOWNLOAD_URL = "https://api.freepik.com/v1/videos/{asset_id}/download"
COVERR_VIDEO_SEARCH_URL = "https://api.coverr.co/videos"
OPENVERSE_IMAGE_SEARCH_URL = "https://api.openverse.org/v1/images/"
DOWNLOAD_ROOT = grouped_output_path("tools", "broll_downloads")
SUPPORTED_PROVIDERS = ("pexels", "pixabay", "freepik", "coverr", "openverse")
DEFAULT_PROVIDER_PRIORITY = ("pexels", "pixabay", "freepik", "coverr", "openverse")
DEFAULT_TIMEOUT = (15, 90)
LANDSCAPE_RATIO = 16 / 9
PORTRAIT_RATIO = 9 / 16
PREFERRED_PLAN_NAME = "subtitle_tr_B_roll_fikirleri.json"
PREFERRED_PLAN_PARENT = "_json_cache"
DEFAULT_MAX_CANDIDATE_DOWNLOAD_OPTIONS = 1
QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "cinematic",
    "close",
    "detailed",
    "for",
    "footage",
    "in",
    "of",
    "on",
    "shot",
    "stock",
    "the",
    "with",
}
SEARCH_CACHE_PATH = grouped_json_output_path("tools", "automatic_broll_search_cache.json")


@dataclass
class ClipCandidate:
    provider: str
    query: str
    asset_id: str
    file_url: str
    page_url: str
    width: int
    height: int
    duration_seconds: float
    author_name: str
    rendition_label: str
    relevance_rank: int
    popularity_score: float
    score: float
    media_type: str = "video"
    file_extension: str = ".mp4"


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_int_env(name: str, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    raw = os.getenv(name, "").strip()
    try:
        value = int(raw or default)
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def normalize_enum_env(name: str, default: str, allowed: tuple[str, ...]) -> str:
    raw = os.getenv(name, default).strip().lower() or default
    return raw if raw in allowed else default


def parse_float_env(name: str, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    raw = os.getenv(name, "").strip().replace(",", ".")
    try:
        value = float(raw or default)
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def normalize_provider_priority(provider_priority: list[str] | None = None) -> list[str]:
    raw = provider_priority or [
        item.strip().lower()
        for item in os.getenv("BROLL_PROVIDER_PRIORITY", ",".join(DEFAULT_PROVIDER_PRIORITY)).split(",")
        if item.strip()
    ]
    normalized = [item for item in raw if item in SUPPORTED_PROVIDERS]
    ordered = list(dict.fromkeys(normalized))
    return ordered or list(DEFAULT_PROVIDER_PRIORITY)


def available_providers(provider_priority: list[str]) -> list[str]:
    available: list[str] = []
    image_fallback_enabled = parse_bool_env("BROLL_ALLOW_IMAGE_FALLBACK", default=False)
    for provider in provider_priority:
        if provider == "pexels" and os.getenv("PEXELS_API_KEY", "").strip():
            available.append(provider)
        elif provider == "pixabay" and os.getenv("PIXABAY_API_KEY", "").strip():
            available.append(provider)
        elif provider == "freepik" and os.getenv("FREEPIK_API_KEY", "").strip():
            available.append(provider)
        elif provider == "coverr" and os.getenv("COVERR_API_KEY", "").strip():
            available.append(provider)
        elif provider == "openverse" and image_fallback_enabled:
            available.append(provider)
    return available


def provider_env_status() -> dict[str, bool]:
    return {
        "pexels": bool(os.getenv("PEXELS_API_KEY", "").strip()),
        "pixabay": bool(os.getenv("PIXABAY_API_KEY", "").strip()),
        "freepik": bool(os.getenv("FREEPIK_API_KEY", "").strip()),
        "coverr": bool(os.getenv("COVERR_API_KEY", "").strip()),
        "openverse_fallback": parse_bool_env("BROLL_ALLOW_IMAGE_FALLBACK", default=False),
    }


def find_broll_plan_files() -> list[Path]:
    candidates = glob_outputs("*_B_roll_fikirleri.json", groups=("youtube",), include_json_cache=True)
    unique = {path.resolve(): path for path in candidates if path.is_file()}
    return sorted(unique.values(), key=lambda item: item.stat().st_mtime, reverse=True)


def find_preferred_plan_file(files: list[Path]) -> Path | None:
    for path in files:
        if path.name == PREFERRED_PLAN_NAME and path.parent.name == PREFERRED_PLAN_PARENT:
            return path
    return None


def select_plan_file(files: list[Path]) -> Path | None:
    if not files:
        return None
    if len(files) == 1:
        return files[0]

    preferred = find_preferred_plan_file(files)
    if preferred:
        logger.info(f"Varsayilan B-Roll plani otomatik secildi: {preferred.name} | {preferred.parent.name}")
        print(f"\nVarsayilan B-Roll plani otomatik secildi: {preferred.name} | {preferred.parent.name}")
        return preferred

    print("\nMevcut B-Roll planlari:")
    for index, path in enumerate(files, start=1):
        print(f"  [{index}] {path.name} | {path.parent.name}")

    raw = input("👉 Indirilecek B-Roll plan numarasi? (bos = en guncel): ").strip()
    if not raw:
        return files[0]
    try:
        return files[int(raw) - 1]
    except (ValueError, IndexError):
        logger.error("Gecersiz plan secimi yapildi.")
        return None


def extract_plan_stem(plan_path: Path) -> str:
    suffix = "_B_roll_fikirleri"
    if plan_path.stem.endswith(suffix):
        return plan_path.stem[: -len(suffix)]
    return plan_path.stem


def load_plan_items(plan_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        raw_items = payload.get("items") or payload.get("data") or []
    else:
        raw_items = payload

    if not isinstance(raw_items, list):
        raise RuntimeError(f"B-Roll plan formati desteklenmiyor: {plan_path.name}")

    normalized_items: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            continue
        english_prompt = str(item.get("english_prompt", "") or "").strip()
        query = str(item.get("stock_search_query", "") or "").strip() or build_stock_search_query(english_prompt)
        timestamp = str(item.get("timestamp", "") or "").strip()
        normalized_items.append(
            {
                "index": index,
                "timestamp": timestamp,
                "reason": str(item.get("reason", "") or "").strip(),
                "english_prompt": english_prompt,
                "stock_search_query": clean_search_query(query),
                "orientation": detect_orientation(item),
            }
        )
    return normalized_items


def clean_search_query(query: str, max_length: int = 100) -> str:
    text = re.sub(r"\s+", " ", str(query or "")).strip()
    text = re.sub(r"[^A-Za-z0-9 \-']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_length].strip() or "cinematic stock footage"


def build_query_variants(query: str, english_prompt: str = "") -> list[str]:
    variants: list[str] = []

    def add(value: str) -> None:
        candidate = clean_search_query(value, max_length=80)
        if not candidate or candidate in variants:
            return
        variants.append(candidate)

    cleaned_query = clean_search_query(query, max_length=80)
    add(cleaned_query)

    tokens = [token for token in cleaned_query.split() if token.lower() not in QUERY_STOPWORDS]
    if len(tokens) >= 4:
        add(" ".join(tokens[:4]))
    if len(tokens) >= 3:
        add(" ".join(tokens[:3]))
    if len(tokens) >= 2:
        add(" ".join(tokens[:2]))

    if english_prompt.strip():
        fallback_query = build_stock_search_query(english_prompt)
        add(fallback_query)
        fallback_tokens = [token for token in fallback_query.split() if token.lower() not in QUERY_STOPWORDS]
        if len(fallback_tokens) >= 3:
            add(" ".join(fallback_tokens[:3]))

    if tokens:
        add(tokens[0])

    limit = parse_int_env("BROLL_QUERY_VARIANT_LIMIT", default=4, minimum=1, maximum=8)
    return variants[:limit] or ["cinematic stock footage"]


def detect_orientation(item: dict[str, Any]) -> str:
    raw_text = " ".join(
        [
            str(item.get("orientation", "") or ""),
            str(item.get("stock_search_query", "") or ""),
            str(item.get("english_prompt", "") or ""),
        ]
    ).lower()
    if any(token in raw_text for token in ("9:16", "vertical", "portrait")):
        return "portrait"
    if any(token in raw_text for token in ("16:9", "horizontal", "landscape")):
        return "landscape"

    default_orientation = os.getenv("BROLL_DEFAULT_ORIENTATION", "landscape").strip().lower()
    return default_orientation if default_orientation in {"landscape", "portrait", "square"} else "landscape"


def orientation_from_dimensions(width: int, height: int) -> str:
    if width > height:
        return "landscape"
    if height > width:
        return "portrait"
    return "square"


def target_dimensions(orientation: str) -> tuple[int, int]:
    if orientation == "portrait":
        return 1080, 1920
    if orientation == "square":
        return 1080, 1080
    return 1920, 1080


def target_aspect_ratio(orientation: str) -> float:
    if orientation == "portrait":
        return PORTRAIT_RATIO
    if orientation == "square":
        return 1.0
    return LANDSCAPE_RATIO


def parse_timestamp_range(timestamp_range: str) -> tuple[float, float]:
    if "-->" not in str(timestamp_range or ""):
        return 0.0, 0.0
    start_raw, end_raw = [item.strip() for item in str(timestamp_range).split("-->", 1)]
    return timestamp_to_seconds(start_raw), timestamp_to_seconds(end_raw)


def format_seconds_for_filename(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours:
        return f"{hours:02d}-{minutes:02d}-{secs:02d}"
    return f"{minutes:02d}-{secs:02d}"


def slugify(value: str, max_words: int = 6, max_length: int = 48) -> str:
    words = re.findall(r"[A-Za-z0-9]+", str(value or "").lower())
    if not words:
        return "stock_clip"
    slug = "_".join(words[:max_words]).strip("_")
    return slug[:max_length].strip("_") or "stock_clip"


def normalize_file_extension(value: str, fallback: str = ".mp4") -> str:
    suffix = Path(urlparse(str(value or "")).path).suffix.strip().lower()
    if suffix and len(suffix) <= 6:
        return suffix
    return fallback


def build_asset_marker(provider: str, asset_id: str = "", file_url: str = "") -> str:
    provider_name = str(provider or "").strip().lower()
    asset_text = str(asset_id or "").strip()
    file_text = str(file_url or "").strip()
    if not provider_name:
        return ""
    return f"{provider_name}:{asset_text or file_text}"


def candidate_asset_marker(candidate: ClipCandidate) -> str:
    return build_asset_marker(candidate.provider, candidate.asset_id, candidate.file_url)


def normalize_path_key(path: Path | str) -> str:
    resolved = Path(path)
    try:
        resolved = resolved.resolve()
    except Exception:
        pass
    return str(resolved).strip().lower()


def build_versioned_asset_path(target_path: Path, candidate: ClipCandidate) -> Path:
    asset_hint = candidate.asset_id or Path(urlparse(candidate.file_url).path).stem or "asset"
    suffix_slug = slugify(f"{candidate.provider}_{asset_hint}", max_words=8, max_length=24)

    for index in range(1, 1000):
        suffix = f"_{suffix_slug}" if index == 1 else f"_{suffix_slug}_{index}"
        candidate_path = target_path.with_name(f"{target_path.stem}{suffix}{target_path.suffix}")
        if not candidate_path.exists():
            return candidate_path

    raise RuntimeError(f"Bos alternatif dosya adi bulunamadi: {target_path.name}")


def resolve_target_path_for_candidate(
    target_path: Path,
    candidate: ClipCandidate,
    known_path_markers: dict[str, str],
) -> Path:
    if not target_path.exists():
        return target_path

    existing_marker = known_path_markers.get(normalize_path_key(target_path), "")
    candidate_marker = candidate_asset_marker(candidate)
    if existing_marker and existing_marker == candidate_marker:
        return target_path

    return build_versioned_asset_path(target_path, candidate)


def build_alternative_asset_path(primary_path: Path, candidate: ClipCandidate, option_index: int) -> Path:
    if option_index <= 1:
        return primary_path

    asset_hint = candidate.asset_id or Path(urlparse(candidate.file_url).path).stem or "asset"
    suffix_slug = slugify(f"{candidate.provider}_{asset_hint}", max_words=8, max_length=24)
    return primary_path.with_name(f"{primary_path.stem}_aday_{option_index}_{suffix_slug}{primary_path.suffix}")


def build_asset_path(
    output_dir: Path,
    item_index: int,
    timestamp_range: str,
    query: str,
    provider: str = "",
    extension: str = ".mp4",
) -> Path:
    start_seconds, _ = parse_timestamp_range(timestamp_range)
    timestamp_label = format_seconds_for_filename(start_seconds)
    query_slug = slugify(query)
    provider_slug = slugify(provider or "provider", max_words=2, max_length=16)
    normalized_extension = extension if str(extension or "").startswith(".") else f".{extension}"
    return output_dir / f"broll_{item_index:02d}_{timestamp_label}_{query_slug}_{provider_slug}{normalized_extension}"


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Youtube-Otomasyon-v2/AutomaticBrollDownloader",
            "Accept": "application/json, */*",
        }
    )
    return session


def _search_cache_key(provider: str, query: str, orientation: str, target_duration: float) -> str:
    return "|".join(
        [
            str(provider or "").strip().lower(),
            clean_search_query(query, max_length=80).casefold(),
            str(orientation or "").strip().lower(),
            f"{round(float(target_duration or 0.0), 2):.2f}",
        ]
    )


def _load_search_cache() -> dict[str, Any]:
    if not SEARCH_CACHE_PATH.exists():
        return {"entries": {}}
    try:
        parsed = json.loads(SEARCH_CACHE_PATH.read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, dict) else {"entries": {}}
    except Exception:
        return {"entries": {}}


def _save_search_cache(cache: dict[str, Any]) -> None:
    SEARCH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEARCH_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _serialize_candidate(candidate: ClipCandidate) -> dict[str, Any]:
    return asdict(candidate)


def _deserialize_candidate(payload: dict[str, Any]) -> ClipCandidate | None:
    if not isinstance(payload, dict):
        return None
    try:
        return ClipCandidate(
            provider=str(payload.get("provider", "") or ""),
            query=str(payload.get("query", "") or ""),
            asset_id=str(payload.get("asset_id", "") or ""),
            file_url=str(payload.get("file_url", "") or ""),
            page_url=str(payload.get("page_url", "") or ""),
            width=int(payload.get("width", 0) or 0),
            height=int(payload.get("height", 0) or 0),
            duration_seconds=float(payload.get("duration_seconds", 0.0) or 0.0),
            author_name=str(payload.get("author_name", "") or ""),
            rendition_label=str(payload.get("rendition_label", "") or ""),
            relevance_rank=int(payload.get("relevance_rank", 0) or 0),
            popularity_score=float(payload.get("popularity_score", 0.0) or 0.0),
            score=float(payload.get("score", 0.0) or 0.0),
            media_type=str(payload.get("media_type", "video") or "video"),
            file_extension=str(payload.get("file_extension", ".mp4") or ".mp4"),
        )
    except Exception:
        return None


def _prune_search_cache(cache: dict[str, Any], ttl_seconds: int) -> bool:
    entries = cache.get("entries", {}) if isinstance(cache.get("entries"), dict) else {}
    if not entries:
        cache["entries"] = {}
        return False
    if ttl_seconds <= 0:
        return False
    now_ts = time.time()
    dirty = False
    for key in list(entries.keys()):
        entry = entries.get(key)
        if not isinstance(entry, dict):
            entries.pop(key, None)
            dirty = True
            continue
        saved_at_ts = float(entry.get("saved_at_ts", 0.0) or 0.0)
        if saved_at_ts <= 0 or (now_ts - saved_at_ts) > ttl_seconds:
            entries.pop(key, None)
            dirty = True
    cache["entries"] = entries
    return dirty


def _get_cached_search_candidates(
    cache: dict[str, Any],
    provider: str,
    query: str,
    orientation: str,
    target_duration: float,
    ttl_seconds: int,
) -> list[ClipCandidate] | None:
    entries = cache.get("entries", {}) if isinstance(cache.get("entries"), dict) else {}
    cache_key = _search_cache_key(provider, query, orientation, target_duration)
    entry = entries.get(cache_key)
    if not isinstance(entry, dict):
        return None
    if ttl_seconds > 0:
        saved_at_ts = float(entry.get("saved_at_ts", 0.0) or 0.0)
        if saved_at_ts <= 0 or (time.time() - saved_at_ts) > ttl_seconds:
            entries.pop(cache_key, None)
            cache["entries"] = entries
            return None

    candidates = []
    for item in entry.get("candidates", []) or []:
        candidate = _deserialize_candidate(item)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _store_search_candidates(
    cache: dict[str, Any],
    provider: str,
    query: str,
    orientation: str,
    target_duration: float,
    candidates: list[ClipCandidate],
) -> None:
    entries = cache.setdefault("entries", {})
    entries[_search_cache_key(provider, query, orientation, target_duration)] = {
        "saved_at_ts": time.time(),
        "candidates": [_serialize_candidate(candidate) for candidate in candidates],
    }


def _search_provider_with_fresh_session(
    provider: str,
    query: str,
    orientation: str,
    target_duration: float,
) -> list[ClipCandidate]:
    session = create_session()
    try:
        return search_candidates_for_provider(session, provider, query, orientation, target_duration)
    finally:
        session.close()


def _should_probe_remote_size(
    candidate: ClipCandidate,
    option_index: int,
    max_size_bytes: int,
    mode: str,
) -> bool:
    if max_size_bytes <= 0 or not str(candidate.file_url or "").strip():
        return False
    normalized_mode = str(mode or "smart").strip().lower()
    if normalized_mode == "never":
        return False
    if normalized_mode == "always":
        return True
    if option_index > 1:
        return True
    if candidate.provider in {"freepik", "coverr"}:
        return True
    if max(candidate.width, candidate.height) >= 2560:
        return True
    if candidate.duration_seconds >= 90:
        return True
    return False


def request_json(
    session: requests.Session,
    url: str,
    description: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    def action() -> dict[str, Any]:
        response = session.get(url, headers=headers, params=params, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        return response.json()

    return retry_with_backoff(action, description, logger, max_attempts=4, base_delay_seconds=15, max_delay_seconds=90)


def extract_total_size_from_headers(headers: requests.structures.CaseInsensitiveDict[str]) -> int | None:
    content_range = str(headers.get("Content-Range", "") or headers.get("content-range", "")).strip()
    if content_range:
        match = re.search(r"/(\d+)$", content_range)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass

    content_length = str(headers.get("Content-Length", "") or headers.get("content-length", "")).strip()
    if content_length.isdigit():
        try:
            return int(content_length)
        except ValueError:
            return None
    return None


def probe_remote_file_size_bytes(session: requests.Session, url: str) -> int | None:
    def request_size(method: str, extra_headers: dict[str, str] | None = None) -> int | None:
        response = session.request(
            method,
            url,
            headers=extra_headers,
            allow_redirects=True,
            stream=True,
            timeout=(15, 60),
        )
        try:
            response.raise_for_status()
            return extract_total_size_from_headers(response.headers)
        finally:
            response.close()

    try:
        size_bytes = request_size("HEAD")
        if size_bytes is not None:
            return size_bytes
    except requests.RequestException:
        pass

    try:
        return request_size("GET", {"Range": "bytes=0-0"})
    except requests.RequestException:
        return None


def exceeds_size_limit(size_bytes: int | None, max_size_bytes: int) -> bool:
    return bool(size_bytes is not None and max_size_bytes > 0 and size_bytes > max_size_bytes)


def format_size_bytes(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "bilinmiyor"
    return f"{round(size_bytes / (1024 * 1024), 1)} MB"


def first_number(*values: Any, default: float = 0.0) -> float:
    for value in values:
        if value in (None, "", False):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return float(default)


def first_text(*values: Any, default: str = "") -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return default


def parse_duration_value(value: Any) -> float:
    if value in (None, "", False):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return 0.0

    try:
        return float(text)
    except (TypeError, ValueError):
        pass

    if ":" in text:
        try:
            parts = [float(part) for part in text.split(":")]
        except ValueError:
            return 0.0
        total = 0.0
        for part in parts:
            total = total * 60 + part
        return total
    return 0.0


def parse_aspect_ratio_text(value: Any) -> tuple[float, float] | None:
    text = str(value or "").strip().lower().replace("-", ":")
    if not text:
        return None
    if ":" in text:
        left, right = text.split(":", 1)
        try:
            width = float(left)
            height = float(right)
            if width > 0 and height > 0:
                return width, height
        except ValueError:
            return None
    return None


def infer_dimensions_from_quality(quality: Any, aspect_ratio: Any = "16:9") -> tuple[int, int]:
    text = str(quality or "").strip().lower()
    if not text:
        return 0, 0

    dimension_match = re.search(r"(\d{3,5})\s*x\s*(\d{3,5})", text)
    if dimension_match:
        return int(dimension_match.group(1)), int(dimension_match.group(2))

    ratio = parse_aspect_ratio_text(aspect_ratio) or (16.0, 9.0)
    ratio_width, ratio_height = ratio

    quality_map = {
        "480p": 480,
        "720p": 720,
        "1080p": 1080,
        "1440p": 1440,
        "2k": 1440,
        "2160p": 2160,
        "4k": 2160,
    }
    target_height = quality_map.get(text)
    if not target_height:
        return 0, 0

    target_width = int(round(target_height * (ratio_width / max(1.0, ratio_height))))
    return target_width, target_height


def score_rendition(width: int, height: int, orientation: str) -> float:
    if not width or not height:
        return -100.0

    target_width, target_height = target_dimensions(orientation)
    aspect = width / max(1, height)
    target_aspect = target_aspect_ratio(orientation)
    orientation_match = orientation_from_dimensions(width, height) == orientation or orientation == "square"
    target_short_side = min(target_width, target_height)
    short_side_ratio = min(width, height) / max(1, target_short_side)

    aspect_score = max(0.0, 60.0 - abs(aspect - target_aspect) * 60.0)
    orientation_score = 35.0 if orientation_match else -35.0
    # Cozunurluk artik sert filtre gibi davranmasin; iyi kadraj ve dogru oran daha agir bassin.
    resolution_score = min(8.0, short_side_ratio * 8.0)
    return orientation_score + aspect_score + resolution_score


def score_candidate(
    width: int,
    height: int,
    duration_seconds: float,
    target_duration: float,
    orientation: str,
    popularity_score: float,
    relevance_rank: int,
) -> float:
    total = score_rendition(width, height, orientation)

    if target_duration > 0 and duration_seconds > 0:
        if duration_seconds >= target_duration:
            total += max(0.0, 18.0 - abs(duration_seconds - target_duration))
        else:
            total += max(-8.0, 8.0 - ((target_duration - duration_seconds) * 2.0))

    total += popularity_score
    total += max(0.0, 12.0 - ((relevance_rank - 1) * 1.5))
    return round(total, 2)


def candidate_raw_sort_key(item: ClipCandidate) -> tuple[float, int, float, int]:
    return (
        item.score,
        item.width * item.height,
        item.duration_seconds,
        -item.relevance_rank,
    )


def candidate_preference_sort_key(item: ClipCandidate, provider_priority: list[str]) -> tuple[float, int, int, float]:
    provider_rank = {provider: index for index, provider in enumerate(provider_priority)}
    return (
        item.score + provider_preference_bonus(item.provider, provider_priority),
        -provider_rank.get(item.provider, 999),
        item.width * item.height,
        item.duration_seconds,
    )


def provider_preference_bonus(provider: str, provider_priority: list[str]) -> float:
    step = parse_float_env("BROLL_PROVIDER_PRIORITY_BONUS_STEP", default=2.5, minimum=0.0, maximum=10.0)
    try:
        provider_index = provider_priority.index(provider)
    except ValueError:
        return 0.0
    return round((len(provider_priority) - provider_index - 1) * step, 2)


def score_image_candidate(
    width: int,
    height: int,
    orientation: str,
    popularity_score: float,
    relevance_rank: int,
) -> float:
    total = score_rendition(width, height, orientation)
    total += popularity_score
    total += max(0.0, 10.0 - ((relevance_rank - 1) * 1.25))
    total -= 12.0
    return round(total, 2)


def extract_freepik_dimensions(item: dict[str, Any]) -> tuple[int, int]:
    thumbnails = item.get("thumbnails") or []
    previews = item.get("previews") or []
    options = item.get("options") or []

    def best_dimensions(blocks: Any) -> tuple[int, int]:
        best_width = 0
        best_height = 0
        best_area = -1
        for block in blocks or []:
            if not isinstance(block, dict):
                continue
            width = int(first_number(block.get("width"), default=0))
            height = int(first_number(block.get("height"), default=0))
            area = width * height
            if width > 0 and height > 0 and area > best_area:
                best_width = width
                best_height = height
                best_area = area
        return best_width, best_height

    option_width, option_height = best_dimensions(options)
    preview_width, preview_height = best_dimensions(previews)
    thumb_width, thumb_height = best_dimensions(thumbnails)
    width = int(
        first_number(
            item.get("width"),
            item.get("w"),
            (item.get("dimensions") or {}).get("width"),
            (item.get("image") or {}).get("width"),
            option_width,
            preview_width,
            thumb_width,
            default=0,
        )
    )
    height = int(
        first_number(
            item.get("height"),
            item.get("h"),
            (item.get("dimensions") or {}).get("height"),
            (item.get("image") or {}).get("height"),
            option_height,
            preview_height,
            thumb_height,
            default=0,
        )
    )

    inferred_width, inferred_height = infer_dimensions_from_quality(
        first_text(
            item.get("quality"),
            (item.get("video") or {}).get("quality"),
            ((options[0] if options else {}) or {}).get("quality"),
            default="",
        ),
        first_text(
            item.get("aspect-ratio"),
            item.get("aspect_ratio"),
            ((options[0] if options else {}) or {}).get("aspect_ratio"),
            ((thumbnails[0] if thumbnails else {}) or {}).get("aspect-ratio"),
            default="16:9",
        ),
    )
    if inferred_width > width and inferred_height > height:
        width, height = inferred_width, inferred_height

    return width, height


def extract_freepik_duration(item: dict[str, Any]) -> float:
    numeric_duration = first_number(
        item.get("duration"),
        item.get("duration_seconds"),
        item.get("video_duration"),
        (item.get("video") or {}).get("duration"),
        default=0.0,
    )
    if numeric_duration > 0:
        return numeric_duration
    return parse_duration_value(
        first_text(
            item.get("duration"),
            item.get("duration_seconds"),
            item.get("video_duration"),
            (item.get("video") or {}).get("duration"),
            default="",
        )
    )


def extract_freepik_author(item: dict[str, Any]) -> str:
    return first_text(
        (item.get("author") or {}).get("name"),
        (item.get("contributor") or {}).get("name"),
        item.get("author_name"),
        default="Freepik",
    )


def extract_freepik_page_url(item: dict[str, Any]) -> str:
    return first_text(
        item.get("url"),
        item.get("page_url"),
        (item.get("links") or {}).get("self"),
        default="",
    )


def extract_freepik_popularity(item: dict[str, Any]) -> float:
    popularity_base = (
        first_number(item.get("downloads"))
        + first_number(item.get("likes")) * 15.0
        + first_number(item.get("views")) * 0.03
        + first_number(item.get("comments")) * 8.0
    )
    return round(min(16.0, math.log1p(popularity_base)), 2) if popularity_base > 0 else 0.0


def resolve_freepik_download(session: requests.Session, asset_id: str) -> tuple[str, str]:
    api_key = os.getenv("FREEPIK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("FREEPIK_API_KEY tanimli degil.")

    payload = request_json(
        session,
        FREEPIK_VIDEO_DOWNLOAD_URL.format(asset_id=asset_id),
        f"Freepik dosya baglantisi ({asset_id})",
        headers={"x-freepik-api-key": api_key},
    )
    data = payload.get("data") or payload
    file_url = first_text(
        (data.get("links") or {}).get("download"),
        data.get("url"),
        data.get("download_url"),
        data.get("link"),
        default="",
    )
    if not file_url:
        raise RuntimeError(f"Freepik download baglantisi donmedi: {asset_id}")

    file_extension = normalize_file_extension(
        first_text(data.get("filename"), data.get("name"), file_url, default=".mp4"),
        fallback=".mp4",
    )
    return file_url, file_extension


def choose_best_pexels_file(video_files: list[dict[str, Any]], orientation: str) -> dict[str, Any] | None:
    best_item: dict[str, Any] | None = None
    best_score = -10_000.0

    for item in video_files or []:
        if str(item.get("file_type", "")).lower() != "video/mp4":
            continue
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        link = str(item.get("link", "") or "").strip()
        if not width or not height or not link:
            continue
        item_score = score_rendition(width, height, orientation)
        if str(item.get("quality", "")).lower() == "hd":
            item_score += 4.0
        if item_score > best_score:
            best_score = item_score
            best_item = item
    return best_item


def choose_best_pixabay_rendition(videos_block: dict[str, Any], orientation: str) -> tuple[str, dict[str, Any]] | None:
    best_name = ""
    best_item: dict[str, Any] | None = None
    best_score = -10_000.0

    for name, item in (videos_block or {}).items():
        if not isinstance(item, dict):
            continue
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        url = str(item.get("url", "") or "").strip()
        if not width or not height or not url:
            continue
        item_score = score_rendition(width, height, orientation)
        if name == "large":
            item_score += 2.0
        if name == "medium":
            item_score += 3.0
        if item_score > best_score:
            best_score = item_score
            best_name = name
            best_item = item

    if not best_item:
        return None
    return best_name, best_item


def search_pexels_candidates(
    session: requests.Session,
    query: str,
    orientation: str,
    target_duration: float,
) -> list[ClipCandidate]:
    api_key = os.getenv("PEXELS_API_KEY", "").strip()
    if not api_key:
        return []

    params = {
        "query": query,
        "orientation": orientation,
        "size": "medium",
        "per_page": min(15, max(3, int(os.getenv("BROLL_PEXELS_PER_PAGE", "8")))),
        "locale": os.getenv("PEXELS_LOCALE", "en-US").strip() or "en-US",
    }
    payload = request_json(
        session,
        PEXELS_SEARCH_URL,
        f"Pexels B-Roll aramasi ({query})",
        headers={"Authorization": api_key},
        params=params,
    )

    candidates: list[ClipCandidate] = []
    for rank, video in enumerate(payload.get("videos", []), start=1):
        selected_file = choose_best_pexels_file(video.get("video_files", []), orientation)
        if not selected_file:
            continue
        width = int(selected_file.get("width") or 0)
        height = int(selected_file.get("height") or 0)
        duration_seconds = float(video.get("duration") or 0.0)
        score = score_candidate(
            width=width,
            height=height,
            duration_seconds=duration_seconds,
            target_duration=target_duration,
            orientation=orientation,
            popularity_score=0.0,
            relevance_rank=rank,
        )
        candidates.append(
            ClipCandidate(
                provider="pexels",
                query=query,
                asset_id=str(video.get("id", "")),
                file_url=str(selected_file.get("link", "")),
                page_url=str(video.get("url", "")),
                width=width,
                height=height,
                duration_seconds=duration_seconds,
                author_name=str((video.get("user") or {}).get("name", "")),
                rendition_label=str(selected_file.get("quality", "mp4")),
                relevance_rank=rank,
                popularity_score=0.0,
                score=score,
            )
        )
    return candidates


def search_pixabay_candidates(
    session: requests.Session,
    query: str,
    orientation: str,
    target_duration: float,
) -> list[ClipCandidate]:
    api_key = os.getenv("PIXABAY_API_KEY", "").strip()
    if not api_key:
        return []

    params = {
        "key": api_key,
        "q": query,
        "lang": os.getenv("PIXABAY_LANG", "en").strip() or "en",
        "order": "popular",
        "safesearch": "true",
        "per_page": min(20, max(3, int(os.getenv("BROLL_PIXABAY_PER_PAGE", "10")))),
    }
    min_width = parse_int_env("BROLL_PIXABAY_MIN_WIDTH", default=0, minimum=0, maximum=4000)
    min_height = parse_int_env("BROLL_PIXABAY_MIN_HEIGHT", default=0, minimum=0, maximum=4000)
    if min_width > 0:
        params["min_width"] = min_width
    if min_height > 0:
        params["min_height"] = min_height
    payload = request_json(session, PIXABAY_SEARCH_URL, f"Pixabay B-Roll aramasi ({query})", params=params)

    candidates: list[ClipCandidate] = []
    for rank, hit in enumerate(payload.get("hits", []), start=1):
        selected = choose_best_pixabay_rendition(hit.get("videos", {}), orientation)
        if not selected:
            continue
        rendition_label, rendition = selected
        width = int(rendition.get("width") or 0)
        height = int(rendition.get("height") or 0)
        duration_seconds = float(hit.get("duration") or 0.0)
        popularity_score = min(
            parse_float_env("BROLL_PROVIDER_POPULARITY_CAP", default=8.0, minimum=0.0, maximum=18.0),
            math.log1p(
                float(hit.get("downloads") or 0)
                + (float(hit.get("views") or 0) * 0.05)
                + (float(hit.get("likes") or 0) * 12)
                + (float(hit.get("comments") or 0) * 8)
            ),
        )
        score = score_candidate(
            width=width,
            height=height,
            duration_seconds=duration_seconds,
            target_duration=target_duration,
            orientation=orientation,
            popularity_score=popularity_score,
            relevance_rank=rank,
        )
        candidates.append(
            ClipCandidate(
                provider="pixabay",
                query=query,
                asset_id=str(hit.get("id", "")),
                file_url=str(rendition.get("url", "")),
                page_url=str(hit.get("pageURL", "")),
                width=width,
                height=height,
                duration_seconds=duration_seconds,
                author_name=str(hit.get("user", "")),
                rendition_label=rendition_label,
                relevance_rank=rank,
                popularity_score=round(popularity_score, 2),
                score=score,
            )
        )
    return candidates


def search_freepik_candidates(
    session: requests.Session,
    query: str,
    orientation: str,
    target_duration: float,
) -> list[ClipCandidate]:
    api_key = os.getenv("FREEPIK_API_KEY", "").strip()
    if not api_key:
        return []

    params = {
        "term": query,
        "page": 1,
        "order": normalize_enum_env("FREEPIK_ORDER", "relevance", ("relevance", "recent", "random")),
    }
    payload = request_json(
        session,
        FREEPIK_VIDEO_SEARCH_URL,
        f"Freepik B-Roll aramasi ({query})",
        headers={"x-freepik-api-key": api_key},
        params=params,
    )
    raw_items = payload.get("data") or payload.get("items") or payload.get("resources") or []
    if not isinstance(raw_items, list):
        logger.warning(
            f"Freepik beklenmeyen cevap formati dondu ({query}). "
            f"Payload anahtarlari: {', '.join(list(payload.keys())[:8])}"
        )
        return []

    include_premium = parse_bool_env("FREEPIK_INCLUDE_PREMIUM", default=False)
    allow_premium_fallback = parse_bool_env("FREEPIK_ALLOW_PREMIUM_FALLBACK", default=False)
    premium_flags = [
        bool(item.get("premium") or item.get("is_premium") or item.get("requires_subscription"))
        for item in raw_items
        if isinstance(item, dict)
    ]
    premium_only_results = bool(premium_flags) and all(premium_flags)
    use_premium_results = include_premium or (allow_premium_fallback and premium_only_results)
    if premium_only_results and not include_premium:
        logger.info(
            f"Freepik sadece premium sonuc verdi: sorgu={query}, raw={len(raw_items)}, "
            f"FREEPIK_INCLUDE_PREMIUM={include_premium}, FREEPIK_ALLOW_PREMIUM_FALLBACK={allow_premium_fallback}"
        )

    candidates: list[ClipCandidate] = []
    premium_filtered = 0
    missing_asset_id = 0
    missing_dimensions = 0
    for rank, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            continue
        is_premium = bool(item.get("premium") or item.get("is_premium") or item.get("requires_subscription"))
        if is_premium and not use_premium_results:
            premium_filtered += 1
            continue

        asset_id = first_text(item.get("id"), item.get("resource_id"), default="")
        if not asset_id:
            missing_asset_id += 1
            continue

        width, height = extract_freepik_dimensions(item)
        if not width or not height:
            missing_dimensions += 1
            continue

        duration_seconds = extract_freepik_duration(item)
        popularity_score = extract_freepik_popularity(item)
        score = score_candidate(
            width=width,
            height=height,
            duration_seconds=duration_seconds,
            target_duration=target_duration,
            orientation=orientation,
            popularity_score=popularity_score,
            relevance_rank=rank,
        )
        candidates.append(
            ClipCandidate(
                provider="freepik",
                query=query,
                asset_id=asset_id,
                file_url="",
                page_url=extract_freepik_page_url(item),
                width=width,
                height=height,
                duration_seconds=duration_seconds,
                author_name=extract_freepik_author(item),
                rendition_label="freepik_video",
                relevance_rank=rank,
                popularity_score=popularity_score,
                score=score,
            )
        )
    if not candidates:
        logger.info(
            f"Freepik aday cikarmadi: sorgu={query}, raw={len(raw_items)}, "
            f"premium_elendi={premium_filtered}, id_eksik={missing_asset_id}, boyut_eksik={missing_dimensions}, "
            f"premium_only={premium_only_results}"
        )
    return candidates


def search_coverr_candidates(
    session: requests.Session,
    query: str,
    orientation: str,
    target_duration: float,
) -> list[ClipCandidate]:
    api_key = os.getenv("COVERR_API_KEY", "").strip()
    if not api_key:
        return []

    params = {
        "api_key": api_key,
        "query": query,
        "page": 0,
        "page_size": parse_int_env("BROLL_COVERR_PAGE_SIZE", default=8, minimum=3, maximum=20),
        "sort": os.getenv("COVERR_SORT", "popular").strip() or "popular",
        "urls": "true",
    }
    payload = request_json(session, COVERR_VIDEO_SEARCH_URL, f"Coverr B-Roll aramasi ({query})", params=params)
    raw_items = (
        payload
        if isinstance(payload, list)
        else payload.get("hits") or payload.get("data") or payload.get("videos") or payload.get("results") or []
    )
    if not isinstance(raw_items, list):
        logger.warning(
            f"Coverr beklenmeyen cevap formati dondu ({query}). "
            f"Payload anahtarlari: {', '.join(list(payload.keys())[:8])}"
        )
        return []

    candidates: list[ClipCandidate] = []
    missing_dimensions = 0
    missing_file_url = 0
    for rank, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            continue

        width = int(first_number(item.get("max_width"), default=0))
        height = int(first_number(item.get("max_height"), default=0))
        urls = item.get("urls") or {}
        file_url = first_text(urls.get("mp4_download"), urls.get("mp4"), default="")
        if not width or not height:
            missing_dimensions += 1
            continue
        if not file_url:
            missing_file_url += 1
            continue

        duration_seconds = first_number(item.get("duration"), default=0.0)
        popularity_score = min(
            parse_float_env("BROLL_PROVIDER_POPULARITY_CAP", default=8.0, minimum=0.0, maximum=18.0),
            math.log1p(first_number(item.get("downloads")) + (first_number(item.get("views")) * 0.1)),
        )
        score = score_candidate(
            width=width,
            height=height,
            duration_seconds=duration_seconds,
            target_duration=target_duration,
            orientation=orientation,
            popularity_score=popularity_score,
            relevance_rank=rank,
        )
        candidates.append(
            ClipCandidate(
                provider="coverr",
                query=query,
                asset_id=first_text(item.get("id"), default=""),
                file_url=file_url,
                page_url=first_text(item.get("thumbnail"), default=""),
                width=width,
                height=height,
                duration_seconds=duration_seconds,
                author_name="Coverr",
                rendition_label="coverr_mp4",
                relevance_rank=rank,
                popularity_score=round(popularity_score, 2),
                score=score,
            )
        )
    if not candidates:
        payload_keys = ", ".join(list(payload.keys())[:8]) if isinstance(payload, dict) else type(payload).__name__
        logger.info(
            f"Coverr aday cikarmadi: sorgu={query}, raw={len(raw_items)}, "
            f"boyut_eksik={missing_dimensions}, url_eksik={missing_file_url}, payload_keys={payload_keys}"
        )
    return candidates


def search_openverse_candidates(
    session: requests.Session,
    query: str,
    orientation: str,
) -> list[ClipCandidate]:
    if not parse_bool_env("BROLL_ALLOW_IMAGE_FALLBACK", default=False):
        return []

    params = {
        "q": query,
        "page_size": parse_int_env("BROLL_OPENVERSE_PAGE_SIZE", default=8, minimum=3, maximum=20),
    }
    payload = request_json(session, OPENVERSE_IMAGE_SEARCH_URL, f"Openverse gorsel aramasi ({query})", params=params)
    raw_items = payload.get("results") or []
    if not isinstance(raw_items, list):
        return []

    candidates: list[ClipCandidate] = []
    for rank, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            continue
        width = int(first_number(item.get("width"), default=0))
        height = int(first_number(item.get("height"), default=0))
        file_url = first_text(item.get("url"), item.get("thumbnail"), default="")
        if not file_url or not width or not height:
            continue

        score = score_image_candidate(
            width=width,
            height=height,
            orientation=orientation,
            popularity_score=0.0,
            relevance_rank=rank,
        )
        candidates.append(
            ClipCandidate(
                provider="openverse",
                query=query,
                asset_id=first_text(item.get("id"), default=file_url),
                file_url=file_url,
                page_url=first_text(item.get("foreign_landing_url"), item.get("creator_url"), default=""),
                width=width,
                height=height,
                duration_seconds=0.0,
                author_name=first_text(item.get("creator"), default="Openverse"),
                rendition_label="image_fallback",
                relevance_rank=rank,
                popularity_score=0.0,
                score=score,
                media_type="image",
                file_extension=normalize_file_extension(file_url, fallback=".jpg"),
            )
        )
    return candidates


def choose_best_candidate(candidates: list[ClipCandidate], provider_priority: list[str]) -> ClipCandidate | None:
    if not candidates:
        return None
    return max(candidates, key=lambda item: candidate_preference_sort_key(item, provider_priority))


def rank_candidates(candidates: list[ClipCandidate], provider_priority: list[str]) -> list[ClipCandidate]:
    return sorted(candidates, key=lambda item: candidate_preference_sort_key(item, provider_priority), reverse=True)


def select_download_candidates(
    candidates: list[ClipCandidate],
    provider_priority: list[str],
    max_count: int,
) -> list[ClipCandidate]:
    if not candidates or max_count <= 0:
        return []

    ranked_candidates = rank_candidates(candidates, provider_priority)
    best_by_provider: list[ClipCandidate] = []
    seen_providers: set[str] = set()

    for candidate in ranked_candidates:
        if candidate.provider in seen_providers:
            continue
        seen_providers.add(candidate.provider)
        best_by_provider.append(candidate)

    target_count = max(max_count, len(best_by_provider))
    selected_by_marker: dict[str, ClipCandidate] = {}

    for candidate in best_by_provider:
        marker = candidate_asset_marker(candidate)
        if marker and marker not in selected_by_marker:
            selected_by_marker[marker] = candidate

    for candidate in ranked_candidates:
        if len(selected_by_marker) >= target_count:
            break
        marker = candidate_asset_marker(candidate)
        if marker and marker in selected_by_marker:
            continue
        if marker:
            selected_by_marker[marker] = candidate

    return rank_candidates(list(selected_by_marker.values()), provider_priority)


def build_download_attempt_pool(
    candidates: list[ClipCandidate],
    provider_priority: list[str],
    max_count: int,
) -> list[ClipCandidate]:
    prioritized_candidates = select_download_candidates(candidates, provider_priority, max_count)
    ordered_candidates: list[ClipCandidate] = []
    seen_markers: set[str] = set()

    for candidate in prioritized_candidates + rank_candidates(candidates, provider_priority):
        marker = candidate_asset_marker(candidate)
        if not marker or marker in seen_markers:
            continue
        seen_markers.add(marker)
        ordered_candidates.append(candidate)

    return ordered_candidates


def merge_unique_candidates(existing_candidates: list[ClipCandidate], new_candidates: list[ClipCandidate]) -> int:
    existing_index_by_marker = {
        candidate_asset_marker(candidate): index for index, candidate in enumerate(existing_candidates)
    }
    added_count = 0

    for candidate in new_candidates:
        marker = candidate_asset_marker(candidate)
        existing_index = existing_index_by_marker.get(marker)
        if existing_index is None:
            existing_index_by_marker[marker] = len(existing_candidates)
            existing_candidates.append(candidate)
            added_count += 1
            continue

        if candidate_raw_sort_key(candidate) > candidate_raw_sort_key(existing_candidates[existing_index]):
            existing_candidates[existing_index] = candidate

    return added_count


def resolve_candidate_download_url(session: requests.Session, candidate: ClipCandidate) -> ClipCandidate:
    if candidate.file_url:
        return candidate
    if candidate.provider == "freepik":
        file_url, file_extension = resolve_freepik_download(session, candidate.asset_id)
        return replace(candidate, file_url=file_url, file_extension=file_extension)
    return candidate


def build_provider_diagnostics(
    provider_candidates_map: dict[str, list[ClipCandidate]],
    provider_priority: list[str],
    excluded_markers: set[str],
) -> dict[str, dict[str, Any]]:
    diagnostics: dict[str, dict[str, Any]] = {}

    for provider in provider_priority:
        candidates = provider_candidates_map.get(provider, [])
        if not candidates:
            diagnostics[provider] = {"candidate_count": 0, "excluded_count": 0}
            continue

        raw_best = max(candidates, key=candidate_raw_sort_key)
        eligible_candidates = [
            candidate for candidate in candidates if candidate_asset_marker(candidate) not in excluded_markers
        ]
        eligible_best = max(eligible_candidates, key=candidate_raw_sort_key) if eligible_candidates else None
        provider_bonus = provider_preference_bonus(provider, provider_priority)

        diagnostics[provider] = {
            "candidate_count": len(candidates),
            "excluded_count": len(candidates) - len(eligible_candidates),
            "raw_best_score": raw_best.score,
            "raw_best_final_score": round(raw_best.score + provider_bonus, 2),
            "raw_best_asset_marker": candidate_asset_marker(raw_best),
            "eligible_best_score": eligible_best.score if eligible_best else None,
            "eligible_best_final_score": round(eligible_best.score + provider_bonus, 2) if eligible_best else None,
            "eligible_best_asset_marker": candidate_asset_marker(eligible_best) if eligible_best else "",
        }

    return diagnostics


def format_provider_diagnostics_text(provider_diagnostics: dict[str, dict[str, Any]]) -> str:
    parts: list[str] = []
    for provider, data in provider_diagnostics.items():
        if not isinstance(data, dict):
            continue

        candidate_count = int(data.get("candidate_count", 0) or 0)
        excluded_count = int(data.get("excluded_count", 0) or 0)
        if candidate_count <= 0:
            parts.append(f"{provider}=0")
            continue

        eligible_score = data.get("eligible_best_score")
        eligible_final = data.get("eligible_best_final_score")
        if eligible_score is None or eligible_final is None:
            parts.append(
                f"{provider}: aday={candidate_count}, elenen={excluded_count}, uygun_yok, "
                f"en_iyi_ham={data.get('raw_best_score')}, en_iyi_final={data.get('raw_best_final_score')}"
            )
            continue

        parts.append(
            f"{provider}: aday={candidate_count}, elenen={excluded_count}, "
            f"uygun_ham={eligible_score}, uygun_final={eligible_final}"
        )

    return " | ".join(parts)


def search_candidates_for_provider(
    session: requests.Session,
    provider: str,
    query: str,
    orientation: str,
    target_duration: float,
) -> list[ClipCandidate]:
    if provider == "pexels":
        return search_pexels_candidates(session, query, orientation, target_duration)
    if provider == "pixabay":
        return search_pixabay_candidates(session, query, orientation, target_duration)
    if provider == "freepik":
        return search_freepik_candidates(session, query, orientation, target_duration)
    if provider == "coverr":
        return search_coverr_candidates(session, query, orientation, target_duration)
    if provider == "openverse":
        return search_openverse_candidates(session, query, orientation)
    return []


def download_asset(
    session: requests.Session,
    candidate: ClipCandidate,
    target_path: Path,
    *,
    overwrite: bool = False,
) -> str:
    if target_path.exists() and not overwrite:
        logger.info(f"Mevcut dosya yeniden indirilmeyecek: {target_path.name}")
        return "existing"
    if target_path.exists() and overwrite:
        logger.info(f"Mevcut dosya silinip yeniden indirilecek: {target_path.name}")

    temporary_path = target_path.with_suffix(target_path.suffix + ".part")
    if temporary_path.exists():
        temporary_path.unlink()

    def action() -> None:
        with session.get(candidate.file_url, stream=True, timeout=(15, 180)) as response:
            response.raise_for_status()
            with open(temporary_path, "wb") as file_handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file_handle.write(chunk)

    retry_with_backoff(
        action,
        f"{candidate.provider.title()} dosya indirme ({target_path.name})",
        logger,
        max_attempts=4,
        base_delay_seconds=15,
        max_delay_seconds=120,
    )

    if target_path.exists():
        target_path.unlink()
    temporary_path.replace(target_path)
    return "downloaded"


def save_download_report(output_dir: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    json_path = output_dir / "automatic_broll_download_report.json"
    txt_path = output_dir / "Automatic_B_Roll_Downloader.txt"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "=== AUTOMATIC B-ROLL DOWNLOADER RAPORU ===",
        "",
        f"Plan dosyasi: {payload.get('plan_path', '')}",
        f"Cikti klasoru: {payload.get('output_dir', '')}",
        f"Provider sirasi: {', '.join(payload.get('provider_priority', []))}",
        f"Downloaded: {payload.get('downloaded_count', 0)}",
        f"Existing: {payload.get('existing_count', 0)}",
        f"Planned: {payload.get('planned_count', 0)}",
        f"Failed: {payload.get('failed_count', 0)}",
        f"Candidate files: {payload.get('candidate_file_count', 0)}",
        f"Dry run: {payload.get('dry_run', False)}",
        f"Overwrite: {payload.get('overwrite', False)}",
        f"Avoid previous assets: {payload.get('avoid_previous_assets', False)}",
        "",
        "PROVIDER OZETI",
        "-" * 72,
    ]

    provider_summary = payload.get("provider_summary", {}) or {}
    for provider_name, provider_data in provider_summary.items():
        if not isinstance(provider_data, dict):
            continue
        lines.append(
            f"{provider_name}: secildi={provider_data.get('chosen_count', 0)}, "
            f"indirilen_dosya={provider_data.get('downloaded_file_count', 0)}, "
            f"aday={provider_data.get('candidate_count', 0)}, hata={provider_data.get('error_count', 0)}"
        )
    lines.extend(
        [
            "",
        "DETAYLAR",
        "-" * 72,
        ]
    )

    for item in payload.get("items", []):
        lines.append(f"[{item.get('index', 0)}] {item.get('timestamp', '')}")
        lines.append(f"Orijinal sorgu: {item.get('stock_search_query', '')}")
        if item.get("search_query_used"):
            lines.append(f"Kullanilan sorgu: {item['search_query_used']}")
        lines.append(f"Durum: {item.get('status', '')}")
        if item.get("provider"):
            lines.append(f"Provider: {item['provider']}")
        if item.get("media_type"):
            lines.append(f"Medya tipi: {item['media_type']}")
        if item.get("asset_marker"):
            lines.append(f"Asset marker: {item['asset_marker']}")
        if item.get("provider_candidate_counts"):
            counts_text = ", ".join(f"{key}={value}" for key, value in item["provider_candidate_counts"].items())
            lines.append(f"Provider adaylari: {counts_text}")
        if item.get("provider_diagnostics"):
            lines.append(f"Provider skorlari: {format_provider_diagnostics_text(item['provider_diagnostics'])}")
        if item.get("previous_asset_count") is not None:
            lines.append(f"Dislanan onceki asset sayisi: {item.get('previous_asset_count', 0)}")
        downloaded_candidates = item.get("downloaded_candidates") or []
        if isinstance(downloaded_candidates, list) and downloaded_candidates:
            lines.append(f"Indirilen aday sayisi: {len(downloaded_candidates)}")
            for candidate_info in downloaded_candidates:
                if not isinstance(candidate_info, dict):
                    continue
                badge = " [PRIMARY]" if candidate_info.get("is_primary") else ""
                lines.append(
                    f"Aday {candidate_info.get('option_index', '?')}{badge}: "
                    f"{candidate_info.get('provider', '')} | skor={candidate_info.get('score', '')} | "
                    f"durum={candidate_info.get('status', '')} | dosya={candidate_info.get('local_path', '')}"
                )
        if item.get("local_path"):
            lines.append(f"Dosya: {item['local_path']}")
        if item.get("score") is not None:
            lines.append(f"Skor: {item['score']}")
        if item.get("resolution"):
            lines.append(f"Cozunurluk: {item['resolution']}")
        if item.get("duration_seconds"):
            lines.append(f"Sure: {item['duration_seconds']} saniye")
        if item.get("page_url"):
            lines.append(f"Kaynak: {item['page_url']}")
        if item.get("error"):
            lines.append(f"Hata: {item['error']}")
        if item.get("provider_errors"):
            lines.append(f"Provider hatalari: {' | '.join(item['provider_errors'])}")
        if item.get("reason"):
            lines.append(f"Neden: {item['reason']}")
        lines.append("")

    txt_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return json_path, txt_path


def load_previous_download_state(output_dir: Path) -> tuple[dict[int, set[str]], dict[str, Path], dict[str, str]]:
    report_path = output_dir / "automatic_broll_download_report.json"
    if not report_path.exists():
        return {}, {}, {}

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return {}, {}, {}

    history: dict[int, set[str]] = {}
    marker_to_path: dict[str, Path] = {}
    path_to_marker: dict[str, str] = {}

    def register_entry(item_index: int, provider: str, asset_id: str, file_url: str, local_path_text: str) -> None:
        if item_index <= 0 or not provider:
            return

        marker = build_asset_marker(provider, asset_id, file_url)
        if marker:
            history.setdefault(item_index, set()).add(marker)

        if not local_path_text:
            return

        local_path = Path(local_path_text)
        if marker:
            path_to_marker[normalize_path_key(local_path)] = marker
            if local_path.exists() and marker not in marker_to_path:
                marker_to_path[marker] = local_path

    for item in payload.get("items", []):
        if not isinstance(item, dict):
            continue
        try:
            item_index = int(item.get("index", 0) or 0)
        except (TypeError, ValueError):
            continue
        register_entry(
            item_index,
            str(item.get("provider", "") or "").strip().lower(),
            str(item.get("asset_id", "") or "").strip(),
            str(item.get("file_url", "") or "").strip(),
            str(item.get("local_path", "") or "").strip(),
        )

        for candidate_info in item.get("downloaded_candidates") or []:
            if not isinstance(candidate_info, dict):
                continue
            register_entry(
                item_index,
                str(candidate_info.get("provider", "") or "").strip().lower(),
                str(candidate_info.get("asset_id", "") or "").strip(),
                str(candidate_info.get("file_url", "") or "").strip(),
                str(candidate_info.get("local_path", "") or "").strip(),
            )

    return history, marker_to_path, path_to_marker


def run_automatic(
    plan_path: Path | None = None,
    *,
    output_dir: Path | None = None,
    provider_priority: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    reload_project_env(override=True)
    plan_files = find_broll_plan_files()
    selected_plan = plan_path or find_preferred_plan_file(plan_files) or (plan_files[0] if plan_files else None)
    if not selected_plan or not selected_plan.exists():
        raise RuntimeError("B-Roll plan dosyasi bulunamadi. Once B-Roll Prompt Uretici calismali.")

    priority = normalize_provider_priority(provider_priority)
    enabled_providers = available_providers(priority)
    if not enabled_providers:
        status = provider_env_status()
        raise RuntimeError(
            "Kullanilabilir provider bulunamadi. `PEXELS_API_KEY`, `PIXABAY_API_KEY`, `FREEPIK_API_KEY`, `COVERR_API_KEY` girin veya "
            "`BROLL_ALLOW_IMAGE_FALLBACK=true` ile Openverse gorsel fallback acin. "
            f"Durum: pexels={status['pexels']}, pixabay={status['pixabay']}, freepik={status['freepik']}, "
            f"coverr={status['coverr']}, openverse_fallback={status['openverse_fallback']}"
        )

    items = load_plan_items(selected_plan)
    if not items:
        raise RuntimeError(f"B-Roll planinda indirilecek gecerli kayit yok: {selected_plan.name}")

    plan_stem = extract_plan_stem(selected_plan)
    destination = ensure_directory(output_dir or (DOWNLOAD_ROOT / plan_stem))
    session = create_session()
    overwrite = parse_bool_env("BROLL_OVERWRITE", default=True)
    avoid_previous_assets = parse_bool_env("BROLL_AVOID_PREVIOUS_ASSETS", default=True)
    max_download_size_mb = parse_int_env("BROLL_MAX_FILE_SIZE_MB", default=300, minimum=0, maximum=10_000)
    max_download_size_bytes = max_download_size_mb * 1024 * 1024
    max_download_options = parse_int_env(
        "BROLL_MAX_DOWNLOAD_OPTIONS",
        default=DEFAULT_MAX_CANDIDATE_DOWNLOAD_OPTIONS,
        minimum=1,
        maximum=8,
    )
    search_cache_ttl_seconds = parse_int_env(
        "BROLL_SEARCH_CACHE_TTL_SECONDS",
        default=21600,
        minimum=0,
        maximum=604800,
    )
    search_parallelism = parse_int_env("BROLL_SEARCH_PARALLELISM", default=3, minimum=1, maximum=8)
    remote_size_probe_mode = normalize_enum_env(
        "BROLL_REMOTE_SIZE_PROBE_MODE",
        "smart",
        ("smart", "always", "never"),
    )
    previous_asset_history, previous_asset_files, known_path_markers = load_previous_download_state(destination)
    blocked_previous_markers = (
        {marker for markers in previous_asset_history.values() for marker in markers}
        if avoid_previous_assets
        else set()
    )
    search_cache = _load_search_cache()
    search_cache_dirty = _prune_search_cache(search_cache, search_cache_ttl_seconds)

    logger.info(
        f"B-Roll ayarlari: overwrite={overwrite}, avoid_previous_assets={avoid_previous_assets}, "
        f"known_assets={len(previous_asset_files)}, max_download_options={max_download_options}, "
        f"search_parallelism={search_parallelism}, remote_size_probe_mode={remote_size_probe_mode}"
    )

    cached_downloads: dict[str, Path] = dict(previous_asset_files)
    remote_size_cache: dict[str, int | None] = {}
    selected_asset_markers: set[str] = set()
    report_items: list[dict[str, Any]] = []
    provider_summary = {
        provider: {"candidate_count": 0, "chosen_count": 0, "downloaded_file_count": 0, "error_count": 0}
        for provider in enabled_providers
    }
    try:
        for item in items:
            item_index = int(item.get("index", 0) or 0)
            original_query = clean_search_query(str(item.get("stock_search_query", "") or ""))
            orientation = str(item.get("orientation", "landscape") or "landscape")
            start_seconds, end_seconds = parse_timestamp_range(str(item.get("timestamp", "") or ""))
            target_duration = max(0.0, end_seconds - start_seconds)
            english_prompt = str(item.get("english_prompt", "") or "")

            logger.info(f"B-Roll araniyor: {original_query} | orientation={orientation}")
            selected_candidates: list[ClipCandidate] = []
            provider_errors: list[str] = []
            query_used = original_query
            provider_candidate_counts: dict[str, int] = {provider: 0 for provider in enabled_providers}
            provider_diagnostics: dict[str, dict[str, Any]] = {}
            duplicate_fallback_candidates: list[ClipCandidate] = []
            download_attempt_candidates: list[ClipCandidate] = []
            duplicate_fallback_attempt_candidates: list[ClipCandidate] = []
            duplicate_fallback_query = original_query
            duplicate_fallback_counts = provider_candidate_counts.copy()
            duplicate_fallback_diagnostics: dict[str, dict[str, Any]] = {}

            aggregated_provider_candidates: dict[str, list[ClipCandidate]] = {
                provider: [] for provider in enabled_providers
            }
            query_variants = build_query_variants(original_query, english_prompt)
            for variant_index, query_variant in enumerate(query_variants, start=1):
                variant_errors: list[str] = []
                excluded_markers = set(selected_asset_markers)
                excluded_markers.update(blocked_previous_markers)

                providers_to_query: list[str] = []
                for provider in enabled_providers:
                    current_candidates = aggregated_provider_candidates[provider]
                    if variant_index == 1:
                        providers_to_query.append(provider)
                        continue

                    has_eligible_candidate = any(
                        candidate_asset_marker(candidate) not in excluded_markers for candidate in current_candidates
                    )
                    if not current_candidates or not has_eligible_candidate:
                        providers_to_query.append(provider)

                if not providers_to_query:
                    break

                providers_to_fetch: list[str] = []
                for provider in providers_to_query:
                    cached_candidates = _get_cached_search_candidates(
                        search_cache,
                        provider,
                        query_variant,
                        orientation,
                        target_duration,
                        search_cache_ttl_seconds,
                    )
                    if cached_candidates is None:
                        providers_to_fetch.append(provider)
                        continue
                    added_count = merge_unique_candidates(
                        aggregated_provider_candidates[provider],
                        cached_candidates,
                    )
                    provider_summary[provider]["candidate_count"] += added_count

                if providers_to_fetch:
                    if search_parallelism > 1 and len(providers_to_fetch) > 1:
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=min(search_parallelism, len(providers_to_fetch))
                        ) as executor:
                            future_map = {
                                executor.submit(
                                    _search_provider_with_fresh_session,
                                    provider,
                                    query_variant,
                                    orientation,
                                    target_duration,
                                ): provider
                                for provider in providers_to_fetch
                            }
                            for future in concurrent.futures.as_completed(future_map):
                                provider = future_map[future]
                                try:
                                    provider_candidates = future.result()
                                    _store_search_candidates(
                                        search_cache,
                                        provider,
                                        query_variant,
                                        orientation,
                                        target_duration,
                                        provider_candidates,
                                    )
                                    search_cache_dirty = True
                                    added_count = merge_unique_candidates(
                                        aggregated_provider_candidates[provider],
                                        provider_candidates,
                                    )
                                    provider_summary[provider]["candidate_count"] += added_count
                                except Exception as exc:
                                    message = f"{provider.title()} aramasi basarisiz oldu: {exc}"
                                    logger.warning(message)
                                    variant_errors.append(message)
                                    provider_summary[provider]["error_count"] += 1
                    else:
                        for provider in providers_to_fetch:
                            try:
                                provider_candidates = search_candidates_for_provider(
                                    session,
                                    provider,
                                    query_variant,
                                    orientation,
                                    target_duration,
                                )
                                _store_search_candidates(
                                    search_cache,
                                    provider,
                                    query_variant,
                                    orientation,
                                    target_duration,
                                    provider_candidates,
                                )
                                search_cache_dirty = True
                                added_count = merge_unique_candidates(
                                    aggregated_provider_candidates[provider],
                                    provider_candidates,
                                )
                                provider_summary[provider]["candidate_count"] += added_count
                            except Exception as exc:
                                message = f"{provider.title()} aramasi basarisiz oldu: {exc}"
                                logger.warning(message)
                                variant_errors.append(message)
                                provider_summary[provider]["error_count"] += 1

                provider_errors.extend(variant_errors)
                provider_candidate_counts = {
                    provider: len(aggregated_provider_candidates[provider]) for provider in enabled_providers
                }
                collected_candidates = [
                    candidate
                    for provider in enabled_providers
                    for candidate in aggregated_provider_candidates[provider]
                ]
                if not collected_candidates:
                    continue

                provider_diagnostics = build_provider_diagnostics(
                    aggregated_provider_candidates,
                    enabled_providers,
                    excluded_markers,
                )
                duplicate_fallback_candidates = select_download_candidates(
                    collected_candidates,
                    enabled_providers,
                    max_download_options,
                )
                duplicate_fallback_attempt_candidates = build_download_attempt_pool(
                    collected_candidates,
                    enabled_providers,
                    max_download_options,
                )
                duplicate_fallback_query = query_variant
                duplicate_fallback_counts = provider_candidate_counts.copy()
                duplicate_fallback_diagnostics = provider_diagnostics

                filtered_candidates = [
                    candidate
                    for candidate in collected_candidates
                    if candidate_asset_marker(candidate) not in excluded_markers
                ]
                if not filtered_candidates:
                    continue

                selected_candidates = select_download_candidates(
                    filtered_candidates,
                    enabled_providers,
                    max_download_options,
                )
                download_attempt_candidates = build_download_attempt_pool(
                    filtered_candidates,
                    enabled_providers,
                    max_download_options,
                )
                query_used = query_variant

                providers_still_missing = []
                for provider in enabled_providers:
                    provider_has_eligible_candidate = any(
                        candidate_asset_marker(candidate) not in excluded_markers
                        for candidate in aggregated_provider_candidates[provider]
                    )
                    if not provider_has_eligible_candidate:
                        providers_still_missing.append(provider)

                if not providers_still_missing:
                    break

            if not selected_candidates and duplicate_fallback_candidates:
                selected_candidates = duplicate_fallback_candidates
                download_attempt_candidates = duplicate_fallback_attempt_candidates
                query_used = duplicate_fallback_query
                provider_candidate_counts = duplicate_fallback_counts
                provider_diagnostics = duplicate_fallback_diagnostics

            if not selected_candidates:
                report_items.append(
                    {
                        "index": item_index,
                        "timestamp": item.get("timestamp", ""),
                        "stock_search_query": original_query,
                        "search_query_used": query_used,
                        "reason": item.get("reason", ""),
                        "english_prompt": english_prompt,
                        "status": "failed",
                        "error": "; ".join(provider_errors) or "Uygun stok medya bulunamadi.",
                        "provider_candidate_counts": provider_candidate_counts,
                        "provider_diagnostics": provider_diagnostics,
                        "provider_errors": provider_errors,
                    }
                )
                continue

            downloaded_candidates: list[dict[str, Any]] = []
            primary_candidate: ClipCandidate | None = None
            primary_status = ""
            primary_local_path = ""
            primary_marker = ""

            if not download_attempt_candidates:
                download_attempt_candidates = list(selected_candidates)

            for raw_candidate in download_attempt_candidates:
                if len(downloaded_candidates) >= max_download_options:
                    break

                try:
                    candidate = resolve_candidate_download_url(session, raw_candidate)
                except Exception as exc:
                    provider_summary[raw_candidate.provider]["error_count"] += 1
                    provider_errors.append(
                        f"{raw_candidate.provider.title()} aday baglantisi alinamadi: {exc}"
                    )
                    continue

                option_index = len(downloaded_candidates) + 1
                candidate_base_path = build_asset_path(
                    destination,
                    item_index,
                    str(item.get("timestamp", "")),
                    original_query,
                    candidate.provider,
                    candidate.file_extension,
                )
                local_path = build_alternative_asset_path(candidate_base_path, candidate, option_index)
                resolved_local_path = resolve_target_path_for_candidate(local_path, candidate, known_path_markers)
                if resolved_local_path != local_path:
                    logger.info(
                        f"Ayni dosya adi farkli assete ait gorundugu icin yeni ad kullaniliyor: "
                        f"{local_path.name} -> {resolved_local_path.name}"
                    )
                local_path = resolved_local_path

                cache_key = candidate_asset_marker(candidate)
                cached_source = cached_downloads.get(cache_key)
                same_cached_target = (
                    bool(cached_source)
                    and cached_source.exists()
                    and normalize_path_key(cached_source) == normalize_path_key(local_path)
                )
                if cached_source and cached_source.exists() and not same_cached_target:
                    cached_size_bytes = cached_source.stat().st_size
                    if exceeds_size_limit(cached_size_bytes, max_download_size_bytes):
                        provider_errors.append(
                            f"{candidate.provider.title()} aday atlandi: {cached_source.name} "
                            f"boyut={format_size_bytes(cached_size_bytes)} > {max_download_size_mb} MB"
                        )
                        continue
                    if dry_run:
                        status = "planned_copy"
                    else:
                        shutil.copy2(cached_source, local_path)
                        status = "copied_from_cache"
                else:
                    existing_size_bytes = local_path.stat().st_size if local_path.exists() else None
                    if exceeds_size_limit(existing_size_bytes, max_download_size_bytes):
                        provider_errors.append(
                            f"{candidate.provider.title()} aday atlandi: {local_path.name} "
                            f"boyut={format_size_bytes(existing_size_bytes)} > {max_download_size_mb} MB"
                        )
                        continue

                    if dry_run:
                        status = "planned_download"
                    else:
                        remote_size_bytes = None
                        if _should_probe_remote_size(
                            candidate,
                            option_index,
                            max_download_size_bytes,
                            remote_size_probe_mode,
                        ):
                            remote_size_bytes = remote_size_cache.get(candidate.file_url)
                            if candidate.file_url and candidate.file_url not in remote_size_cache:
                                remote_size_bytes = probe_remote_file_size_bytes(session, candidate.file_url)
                                remote_size_cache[candidate.file_url] = remote_size_bytes

                        if exceeds_size_limit(remote_size_bytes, max_download_size_bytes):
                            provider_errors.append(
                                f"{candidate.provider.title()} aday atlandi: {candidate.asset_id or candidate.file_url} "
                                f"boyut={format_size_bytes(remote_size_bytes)} > {max_download_size_mb} MB"
                            )
                            continue

                        try:
                            status = download_asset(session, candidate, local_path, overwrite=overwrite)
                        except Exception as exc:
                            provider_summary[candidate.provider]["error_count"] += 1
                            provider_errors.append(f"{candidate.provider.title()} aday {option_index} indirilemedi: {exc}")
                            continue

                if cache_key and status in {"downloaded", "existing", "copied_from_cache"}:
                    cached_downloads[cache_key] = local_path
                    selected_asset_markers.add(cache_key)
                    known_path_markers[normalize_path_key(local_path)] = cache_key
                    provider_summary[candidate.provider]["downloaded_file_count"] += 1

                candidate_record = {
                    "option_index": option_index,
                    "provider": candidate.provider,
                    "asset_marker": cache_key,
                    "asset_id": candidate.asset_id,
                    "status": status,
                    "score": candidate.score,
                    "final_score": round(
                        candidate.score + provider_preference_bonus(candidate.provider, enabled_providers), 2
                    ),
                    "resolution": f"{candidate.width}x{candidate.height}",
                    "duration_seconds": round(candidate.duration_seconds, 2),
                    "page_url": candidate.page_url,
                    "file_url": candidate.file_url,
                    "local_path": str(local_path),
                    "is_primary": False,
                }
                downloaded_candidates.append(candidate_record)

                if primary_candidate is None:
                    primary_candidate = candidate
                    primary_status = status
                    primary_local_path = str(local_path)
                    primary_marker = cache_key
                    candidate_record["is_primary"] = True

            if not primary_candidate:
                report_items.append(
                    {
                        "index": item_index,
                        "timestamp": item.get("timestamp", ""),
                        "stock_search_query": original_query,
                        "search_query_used": query_used,
                        "reason": item.get("reason", ""),
                        "english_prompt": english_prompt,
                        "status": "failed",
                        "error": "; ".join(provider_errors) or "Adaylar bulundu ama indirilemedi.",
                        "provider_candidate_counts": provider_candidate_counts,
                        "provider_diagnostics": provider_diagnostics,
                        "provider_errors": provider_errors,
                    }
                )
                continue

            provider_summary[primary_candidate.provider]["chosen_count"] += 1

            report_items.append(
                {
                    "index": item_index,
                    "timestamp": item.get("timestamp", ""),
                    "stock_search_query": original_query,
                    "search_query_used": primary_candidate.query or query_used,
                    "reason": item.get("reason", ""),
                    "english_prompt": english_prompt,
                    "orientation": orientation,
                    "status": primary_status,
                    "provider": primary_candidate.provider,
                    "media_type": primary_candidate.media_type,
                    "provider_candidate_counts": provider_candidate_counts,
                    "provider_diagnostics": provider_diagnostics,
                    "provider_errors": provider_errors,
                    "previous_asset_count": len(previous_asset_history.get(item_index, set())),
                    "asset_marker": primary_marker,
                    "asset_id": primary_candidate.asset_id,
                    "resolution": f"{primary_candidate.width}x{primary_candidate.height}",
                    "duration_seconds": round(primary_candidate.duration_seconds, 2),
                    "score": primary_candidate.score,
                    "page_url": primary_candidate.page_url,
                    "file_url": primary_candidate.file_url,
                    "author_name": primary_candidate.author_name,
                    "rendition_label": primary_candidate.rendition_label,
                    "local_path": primary_local_path,
                    "candidate": asdict(primary_candidate),
                    "downloaded_candidates": downloaded_candidates,
                }
            )
    finally:
        session.close()
        if search_cache_dirty:
            _save_search_cache(search_cache)

    payload = {
        "title": "Automatic B-Roll Downloader",
        "plan_path": str(selected_plan),
        "output_dir": str(destination),
        "provider_priority": enabled_providers,
        "max_download_options": max_download_options,
        "dry_run": dry_run,
        "overwrite": overwrite,
        "avoid_previous_assets": avoid_previous_assets,
        "downloaded_count": sum(1 for item in report_items if item.get("status") in {"downloaded", "copied_from_cache"}),
        "existing_count": sum(1 for item in report_items if item.get("status") == "existing"),
        "planned_count": sum(1 for item in report_items if item.get("status") in {"planned_download", "planned_copy"}),
        "failed_count": sum(1 for item in report_items if item.get("status") == "failed"),
        "candidate_file_count": sum(
            len(item.get("downloaded_candidates") or [])
            for item in report_items
            if isinstance(item, dict)
        ),
        "provider_summary": provider_summary,
        "items": report_items,
    }
    json_path, txt_path = save_download_report(destination, payload)
    payload["json_path"] = str(json_path)
    payload["txt_path"] = str(txt_path)
    return payload


def run() -> dict[str, Any] | None:
    print("\n" + "=" * 60)
    print("AUTOMATIC B-ROLL DOWNLOADER")
    print("=" * 60)
    print("Desteklenen providerlar: Pexels, Pixabay, Freepik, Coverr")
    print("Opsiyonel gorsel fallback: Openverse (BROLL_ALLOW_IMAGE_FALLBACK=true)")
    print("API key ortam degiskenleri: PEXELS_API_KEY, PIXABAY_API_KEY, FREEPIK_API_KEY, COVERR_API_KEY")
    print("Mixkit resmi API sunmadigi icin entegre edilmedi.")

    plan_files = find_broll_plan_files()
    if not plan_files:
        logger.error("B-Roll plan dosyasi bulunamadi. Once B-Roll Prompt Uretici calismali.")
        print("B-Roll plan dosyasi bulunamadi. Once B-Roll Prompt Uretici calismali.")
        return None

    selected_plan = select_plan_file(plan_files)
    if not selected_plan:
        return None

    try:
        result = run_automatic(selected_plan, dry_run=False)
    except Exception as exc:
        logger.error(f"Automatic B-Roll Downloader basarisiz oldu: {exc}")
        print(f"Automatic B-Roll Downloader basarisiz oldu: {exc}")
        return None

    print(f"\nJSON rapor: {result['json_path']}")
    print(f"TXT rapor:  {result['txt_path']}")
    print(f"Indirme klasoru: {result['output_dir']}")
    print("=" * 60)
    return result
