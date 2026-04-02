"""Prepared folder parsing for the YouTube draft uploader."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from uploader.constants import SUBTITLE_FILE_MAP, THUMBNAIL_EXTENSIONS, VIDEO_EXTENSIONS
from uploader.errors import MetadataError
from uploader.models import PreparedUploadJob, UploadDefaults, UploaderConfig


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").strip()


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _normalize_tags(raw_text: str | list[str]) -> list[str]:
    if isinstance(raw_text, list):
        candidates = raw_text
    else:
        candidates = re.split(r"[\n,;]+", raw_text)

    final: list[str] = []
    for item in candidates:
        cleaned = str(item).strip().lstrip("#").strip()
        if not cleaned:
            continue
        final.append(cleaned[:30])
    return _dedupe(final)[:25]


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_files(job_dir: Path, pattern: str) -> list[Path]:
    return sorted(path for path in job_dir.glob(pattern) if path.is_file())


def _find_video_path(job_dir: Path) -> Path:
    for ext in VIDEO_EXTENSIONS:
        candidate = job_dir / f"video{ext}"
        if candidate.exists():
            return candidate

    all_candidates = sorted(
        [path for path in job_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS],
        key=lambda item: item.name.lower(),
    )
    if all_candidates:
        return all_candidates[0]
    raise MetadataError(f"Video dosyasi bulunamadi: {job_dir}")


def _find_thumbnail_path(job_dir: Path) -> Path | None:
    for stem in ("thumbnail", "thumb", "cover"):
        for ext in THUMBNAIL_EXTENSIONS:
            candidate = job_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate

    thumbnails = sorted(
        [path for path in job_dir.iterdir() if path.is_file() and path.suffix.lower() in THUMBNAIL_EXTENSIONS],
        key=lambda item: item.name.lower(),
    )
    return thumbnails[0] if thumbnails else None


def _extract_language_bucket(metadata_payload: dict[str, Any], preferred_language: str) -> dict[str, Any]:
    if isinstance(metadata_payload.get(preferred_language), dict):
        return metadata_payload[preferred_language]

    if isinstance(metadata_payload.get("titles"), list) or isinstance(metadata_payload.get("description"), dict):
        return metadata_payload

    for value in metadata_payload.values():
        if isinstance(value, dict) and (
            isinstance(value.get("titles"), list) or isinstance(value.get("description"), dict)
        ):
            return value

    return {}


def _load_metadata_json(job_dir: Path, preferred_language: str) -> tuple[dict[str, Any], str]:
    candidates = []
    metadata_json = job_dir / "metadata.json"
    if metadata_json.exists():
        candidates.append(metadata_json)
    candidates.extend(_candidate_files(job_dir, "*_metadata.json"))

    for path in candidates:
        try:
            payload = _load_json(path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        bucket = _extract_language_bucket(payload, preferred_language)
        if bucket:
            return bucket, path.name
    return {}, ""


def _build_description(bucket: dict[str, Any]) -> str:
    description_value = bucket.get("description")
    if isinstance(description_value, dict):
        hook = str(description_value.get("hook", "")).strip()
        seo = str(description_value.get("seo", "")).strip()
        return "\n\n".join(part for part in (hook, seo) if part)
    if isinstance(description_value, str):
        return description_value.strip()
    return ""


def _sample_file_signature(path: Path) -> bytes:
    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode("utf-8"))
    digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    window_size = 65536
    with path.open("rb") as handle:
        digest.update(handle.read(window_size))
        if stat.st_size > window_size:
            handle.seek(max(stat.st_size - window_size, 0))
            digest.update(handle.read(window_size))
    return digest.digest()


def _build_fingerprint(video_path: Path, title: str, description: str) -> str:
    digest = hashlib.sha256()
    digest.update(video_path.name.encode("utf-8"))
    digest.update(_sample_file_signature(video_path))
    digest.update(title.encode("utf-8"))
    digest.update(description.encode("utf-8"))
    return digest.hexdigest()


def _merge_settings(config: UploaderConfig, raw_settings: dict[str, Any], warnings: list[str]) -> UploadDefaults:
    defaults = config.defaults
    privacy_status = str(raw_settings.get("privacyStatus", defaults.privacy_status)).lower()
    if privacy_status != "private":
        warnings.append("privacyStatus guvenlik icin private olarak zorlandi.")
        privacy_status = "private"

    made_for_kids = bool(raw_settings.get("madeForKids", defaults.made_for_kids))
    self_declared = bool(raw_settings.get("selfDeclaredMadeForKids", made_for_kids))
    publish_at = raw_settings.get("publishAt", defaults.publish_at)
    if publish_at and not config.allow_scheduled_publish:
        warnings.append("publishAt config geregi devre disi birakildi; video private kalacak.")
        publish_at = None

    return UploadDefaults(
        privacy_status=privacy_status,
        category_id=str(raw_settings.get("categoryId", defaults.category_id)),
        default_language=str(raw_settings.get("defaultLanguage", defaults.default_language)),
        default_audio_language=str(raw_settings.get("defaultAudioLanguage", defaults.default_audio_language)),
        made_for_kids=made_for_kids,
        self_declared_made_for_kids=self_declared,
        embeddable=bool(raw_settings.get("embeddable", defaults.embeddable)),
        license_name=str(raw_settings.get("license", defaults.license_name)),
        public_stats_viewable=bool(raw_settings.get("publicStatsViewable", defaults.public_stats_viewable)),
        publish_at=publish_at,
        playlist_id=raw_settings.get("playlistId", defaults.playlist_id),
    )


def _load_settings(job_dir: Path, config: UploaderConfig, warnings: list[str]) -> UploadDefaults:
    settings_path = job_dir / "settings.json"
    if not settings_path.exists():
        return _merge_settings(config, {}, warnings)
    try:
        raw = _load_json(settings_path)
    except Exception as exc:
        raise MetadataError(f"settings.json okunamadi: {settings_path}") from exc
    if not isinstance(raw, dict):
        raise MetadataError(f"settings.json obje formatinda olmali: {settings_path}")
    return _merge_settings(config, raw, warnings)


def _discover_subtitles(job_dir: Path) -> dict[str, Path]:
    subtitles: dict[str, Path] = {}
    for language_code, expected_name in SUBTITLE_FILE_MAP.items():
        exact = job_dir / expected_name
        if exact.exists():
            subtitles[language_code] = exact
            continue

        fallback_candidates = []
        fallback_candidates.extend(_candidate_files(job_dir, f"*_{language_code}.srt"))
        fallback_candidates.extend(_candidate_files(job_dir, f"*_{language_code.upper()}.srt"))
        if fallback_candidates:
            subtitles[language_code] = fallback_candidates[0]
    return subtitles


def parse_prepared_upload_job(job_dir: Path, config: UploaderConfig) -> PreparedUploadJob:
    """Parse a prepared upload folder into a validated upload job."""
    job_dir = job_dir.resolve()
    if not job_dir.exists() or not job_dir.is_dir():
        raise MetadataError(f"Upload klasoru bulunamadi: {job_dir}")

    warnings: list[str] = []
    metadata_bucket, metadata_source = _load_metadata_json(job_dir, config.preferred_metadata_language)

    video_path = _find_video_path(job_dir)

    title_path = job_dir / "title.txt"
    if title_path.exists():
        title = _read_text(title_path)
    else:
        titles = metadata_bucket.get("titles", []) if isinstance(metadata_bucket.get("titles"), list) else []
        title = str(titles[0]).strip() if titles else ""
        if title:
            warnings.append(f"title.txt yok; baslik {metadata_source} icinden secildi.")
    if not title:
        raise MetadataError(f"title.txt eksik veya bos: {job_dir}")
    if len(title) > 100:
        title = title[:100].rstrip()
        warnings.append("Baslik YouTube limiti icin 100 karaktere kisaltildi.")

    description_path = job_dir / "description.txt"
    if description_path.exists():
        description = _read_text(description_path)
    else:
        description = _build_description(metadata_bucket)
        if description:
            warnings.append(f"description.txt yok; aciklama {metadata_source} icinden derlendi.")
    if not description:
        raise MetadataError(f"description.txt eksik veya bos: {job_dir}")
    if len(description) > 5000:
        description = description[:5000].rstrip()
        warnings.append("Aciklama YouTube limiti icin 5000 karaktere kisaltildi.")

    tags_path = job_dir / "tags.txt"
    if tags_path.exists():
        tags = _normalize_tags(_read_text(tags_path))
    else:
        tags = _normalize_tags(metadata_bucket.get("tags", []))
        if tags:
            warnings.append(f"tags.txt yok; etiketler {metadata_source} icinden alindi.")

    settings = _load_settings(job_dir, config, warnings)
    thumbnail_path = _find_thumbnail_path(job_dir)
    if thumbnail_path is None:
        warnings.append("Thumbnail bulunamadi; video thumbnailsiz yuklenecek.")

    subtitle_paths = _discover_subtitles(job_dir)
    for language_code in SUBTITLE_FILE_MAP:
        if language_code not in subtitle_paths:
            warnings.append(f"{language_code} altyazisi bulunamadi; bu dil atlanacak.")

    fingerprint = _build_fingerprint(video_path, title, description)
    return PreparedUploadJob(
        job_dir=job_dir,
        job_name=job_dir.name,
        fingerprint=fingerprint,
        video_path=video_path,
        title=title,
        description=description,
        tags=tags,
        thumbnail_path=thumbnail_path,
        settings=settings,
        subtitle_paths=subtitle_paths,
        warnings=warnings,
        metadata_source=metadata_source or "folder_files",
    )
