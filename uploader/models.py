"""Typed models for the YouTube draft uploader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class RetrySettings:
    """Retry configuration."""

    max_attempts: int = 4
    initial_delay_seconds: int = 5
    max_delay_seconds: int = 60


@dataclass(slots=True)
class WatchSettings:
    """Watch mode configuration."""

    poll_interval_seconds: int = 30


@dataclass(slots=True)
class OAuthSettings:
    """OAuth file locations and interaction behavior."""

    client_secret_file: Path
    token_file: Path
    open_browser_for_oauth: bool = True


@dataclass(slots=True)
class UploadDefaults:
    """Default YouTube video settings."""

    privacy_status: str = "private"
    category_id: str = "22"
    default_language: str = "tr"
    default_audio_language: str = "tr"
    made_for_kids: bool = False
    self_declared_made_for_kids: bool = False
    embeddable: bool = True
    license_name: str = "youtube"
    public_stats_viewable: bool = True
    publish_at: str | None = None
    playlist_id: str | None = None


@dataclass(slots=True)
class UploaderConfig:
    """Resolved runtime configuration."""

    input_root: Path
    success_root: Path
    failed_root: Path
    state_file: Path
    lock_dir: Path
    log_file: Path
    preferred_metadata_language: str
    allow_scheduled_publish: bool
    move_successful_folders: bool
    move_failed_folders: bool
    oauth: OAuthSettings
    retry: RetrySettings
    watch: WatchSettings
    defaults: UploadDefaults


@dataclass(slots=True)
class PreparedUploadJob:
    """Fully parsed upload bundle for one prepared folder."""

    job_dir: Path
    job_name: str
    fingerprint: str
    video_path: Path
    title: str
    description: str
    tags: list[str]
    thumbnail_path: Path | None
    settings: UploadDefaults
    subtitle_paths: dict[str, Path]
    warnings: list[str] = field(default_factory=list)
    metadata_source: str = "folder_files"


@dataclass(slots=True)
class UploadOutcome:
    """Execution outcome for one upload job."""

    job_dir: Path
    status: str
    dry_run: bool
    video_id: str | None = None
    video_url: str | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    uploaded_captions: list[str] = field(default_factory=list)
    thumbnail_uploaded: bool = False
    playlist_assigned: bool = False

