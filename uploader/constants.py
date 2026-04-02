"""Central constants for the YouTube draft uploader."""

from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
DEFAULT_CONFIG_FILENAME = "youtube_uploader.json"
EXAMPLE_CONFIG_FILENAME = "youtube_uploader.example.json"
STATE_FILE_VERSION = 1

STATUS_PENDING = "pending"
STATUS_UPLOADING = "uploading"
STATUS_UPLOADED = "uploaded"
STATUS_FAILED = "failed"
STATUS_VALUES = {
    STATUS_PENDING,
    STATUS_UPLOADING,
    STATUS_UPLOADED,
    STATUS_FAILED,
}

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".m4v", ".avi", ".webm"}
THUMBNAIL_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SUBTITLE_FILE_MAP = {
    "tr": "subtitle_tr.srt",
    "en": "subtitle_en.srt",
    "de": "subtitle_de.srt",
}
SUBTITLE_TRACK_NAMES = {
    "tr": "Turkish",
    "en": "English",
    "de": "German",
}
SUPPORTED_LANGUAGE_CODES = tuple(SUBTITLE_FILE_MAP.keys())

YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",
]

DEFAULT_STEP_NAMES = (
    "video_upload",
    "thumbnail_upload",
    "playlist_assignment",
)
