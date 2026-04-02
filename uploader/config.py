"""Configuration loading for the YouTube draft uploader."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from uploader.constants import CONFIG_DIR, DEFAULT_CONFIG_FILENAME, ROOT_DIR
from uploader.errors import ConfigError
from uploader.models import OAuthSettings, RetrySettings, UploadDefaults, UploaderConfig, WatchSettings

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False


DEFAULT_CONFIG: dict[str, Any] = {
    "input_root": "input/youtube_uploads",
    "success_root": "success/youtube_uploaded",
    "failed_root": "failed/youtube_failed",
    "state_file": "state/youtube_upload_state.json",
    "lock_dir": "state/youtube_locks",
    "log_file": "logs/youtube_draft_uploader.log",
    "preferred_metadata_language": "Türkçe",
    "allow_scheduled_publish": False,
    "move_successful_folders": True,
    "move_failed_folders": True,
    "open_browser_for_oauth": True,
    "oauth": {
        "client_secret_file": "00_Inputs/oauth/google_client_secret.json",
        "token_file": "00_Inputs/oauth/youtube_draft_upload_token.json",
    },
    "watch": {
        "poll_interval_seconds": 30,
    },
    "retry": {
        "max_attempts": 4,
        "initial_delay_seconds": 5,
        "max_delay_seconds": 60,
    },
    "defaults": {
        "privacyStatus": "private",
        "categoryId": "22",
        "defaultLanguage": "tr",
        "defaultAudioLanguage": "tr",
        "madeForKids": False,
        "selfDeclaredMadeForKids": False,
        "embeddable": True,
        "license": "youtube",
        "publicStatsViewable": True,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"Gecersiz boolean config degeri: {value}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Config dosyasi bulunamadi: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Config JSON gecersiz: {path}") from exc


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    merged = _deep_merge({}, raw)
    env_map = {
        "YT_UPLOADER_INPUT_ROOT": ("input_root", str),
        "YT_UPLOADER_SUCCESS_ROOT": ("success_root", str),
        "YT_UPLOADER_FAILED_ROOT": ("failed_root", str),
        "YT_UPLOADER_STATE_FILE": ("state_file", str),
        "YT_UPLOADER_LOCK_DIR": ("lock_dir", str),
        "YT_UPLOADER_LOG_FILE": ("log_file", str),
        "YT_UPLOADER_PREFERRED_METADATA_LANGUAGE": ("preferred_metadata_language", str),
        "YT_UPLOADER_ALLOW_SCHEDULED_PUBLISH": ("allow_scheduled_publish", _parse_bool),
        "YT_UPLOADER_MOVE_SUCCESSFUL_FOLDERS": ("move_successful_folders", _parse_bool),
        "YT_UPLOADER_MOVE_FAILED_FOLDERS": ("move_failed_folders", _parse_bool),
        "YT_UPLOADER_OPEN_BROWSER_FOR_OAUTH": ("open_browser_for_oauth", _parse_bool),
        "YT_UPLOADER_OAUTH_CLIENT_SECRET_FILE": ("oauth.client_secret_file", str),
        "YT_UPLOADER_OAUTH_TOKEN_FILE": ("oauth.token_file", str),
        "YT_UPLOADER_WATCH_POLL_INTERVAL_SECONDS": ("watch.poll_interval_seconds", int),
        "YT_UPLOADER_RETRY_MAX_ATTEMPTS": ("retry.max_attempts", int),
        "YT_UPLOADER_RETRY_INITIAL_DELAY_SECONDS": ("retry.initial_delay_seconds", int),
        "YT_UPLOADER_RETRY_MAX_DELAY_SECONDS": ("retry.max_delay_seconds", int),
    }

    for env_name, (path_text, caster) in env_map.items():
        value = os.getenv(env_name)
        if value is None:
            continue
        parts = path_text.split(".")
        target = merged
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = caster(value)
    return merged


def load_uploader_config(config_path: str | None = None) -> UploaderConfig:
    """Load config from defaults, optional JSON file, and environment variables."""
    load_dotenv(ROOT_DIR / ".env")
    load_dotenv(ROOT_DIR / "moduller" / ".env")

    raw = _deep_merge({}, DEFAULT_CONFIG)
    candidate = Path(config_path).resolve() if config_path else (CONFIG_DIR / DEFAULT_CONFIG_FILENAME)
    if candidate.exists():
        raw = _deep_merge(raw, _load_json(candidate))

    raw = _apply_env_overrides(raw)

    retry = RetrySettings(
        max_attempts=int(raw["retry"]["max_attempts"]),
        initial_delay_seconds=int(raw["retry"]["initial_delay_seconds"]),
        max_delay_seconds=int(raw["retry"]["max_delay_seconds"]),
    )
    if retry.max_attempts < 1:
        raise ConfigError("retry.max_attempts en az 1 olmali.")

    watch = WatchSettings(poll_interval_seconds=int(raw["watch"]["poll_interval_seconds"]))
    oauth = OAuthSettings(
        client_secret_file=_resolve_path(str(raw["oauth"]["client_secret_file"])),
        token_file=_resolve_path(str(raw["oauth"]["token_file"])),
        open_browser_for_oauth=bool(raw["open_browser_for_oauth"]),
    )
    defaults = UploadDefaults(
        privacy_status=str(raw["defaults"].get("privacyStatus", "private")).lower(),
        category_id=str(raw["defaults"].get("categoryId", "22")),
        default_language=str(raw["defaults"].get("defaultLanguage", "tr")),
        default_audio_language=str(raw["defaults"].get("defaultAudioLanguage", "tr")),
        made_for_kids=bool(raw["defaults"].get("madeForKids", False)),
        self_declared_made_for_kids=bool(raw["defaults"].get("selfDeclaredMadeForKids", False)),
        embeddable=bool(raw["defaults"].get("embeddable", True)),
        license_name=str(raw["defaults"].get("license", "youtube")),
        public_stats_viewable=bool(raw["defaults"].get("publicStatsViewable", True)),
        publish_at=raw["defaults"].get("publishAt"),
        playlist_id=raw["defaults"].get("playlistId"),
    )
    if defaults.privacy_status not in {"private", "unlisted", "public"}:
        raise ConfigError("defaults.privacyStatus sadece private, unlisted veya public olabilir.")

    config = UploaderConfig(
        input_root=_resolve_path(str(raw["input_root"])),
        success_root=_resolve_path(str(raw["success_root"])),
        failed_root=_resolve_path(str(raw["failed_root"])),
        state_file=_resolve_path(str(raw["state_file"])),
        lock_dir=_resolve_path(str(raw["lock_dir"])),
        log_file=_resolve_path(str(raw["log_file"])),
        preferred_metadata_language=str(raw["preferred_metadata_language"]),
        allow_scheduled_publish=bool(raw["allow_scheduled_publish"]),
        move_successful_folders=bool(raw["move_successful_folders"]),
        move_failed_folders=bool(raw["move_failed_folders"]),
        oauth=oauth,
        retry=retry,
        watch=watch,
        defaults=defaults,
    )

    for path in (
        config.input_root,
        config.success_root,
        config.failed_root,
        config.lock_dir,
        config.state_file.parent,
        config.log_file.parent,
        config.oauth.token_file.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)

    return config
