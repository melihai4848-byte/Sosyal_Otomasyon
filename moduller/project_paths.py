from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = BASE_DIR / "workspace"

INPUTS_DIR = WORKSPACE_DIR / "00_Inputs"
OUTPUTS_DIR = WORKSPACE_DIR / "00_Outputs"
LOGS_DIR = WORKSPACE_DIR / "logs"
STATE_DIR = WORKSPACE_DIR / "state"
MODELS_DIR = INPUTS_DIR / "models"
OAUTH_DIR = INPUTS_DIR / "oauth"

UPLOADER_DIR = WORKSPACE_DIR / "uploader"
UPLOADER_INPUT_ROOT = UPLOADER_DIR / "input" / "youtube_uploads"
UPLOADER_SUCCESS_ROOT = UPLOADER_DIR / "success" / "youtube_uploaded"
UPLOADER_FAILED_ROOT = UPLOADER_DIR / "failed" / "youtube_failed"
UPLOADER_STATE_FILE = STATE_DIR / "youtube_upload_state.json"
UPLOADER_LOCK_DIR = STATE_DIR / "youtube_locks"
UPLOADER_LOG_FILE = LOGS_DIR / "youtube_draft_uploader.log"


def ensure_workspace_structure() -> None:
    for path in (
        WORKSPACE_DIR,
        INPUTS_DIR,
        OUTPUTS_DIR,
        LOGS_DIR,
        STATE_DIR,
        MODELS_DIR,
        OAUTH_DIR,
        UPLOADER_INPUT_ROOT,
        UPLOADER_SUCCESS_ROOT,
        UPLOADER_FAILED_ROOT,
        UPLOADER_LOCK_DIR,
        UPLOADER_STATE_FILE.parent,
        UPLOADER_LOG_FILE.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)
