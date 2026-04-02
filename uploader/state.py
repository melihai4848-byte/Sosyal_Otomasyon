"""Central JSON state store for prepared YouTube upload jobs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from uploader.constants import STATE_FILE_VERSION, STATUS_FAILED, STATUS_PENDING, STATUS_UPLOADED, STATUS_UPLOADING
from uploader.errors import DuplicateUploadError, StateError
from uploader.models import PreparedUploadJob, UploaderConfig


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _safe_lock_name(job_key: str) -> str:
    return hashlib.sha256(job_key.encode("utf-8")).hexdigest() + ".lock"


class FileLock:
    """Very small cross-platform file lock based on exclusive file creation."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._locked = False

    def __enter__(self) -> "FileLock":
        try:
            descriptor = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(_utc_now())
            self._locked = True
            return self
        except FileExistsError as exc:
            raise StateError(f"Kilitleme dosyasi zaten mevcut: {self.path}") from exc

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if self._locked:
            self.path.unlink(missing_ok=True)
            self._locked = False


class UploadStateStore:
    """Persist uploader state in one central JSON file."""

    def __init__(self, config: UploaderConfig) -> None:
        self.config = config
        self.state_file = config.state_file
        self.store_lock = config.lock_dir / ".youtube_uploader_state.lock"
        self._ensure_state_file()

    def _ensure_state_file(self) -> None:
        if self.state_file.exists():
            return
        _atomic_write_json(
            self.state_file,
            {
                "version": STATE_FILE_VERSION,
                "updated_at": _utc_now(),
                "jobs": {},
                "fingerprints": {},
            },
        )

    def _read(self) -> dict[str, Any]:
        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self._ensure_state_file()
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StateError(f"State dosyasi bozuk: {self.state_file}") from exc

    @contextmanager
    def transaction(self) -> Iterator[dict[str, Any]]:
        with FileLock(self.store_lock):
            payload = self._read()
            yield payload
            payload["updated_at"] = _utc_now()
            _atomic_write_json(self.state_file, payload)

    def acquire_job_lock(self, job_key: str) -> FileLock:
        return FileLock(self.config.lock_dir / _safe_lock_name(job_key))

    def get_job(self, job_key: str) -> dict[str, Any] | None:
        return self._read().get("jobs", {}).get(job_key)

    def register_job(self, job: PreparedUploadJob, *, force: bool = False) -> dict[str, Any]:
        job_key = str(job.job_dir.resolve())
        now = _utc_now()
        with self.transaction() as payload:
            jobs = payload["jobs"]
            fingerprint_map = payload["fingerprints"]

            existing_by_key = jobs.get(job_key)
            existing_key_by_fingerprint = fingerprint_map.get(job.fingerprint)
            existing_by_fingerprint = jobs.get(existing_key_by_fingerprint) if existing_key_by_fingerprint else None

            if (
                not force
                and existing_by_fingerprint
                and existing_by_fingerprint.get("status") == STATUS_UPLOADED
                and existing_key_by_fingerprint != job_key
            ):
                raise DuplicateUploadError(
                    f"Ayni icerik zaten yuklenmis gorunuyor: {existing_key_by_fingerprint}"
                )

            if existing_by_key is None:
                existing_by_key = {
                    "job_key": job_key,
                    "job_dir": job_key,
                    "job_name": job.job_name,
                    "fingerprint": job.fingerprint,
                    "status": STATUS_PENDING,
                    "attempt_count": 0,
                    "video_id": None,
                    "video_url": None,
                    "archived_path": None,
                    "last_error": None,
                    "warnings": [],
                    "steps": {},
                    "created_at": now,
                    "updated_at": now,
                }
                jobs[job_key] = existing_by_key

            if force and existing_by_key.get("status") == STATUS_UPLOADED:
                existing_by_key["status"] = STATUS_PENDING
                existing_by_key["video_id"] = None
                existing_by_key["video_url"] = None
                existing_by_key["archived_path"] = None
                existing_by_key["last_error"] = None
                existing_by_key["warnings"] = []
                existing_by_key["steps"] = {}
                existing_by_key.pop("completed_at", None)
                existing_by_key.pop("started_at", None)

            existing_by_key["job_name"] = job.job_name
            existing_by_key["fingerprint"] = job.fingerprint
            existing_by_key["job_dir"] = job_key
            existing_by_key["updated_at"] = now
            fingerprint_map[job.fingerprint] = job_key
            return existing_by_key

    def start_job(self, job_key: str) -> dict[str, Any]:
        now = _utc_now()
        with self.transaction() as payload:
            job = payload["jobs"][job_key]
            job["status"] = STATUS_UPLOADING
            job["attempt_count"] = int(job.get("attempt_count", 0)) + 1
            job["started_at"] = now
            job["updated_at"] = now
            job["last_error"] = None
            return job

    def update_step(
        self,
        job_key: str,
        step_name: str,
        *,
        status: str,
        error: str | None = None,
        payload_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = _utc_now()
        with self.transaction() as payload:
            job = payload["jobs"][job_key]
            job["steps"].setdefault(step_name, {})
            step = job["steps"][step_name]
            step["status"] = status
            step["updated_at"] = now
            if error:
                step["error"] = error
            else:
                step.pop("error", None)
            if payload_data:
                step.update(payload_data)
            job["updated_at"] = now
            return job

    def mark_uploaded(
        self,
        job_key: str,
        *,
        video_id: str,
        video_url: str,
        warnings: list[str],
        archived_path: str | None,
    ) -> dict[str, Any]:
        now = _utc_now()
        with self.transaction() as payload:
            job = payload["jobs"][job_key]
            job["status"] = STATUS_UPLOADED
            job["video_id"] = video_id
            job["video_url"] = video_url
            job["warnings"] = warnings
            job["archived_path"] = archived_path
            job["last_error"] = None
            job["updated_at"] = now
            job["completed_at"] = now
            return job

    def mark_failed(
        self,
        job_key: str,
        *,
        last_error: str,
        warnings: list[str],
        archived_path: str | None,
    ) -> dict[str, Any]:
        now = _utc_now()
        with self.transaction() as payload:
            job = payload["jobs"][job_key]
            job["status"] = STATUS_FAILED
            job["warnings"] = warnings
            job["last_error"] = last_error
            job["archived_path"] = archived_path
            job["updated_at"] = now
            return job

    def update_video_reference(self, job_key: str, *, video_id: str, video_url: str) -> dict[str, Any]:
        with self.transaction() as payload:
            job = payload["jobs"][job_key]
            job["video_id"] = video_id
            job["video_url"] = video_url
            job["updated_at"] = _utc_now()
            return job

    def mark_pending(self, job_key: str, *, warnings: list[str]) -> dict[str, Any]:
        with self.transaction() as payload:
            job = payload["jobs"][job_key]
            job["status"] = STATUS_PENDING
            job["warnings"] = warnings
            job["updated_at"] = _utc_now()
            return job
