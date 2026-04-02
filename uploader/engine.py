"""Execution engine for prepared YouTube draft uploads."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from uploader.constants import STATUS_FAILED, STATUS_PENDING, STATUS_UPLOADED
from uploader.errors import DuplicateUploadError, MetadataError
from uploader.metadata import parse_prepared_upload_job
from uploader.models import UploadOutcome, UploaderConfig
from uploader.state import UploadStateStore
from uploader.youtube_api import YouTubeApiClient


class DraftUploadEngine:
    """Coordinate parsing, state, retries, and YouTube API calls."""

    def __init__(self, config: UploaderConfig, logger) -> None:
        self.config = config
        self.logger = logger
        self.state_store = UploadStateStore(config)
        self._api_client: YouTubeApiClient | None = None

    @property
    def api_client(self) -> YouTubeApiClient:
        if self._api_client is None:
            self._api_client = YouTubeApiClient(self.config, self.logger)
        return self._api_client

    def _candidate_job_dirs(self, root_dir: Path) -> list[Path]:
        if not root_dir.exists():
            return []
        return sorted(
            [path for path in root_dir.iterdir() if path.is_dir() and not path.name.startswith(".")],
            key=lambda item: item.name.lower(),
        )

    def _build_receipt_payload(self, outcome: UploadOutcome) -> dict:
        return {
            "job_dir": str(outcome.job_dir),
            "status": outcome.status,
            "dry_run": outcome.dry_run,
            "video_id": outcome.video_id,
            "video_url": outcome.video_url,
            "warnings": outcome.warnings,
            "errors": outcome.errors,
            "uploaded_captions": outcome.uploaded_captions,
            "thumbnail_uploaded": outcome.thumbnail_uploaded,
            "playlist_assigned": outcome.playlist_assigned,
        }

    def _write_receipt(self, job_dir: Path, outcome: UploadOutcome) -> None:
        receipt_path = job_dir / "youtube_upload_result.json"
        receipt_path.write_text(
            json.dumps(self._build_receipt_payload(outcome), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _dedupe_messages(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result

    def _archive_folder(self, job_dir: Path, target_root: Path) -> Path:
        target_root.mkdir(parents=True, exist_ok=True)
        if job_dir.parent == target_root:
            return job_dir

        candidate = target_root / job_dir.name
        if candidate.exists():
            candidate = target_root / f"{job_dir.name}_{time.strftime('%Y%m%d_%H%M%S')}"
        moved_path = Path(shutil.move(str(job_dir), str(candidate)))
        return moved_path

    def _mark_optional_missing(
        self,
        *,
        job_key: str,
        step_name: str,
        warnings: list[str],
        message: str,
    ) -> None:
        warnings.append(message)
        self.state_store.update_step(
            job_key,
            step_name,
            status=STATUS_UPLOADED,
            payload_data={"detail": "optional_missing", "message": message},
        )

    def process_folder(self, folder_path: Path, *, dry_run: bool = False, force: bool = False) -> UploadOutcome:
        """Upload one prepared folder to YouTube as private/draft."""
        folder_path = folder_path.resolve()
        job = parse_prepared_upload_job(folder_path, self.config)
        record = self.state_store.register_job(job, force=force)
        job_key = str(job.job_dir.resolve())

        if record.get("status") == STATUS_UPLOADED and not force:
            return UploadOutcome(
                job_dir=job.job_dir,
                status=STATUS_UPLOADED,
                dry_run=dry_run,
                video_id=record.get("video_id"),
                video_url=record.get("video_url"),
                warnings=["Bu klasor zaten yuklenmis gorunuyor; duplicate upload engellendi."],
            )

        warnings = list(job.warnings)
        errors: list[str] = []
        uploaded_captions: list[str] = []
        thumbnail_uploaded = False
        playlist_assigned = False

        with self.state_store.acquire_job_lock(job_key):
            self.state_store.start_job(job_key)

            if dry_run:
                self.logger.info(f"[DRY-RUN] {job.job_name} klasoru dogrulandi. Gercek upload yapilmadi.")
                warnings = self._dedupe_messages(warnings)
                self.state_store.mark_pending(job_key, warnings=warnings)
                outcome = UploadOutcome(
                    job_dir=job.job_dir,
                    status=STATUS_PENDING,
                    dry_run=True,
                    warnings=warnings,
                )
                self._write_receipt(job.job_dir, outcome)
                return outcome

            existing = self.state_store.get_job(job_key) or {}
            steps = existing.get("steps", {})
            video_id = existing.get("video_id")
            video_url = existing.get("video_url")

            if steps.get("video_upload", {}).get("status") == STATUS_UPLOADED and video_id and video_url:
                self.logger.info(f"{job.job_name} mevcut video_id ile devam ediyor: {video_id}")
            else:
                try:
                    video_id, video_url = self.api_client.upload_video(job)
                    self.state_store.update_video_reference(job_key, video_id=video_id, video_url=video_url)
                    self.state_store.update_step(
                        job_key,
                        "video_upload",
                        status=STATUS_UPLOADED,
                        payload_data={"video_id": video_id, "video_url": video_url},
                    )
                    self.logger.info(f"{job.job_name} private taslak olarak yuklendi. Video ID: {video_id}")
                except Exception as exc:  # noqa: BLE001
                    error_message = f"Video upload basarisiz: {exc}"
                    errors.append(error_message)
                    self.state_store.update_step(job_key, "video_upload", status=STATUS_FAILED, error=error_message)
                    archived_path = None
                    if self.config.move_failed_folders:
                        archived_path = str(self._archive_folder(job.job_dir, self.config.failed_root))
                    outcome = UploadOutcome(
                        job_dir=Path(archived_path) if archived_path else job.job_dir,
                        status=STATUS_FAILED,
                        dry_run=False,
                        warnings=warnings,
                        errors=errors,
                    )
                    self._write_receipt(outcome.job_dir, outcome)
                    self.state_store.mark_failed(
                        job_key,
                        last_error=error_message,
                        warnings=self._dedupe_messages(warnings),
                        archived_path=archived_path,
                    )
                    return outcome

            if not video_id or not video_url:
                raise MetadataError("Video yukleme sonrasi video_id veya url olusmadi.")

            if job.thumbnail_path is None:
                self._mark_optional_missing(
                    job_key=job_key,
                    step_name="thumbnail_upload",
                    warnings=warnings,
                    message="Thumbnail bulunmadigi icin adim warning ile atlandi.",
                )
            else:
                try:
                    self.api_client.set_thumbnail(video_id, job.thumbnail_path)
                    self.state_store.update_step(
                        job_key,
                        "thumbnail_upload",
                        status=STATUS_UPLOADED,
                        payload_data={"thumbnail_path": str(job.thumbnail_path)},
                    )
                    thumbnail_uploaded = True
                except Exception as exc:  # noqa: BLE001
                    error_message = f"Thumbnail upload basarisiz: {exc}"
                    errors.append(error_message)
                    self.state_store.update_step(
                        job_key,
                        "thumbnail_upload",
                        status=STATUS_FAILED,
                        error=error_message,
                    )

            for language_code, subtitle_path in job.subtitle_paths.items():
                step_name = f"caption_{language_code}"
                try:
                    self.api_client.upload_caption(video_id, language_code, subtitle_path)
                    self.state_store.update_step(
                        job_key,
                        step_name,
                        status=STATUS_UPLOADED,
                        payload_data={"subtitle_path": str(subtitle_path)},
                    )
                    uploaded_captions.append(language_code)
                except Exception as exc:  # noqa: BLE001
                    error_message = f"{language_code} altyazi yukleme basarisiz: {exc}"
                    warnings.append(error_message)
                    errors.append(error_message)
                    self.state_store.update_step(job_key, step_name, status=STATUS_FAILED, error=error_message)

            for language_code in ("tr", "en", "de"):
                step_name = f"caption_{language_code}"
                if language_code not in job.subtitle_paths:
                    self._mark_optional_missing(
                        job_key=job_key,
                        step_name=step_name,
                        warnings=warnings,
                        message=f"{language_code} altyazisi bulunmadigi icin adim warning ile atlandi.",
                    )

            if job.settings.playlist_id:
                try:
                    self.api_client.add_to_playlist(video_id, str(job.settings.playlist_id))
                    self.state_store.update_step(
                        job_key,
                        "playlist_assignment",
                        status=STATUS_UPLOADED,
                        payload_data={"playlist_id": str(job.settings.playlist_id)},
                    )
                    playlist_assigned = True
                except Exception as exc:  # noqa: BLE001
                    error_message = f"Playlist ekleme basarisiz: {exc}"
                    errors.append(error_message)
                    self.state_store.update_step(
                        job_key,
                        "playlist_assignment",
                        status=STATUS_FAILED,
                        error=error_message,
                    )
            else:
                self._mark_optional_missing(
                    job_key=job_key,
                    step_name="playlist_assignment",
                    warnings=warnings,
                    message="playlistId tanimlanmadigi icin playlist adimi warning ile atlandi.",
                )

            final_status = STATUS_UPLOADED if not errors else STATUS_FAILED
            warnings = self._dedupe_messages(warnings)
            errors = self._dedupe_messages(errors)
            archive_target = None
            final_job_dir = job.job_dir

            if final_status == STATUS_UPLOADED and self.config.move_successful_folders:
                final_job_dir = self._archive_folder(job.job_dir, self.config.success_root)
                archive_target = str(final_job_dir)
            elif final_status == STATUS_FAILED and self.config.move_failed_folders:
                final_job_dir = self._archive_folder(job.job_dir, self.config.failed_root)
                archive_target = str(final_job_dir)

            outcome = UploadOutcome(
                job_dir=final_job_dir,
                status=final_status,
                dry_run=False,
                video_id=video_id,
                video_url=video_url,
                warnings=warnings,
                errors=errors,
                uploaded_captions=uploaded_captions,
                thumbnail_uploaded=thumbnail_uploaded,
                playlist_assigned=playlist_assigned,
            )
            self._write_receipt(final_job_dir, outcome)

            if final_status == STATUS_UPLOADED:
                self.state_store.mark_uploaded(
                    job_key,
                    video_id=video_id,
                    video_url=video_url,
                    warnings=warnings,
                    archived_path=archive_target,
                )
            else:
                self.state_store.mark_failed(
                    job_key,
                    last_error=" | ".join(errors),
                    warnings=warnings,
                    archived_path=archive_target,
                )
            return outcome

    def process_batch(self, root_dir: Path | None = None, *, dry_run: bool = False, force: bool = False) -> list[UploadOutcome]:
        """Process all prepared folders in the configured input root."""
        target_root = root_dir.resolve() if root_dir else self.config.input_root
        outcomes: list[UploadOutcome] = []
        for job_dir in self._candidate_job_dirs(target_root):
            try:
                outcomes.append(self.process_folder(job_dir, dry_run=dry_run, force=force))
            except DuplicateUploadError as exc:
                self.logger.warning(str(exc))
                outcomes.append(
                    UploadOutcome(job_dir=job_dir, status=STATUS_UPLOADED, dry_run=dry_run, warnings=[str(exc)])
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.exception(f"{job_dir.name} batch icinde islenemedi: {exc}")
                outcomes.append(
                    UploadOutcome(job_dir=job_dir, status=STATUS_FAILED, dry_run=dry_run, errors=[str(exc)])
                )
        return outcomes

    def watch(self, root_dir: Path | None = None, *, dry_run: bool = False, force: bool = False) -> None:
        """Watch the input root and process new folders continuously."""
        target_root = root_dir.resolve() if root_dir else self.config.input_root
        self.logger.info(
            f"Watch mode basladi. Klasor: {target_root} | interval: {self.config.watch.poll_interval_seconds} sn"
        )
        try:
            while True:
                self.process_batch(target_root, dry_run=dry_run, force=force)
                time.sleep(self.config.watch.poll_interval_seconds)
        except KeyboardInterrupt:
            self.logger.info("Watch mode kullanici tarafindan durduruldu.")
