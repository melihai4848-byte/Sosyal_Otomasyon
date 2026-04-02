"""YouTube Data API client helpers for draft uploads."""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Any, Optional

from uploader.constants import SUBTITLE_TRACK_NAMES, YOUTUBE_SCOPES
from uploader.errors import AuthenticationError, PermanentApiError
from uploader.models import PreparedUploadJob, UploadDefaults, UploaderConfig
from uploader.retry_utils import execute_with_retry, is_retryable_exception


class YouTubeApiClient:
    """Thin YouTube Data API wrapper with resumable upload support."""

    def __init__(self, config: UploaderConfig, logger) -> None:
        self.config = config
        self.logger = logger
        self._service = None

    def _import_google_dependencies(self) -> dict[str, Any]:
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload
        except ImportError as exc:  # pragma: no cover
            raise AuthenticationError(
                "Google API kutuphaneleri eksik. requirements.txt ile bagimliliklari kurun."
            ) from exc

        return {
            "Request": Request,
            "Credentials": Credentials,
            "InstalledAppFlow": InstalledAppFlow,
            "build": build,
            "MediaFileUpload": MediaFileUpload,
        }

    def _load_credentials(self):
        libs = self._import_google_dependencies()
        token_path = self.config.oauth.token_file
        client_secret_path = self.config.oauth.client_secret_file

        if not client_secret_path.exists():
            client_id = os.getenv("YOUTUBE_OAUTH_CLIENT_ID", "").strip()
            client_secret = os.getenv("YOUTUBE_OAUTH_CLIENT_SECRET", "").strip()
            if not client_id or not client_secret:
                raise AuthenticationError(
                    f"OAuth client secret dosyasi bulunamadi: {client_secret_path}"
                )

        credentials = None
        Credentials = libs["Credentials"]
        Request = libs["Request"]
        InstalledAppFlow = libs["InstalledAppFlow"]

        if token_path.exists():
            credentials = Credentials.from_authorized_user_file(str(token_path), YOUTUBE_SCOPES)

        if credentials and credentials.valid:
            return credentials

        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            if client_secret_path.exists():
                flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_path), YOUTUBE_SCOPES)
            else:
                flow = InstalledAppFlow.from_client_config(
                    {
                        "installed": {
                            "client_id": os.getenv("YOUTUBE_OAUTH_CLIENT_ID", "").strip(),
                            "client_secret": os.getenv("YOUTUBE_OAUTH_CLIENT_SECRET", "").strip(),
                            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                            "token_uri": "https://oauth2.googleapis.com/token",
                            "redirect_uris": ["http://localhost"],
                        }
                    },
                    YOUTUBE_SCOPES,
                )
            if self.config.oauth.open_browser_for_oauth:
                credentials = flow.run_local_server(port=0, open_browser=True)
            else:
                credentials = flow.run_console()

        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(credentials.to_json(), encoding="utf-8")
        return credentials

    @property
    def service(self):
        if self._service is None:
            libs = self._import_google_dependencies()
            credentials = self._load_credentials()
            self._service = libs["build"](
                "youtube",
                "v3",
                credentials=credentials,
                cache_discovery=False,
            )
        return self._service

    def get_authenticated_channel_profile(self) -> dict[str, Any]:
        response = execute_with_retry(
            lambda: self.service.channels()
            .list(part="snippet,contentDetails", mine=True, maxResults=1)
            .execute(),
            description="authenticated channel profile",
            retry=self.config.retry,
            logger=self.logger,
        )
        items = response.get("items", [])
        if not isinstance(items, list) or not items:
            raise PermanentApiError("Kimligi dogrulanmis YouTube kanali bulunamadi.")

        channel = items[0]
        uploads_playlist_id = (
            channel.get("contentDetails", {})
            .get("relatedPlaylists", {})
            .get("uploads")
        )
        return {
            "channel_id": channel.get("id"),
            "channel_title": channel.get("snippet", {}).get("title", ""),
            "uploads_playlist_id": uploads_playlist_id,
        }

    def list_recent_private_videos(self, limit: int = 10) -> list[dict[str, Any]]:
        profile = self.get_authenticated_channel_profile()
        uploads_playlist_id = str(profile.get("uploads_playlist_id") or "").strip()
        if not uploads_playlist_id:
            raise PermanentApiError("Authenticated kanal icin uploads playlist bulunamadi.")

        results: list[dict[str, Any]] = []
        next_page_token: str | None = None
        while len(results) < limit:
            playlist_response = execute_with_retry(
                lambda: self.service.playlistItems()
                .list(
                    part="snippet,contentDetails",
                    playlistId=uploads_playlist_id,
                    maxResults=25,
                    pageToken=next_page_token,
                )
                .execute(),
                description="private draft playlist list",
                retry=self.config.retry,
                logger=self.logger,
            )
            playlist_items = playlist_response.get("items", []) or []
            if not playlist_items:
                break

            video_ids: list[str] = []
            for item in playlist_items:
                video_id = str(item.get("contentDetails", {}).get("videoId") or "").strip()
                if video_id:
                    video_ids.append(video_id)
            if not video_ids:
                next_page_token = str(playlist_response.get("nextPageToken") or "").strip() or None
                if not next_page_token:
                    break
                continue

            videos_response = execute_with_retry(
                lambda: self.service.videos()
                .list(part="snippet,status,contentDetails", id=",".join(video_ids))
                .execute(),
                description="private draft video details",
                retry=self.config.retry,
                logger=self.logger,
            )
            videos_by_id = {
                str(item.get("id") or "").strip(): item
                for item in (videos_response.get("items", []) or [])
                if str(item.get("id") or "").strip()
            }

            for video_id in video_ids:
                item = videos_by_id.get(video_id)
                if not item:
                    continue
                status = item.get("status", {}) or {}
                if str(status.get("privacyStatus") or "").strip().lower() != "private":
                    continue
                snippet = item.get("snippet", {}) or {}
                content = item.get("contentDetails", {}) or {}
                results.append(
                    {
                        "video_id": video_id,
                        "title": str(snippet.get("title") or "").strip(),
                        "published_at": str(snippet.get("publishedAt") or "").strip(),
                        "channel_title": str(snippet.get("channelTitle") or profile.get("channel_title") or "").strip(),
                        "privacy_status": "private",
                        "duration_iso": str(content.get("duration") or "").strip(),
                    }
                )
                if len(results) >= limit:
                    break

            next_page_token = str(playlist_response.get("nextPageToken") or "").strip() or None
            if not next_page_token:
                break
        return results

    def get_video_resource(self, video_id: str, part: str = "snippet,status,localizations") -> dict[str, Any]:
        response = execute_with_retry(
            lambda: self.service.videos().list(part=part, id=video_id).execute(),
            description=f"{video_id} video fetch",
            retry=self.config.retry,
            logger=self.logger,
        )
        items = response.get("items", []) or []
        if not items:
            raise PermanentApiError(f"Video bulunamadi: {video_id}")
        return items[0]

    def update_existing_private_video(
        self,
        video_id: str,
        *,
        title: str,
        description: str,
        settings: UploadDefaults,
        localizations: dict[str, dict[str, str]] | None = None,
        current_resource: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        current = current_resource or self.get_video_resource(video_id)
        current_snippet = current.get("snippet", {}) or {}
        current_status = current.get("status", {}) or {}
        current_privacy = str(current_status.get("privacyStatus") or "").strip().lower()
        if current_privacy != "private":
            raise PermanentApiError(
                f"Bu modul yalnizca private taslaklari gunceller. Secilen video private degil: {video_id}"
            )

        snippet: dict[str, Any] = {
            "title": title,
            "description": description,
            "categoryId": str(current_snippet.get("categoryId") or settings.category_id),
            "defaultLanguage": str(settings.default_language or current_snippet.get("defaultLanguage") or "tr"),
            "defaultAudioLanguage": str(
                settings.default_audio_language or current_snippet.get("defaultAudioLanguage") or "tr"
            ),
        }
        current_tags = current_snippet.get("tags")
        if isinstance(current_tags, list) and current_tags:
            snippet["tags"] = current_tags

        status: dict[str, Any] = {
            "privacyStatus": "private",
            "embeddable": bool(settings.embeddable),
            "license": str(settings.license_name),
            "publicStatsViewable": bool(settings.public_stats_viewable),
            "selfDeclaredMadeForKids": bool(settings.self_declared_made_for_kids),
        }

        merged_localizations = dict(current.get("localizations") or {})
        for language_code, payload in (localizations or {}).items():
            if not isinstance(payload, dict):
                continue
            localized_title = str(payload.get("title") or "").strip()
            localized_description = str(payload.get("description") or "").strip()
            if not localized_title and not localized_description:
                continue
            existing_bucket = dict(merged_localizations.get(language_code) or {})
            if localized_title:
                existing_bucket["title"] = localized_title
            if localized_description:
                existing_bucket["description"] = localized_description
            merged_localizations[language_code] = existing_bucket

        body: dict[str, Any] = {
            "id": video_id,
            "snippet": snippet,
            "status": status,
        }
        parts = ["snippet", "status"]
        if merged_localizations:
            body["localizations"] = merged_localizations
            parts.append("localizations")

        return execute_with_retry(
            lambda: self.service.videos().update(part=",".join(parts), body=body).execute(),
            description=f"{video_id} metadata update",
            retry=self.config.retry,
            logger=self.logger,
        )

    def _build_video_body(self, job: PreparedUploadJob) -> dict[str, Any]:
        snippet: dict[str, Any] = {
            "title": job.title,
            "description": job.description,
            "categoryId": job.settings.category_id,
            "defaultLanguage": job.settings.default_language,
            "defaultAudioLanguage": job.settings.default_audio_language,
        }
        if job.tags:
            snippet["tags"] = job.tags

        status: dict[str, Any] = {
            "privacyStatus": job.settings.privacy_status,
            "embeddable": job.settings.embeddable,
            "license": job.settings.license_name,
            "publicStatsViewable": job.settings.public_stats_viewable,
            "selfDeclaredMadeForKids": job.settings.self_declared_made_for_kids,
        }
        if job.settings.publish_at:
            status["publishAt"] = job.settings.publish_at

        return {
            "snippet": snippet,
            "status": status,
        }

    def _run_resumable_request(self, request, description: str) -> dict[str, Any]:
        response = None
        attempt = 0
        last_progress = -1
        while response is None:
            try:
                progress, response = request.next_chunk()
                if progress is not None:
                    percentage = int(progress.progress() * 100)
                    if percentage != last_progress:
                        self.logger.info(f"{description} ilerleme: %{percentage}")
                        last_progress = percentage
            except Exception as exc:  # noqa: BLE001
                attempt += 1
                if attempt >= self.config.retry.max_attempts or not is_retryable_exception(exc):
                    raise
                delay = min(
                    self.config.retry.initial_delay_seconds * (2 ** max(attempt - 1, 0)),
                    self.config.retry.max_delay_seconds,
                )
                self.logger.warning(
                    f"{description} gecici hata verdi. {delay} saniye sonra devam edilecek. Hata: {exc}"
                )
                import time

                time.sleep(delay)
        return response

    def upload_video(self, job: PreparedUploadJob) -> tuple[str, str]:
        libs = self._import_google_dependencies()
        media = libs["MediaFileUpload"](
            str(job.video_path),
            mimetype="application/octet-stream",
            resumable=True,
        )
        request = self.service.videos().insert(
            part="snippet,status",
            body=self._build_video_body(job),
            media_body=media,
            notifySubscribers=False,
        )
        response = self._run_resumable_request(request, f"{job.job_name} video upload")
        video_id = response["id"]
        return video_id, f"https://www.youtube.com/watch?v={video_id}"

    def set_thumbnail(self, video_id: str, thumbnail_path: Path) -> None:
        libs = self._import_google_dependencies()
        mime_type, _ = mimetypes.guess_type(thumbnail_path.name)
        media = libs["MediaFileUpload"](
            str(thumbnail_path),
            mimetype=mime_type or "application/octet-stream",
            resumable=False,
        )
        execute_with_retry(
            lambda: self.service.thumbnails().set(videoId=video_id, media_body=media).execute(),
            description=f"{video_id} thumbnail upload",
            retry=self.config.retry,
            logger=self.logger,
        )

    def _list_captions(self, video_id: str) -> list[dict[str, Any]]:
        response = execute_with_retry(
            lambda: self.service.captions().list(part="snippet", videoId=video_id).execute(),
            description=f"{video_id} caption list",
            retry=self.config.retry,
            logger=self.logger,
        )
        items = response.get("items", [])
        return items if isinstance(items, list) else []

    def get_caption_index(self, video_id: str) -> dict[str, dict[str, Any]]:
        index: dict[str, dict[str, Any]] = {}
        for item in self._list_captions(video_id):
            snippet = item.get("snippet", {}) or {}
            language = str(snippet.get("language") or "").strip().lower()
            if not language:
                continue
            index[language] = item
        return index

    def _delete_caption(self, caption_id: str) -> None:
        execute_with_retry(
            lambda: self.service.captions().delete(id=caption_id).execute(),
            description=f"{caption_id} caption delete",
            retry=self.config.retry,
            logger=self.logger,
        )

    def upload_caption(
        self,
        video_id: str,
        language_code: str,
        subtitle_path: Path,
        existing_captions: Optional[dict[str, dict[str, Any]]] = None,
    ) -> None:
        libs = self._import_google_dependencies()
        normalized_language = str(language_code or "").strip().lower()
        caption_index = existing_captions or self.get_caption_index(video_id)
        existing_item = caption_index.get(normalized_language) if isinstance(caption_index, dict) else None
        if isinstance(existing_item, dict) and existing_item.get("id"):
            self._delete_caption(str(existing_item["id"]))

        media = libs["MediaFileUpload"](
            str(subtitle_path),
            mimetype="application/octet-stream",
            resumable=False,
        )
        body = {
            "snippet": {
                "videoId": video_id,
                "language": normalized_language,
                "name": SUBTITLE_TRACK_NAMES.get(normalized_language, normalized_language),
                "isDraft": False,
            }
        }
        execute_with_retry(
            lambda: self.service.captions().insert(part="snippet", body=body, media_body=media).execute(),
            description=f"{video_id} {normalized_language} caption upload",
            retry=self.config.retry,
            logger=self.logger,
        )

    def add_to_playlist(self, video_id: str, playlist_id: str) -> None:
        body = {
            "snippet": {
                "playlistId": playlist_id,
                "resourceId": {
                    "kind": "youtube#video",
                    "videoId": video_id,
                },
            }
        }
        execute_with_retry(
            lambda: self.service.playlistItems().insert(part="snippet", body=body).execute(),
            description=f"{video_id} playlist assignment",
            retry=self.config.retry,
            logger=self.logger,
        )
