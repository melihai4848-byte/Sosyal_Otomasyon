"""Interactive wrapper for updating existing private YouTube drafts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

from moduller.output_paths import find_existing_output, glob_outputs, grouped_json_output_path
from uploader.config import load_uploader_config
from uploader.logging_utils import configure_logging
from uploader.youtube_api import YouTubeApiClient


DESCRIPTION_FILES = {
    "tr": "YT-Metadata_TR.txt",
    "en": "YT-Metadata_EN.txt",
    "de": "YT-Metadata_DE.txt",
}

SUBTITLE_FILES = {
    "tr": "subtitle_tr.srt",
    "en": "subtitle_en.srt",
    "de": "subtitle_de.srt",
}

LANGUAGE_LABELS = {
    "tr": "TR",
    "en": "EN",
    "de": "DE",
}

CHAPTER_HEADERS = {
    "tr": "KISIMLAR",
    "en": "CHAPTERS",
    "de": "KAPITEL",
}

COMBINED_METADATA_CACHE_PATH = grouped_json_output_path("tools", "YouTube_Draft_Sync_Cache.json")
COMBINED_METADATA_LANGUAGE_MAP = {
    "Türkçe": "tr",
    "Turkce": "tr",
    "İngilizce": "en",
    "Ingilizce": "en",
    "Almanca": "de",
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").strip()


def _parse_best_title(path: Path) -> str:
    text = _read_text(path)
    match = re.search(r"^En Iyi Baslik:\s*(.+)$", text, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()

    current_section = ""
    title_candidates: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "BASLIK ONERILERI":
            current_section = stripped
            continue
        if stripped and set(stripped) == {"-"}:
            continue
        if current_section == "BASLIK ONERILERI":
            if stripped:
                title_candidates.append(stripped)
                continue
            if title_candidates:
                break

    if title_candidates:
        return title_candidates[0]

    lines = [line.strip() for line in text.splitlines()]
    for index, line in enumerate(lines):
        if line.startswith("#1"):
            for candidate in lines[index + 1 :]:
                if candidate and not set(candidate) <= {"-"}:
                    return candidate
    for line in lines:
        if line and not line.startswith("===") and ":" not in line:
            return line
    return ""


def _parse_description_body(path: Path) -> str:
    text = _read_text(path)
    lowered_name = path.name.lower()
    chapter_header = "KISIMLAR"
    for language_code, header in CHAPTER_HEADERS.items():
        if f"_{language_code}.txt" in lowered_name:
            chapter_header = header
            break
    section_names = {"HOOK", "DESCRIPTION", "KISIMLAR", "HASHTAG SATIRI", "SEARCH TERMS", "OPTIMIZATION NOTE (TR)"}
    current_section = ""
    section_lines: dict[str, list[str]] = {}
    start_capture = False

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Karakter Sayisi:"):
            start_capture = True
            continue
        if not start_capture:
            continue
        if stripped in section_names:
            current_section = stripped
            section_lines.setdefault(current_section, [])
            continue
        if stripped and set(stripped) == {"-"}:
            continue
        if current_section:
            section_lines.setdefault(current_section, []).append(line.rstrip())

    hook = "\n".join(section_lines.get("HOOK", [])).strip()
    description = "\n".join(section_lines.get("DESCRIPTION", [])).strip()
    chapters = "\n".join(section_lines.get("KISIMLAR", [])).strip()
    hashtags = "\n".join(section_lines.get("HASHTAG SATIRI", [])).strip()

    if hook or description or chapters or hashtags:
        parts: list[str] = []
        if hook:
            parts.append(hook)
        if description:
            parts.append(description)
        if chapters:
            parts.append(f"{chapter_header}\n{chapters}")
        if hashtags:
            parts.append(hashtags)
        return "\n\n".join(part for part in parts if part.strip()).strip()

    lines = text.splitlines()
    capture = False
    collected: list[str] = []
    hashtag_line = ""
    in_hashtag_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Karakter Sayisi:"):
            capture = True
            continue
        if not capture:
            continue
        if stripped == "HASHTAG SATIRI":
            in_hashtag_section = True
            continue
        if stripped and set(stripped) == {"-"}:
            continue
        if in_hashtag_section:
            if stripped:
                hashtag_line = stripped
                break
            continue
        collected.append(line)

    assembled = "\n".join(collected).strip()
    if hashtag_line:
        assembled = f"{assembled}\n\n{hashtag_line}".strip()
    return assembled


def _load_sync_cache() -> dict:
    if not COMBINED_METADATA_CACHE_PATH.exists():
        return {"videos": {}}
    try:
        parsed = json.loads(COMBINED_METADATA_CACHE_PATH.read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, dict) else {"videos": {}}
    except Exception:
        return {"videos": {}}


def _save_sync_cache(cache: dict) -> None:
    COMBINED_METADATA_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMBINED_METADATA_CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _hash_text(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _hash_payload(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _latest_combined_metadata_path() -> Optional[Path]:
    candidates = glob_outputs("*_metadata.json", groups=["youtube"], include_json_cache=True)
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _extract_structured_metadata(path: Path) -> tuple[dict[str, str], dict[str, str], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}, {}, []

    titles: dict[str, str] = {}
    descriptions: dict[str, str] = {}
    messages: list[str] = [f"✅ Birlesik metadata JSON bulundu: {path.name}"]

    for source_language, target_code in COMBINED_METADATA_LANGUAGE_MAP.items():
        item = payload.get(source_language)
        if not isinstance(item, dict):
            continue
        best_title = str(item.get("best_title") or "").strip()
        description_with_hashtags = str(item.get("description_with_hashtags") or "").strip()
        if best_title:
            titles[target_code] = best_title
            messages.append(f"✅ {LANGUAGE_LABELS[target_code]} baslik structured metadata'dan alindi.")
        if description_with_hashtags:
            descriptions[target_code] = description_with_hashtags
            messages.append(f"✅ {LANGUAGE_LABELS[target_code]} aciklama structured metadata'dan alindi.")
    return titles, descriptions, messages


def _find_required_outputs() -> tuple[dict[str, str], dict[str, str], dict[str, Path], list[str]]:
    titles: dict[str, str] = {}
    descriptions: dict[str, str] = {}
    subtitles: dict[str, Path] = {}
    messages: list[str] = []

    structured_path = _latest_combined_metadata_path()
    if structured_path:
        try:
            structured_titles, structured_descriptions, structured_messages = _extract_structured_metadata(structured_path)
            titles.update(structured_titles)
            descriptions.update(structured_descriptions)
            messages.extend(structured_messages)
        except Exception as exc:
            messages.append(f"⚠️ Birlesik metadata JSON okunamadi: {structured_path.name} ({exc})")

    for language_code, filename in DESCRIPTION_FILES.items():
        path = find_existing_output(filename, groups=["youtube"])
        if not path:
            messages.append(f"❌ {LANGUAGE_LABELS[language_code]} aciklama dosyasi bulunamadi: {filename}")
            continue
        if not descriptions.get(language_code):
            parsed = _parse_description_body(path)
            if parsed:
                descriptions[language_code] = parsed
                messages.append(f"✅ {LANGUAGE_LABELS[language_code]} aciklama dosyasi bulundu: {path.name}")
            else:
                messages.append(f"⚠️ {LANGUAGE_LABELS[language_code]} aciklama dosyasi bulundu ama metin parse edilemedi: {path.name}")
        if not titles.get(language_code):
            parsed_title = _parse_best_title(path)
            if parsed_title:
                titles[language_code] = parsed_title
                messages.append(f"✅ {LANGUAGE_LABELS[language_code]} baslik aciklama dosyasindan alindi: {path.name}")
            else:
                messages.append(f"⚠️ {LANGUAGE_LABELS[language_code]} baslik aciklama dosyasindan parse edilemedi: {path.name}")

    for language_code, filename in SUBTITLE_FILES.items():
        path = find_existing_output(filename, groups=["subtitle"])
        if path and path.exists():
            subtitles[language_code] = path
            messages.append(f"✅ {LANGUAGE_LABELS[language_code]} altyazisi bulundu: {path.name}")
        else:
            messages.append(f"⚠️ {LANGUAGE_LABELS[language_code]} altyazisi bulunamadi: {filename}")

    return titles, descriptions, subtitles, messages


def _print_generation_summary(titles: dict[str, str], descriptions: dict[str, str], subtitles: dict[str, Path], messages: list[str]) -> bool:
    print("\n" + "-" * 72)
    print("URETILMIS YOUTUBE DOSYALARI KONTROL EDILIYOR")
    print("-" * 72)
    for item in messages:
        print(item)

    if "tr" not in titles or "tr" not in descriptions:
        print("-" * 72)
        print("❌ En az TR baslik ve TR aciklama bulunmali. Modul durduruldu.")
        print("-" * 72)
        return False

    localization_count = sum(
        1 for language_code in ("en", "de") if titles.get(language_code) and descriptions.get(language_code)
    )
    print("-" * 72)
    print(f"TR metadata hazir. Ek lokalizasyon sayisi: {localization_count}")
    print(f"Bulunan altyazi sayisi: {len(subtitles)}")
    print("-" * 72)
    return True


def _choose_private_draft(api_client: YouTubeApiClient) -> dict | None:
    drafts = api_client.list_recent_private_videos(limit=15)
    if not drafts:
        print("❌ Kanaldaki private taslak video listesi bos.")
        return None

    channel_title = drafts[0].get("channel_title", "")
    print("\n" + "=" * 72)
    print("MEVCUT PRIVATE YOUTUBE TASLAKLARI")
    print("=" * 72)
    if channel_title:
        print(f"Kanal: {channel_title}")
        print("-" * 72)

    for index, item in enumerate(drafts, start=1):
        published_at = str(item.get("published_at") or "").replace("T", " ").replace("Z", "")
        title = str(item.get("title") or "").strip() or "(Baslik yok)"
        print(f"[{index}] {title}")
        print(f"     Video ID: {item.get('video_id', '')}")
        if published_at:
            print(f"     Yuklenme: {published_at}")

    try:
        selection = int(input("\n👉 Guncellenecek private taslak no: ").strip())
        if selection < 1 or selection > len(drafts):
            raise ValueError
    except Exception:
        print("❌ Gecersiz video secimi.")
        return None
    return drafts[selection - 1]


def _build_localizations(titles: dict[str, str], descriptions: dict[str, str]) -> dict[str, dict[str, str]]:
    localizations: dict[str, dict[str, str]] = {}
    for language_code in ("tr", "en", "de"):
        title = str(titles.get(language_code) or "").strip()
        description = str(descriptions.get(language_code) or "").strip()
        if not title and not description:
            continue
        payload: dict[str, str] = {}
        if title:
            payload["title"] = title
        if description:
            payload["description"] = description
        localizations[language_code] = payload
    return localizations


def _desired_metadata_payload(titles: dict[str, str], descriptions: dict[str, str]) -> dict:
    return {
        "title": str(titles.get("tr") or "").strip(),
        "description": str(descriptions.get("tr") or "").strip(),
        "localizations": _build_localizations(titles, descriptions),
    }


def _normalize_localized_text(value: str) -> str:
    return str(value or "").replace("\r\n", "\n").strip()


def _metadata_sync_needed(current_resource: dict, desired_payload: dict, defaults) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    current_snippet = current_resource.get("snippet", {}) or {}
    current_status = current_resource.get("status", {}) or {}
    current_localizations = current_resource.get("localizations", {}) or {}

    desired_title = _normalize_localized_text(desired_payload.get("title", ""))
    desired_description = _normalize_localized_text(desired_payload.get("description", ""))
    if _normalize_localized_text(current_snippet.get("title", "")) != desired_title:
        reasons.append("TR baslik farkli")
    if _normalize_localized_text(current_snippet.get("description", "")) != desired_description:
        reasons.append("TR aciklama farkli")

    default_language = str(defaults.default_language or current_snippet.get("defaultLanguage") or "tr").strip()
    default_audio_language = str(
        defaults.default_audio_language or current_snippet.get("defaultAudioLanguage") or "tr"
    ).strip()
    if str(current_snippet.get("defaultLanguage") or "").strip() != default_language:
        reasons.append("defaultLanguage farkli")
    if str(current_snippet.get("defaultAudioLanguage") or "").strip() != default_audio_language:
        reasons.append("defaultAudioLanguage farkli")
    if str(current_status.get("privacyStatus") or "").strip().lower() != "private":
        reasons.append("privacy private degil")
    if bool(current_status.get("embeddable", True)) != bool(defaults.embeddable):
        reasons.append("embeddable farkli")
    if str(current_status.get("license") or "").strip() != str(defaults.license_name):
        reasons.append("license farkli")
    if bool(current_status.get("publicStatsViewable", True)) != bool(defaults.public_stats_viewable):
        reasons.append("publicStatsViewable farkli")
    if bool(current_status.get("selfDeclaredMadeForKids", False)) != bool(defaults.self_declared_made_for_kids):
        reasons.append("madeForKids farkli")

    for language_code, localized in (desired_payload.get("localizations") or {}).items():
        if not isinstance(localized, dict):
            continue
        current_localized = current_localizations.get(language_code, {}) if isinstance(current_localizations, dict) else {}
        desired_local_title = _normalize_localized_text(localized.get("title", ""))
        desired_local_description = _normalize_localized_text(localized.get("description", ""))
        if _normalize_localized_text(current_localized.get("title", "")) != desired_local_title:
            reasons.append(f"{language_code.upper()} baslik farkli")
        if _normalize_localized_text(current_localized.get("description", "")) != desired_local_description:
            reasons.append(f"{language_code.upper()} aciklama farkli")

    return bool(reasons), reasons


def _video_cache_bucket(cache: dict, video_id: str) -> dict:
    videos = cache.setdefault("videos", {})
    bucket = videos.setdefault(video_id, {})
    bucket.setdefault("subtitles", {})
    return bucket


def _should_skip_caption_sync(cache: dict, video_id: str, language_code: str, subtitle_hash: str, caption_index: dict) -> bool:
    bucket = _video_cache_bucket(cache, video_id)
    subtitle_cache = bucket.get("subtitles", {}) if isinstance(bucket.get("subtitles"), dict) else {}
    cached_hash = str((subtitle_cache.get(language_code, {}) or {}).get("hash") or "").strip()
    return bool(cached_hash and cached_hash == subtitle_hash and language_code in caption_index)


def _mark_caption_synced(cache: dict, video_id: str, language_code: str, subtitle_hash: str) -> None:
    bucket = _video_cache_bucket(cache, video_id)
    subtitles = bucket.setdefault("subtitles", {})
    subtitles[language_code] = {"hash": subtitle_hash}


def run() -> None:
    config = load_uploader_config()
    logger = configure_logging(config)
    api_client = YouTubeApiClient(config, logger)

    print("\n" + "=" * 72)
    print("YOUTUBE DRAFT METADATA SYNC | PRIVATE DRAFT ONLY")
    print("=" * 72)
    print("Bu modul ASLA video upload etmez.")
    print("Sadece kanaldaki mevcut private taslagi gunceller.")
    print("Thumbnail bu modul tarafindan degistirilmez.")

    titles, descriptions, subtitles, messages = _find_required_outputs()
    if not _print_generation_summary(titles, descriptions, subtitles, messages):
        return

    try:
        selected_video = _choose_private_draft(api_client)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Private taslak listesi alinamadi: {exc}")
        return
    if not selected_video:
        return

    selected_title = str(selected_video.get("title") or "").strip()
    selected_video_id = str(selected_video.get("video_id") or "").strip()
    localizations = _build_localizations(titles, descriptions)
    desired_metadata = _desired_metadata_payload(titles, descriptions)
    extra_languages = [lang.upper() for lang in sorted(localizations) if lang != "tr"]
    print("\n" + "-" * 72)
    print(f"Secilen Taslak: {selected_title}")
    print(f"Video ID: {selected_video_id}")
    print("-" * 72)
    print(f"TR Baslik -> {titles.get('tr', '')}")
    print(f"TR Aciklama -> {len(descriptions.get('tr', ''))} karakter")
    print(f"Guncellenecek lokalizasyonlar -> {', '.join(sorted(localizations.keys())).upper()}")
    print(f"Guncellenecek altyazilar -> {', '.join(sorted(subtitles.keys())).upper() if subtitles else 'yok'}")
    confirm = input("👉 Bu private taslagi guncellemek istiyor musun? (e/h): ").strip().lower()
    if confirm not in {"e", "evet", "y", "yes"}:
        print("Islem iptal edildi.")
        return

    sync_cache = _load_sync_cache()
    current_resource = None
    try:
        current_resource = api_client.get_video_resource(selected_video_id)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Video mevcut metadata bilgisi alinamadi: {exc}")
        print(f"❌ Video mevcut metadata bilgisi alinamadi: {exc}")
        return

    metadata_needs_update, metadata_reasons = _metadata_sync_needed(current_resource, desired_metadata, config.defaults)
    metadata_hash = _hash_payload(desired_metadata)
    if metadata_needs_update:
        try:
            api_client.update_existing_private_video(
                selected_video_id,
                title=titles["tr"],
                description=descriptions["tr"],
                settings=config.defaults,
                localizations=localizations,
                current_resource=current_resource,
            )
            bucket = _video_cache_bucket(sync_cache, selected_video_id)
            bucket["metadata_hash"] = metadata_hash
            print(f"✅ Baslik, aciklama ve gerekli ayarlar guncellendi. ({', '.join(metadata_reasons[:5])})")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Video metadata guncellenemedi: {exc}")
            print(f"❌ Video metadata guncellenemedi: {exc}")
            return
    else:
        bucket = _video_cache_bucket(sync_cache, selected_video_id)
        bucket["metadata_hash"] = metadata_hash
        print("⏭️ Metadata degismedigi icin TR/localization guncellemesi atlandi.")

    try:
        caption_index = api_client.get_caption_index(selected_video_id)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Caption listesi alinamadi: {exc}")
        caption_index = {}

    uploaded_captions: list[str] = []
    skipped_captions: list[str] = []
    for language_code, subtitle_path in subtitles.items():
        subtitle_hash = _hash_file(subtitle_path)
        if _should_skip_caption_sync(sync_cache, selected_video_id, language_code, subtitle_hash, caption_index):
            skipped_captions.append(language_code.upper())
            print(f"⏭️ {language_code.upper()} altyazisi degismedigi icin atlandi.")
            continue
        try:
            api_client.upload_caption(
                selected_video_id,
                language_code,
                subtitle_path,
                existing_captions=caption_index,
            )
            caption_index[language_code] = {"id": f"uploaded:{language_code}", "snippet": {"language": language_code}}
            _mark_caption_synced(sync_cache, selected_video_id, language_code, subtitle_hash)
            uploaded_captions.append(language_code.upper())
            print(f"✅ {language_code.upper()} altyazisi guncellendi.")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"{language_code.upper()} altyazisi guncellenemedi: {exc}")
            print(f"⚠️ {language_code.upper()} altyazisi guncellenemedi: {exc}")

    _save_sync_cache(sync_cache)

    print("\n" + "=" * 72)
    print("YOUTUBE DRAFT METADATA SYNC TAMAMLANDI")
    print("=" * 72)
    print(f"Guncellenen private taslak: {selected_title}")
    print(f"Video ID: {selected_video_id}")
    print(f"Baslik / Aciklama: {'TR + ' + ', '.join(extra_languages) if extra_languages else 'yalnizca TR'}")
    if uploaded_captions or skipped_captions:
        summary_parts = []
        if uploaded_captions:
            summary_parts.append(f"Guncellendi: {', '.join(uploaded_captions)}")
        if skipped_captions:
            summary_parts.append(f"Atlandi: {', '.join(skipped_captions)}")
        print(f"Altyazilar: {' | '.join(summary_parts)}")
    else:
        print("Altyazilar: Yuklenemedi / bulunamadi")
    print("Thumbnail degistirilmedi.")
    print("Video upload yapilmadi.")
    print("Video private olarak birakildi.")
    print("=" * 72)


def main_cli(argv: Optional[list[str]] = None) -> int:
    _ = argv
    run()
    return 0
