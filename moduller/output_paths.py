import shutil
from pathlib import Path
from typing import Iterable

from moduller.config import OUTPUTS_DIR


GROUP_DIR_NAMES = {
    "subtitle": "100_Altyazı",
    "youtube": "200_YouTube",
    "instagram": "300_Instagram",
    "research": "400_Arastirma_Sonuclari",
    "tools": "500_Araclar_Sonuclari",
}


ROOT_JSON_CACHE_DIR = OUTPUTS_DIR / "_json_cache"


TXT_OUTPUT_NAMES = {
    "video_critic": "YT-Video_Analysis_TR.txt",
    "hook_rewrite": "YT-Hook_Analysis.txt",
    "trim_suggestions": "YT-Editing_Anaylsis_TR.txt",
    "broll": "YT-B-Roll_Prompts.txt",
    "reels_ideas": "IG-Reels_Fikirleri.txt",
    "music_prompts": "YT-Background_Music_Prompts.txt",
    "grammar": "Gramer_Duzenleyici_Raporu.txt",
    "live_trends": "Youtube_Trends-Trend_Analizi.txt",
    "analytics_feedback": "Youtube_Analytics-Secilen_Video_Feedback.txt",
    "analytics_channel_analysis": "YouTube_Analytics-Kanal_Analizi.txt",
    "analytics_video_analysis": "YouTube_Analytics-Secilen_Video_Analizi.txt",
    "analytics_action_plan": "YouTube_Analytics-Aksiyon_Plani.txt",
    "analytics_channel_prompt": "Youtube_Analytics_Kanal_Prompt.txt",
    "analytics_video_prompt": "Youtube_Analytics_Video_Prompt.txt",
    "instagram_carousel": "IG-Carousel_Fikirleri.txt",
    "instagram_metadata": "IG-Paylasim_Takvimi.txt",
    "instagram_story": "IG-Story_Fikirleri.txt",
    "main_video_thumbnails": "YT-Thumbnail_Prompts.txt",
    "reels_creator_report": "Reel_Shorts_Olusturucu_Raporu.txt",
    "master_pipeline": "Master_Pipeline_Raporu.txt",
    "topic_selector": "Youtube_Trends-Konu_Fikirleri.txt",
}


JSON_OUTPUT_NAMES = {
    "reels_ideas": "Reels_Shorts_Fikirleri.json",
    "reels_creator_report": "Reel_Shorts_Olusturucu_Raporu.json",
    "main_video_thumbnails": "Ana_Video_Thumbnail_Fikirleri.json",
    "live_trends": "Canli_Trend_ve_Veri_Motoru.json",
    "analytics_feedback": "Analitik_Geri_Bildirim_Dongusu.json",
    "analytics_channel_report": "YouTube_Analytics-Kanal_Analizi.json",
    "master_pipeline": "Master_Pipeline_Raporu.json",
    "topic_selector": "YouTube_Konu_Bulucu.json",
}


OUTPUT_KEY_GROUPS = {
    "video_critic": "youtube",
    "hook_rewrite": "youtube",
    "trim_suggestions": "youtube",
    "broll": "youtube",
    "music_prompts": "youtube",
    "main_video_thumbnails": "youtube",
    "grammar": "subtitle",
    "reels_ideas": "instagram",
    "instagram_carousel": "instagram",
    "instagram_metadata": "instagram",
    "instagram_story": "instagram",
    "live_trends": "research",
    "analytics_feedback": "research",
    "analytics_channel_analysis": "research",
    "analytics_video_analysis": "research",
    "analytics_action_plan": "research",
    "analytics_channel_prompt": "research",
    "analytics_video_prompt": "research",
    "master_pipeline": "research",
    "topic_selector": "research",
    "analytics_channel_report": "research",
    "reels_creator_report": "tools",
}


JSON_CACHE_DIR = ROOT_JSON_CACHE_DIR

_TXT_FILENAME_TO_GROUP = {
    filename: OUTPUT_KEY_GROUPS[key]
    for key, filename in TXT_OUTPUT_NAMES.items()
    if key in OUTPUT_KEY_GROUPS
}
_JSON_FILENAME_TO_GROUP = {
    filename: OUTPUT_KEY_GROUPS[key]
    for key, filename in JSON_OUTPUT_NAMES.items()
    if key in OUTPUT_KEY_GROUPS
}
_DIRECTORY_GROUP_HINTS = {
    "_archive": "subtitle",
    "checkpoints": "research",
    "broll_downloads": "tools",
    "premiere_xml": "tools",
    "reels_render": "tools",
}


def output_group_dir(group: str) -> Path:
    path = OUTPUTS_DIR / GROUP_DIR_NAMES[group]
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_cache_dir(group: str) -> Path:
    path = output_group_dir(group) / "_json_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def grouped_output_path(group: str, filename: str) -> Path:
    return output_group_dir(group) / filename


def grouped_json_output_path(group: str, filename: str) -> Path:
    return json_cache_dir(group) / filename


def output_group_for_key(key: str) -> str:
    return OUTPUT_KEY_GROUPS[key]


def txt_output_path(key: str) -> Path:
    if key == "master_pipeline":
        return OUTPUTS_DIR / TXT_OUTPUT_NAMES[key]
    return grouped_output_path(output_group_for_key(key), TXT_OUTPUT_NAMES[key])


def json_output_path(key: str) -> Path:
    return grouped_json_output_path(output_group_for_key(key), JSON_OUTPUT_NAMES[key])


def stem_json_output_path(stem: str, suffix: str, group: str = "youtube") -> Path:
    return grouped_json_output_path(group, f"{stem}{suffix}")


def _candidate_search_dirs(groups: Iterable[str] | None = None, include_json_cache: bool = False) -> list[Path]:
    ordered_groups = list(groups or GROUP_DIR_NAMES.keys())
    dirs: list[Path] = []

    for group in ordered_groups:
        dirs.append(output_group_dir(group))
        if include_json_cache:
            dirs.append(json_cache_dir(group))

    # Legacy fallback
    dirs.append(OUTPUTS_DIR)
    if include_json_cache and ROOT_JSON_CACHE_DIR.exists():
        dirs.append(ROOT_JSON_CACHE_DIR)

    unique: list[Path] = []
    seen = set()
    for path in dirs:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def find_existing_output(
    filename: str,
    groups: Iterable[str] | None = None,
    include_json_cache: bool = False,
) -> Path | None:
    for base_dir in _candidate_search_dirs(groups=groups, include_json_cache=include_json_cache):
        candidate = base_dir / filename
        if candidate.exists():
            return candidate
    return None


def glob_outputs(
    pattern: str,
    groups: Iterable[str] | None = None,
    include_json_cache: bool = False,
) -> list[Path]:
    matches: list[Path] = []
    seen = set()
    for base_dir in _candidate_search_dirs(groups=groups, include_json_cache=include_json_cache):
        for path in base_dir.glob(pattern):
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            matches.append(path)
    return matches


def infer_output_group(filename: str) -> str | None:
    name = str(filename or "").strip()
    if not name:
        return None

    if name in _TXT_FILENAME_TO_GROUP:
        return _TXT_FILENAME_TO_GROUP[name]
    if name in _JSON_FILENAME_TO_GROUP:
        return _JSON_FILENAME_TO_GROUP[name]

    lowered = name.casefold()
    if lowered.endswith(".srt"):
        return "subtitle"

    if lowered.startswith("main_video-"):
        return "youtube"
    if lowered.startswith("ig-") or "instagram" in lowered:
        return "instagram"
    if lowered.startswith("youtube_analytics") or lowered.startswith("youtube_trends"):
        return "research"

    if lowered in {
        "reels_shorts_fikirleri.json",
        "shorts_reels_thumbnail_fikirleri.json",
        "shorts_reels_thumbnail_fikirleri.txt",
        "carousel_fikirleri.json",
        "carousel_fikirleri.txt",
    }:
        return "instagram"

    if lowered in {
        "automatic_broll_download_report.json",
        "automatic_broll_download_report.txt",
        "premiere_xml_report.json",
        "premiere_xml_report.txt",
        "reel_shorts_olusturucu_raporu.json",
        "reel_shorts_olusturucu_raporu.txt",
    }:
        return "tools"

    if lowered.startswith("youtube_konu_bulucu") or lowered.startswith("analitik_geri_bildirim") or lowered.startswith("master_pipeline"):
        return "research"

    if lowered.endswith("_video_description.json") or lowered.endswith("_video_titles.json"):
        return "youtube"
    if lowered.endswith("_video_critic.json") or lowered.endswith("_hook_rewrite.json"):
        return "youtube"
    if lowered.endswith("_trim_suggestions.json") or lowered.endswith("_b_roll_fikirleri.json"):
        return "youtube"
    if lowered.endswith("_music_prompts.json") or lowered.endswith("_metadata.json"):
        return "youtube"
    if lowered.endswith("_instagram_carousel.json") or lowered.endswith("_instagram_story_plani.json"):
        return "instagram"
    if lowered.endswith("_instagram_schedule_plan.json"):
        return "instagram"
    if lowered.endswith("_coklu_secim_checkpoint.json"):
        return "research"

    return None


def _move_file_to_target(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source == target:
        return

    if target.exists():
        try:
            source_mtime = source.stat().st_mtime
            target_mtime = target.stat().st_mtime
        except OSError:
            source_mtime = 0.0
            target_mtime = 0.0

        if target_mtime >= source_mtime:
            source.unlink(missing_ok=True)
            return

        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()

    source.replace(target)


def _merge_directory(source: Path, target: Path) -> None:
    if source == target:
        return

    target.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        child_target = target / child.name
        if child.is_dir():
            _merge_directory(child, child_target)
        else:
            _move_file_to_target(child, child_target)

    try:
        source.rmdir()
    except OSError:
        pass


def _route_legacy_file(path: Path) -> Path | None:
    group = infer_output_group(path.name)
    if not group:
        return None

    if path.suffix.lower() == ".json":
        return grouped_json_output_path(group, path.name)
    return grouped_output_path(group, path.name)


def cleanup_hidden_outputs() -> None:
    for group in GROUP_DIR_NAMES:
        output_group_dir(group)
        json_cache_dir(group)

    for path in OUTPUTS_DIR.iterdir():
        if path.name in set(GROUP_DIR_NAMES.values()) | {"_json_cache"}:
            continue

        if path.is_dir():
            if path.name in _DIRECTORY_GROUP_HINTS:
                _merge_directory(path, grouped_output_path(_DIRECTORY_GROUP_HINTS[path.name], path.name))
                continue
            if path.name in {"analitik_geri_bildirim", "reel_shorts_olusturucu", "topic_selector"}:
                for child in path.iterdir():
                    if not child.is_file():
                        continue
                    hedef = _route_legacy_file(child)
                    if hedef is None:
                        continue
                    _move_file_to_target(child, hedef)
                try:
                    path.rmdir()
                except OSError:
                    pass
            continue

        if not path.is_file():
            continue

        hedef = _route_legacy_file(path)
        if hedef is None:
            continue
        _move_file_to_target(path, hedef)

    if ROOT_JSON_CACHE_DIR.exists():
        for child in ROOT_JSON_CACHE_DIR.iterdir():
            if not child.is_file():
                continue
            hedef = _route_legacy_file(child)
            if hedef is None:
                continue
            _move_file_to_target(child, hedef)
        try:
            next(ROOT_JSON_CACHE_DIR.iterdir())
        except StopIteration:
            try:
                ROOT_JSON_CACHE_DIR.rmdir()
            except OSError:
                pass
