from pathlib import Path
from typing import Iterable

from moduller.output_paths import find_existing_output, glob_outputs, grouped_output_path


SUBTITLE_GROUP = "subtitle"
SUBTITLE_INTERMEDIATE_DIRNAME = "_ara_ciktilar"
FINAL_SUBTITLE_OUTPUT_NAMES = {
    "subtitle_tr.srt",
    "subtitle_en.srt",
    "subtitle_de.srt",
    "subtitle_shorts.srt",
}
KNOWN_INTERMEDIATE_SUBTITLE_NAMES = {
    "subtitle_raw_tr.srt",
    "subtitle_raw_en.srt",
    "subtitle_raw_shorts.srt",
    "subtitle_raw_tr_glossary_fixed.srt",
    "subtitle_raw_shorts_glossary_fixed.srt",
    "subtitle_shorts_tr.srt",
    "subtitle_tr_eski.srt",
    "subtitle_llm_en.srt",
    "grammar_video_glossary.json",
    "grammar_llm_debug.txt",
    "translation_llm_debug.txt",
    "Gramer_Duzenleyici_Raporu.txt",
    "subtitle_whisper_en.srt",
}


def subtitle_output_path(filename: str) -> Path:
    return grouped_output_path(SUBTITLE_GROUP, filename)


def subtitle_intermediate_dir() -> Path:
    path = subtitle_output_path(SUBTITLE_INTERMEDIATE_DIRNAME)
    path.mkdir(parents=True, exist_ok=True)
    return path


def subtitle_intermediate_output_path(filename: str) -> Path:
    return subtitle_intermediate_dir() / filename


def find_subtitle_artifact(filename: str) -> Path | None:
    direct = subtitle_output_path(filename)
    if direct.exists():
        return direct

    intermediate = subtitle_intermediate_output_path(filename)
    if intermediate.exists():
        return intermediate

    return find_existing_output(filename, groups=(SUBTITLE_GROUP,), include_json_cache=False)


def find_subtitle_file(filename: str) -> Path | None:
    return find_existing_output(filename, groups=(SUBTITLE_GROUP,), include_json_cache=False)


def list_subtitle_files() -> list[Path]:
    files = [path for path in glob_outputs("*.srt", groups=(SUBTITLE_GROUP,), include_json_cache=False) if path.is_file()]
    return sorted(files, key=lambda item: item.name.lower())


def find_first_existing_subtitle(filenames: Iterable[str]) -> Path | None:
    for filename in filenames:
        found = find_subtitle_file(filename)
        if found:
            return found
    return None


def relocate_known_subtitle_intermediates() -> None:
    for filename in KNOWN_INTERMEDIATE_SUBTITLE_NAMES:
        source = subtitle_output_path(filename)
        if not source.exists() or not source.is_file():
            continue
        target = subtitle_intermediate_output_path(filename)
        if source == target:
            continue
        if target.exists():
            target.unlink()
        source.replace(target)
