from __future__ import annotations

import json
import subprocess
import xml.etree.ElementTree as ET
from bisect import bisect_right
from pathlib import Path
from typing import Any
from xml.dom import minidom

from moduller.config import INPUTS_DIR
from moduller.logger import get_logger
from moduller.output_paths import glob_outputs, grouped_json_output_path, grouped_output_path
from moduller.video_edit_utils import ffprobe_binary, timestamp_to_seconds

logger = get_logger("premiere_xml")

PREMIERE_OUTPUT_DIR = grouped_output_path("tools", "premiere_xml")
PREMIERE_MEDIA_CACHE_PATH = grouped_json_output_path("tools", "premiere_xml_media_cache.json")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
STEM_SUFFIXES = [
    "_trim_suggestions",
    "_B_roll_fikirleri",
    "_instagram_carousel",
    "_standart_tr_grammar_fixed",
    "_standart_tr",
    "_raw_grammar_fixed",
    "_raw",
    "_shorts",
    "subtitle_raw_shorts",
    "subtitle_shorts",
    "subtitle_raw_en",
    "subtitle_llm_en",
    "subtitle_tr",
    "subtitle_en",
    "subtitle_de",
    "_tr_grammar_fixed",
    "_grammar_fixed",
    "_tr",
    "_en",
    "_de",
]


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_asset_stem(value: str | Path) -> str:
    stem = Path(str(value)).stem
    normalized = stem
    for suffix in STEM_SUFFIXES:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    return normalized.rstrip("_- ")


def find_video_files() -> list[Path]:
    return sorted(
        [
            path
            for path in INPUTS_DIR.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ]
    )


def find_trim_files() -> list[Path]:
    candidates = glob_outputs("*_trim_suggestions.json", groups=("youtube",), include_json_cache=True)
    unique = {path.resolve(): path for path in candidates if path.is_file()}
    return sorted(unique.values(), key=lambda item: item.stat().st_mtime, reverse=True)


def find_broll_report_files() -> list[Path]:
    base_dir = grouped_output_path("tools", "broll_downloads")
    if not base_dir.exists():
        return []
    return sorted(base_dir.rglob("automatic_broll_download_report.json"), key=lambda item: item.stat().st_mtime, reverse=True)


def select_file(files: list[Path], title: str) -> Path | None:
    if not files:
        return None
    if len(files) == 1:
        return files[0]

    print(f"\n{title}")
    for index, path in enumerate(files, start=1):
        print(f"  [{index}] {path.name}")

    raw = input("👉 Secim (bos = ilk dosya, 0 = atla): ").strip()
    if not raw:
        return files[0]
    if raw == "0":
        return None
    try:
        return files[int(raw) - 1]
    except (ValueError, IndexError):
        logger.error("Gecersiz secim yapildi.")
        return None


def match_trim_file(video_path: Path, trim_files: list[Path]) -> Path | None:
    video_stem = normalize_asset_stem(video_path.stem)
    for path in trim_files:
        candidate_stem = normalize_asset_stem(path.stem.replace("_trim_suggestions", ""))
        if candidate_stem == video_stem:
            return path
    return trim_files[0] if len(trim_files) == 1 else None


def match_broll_report(video_path: Path, report_files: list[Path]) -> Path | None:
    video_stem = normalize_asset_stem(video_path.stem)
    for path in report_files:
        if normalize_asset_stem(path.parent.name) == video_stem:
            return path
    return report_files[0] if len(report_files) == 1 else None


def load_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_media_cache() -> dict[str, Any]:
    if not PREMIERE_MEDIA_CACHE_PATH.exists():
        return {"entries": {}}
    try:
        payload = json.loads(PREMIERE_MEDIA_CACHE_PATH.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            entries = payload.get("entries", {})
            return {"entries": entries if isinstance(entries, dict) else {}}
    except Exception:
        pass
    return {"entries": {}}


def save_media_cache(cache: dict[str, Any]) -> None:
    PREMIERE_MEDIA_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREMIERE_MEDIA_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def media_cache_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
    }


def media_cache_key(path: Path) -> str:
    return str(path.resolve()).lower()


def parse_resolution_text(value: str) -> tuple[int, int]:
    text = str(value or "").strip().lower()
    if "x" not in text:
        return 0, 0
    left, right = text.split("x", 1)
    try:
        width = int(left.strip())
        height = int(right.strip())
    except ValueError:
        return 0, 0
    return max(0, width), max(0, height)


def parse_timestamp_range(timestamp_range: str) -> tuple[float, float]:
    if "-->" not in str(timestamp_range or ""):
        return 0.0, 0.0
    start_raw, end_raw = [item.strip() for item in str(timestamp_range).split("-->", 1)]
    return timestamp_to_seconds(start_raw), timestamp_to_seconds(end_raw)


def merge_ranges(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    cleaned = sorted((max(0.0, start), max(0.0, end)) for start, end in ranges if end > start)
    if not cleaned:
        return []

    merged: list[list[float]] = [[cleaned[0][0], cleaned[0][1]]]
    for start, end in cleaned[1:]:
        current = merged[-1]
        if start <= current[1]:
            current[1] = max(current[1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def build_keep_ranges(total_duration: float, cut_ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if total_duration <= 0:
        return []
    if not cut_ranges:
        return [(0.0, total_duration)]

    keep_ranges: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in cut_ranges:
        if start > cursor:
            keep_ranges.append((cursor, min(start, total_duration)))
        cursor = max(cursor, end)
        if cursor >= total_duration:
            break

    if cursor < total_duration:
        keep_ranges.append((cursor, total_duration))
    return [(start, end) for start, end in keep_ranges if end - start > 0.05]


def build_cut_timeline_map(cut_ranges: list[tuple[float, float]]) -> dict[str, list[float]]:
    starts: list[float] = []
    ends: list[float] = []
    removed_before_end: list[float] = []
    removed_total = 0.0
    for start, end in cut_ranges:
        starts.append(float(start))
        ends.append(float(end))
        removed_total += max(0.0, float(end) - float(start))
        removed_before_end.append(removed_total)
    return {
        "starts": starts,
        "ends": ends,
        "removed_before_end": removed_before_end,
    }


def removed_duration_before(
    source_seconds: float,
    cut_ranges: list[tuple[float, float]],
    cut_timeline_map: dict[str, list[float]] | None = None,
) -> float:
    if not cut_ranges:
        return 0.0
    timeline_map = cut_timeline_map or build_cut_timeline_map(cut_ranges)
    starts = timeline_map.get("starts", [])
    ends = timeline_map.get("ends", [])
    removed_before_end = timeline_map.get("removed_before_end", [])
    if not starts:
        return 0.0

    completed_index = bisect_right(ends, source_seconds) - 1
    removed = removed_before_end[completed_index] if completed_index >= 0 else 0.0

    next_index = completed_index + 1
    if 0 <= next_index < len(starts) and starts[next_index] < source_seconds < ends[next_index]:
        removed += source_seconds - starts[next_index]
    return removed


def map_source_to_sequence_time(
    source_seconds: float,
    cut_ranges: list[tuple[float, float]],
    cut_timeline_map: dict[str, list[float]] | None = None,
) -> float:
    if not cut_ranges:
        return max(0.0, source_seconds)
    timeline_map = cut_timeline_map or build_cut_timeline_map(cut_ranges)
    starts = timeline_map.get("starts", [])
    ends = timeline_map.get("ends", [])
    insertion_index = bisect_right(starts, source_seconds) - 1
    if 0 <= insertion_index < len(starts) and starts[insertion_index] <= source_seconds < ends[insertion_index]:
        cut_start = starts[insertion_index]
        return max(0.0, cut_start - removed_duration_before(cut_start, cut_ranges, timeline_map))
    return max(0.0, source_seconds - removed_duration_before(source_seconds, cut_ranges, timeline_map))


def seconds_to_frames(seconds: float, fps: float) -> int:
    return max(0, int(round(max(0.0, seconds) * fps)))


def frame_rate_info(fps: float) -> tuple[int, str]:
    target_fps = fps or 30.0
    timebase = int(round(target_fps))
    ntsc = "TRUE" if abs(target_fps - timebase) > 0.01 else "FALSE"
    return max(1, timebase), ntsc


def probe_video_media(
    path: Path,
    *,
    media_cache: dict[str, Any] | None = None,
    known_fields: dict[str, Any] | None = None,
    default_fps: float | None = None,
    assume_silent: bool = False,
) -> dict[str, Any]:
    resolved = path.resolve()
    cache_entries = media_cache.setdefault("entries", {}) if isinstance(media_cache, dict) else {}
    cache_key = media_cache_key(resolved)
    signature = media_cache_signature(resolved)
    cache_entry = cache_entries.get(cache_key) if isinstance(cache_entries, dict) else None
    if isinstance(cache_entry, dict) and cache_entry.get("signature") == signature:
        cached_payload = cache_entry.get("media_info")
        if isinstance(cached_payload, dict):
            return dict(cached_payload)

    known_fields = known_fields or {}
    known_width = int(known_fields.get("width", 0) or 0)
    known_height = int(known_fields.get("height", 0) or 0)
    try:
        known_duration_seconds = float(known_fields.get("duration_seconds", 0.0) or 0.0)
    except Exception:
        known_duration_seconds = 0.0

    if assume_silent and known_width > 0 and known_height > 0 and known_duration_seconds > 0:
        fps = float(default_fps or 30.0 or 30.0)
        timebase, ntsc = frame_rate_info(fps)
        media_info = {
            "path": str(resolved),
            "name": resolved.name,
            "width": known_width,
            "height": known_height,
            "fps": fps,
            "timebase": timebase,
            "ntsc": ntsc,
            "duration_seconds": known_duration_seconds,
            "duration_frames": seconds_to_frames(known_duration_seconds, fps),
            "has_audio": False,
            "sample_rate": 48000,
            "pathurl": resolved.as_uri(),
        }
        if isinstance(cache_entries, dict):
            cache_entries[cache_key] = {"signature": signature, "media_info": media_info}
        return media_info

    ffprobe = ffprobe_binary()
    if not ffprobe:
        raise RuntimeError("ffprobe bulunamadi. Premiere XML icin FFmpeg kurulumu gerekiyor.")

    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,width,height,r_frame_rate,sample_rate:format=duration",
        "-of",
        "json",
        str(resolved),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    payload = json.loads(completed.stdout or "{}")
    streams = payload.get("streams", [])
    video_stream = next((item for item in streams if item.get("codec_type") == "video"), {})
    audio_stream = next((item for item in streams if item.get("codec_type") == "audio"), {})
    fps_raw = str(video_stream.get("r_frame_rate") or "30/1")

    if "/" in fps_raw:
        numerator, denominator = fps_raw.split("/", 1)
        fps = float(numerator) / max(1.0, float(denominator or 1))
    else:
        fps = float(fps_raw or 30.0)

    try:
        duration_seconds = float((payload.get("format") or {}).get("duration") or 0.0)
    except Exception:
        duration_seconds = 0.0

    width = int(video_stream.get("width") or known_width or 1920)
    height = int(video_stream.get("height") or known_height or 1080)
    if duration_seconds <= 0 and known_duration_seconds > 0:
        duration_seconds = known_duration_seconds
    sample_rate = int(audio_stream.get("sample_rate") or 48000)
    timebase, ntsc = frame_rate_info(fps)

    media_info = {
        "path": str(resolved),
        "name": resolved.name,
        "width": width,
        "height": height,
        "fps": fps or 30.0,
        "timebase": timebase,
        "ntsc": ntsc,
        "duration_seconds": duration_seconds,
        "duration_frames": seconds_to_frames(duration_seconds, fps or 30.0),
        "has_audio": bool(audio_stream),
        "sample_rate": sample_rate,
        "pathurl": resolved.as_uri(),
    }
    if isinstance(cache_entries, dict):
        cache_entries[cache_key] = {"signature": signature, "media_info": media_info}
    return media_info


def extract_cut_ranges(trim_path: Path | None) -> tuple[list[tuple[float, float]], dict[str, Any] | None]:
    if not trim_path or not trim_path.exists():
        return [], None

    payload = load_json(trim_path)
    if not isinstance(payload, dict):
        return [], None

    ranges: list[tuple[float, float]] = []
    for item in payload.get("trim_targets", []):
        if not isinstance(item, dict):
            continue
        start_seconds, end_seconds = parse_timestamp_range(str(item.get("timestamp", "") or ""))
        if end_seconds > start_seconds:
            ranges.append((start_seconds, end_seconds))
    return merge_ranges(ranges), payload


def extract_broll_items(
    report_path: Path | None,
    cut_ranges: list[tuple[float, float]],
    sequence_duration: float,
    *,
    cut_timeline_map: dict[str, list[float]] | None = None,
    media_cache: dict[str, Any] | None = None,
    sequence_fps: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not report_path or not report_path.exists():
        return [], None

    payload = load_json(report_path)
    if not isinstance(payload, dict):
        return [], None

    selected_items: list[dict[str, Any]] = []
    for item in payload.get("items", []):
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "") or "")
        if status not in {"downloaded", "existing", "copied_from_cache"}:
            continue

        local_path_raw = str(item.get("local_path", "") or "").strip()
        if not local_path_raw:
            continue
        local_path = Path(local_path_raw)
        if not local_path.exists():
            continue

        source_start, source_end = parse_timestamp_range(str(item.get("timestamp", "") or ""))
        timeline_start = map_source_to_sequence_time(source_start, cut_ranges, cut_timeline_map)
        requested_duration = max(0.0, source_end - source_start)

        reported_width, reported_height = parse_resolution_text(str(item.get("resolution", "") or ""))
        media_info = probe_video_media(
            local_path,
            media_cache=media_cache,
            known_fields={
                "width": reported_width,
                "height": reported_height,
                "duration_seconds": item.get("duration_seconds"),
            },
            default_fps=sequence_fps,
            assume_silent=True,
        )
        overlay_duration = requested_duration if requested_duration > 0 else min(5.0, float(media_info.get("duration_seconds") or 5.0))
        overlay_duration = min(overlay_duration, float(media_info.get("duration_seconds") or overlay_duration))
        overlay_duration = min(overlay_duration, max(0.0, sequence_duration - timeline_start))
        if overlay_duration <= 0.05:
            continue

        selected_items.append(
            {
                "timestamp": str(item.get("timestamp", "") or ""),
                "source_start": source_start,
                "timeline_start": timeline_start,
                "duration_seconds": overlay_duration,
                "local_path": local_path,
                "media_info": media_info,
                "provider": str(item.get("provider", "") or ""),
                "query": str(item.get("stock_search_query", "") or ""),
            }
        )
    return selected_items, payload


def add_text(parent: ET.Element, tag: str, value: Any) -> ET.Element:
    element = ET.SubElement(parent, tag)
    element.text = str(value)
    return element


def add_rate(parent: ET.Element, timebase: int, ntsc: str) -> ET.Element:
    rate = ET.SubElement(parent, "rate")
    add_text(rate, "timebase", timebase)
    add_text(rate, "ntsc", ntsc)
    return rate


def add_sample_characteristics(parent: ET.Element, media_info: dict[str, Any]) -> None:
    sample = ET.SubElement(parent, "samplecharacteristics")
    add_rate(sample, int(media_info["timebase"]), str(media_info["ntsc"]))
    add_text(sample, "width", int(media_info["width"]))
    add_text(sample, "height", int(media_info["height"]))
    add_text(sample, "anamorphic", "FALSE")
    add_text(sample, "pixelaspectratio", "square")
    add_text(sample, "fielddominance", "none")


def append_file_reference(parent: ET.Element, file_id: str, media_info: dict[str, Any], expanded_ids: set[str]) -> None:
    file_element = ET.SubElement(parent, "file", id=file_id)
    if file_id in expanded_ids:
        return

    expanded_ids.add(file_id)
    add_text(file_element, "name", media_info["name"])
    add_text(file_element, "pathurl", media_info["pathurl"])
    add_rate(file_element, int(media_info["timebase"]), str(media_info["ntsc"]))
    add_text(file_element, "duration", int(media_info["duration_frames"]))

    media = ET.SubElement(file_element, "media")
    video = ET.SubElement(media, "video")
    add_sample_characteristics(video, media_info)

    if media_info.get("has_audio"):
        audio = ET.SubElement(media, "audio")
        sample = ET.SubElement(audio, "samplecharacteristics")
        add_text(sample, "depth", "16")
        add_text(sample, "samplerate", int(media_info.get("sample_rate", 48000)))


def add_track(parent: ET.Element) -> ET.Element:
    track = ET.SubElement(parent, "track")
    add_text(track, "locked", "FALSE")
    add_text(track, "enabled", "TRUE")
    return track


def add_clipitem(
    track: ET.Element,
    *,
    clip_id: str,
    file_id: str,
    name: str,
    media_info: dict[str, Any],
    start_frames: int,
    end_frames: int,
    in_frames: int,
    out_frames: int,
    mediatype: str,
    expanded_ids: set[str],
    track_index: int = 1,
) -> None:
    clipitem = ET.SubElement(track, "clipitem", id=clip_id)
    add_text(clipitem, "name", name)
    add_text(clipitem, "duration", int(media_info["duration_frames"]))
    add_rate(clipitem, int(media_info["timebase"]), str(media_info["ntsc"]))
    add_text(clipitem, "start", start_frames)
    add_text(clipitem, "end", end_frames)
    add_text(clipitem, "enabled", "TRUE")
    add_text(clipitem, "in", in_frames)
    add_text(clipitem, "out", out_frames)
    append_file_reference(clipitem, file_id, media_info, expanded_ids)

    source_track = ET.SubElement(clipitem, "sourcetrack")
    add_text(source_track, "mediatype", mediatype)
    add_text(source_track, "trackindex", track_index)


def pack_broll_tracks(broll_items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not broll_items:
        return []
    sorted_items = sorted(
        broll_items,
        key=lambda item: (float(item.get("timeline_start", 0.0) or 0.0), float(item.get("duration_seconds", 0.0) or 0.0)),
    )
    tracks: list[list[dict[str, Any]]] = []
    track_end_times: list[float] = []
    for item in sorted_items:
        start_seconds = float(item.get("timeline_start", 0.0) or 0.0)
        end_seconds = start_seconds + float(item.get("duration_seconds", 0.0) or 0.0)
        placed = False
        for index, current_end in enumerate(track_end_times):
            if start_seconds >= current_end - 0.001:
                tracks[index].append(item)
                track_end_times[index] = end_seconds
                placed = True
                break
        if not placed:
            tracks.append([item])
            track_end_times.append(end_seconds)
    return tracks


def pretty_xml_bytes(root: ET.Element) -> bytes:
    rough_string = ET.tostring(root, encoding="utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ", encoding="utf-8")


def build_sequence_xml(
    video_path: Path,
    video_info: dict[str, Any],
    keep_ranges: list[tuple[float, float]],
    broll_items: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    fps = float(video_info["fps"])
    sequence_duration_seconds = sum(end - start for start, end in keep_ranges) if keep_ranges else float(video_info["duration_seconds"])
    sequence_duration_frames = seconds_to_frames(sequence_duration_seconds, fps)

    xmeml = ET.Element("xmeml", version="5")
    sequence = ET.SubElement(xmeml, "sequence", id="sequence-1")
    add_text(sequence, "name", f"{video_path.stem}_rough_cut")
    add_rate(sequence, int(video_info["timebase"]), str(video_info["ntsc"]))
    add_text(sequence, "duration", sequence_duration_frames)
    add_text(sequence, "in", 0)
    add_text(sequence, "out", sequence_duration_frames)

    media = ET.SubElement(sequence, "media")
    video = ET.SubElement(media, "video")
    sequence_format = ET.SubElement(video, "format")
    add_sample_characteristics(sequence_format, video_info)

    main_track = add_track(video)
    packed_broll_tracks = pack_broll_tracks(broll_items)
    overlay_tracks = [add_track(video) for _ in packed_broll_tracks] or [add_track(video)]

    audio = ET.SubElement(media, "audio")
    audio_track = add_track(audio)

    expanded_ids: set[str] = set()
    main_file_id = "file-main-video"
    current_timeline_seconds = 0.0
    main_segments_summary: list[dict[str, Any]] = []

    if not keep_ranges:
        keep_ranges = [(0.0, float(video_info["duration_seconds"]))]

    for index, (start_seconds, end_seconds) in enumerate(keep_ranges, start=1):
        duration_seconds = max(0.0, end_seconds - start_seconds)
        if duration_seconds <= 0.05:
            continue

        clip_start_frames = seconds_to_frames(current_timeline_seconds, fps)
        clip_end_frames = seconds_to_frames(current_timeline_seconds + duration_seconds, fps)
        in_frames = seconds_to_frames(start_seconds, fps)
        out_frames = seconds_to_frames(end_seconds, fps)

        add_clipitem(
            main_track,
            clip_id=f"main-video-{index}",
            file_id=main_file_id,
            name=video_info["name"],
            media_info=video_info,
            start_frames=clip_start_frames,
            end_frames=clip_end_frames,
            in_frames=in_frames,
            out_frames=out_frames,
            mediatype="video",
            expanded_ids=expanded_ids,
        )

        if video_info.get("has_audio"):
            add_clipitem(
                audio_track,
                clip_id=f"main-audio-{index}",
                file_id=main_file_id,
                name=video_info["name"],
                media_info=video_info,
                start_frames=clip_start_frames,
                end_frames=clip_end_frames,
                in_frames=in_frames,
                out_frames=out_frames,
                mediatype="audio",
                expanded_ids=expanded_ids,
            )

        main_segments_summary.append(
            {
                "source_start_seconds": round(start_seconds, 3),
                "source_end_seconds": round(end_seconds, 3),
                "timeline_start_seconds": round(current_timeline_seconds, 3),
                "timeline_end_seconds": round(current_timeline_seconds + duration_seconds, 3),
            }
        )
        current_timeline_seconds += duration_seconds

    overlay_summary: list[dict[str, Any]] = []
    overlay_index = 0
    for track_index, track_items in enumerate(packed_broll_tracks, start=1):
        overlay_track = overlay_tracks[track_index - 1]
        for item in track_items:
            overlay_index += 1
            media_info = item["media_info"]
            overlay_fps = float(media_info["fps"])
            file_id = f"file-broll-{overlay_index}"
            timeline_start_seconds = float(item["timeline_start"])
            duration_seconds = float(item["duration_seconds"])

            start_frames = seconds_to_frames(timeline_start_seconds, fps)
            end_frames = seconds_to_frames(timeline_start_seconds + duration_seconds, fps)
            out_frames = seconds_to_frames(duration_seconds, overlay_fps)

            add_clipitem(
                overlay_track,
                clip_id=f"broll-video-{overlay_index}",
                file_id=file_id,
                name=media_info["name"],
                media_info=media_info,
                start_frames=start_frames,
                end_frames=end_frames,
                in_frames=0,
                out_frames=out_frames,
                mediatype="video",
                expanded_ids=expanded_ids,
            )

            overlay_summary.append(
                {
                    "timestamp": item["timestamp"],
                    "local_path": str(item["local_path"]),
                    "provider": item["provider"],
                    "query": item["query"],
                    "track_index": track_index,
                    "timeline_start_seconds": round(timeline_start_seconds, 3),
                    "duration_seconds": round(duration_seconds, 3),
                }
            )

    output_path.write_bytes(pretty_xml_bytes(xmeml))
    return {
        "sequence_duration_seconds": round(sequence_duration_seconds, 3),
        "sequence_duration_frames": sequence_duration_frames,
        "main_segments": main_segments_summary,
        "broll_track_count": max(1, len(packed_broll_tracks)),
        "broll_overlays": overlay_summary,
    }


def save_summary(output_dir: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    json_path = output_dir / "premiere_xml_report.json"
    txt_path = output_dir / "Premiere_Pro_XML_Entegrasyonu.txt"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "=== PREMIERE PRO XML ENTEGRASYON RAPORU ===",
        "",
        f"Kaynak video: {payload.get('video_path', '')}",
        f"Trim dosyasi: {payload.get('trim_path', '') or 'Kullanilmadi'}",
        f"B-Roll raporu: {payload.get('broll_report_path', '') or 'Kullanilmadi'}",
        f"Olusan XML: {payload.get('xml_path', '')}",
        f"Ana timeline segment sayisi: {payload.get('main_segment_count', 0)}",
        f"B-Roll overlay sayisi: {payload.get('broll_overlay_count', 0)}",
        f"B-Roll track sayisi: {payload.get('broll_track_count', 0)}",
        "",
        "AKTARIM ADIMLARI",
        "1. Premiere Pro ac.",
        "2. File > Import ile XML dosyasini ice aktar.",
        "3. Project panelinde olusan sequence'i ac.",
        "4. Muzik seviyesi, renk ve ince kesim ayarlarini yap.",
        "",
    ]

    for item in payload.get("broll_overlays", []):
        lines.append(
            f"- Track {item.get('track_index', 1)} | {item.get('timeline_start_seconds', 0)} sn | {item.get('provider', '')} | {item.get('query', '')} | {item.get('local_path', '')}"
        )

    txt_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return json_path, txt_path


def run_automatic(
    video_path: Path,
    *,
    trim_path: Path | None = None,
    broll_report_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    if not video_path.exists():
        raise RuntimeError(f"Kaynak video bulunamadi: {video_path}")

    destination = ensure_directory(output_dir or PREMIERE_OUTPUT_DIR / video_path.stem)
    media_cache = load_media_cache()
    video_info = probe_video_media(video_path, media_cache=media_cache)

    resolved_trim_path = trim_path or match_trim_file(video_path, find_trim_files())
    resolved_broll_path = broll_report_path or match_broll_report(video_path, find_broll_report_files())

    cut_ranges, trim_payload = extract_cut_ranges(resolved_trim_path)
    cut_timeline_map = build_cut_timeline_map(cut_ranges)
    keep_ranges = build_keep_ranges(float(video_info["duration_seconds"]), cut_ranges)
    if not keep_ranges:
        keep_ranges = [(0.0, float(video_info["duration_seconds"]))]

    sequence_duration = sum(end - start for start, end in keep_ranges)
    broll_items, broll_payload = extract_broll_items(
        resolved_broll_path,
        cut_ranges,
        sequence_duration,
        cut_timeline_map=cut_timeline_map,
        media_cache=media_cache,
        sequence_fps=float(video_info["fps"]),
    )

    xml_path = destination / f"{video_path.stem}_premiere_rough_cut.xml"
    xml_summary = build_sequence_xml(video_path, video_info, keep_ranges, broll_items, xml_path)
    save_media_cache(media_cache)

    payload = {
        "title": "Premiere Pro XML Integration",
        "video_path": str(video_path),
        "trim_path": str(resolved_trim_path) if resolved_trim_path else "",
        "broll_report_path": str(resolved_broll_path) if resolved_broll_path else "",
        "xml_path": str(xml_path),
        "main_segment_count": len(xml_summary["main_segments"]),
        "broll_overlay_count": len(xml_summary["broll_overlays"]),
        "broll_track_count": int(xml_summary.get("broll_track_count", 1) or 1),
        "cut_ranges": [{"start_seconds": round(start, 3), "end_seconds": round(end, 3)} for start, end in cut_ranges],
        "main_segments": xml_summary["main_segments"],
        "broll_overlays": xml_summary["broll_overlays"],
        "trim_summary": trim_payload.get("summary", "") if isinstance(trim_payload, dict) else "",
        "estimated_retention_gain": trim_payload.get("estimated_retention_gain", "") if isinstance(trim_payload, dict) else "",
        "broll_downloaded_count": int(broll_payload.get("downloaded_count", 0)) if isinstance(broll_payload, dict) else 0,
    }
    json_path, txt_path = save_summary(destination, payload)
    payload["json_path"] = str(json_path)
    payload["txt_path"] = str(txt_path)
    return payload


def run() -> None:
    print("\n" + "=" * 60)
    print("PREMIERE PRO XML ENTEGRASYONU")
    print("=" * 60)

    video_files = find_video_files()
    if not video_files:
        logger.error("00_Inputs klasorunde islenecek video bulunamadi.")
        print("00_Inputs klasorunde islenecek video bulunamadi.")
        return

    selected_video = select_file(video_files, "Mevcut video dosyalari:")
    if not selected_video:
        logger.error("Video secilemedi.")
        return

    trim_path = match_trim_file(selected_video, find_trim_files())
    if not trim_path:
        trim_path = select_file(find_trim_files(), "Trim dosyasi secin (0 = trimsiz rough cut):")

    broll_report_path = match_broll_report(selected_video, find_broll_report_files())
    if not broll_report_path:
        broll_report_path = select_file(find_broll_report_files(), "B-Roll raporu secin (0 = B-Roll ekleme):")

    try:
        result = run_automatic(selected_video, trim_path=trim_path, broll_report_path=broll_report_path)
    except Exception as exc:
        logger.error(f"Premiere XML uretimi basarisiz oldu: {exc}")
        print(f"Premiere XML uretimi basarisiz oldu: {exc}")
        return

    print(f"\nXML:        {result['xml_path']}")
    print(f"JSON rapor: {result['json_path']}")
    print(f"TXT rapor:  {result['txt_path']}")
    print("=" * 60)
