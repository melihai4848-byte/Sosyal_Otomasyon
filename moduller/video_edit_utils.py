import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional


def timestamp_to_seconds(value: str) -> float:
    text = str(value or "").strip().replace(",", ".")
    if not text:
        return 0.0

    parts = text.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    return float(text)


def seconds_to_timestamp(seconds: float) -> str:
    total = max(0.0, float(seconds))
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")


def duration_between(start: str, end: str) -> float:
    return max(0.0, timestamp_to_seconds(end) - timestamp_to_seconds(start))


def total_duration_from_segments(segments: Iterable[dict]) -> float:
    return sum(duration_between(item.get("start", ""), item.get("end", "")) for item in segments or [])


def ffmpeg_binary() -> Optional[str]:
    return shutil.which("ffmpeg")


def ffprobe_binary() -> Optional[str]:
    return shutil.which("ffprobe")


def ffmpeg_available() -> bool:
    return bool(ffmpeg_binary() and ffprobe_binary())


def _env_int(name: str, default: int, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    raw = os.getenv(name, "").strip()
    try:
        value = int(raw) if raw else int(default)
    except Exception:
        value = int(default)
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def _env_float(name: str, default: float, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
    raw = os.getenv(name, "").strip().replace(",", ".")
    try:
        value = float(raw) if raw else float(default)
    except Exception:
        value = float(default)
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "evet", "on"}


def _ass_color_from_env(name: str, default_hex: str) -> str:
    raw = os.getenv(name, default_hex).strip() or default_hex
    value = raw.lstrip("#")
    if len(value) != 6:
        value = default_hex.lstrip("#")
    rr = value[0:2]
    gg = value[2:4]
    bb = value[4:6]
    return f"&H00{bb}{gg}{rr}"


def build_subtitle_force_style() -> str:
    font_name = os.getenv("REELS_BURN_SUBTITLE_FONT", "Arial").strip() or "Arial"
    font_size = _env_float("REELS_BURN_SUBTITLE_SIZE", 24, min_value=8, max_value=120)
    primary_color = _ass_color_from_env("REELS_BURN_SUBTITLE_PRIMARY_COLOR", "#FFFFFF")
    outline_color = _ass_color_from_env("REELS_BURN_SUBTITLE_OUTLINE_COLOR", "#000000")
    bold = -1 if _env_bool("REELS_BURN_SUBTITLE_BOLD", True) else 0
    alignment = _env_int("REELS_BURN_SUBTITLE_ALIGNMENT", 2, min_value=1, max_value=9)
    margin_v = _env_int("REELS_BURN_SUBTITLE_MARGIN_V", 70, min_value=0, max_value=500)
    outline = _env_float("REELS_BURN_SUBTITLE_OUTLINE", 2.0, min_value=0.0, max_value=20.0)
    shadow = _env_float("REELS_BURN_SUBTITLE_SHADOW", 0.0, min_value=0.0, max_value=20.0)
    safe_font_name = font_name.replace(",", " ").replace("'", "\\'")

    style_parts = [
        f"FontName={safe_font_name}",
        f"FontSize={font_size:g}",
        f"PrimaryColour={primary_color}",
        f"OutlineColour={outline_color}",
        f"Bold={bold}",
        "BorderStyle=1",
        f"Outline={outline:g}",
        f"Shadow={shadow:g}",
        f"Alignment={alignment}",
        f"MarginV={margin_v}",
    ]
    return ",".join(style_parts)


def _video_encode_settings() -> dict:
    return {
        "video_crf": _env_int("REELS_RENDER_CRF", 17, min_value=0, max_value=35),
        "video_preset": os.getenv("REELS_RENDER_PRESET", "veryfast").strip() or "veryfast",
        "audio_codec": os.getenv("REELS_RENDER_AUDIO_CODEC", "aac").strip() or "aac",
        "audio_bitrate": os.getenv("REELS_RENDER_AUDIO_BITRATE", "192k").strip() or "192k",
    }


def probe_video(path: Path) -> dict:
    ffprobe = ffprobe_binary()
    if not ffprobe:
        raise RuntimeError("ffprobe bulunamadi. FFmpeg kurulu olmali ve PATH'e eklenmeli.")

    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,width,height:format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    payload = json.loads(completed.stdout or "{}")
    streams = payload.get("streams", [])
    video_stream = next((item for item in streams if item.get("codec_type") == "video"), {})
    has_audio = any(item.get("codec_type") == "audio" for item in streams)
    try:
        duration_seconds = float((payload.get("format") or {}).get("duration") or 0)
    except Exception:
        duration_seconds = 0.0
    return {
        "width": int(video_stream.get("width", 0) or 0),
        "height": int(video_stream.get("height", 0) or 0),
        "duration_seconds": duration_seconds,
        "has_audio": has_audio,
    }


def is_vertical_9_16(path: Path, tolerance: float = 0.03) -> bool:
    meta = probe_video(path)
    width = int(meta.get("width", 0) or 0)
    height = int(meta.get("height", 0) or 0)
    if not width or not height:
        return False
    ratio = width / height
    return abs(ratio - (9 / 16)) <= tolerance


def _is_vertical_from_meta(meta: Optional[dict], tolerance: float = 0.03) -> bool:
    if not isinstance(meta, dict):
        return False
    width = int(meta.get("width", 0) or 0)
    height = int(meta.get("height", 0) or 0)
    if not width or not height:
        return False
    ratio = width / height
    return abs(ratio - (9 / 16)) <= tolerance


def build_vertical_filter(width: int, height: int, target_width: int = 1080, target_height: int = 1920) -> str:
    if not width or not height:
        return f"scale={target_width}:{target_height}"

    src_ratio = width / height
    dst_ratio = target_width / target_height
    if abs(src_ratio - dst_ratio) <= 0.03:
        return f"scale={target_width}:{target_height}"

    if src_ratio > dst_ratio:
        return (
            f"scale=-2:{target_height},"
            f"crop={target_width}:{target_height}"
        )
    return (
        f"scale={target_width}:-2,"
        f"crop={target_width}:{target_height}"
    )


def escape_subtitle_filter_path(path: Path) -> str:
    resolved = str(path.resolve()).replace("\\", "/")
    escaped = resolved.replace(":", r"\:")
    escaped = escaped.replace("'", r"\'")
    escaped = escaped.replace(",", r"\,")
    escaped = escaped.replace("[", r"\[")
    escaped = escaped.replace("]", r"\]")
    return escaped


def render_vertical_master(
    input_video: Path,
    output_video: Path,
    input_video_meta: Optional[dict] = None,
) -> None:
    ffmpeg = ffmpeg_binary()
    if not ffmpeg:
        raise RuntimeError("ffmpeg bulunamadi. FFmpeg kurulu olmali ve PATH'e eklenmeli.")

    probe = input_video_meta or probe_video(input_video)
    width = int(probe.get("width", 0) or 0)
    height = int(probe.get("height", 0) or 0)
    has_audio = bool(probe.get("has_audio"))
    settings = _video_encode_settings()

    output_video.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(input_video),
        "-vf",
        build_vertical_filter(width, height),
        "-c:v",
        "libx264",
        "-preset",
        settings["video_preset"],
        "-crf",
        str(settings["video_crf"]),
        "-pix_fmt",
        "yuv420p",
    ]
    if has_audio:
        command.extend(
            [
                "-c:a",
                settings["audio_codec"],
                "-b:a",
                settings["audio_bitrate"],
            ]
        )
    command.extend(["-movflags", "+faststart", str(output_video)])
    subprocess.run(command, check=True)


def _render_single_segment(
    input_video: Path,
    output_video: Path,
    segment: dict,
    *,
    force_vertical: bool = True,
    subtitle_path: Optional[Path] = None,
    input_video_meta: Optional[dict] = None,
) -> None:
    ffmpeg = ffmpeg_binary()
    if not ffmpeg:
        raise RuntimeError("ffmpeg bulunamadi. FFmpeg kurulu olmali ve PATH'e eklenmeli.")

    probe = input_video_meta or probe_video(input_video)
    width = int(probe.get("width", 0) or 0)
    height = int(probe.get("height", 0) or 0)
    has_audio = bool(probe.get("has_audio"))
    source_is_vertical = _is_vertical_from_meta(probe)
    settings = _video_encode_settings()

    start = timestamp_to_seconds(segment.get("start", ""))
    end = timestamp_to_seconds(segment.get("end", ""))
    duration = max(0.01, end - start)

    vf_parts: list[str] = []
    if force_vertical and not source_is_vertical:
        vf_parts.append(build_vertical_filter(width, height))
    if subtitle_path:
        escaped_subtitle_path = escape_subtitle_filter_path(subtitle_path)
        force_style = build_subtitle_force_style()
        vf_parts.append(
            f"subtitles='{escaped_subtitle_path}':charenc=UTF-8:force_style='{force_style}'"
        )

    output_video.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(input_video),
    ]
    if vf_parts:
        command.extend(["-vf", ",".join(vf_parts)])
    command.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            settings["video_preset"],
            "-crf",
            str(settings["video_crf"]),
            "-pix_fmt",
            "yuv420p",
        ]
    )
    if has_audio:
        command.extend(
            [
                "-c:a",
                settings["audio_codec"],
                "-b:a",
                settings["audio_bitrate"],
            ]
        )
    command.extend(["-movflags", "+faststart", str(output_video)])
    subprocess.run(command, check=True)


def render_concat_segments(
    input_video: Path,
    output_video: Path,
    segments: list[dict],
    force_vertical: bool = True,
    subtitle_path: Optional[Path] = None,
    input_video_meta: Optional[dict] = None,
) -> None:
    ffmpeg = ffmpeg_binary()
    if not ffmpeg:
        raise RuntimeError("ffmpeg bulunamadi. FFmpeg kurulu olmali ve PATH'e eklenmeli.")

    if not segments:
        raise ValueError("Kesilecek segment listesi bos.")

    if len(segments) == 1:
        _render_single_segment(
            input_video=input_video,
            output_video=output_video,
            segment=segments[0],
            force_vertical=force_vertical,
            subtitle_path=subtitle_path,
            input_video_meta=input_video_meta,
        )
        return

    probe = input_video_meta or probe_video(input_video)
    width = int(probe.get("width", 0) or 0)
    height = int(probe.get("height", 0) or 0)
    has_audio = bool(probe.get("has_audio"))
    source_is_vertical = _is_vertical_from_meta(probe)

    parts = []
    filter_parts = []
    for index, item in enumerate(segments):
        start = timestamp_to_seconds(item.get("start", ""))
        end = timestamp_to_seconds(item.get("end", ""))
        duration = max(0.01, end - start)
        parts.extend(["-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(input_video)])

        video_label = f"v{index}"
        audio_label = f"a{index}"
        if force_vertical and not source_is_vertical:
            filter_parts.append(
                f"[{index}:v]setpts=PTS-STARTPTS,{build_vertical_filter(width, height)}[{video_label}]"
            )
        else:
            filter_parts.append(f"[{index}:v]setpts=PTS-STARTPTS[{video_label}]")
        if has_audio:
            filter_parts.append(f"[{index}:a]asetpts=PTS-STARTPTS[{audio_label}]")

    video_output_label = "concatv" if subtitle_path else "outv"

    if has_audio:
        concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(len(segments)))
        filter_parts.append(f"{concat_inputs}concat=n={len(segments)}:v=1:a=1[{video_output_label}][outa]")
    else:
        concat_inputs = "".join(f"[v{i}]" for i in range(len(segments)))
        filter_parts.append(f"{concat_inputs}concat=n={len(segments)}:v=1:a=0[{video_output_label}]")

    if subtitle_path:
        escaped_subtitle_path = escape_subtitle_filter_path(subtitle_path)
        force_style = build_subtitle_force_style()
        filter_parts.append(
            f"[{video_output_label}]subtitles='{escaped_subtitle_path}':charenc=UTF-8:force_style='{force_style}'[outv]"
        )

    settings = _video_encode_settings()
    output_video.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg,
        "-y",
        *parts,
        "-filter_complex",
        ";".join(filter_parts),
        "-map",
        "[outv]",
        "-c:v",
        "libx264",
        "-preset",
        settings["video_preset"],
        "-crf",
        str(settings["video_crf"]),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    if has_audio:
        command[command.index("-c:v") : command.index("-movflags")] = [
            "-map",
            "[outa]",
            "-c:v",
            "libx264",
            "-preset",
            settings["video_preset"],
            "-crf",
            str(settings["video_crf"]),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            settings["audio_codec"],
            "-b:a",
            settings["audio_bitrate"],
        ]
    subprocess.run(command, check=True)
