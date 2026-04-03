import json
import re
import tempfile
from pathlib import Path
from typing import Optional

from moduller._module_alias import load_numbered_module
from moduller.config import INPUTS_DIR
from moduller.logger import get_logger
from moduller.output_paths import grouped_output_path, json_output_path, txt_output_path
from moduller.srt_utils import parse_srt_blocks, read_srt_file
from moduller.subtitle_output_utils import find_subtitle_file, list_subtitle_files, subtitle_output_path
from moduller.transcriber import WhisperMotor, resolve_shorts_word_limit
from moduller.video_edit_utils import (
    ffmpeg_available,
    probe_video,
    render_vertical_master,
    render_concat_segments,
    seconds_to_timestamp,
    timestamp_to_seconds,
)

_REELS_MODULE = load_numbered_module("302_reel_olusturucu.py")
load_latest_reels_data = _REELS_MODULE.load_latest_reels_data

logger = get_logger("reels_render")

REELS_RENDER_DIR = grouped_output_path("tools", "reels_render")
DEFAULT_VERTICAL_VIDEO_NAME = "Vertical_9_16_reframed_video.mp4"
DEFAULT_REEL_SUBTITLE_NAME = "subtitle_shorts.srt"


def _notify(message: str) -> None:
    print(message)
    logger.info(message)


def _safe_name(value: str) -> str:
    temiz = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip())
    return temiz.strip("_") or "klip"


def _video_list() -> list[Path]:
    return sorted(
        list(INPUTS_DIR.glob("*.mp4"))
        + list(INPUTS_DIR.glob("*.mov"))
        + list(INPUTS_DIR.glob("*.mkv"))
        + list(INPUTS_DIR.glob("*.m4v"))
    )


def prompt_burn_subtitles() -> bool:
    secim = input("👉 Dinamik altyazi videoya gomulsun mu? (e/h, bos = h): ").strip().lower()
    return secim in {"e", "evet", "y", "yes", "1"}


def _select_source_video() -> Optional[Path]:
    default_video = INPUTS_DIR / DEFAULT_VERTICAL_VIDEO_NAME
    if default_video.exists():
        _notify(f"✅ 9:16 kaynak video bulundu: {default_video.name}")
        return default_video

    video_files = _video_list()
    if not video_files:
        logger.error("❌ workspace/00_Inputs klasorunde islenecek video bulunamadi.")
        return None

    print("\n📂 workspace/00_Inputs icindeki videolar:")
    for idx, video in enumerate(video_files, start=1):
        print(f"  [{idx}] {video.name}")

    secim = input("👉 Kullanilacak video numarasi: ").strip()
    try:
        secilen = video_files[int(secim) - 1]
    except Exception:
        logger.error("Gecersiz video secimi.")
        return None

    _notify(f"✅ Kaynak video secildi: {secilen.name}")
    return secilen


def _select_reel_subtitle() -> Optional[Path]:
    default_subtitle = find_subtitle_file(DEFAULT_REEL_SUBTITLE_NAME)
    if default_subtitle and default_subtitle.exists():
        _notify(f"✅ Reel altyazisi bulundu: {default_subtitle.name}")
        return default_subtitle

    srt_files = list_subtitle_files()
    if not srt_files:
        logger.error("❌ 100_Altyazı klasorunde kullanilabilir bir altyazi bulunamadi.")
        return None

    print("\n📂 100_Altyazı icindeki altyazilar:")
    for idx, srt in enumerate(srt_files, start=1):
        print(f"  [{idx}] {srt.name}")

    secim = input("👉 Kullanilacak altyazi numarasi: ").strip()
    try:
        secilen = srt_files[int(secim) - 1]
    except Exception:
        logger.error("Gecersiz altyazi secimi.")
        return None

    _notify(f"✅ Reel altyazisi secildi: {secilen.name}")
    return secilen


def _load_reel_plan() -> tuple[Optional[Path], Optional[dict]]:
    ideas_path = json_output_path("reels_ideas")
    if not ideas_path.exists():
        logger.error("❌ Reel segment plani bulunamadi. Once 302 numarali Reels Fikir Uretici modulu calismali.")
        return None, None

    payload = load_latest_reels_data()
    ideas = []
    if isinstance(payload, dict):
        ideas = payload.get("ideas") or payload.get("reel_candidates") or []
    if not payload or not ideas:
        logger.error("❌ Reel segment plani bulundu ama kullanilabilir fikir icermiyor.")
        return None, None

    _notify(f"✅ Reel segment plani bulundu: {ideas_path.name}")
    _notify(f"✅ Toplam {len(ideas)} reel plani yüklendi.")
    return ideas_path, payload


def _parse_srt_timing_line(timing_line: str) -> tuple[float, float]:
    if "-->" not in str(timing_line or ""):
        return 0.0, 0.0
    start_raw, end_raw = [item.strip() for item in str(timing_line).split("-->", 1)]
    return timestamp_to_seconds(start_raw), timestamp_to_seconds(end_raw)


def _resolve_dynamic_subtitle_path(video_path: Path, auto_generate: bool = False) -> Optional[Path]:
    candidates = [
        find_subtitle_file("subtitle_shorts.srt"),
        find_subtitle_file("subtitle_tr.srt"),
        subtitle_output_path(f"{video_path.stem}_shorts.srt"),
        subtitle_output_path(f"{video_path.stem}_reels_shorts.srt"),
        subtitle_output_path(f"{video_path.stem}_standart_tr.srt"),
    ]

    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate

    if not auto_generate:
        return None

    generated_path = subtitle_output_path("subtitle_shorts.srt")
    logger.info("Dinamik altyazi bulunamadi. Gomme icin otomatik olarak yeni SRT uretiliyor...")
    WhisperMotor.generate_dynamic_transcript(
        str(video_path),
        str(generated_path),
        kelime_siniri=resolve_shorts_word_limit(),
    )
    return generated_path if generated_path.exists() else None


def _build_clip_subtitle_file(blocks: list, idea: dict, output_srt: Path) -> Optional[Path]:
    subtitle_entries: list[str] = []
    clip_offset = 0.0
    subtitle_index = 1

    for segment in idea.get("segments", []):
        segment_start = timestamp_to_seconds(segment.get("start", ""))
        segment_end = timestamp_to_seconds(segment.get("end", ""))
        if segment_end <= segment_start:
            continue

        for block in blocks:
            if not block.is_processable or not block.timing_line or not block.text_content:
                continue
            block_start, block_end = _parse_srt_timing_line(block.timing_line)
            if block_end <= segment_start or block_start >= segment_end:
                continue

            clipped_start = max(block_start, segment_start)
            clipped_end = min(block_end, segment_end)
            if clipped_end <= clipped_start:
                continue

            local_start = clip_offset + (clipped_start - segment_start)
            local_end = clip_offset + (clipped_end - segment_start)
            subtitle_entries.append(
                "\n".join(
                    [
                        str(subtitle_index),
                        f"{seconds_to_timestamp(local_start)} --> {seconds_to_timestamp(local_end)}",
                        block.text_content,
                    ]
                )
            )
            subtitle_index += 1

        clip_offset += segment_end - segment_start

    if not subtitle_entries:
        clip_offset = 0.0
        for segment in idea.get("segments", []):
            segment_start = timestamp_to_seconds(segment.get("start", ""))
            segment_end = timestamp_to_seconds(segment.get("end", ""))
            if segment_end <= segment_start:
                continue
            text = str(segment.get("text") or segment.get("purpose_tr") or "").strip()
            if not text:
                clip_offset += segment_end - segment_start
                continue
            subtitle_entries.append(
                "\n".join(
                    [
                        str(subtitle_index),
                        f"{seconds_to_timestamp(clip_offset)} --> {seconds_to_timestamp(clip_offset + (segment_end - segment_start))}",
                        text,
                    ]
                )
            )
            subtitle_index += 1
            clip_offset += segment_end - segment_start

    if not subtitle_entries:
        return None

    output_srt.parent.mkdir(parents=True, exist_ok=True)
    output_srt.write_text("\n\n".join(subtitle_entries).strip() + "\n", encoding="utf-8")
    return output_srt


def _is_vertical_from_meta(video_meta: dict) -> bool:
    width = int(video_meta.get("width", 0) or 0)
    height = int(video_meta.get("height", 0) or 0)
    if not width or not height:
        return False
    return abs((width / height) - (9 / 16)) <= 0.03


def _build_vertical_master_meta(video_meta: dict) -> dict:
    return {
        "width": 1080,
        "height": 1920,
        "duration_seconds": float(video_meta.get("duration_seconds") or 0.0),
        "has_audio": bool(video_meta.get("has_audio")),
    }


def _ensure_vertical_master(video_path: Path, output_root: Path, video_meta: dict) -> tuple[Path, dict]:
    master_path = output_root / f"{_safe_name(video_path.stem)}__vertical_master.mp4"
    if master_path.exists():
        try:
            if master_path.stat().st_size > 0 and master_path.stat().st_mtime >= video_path.stat().st_mtime:
                _notify(f"✅ Ortak vertical master yeniden kullaniliyor: {master_path.name}")
                return master_path, _build_vertical_master_meta(video_meta)
        except Exception:
            pass

    _notify("🎞️ Kaynak video once tek seferlik 9:16 master videoya donusturuluyor...")
    render_vertical_master(video_path, master_path, input_video_meta=video_meta)
    _notify(f"✅ Ortak vertical master hazir: {master_path.name}")
    return master_path, _build_vertical_master_meta(video_meta)


def _normalize_segments_for_video(idea: dict, video_duration_seconds: float) -> Optional[dict]:
    validated_segments = []
    previous_end = 0.0
    adjustments = []

    for segment in idea.get("segments", []):
        start_seconds = max(0.0, timestamp_to_seconds(segment.get("start", "")))
        end_seconds = min(float(video_duration_seconds or 0.0), timestamp_to_seconds(segment.get("end", "")))

        if start_seconds < previous_end:
            start_seconds = previous_end
            adjustments.append("Segmentler arasindaki cakisma otomatik duzeltildi.")

        if end_seconds <= start_seconds:
            continue

        updated = dict(segment)
        updated["start"] = seconds_to_timestamp(start_seconds)
        updated["end"] = seconds_to_timestamp(end_seconds)
        updated["duration_seconds"] = round(end_seconds - start_seconds, 2)
        validated_segments.append(updated)
        previous_end = end_seconds

    if not validated_segments:
        return None

    normalized_idea = dict(idea)
    normalized_idea["segments"] = validated_segments
    if adjustments:
        editing_notes = list(normalized_idea.get("editing_notes_tr", []) or [])
        for item in adjustments:
            if item not in editing_notes:
                editing_notes.insert(0, item)
        normalized_idea["editing_notes_tr"] = editing_notes
    return normalized_idea


def save_report(
    video_path: Path,
    subtitle_path: Path,
    ideas_path: Path,
    ideas: list[dict],
    outputs: list[Path],
    *,
    burn_subtitles: bool = False,
    subtitle_outputs: Optional[list[Path]] = None,
    output_root: Optional[Path] = None,
) -> tuple[Path, Path]:
    json_yolu = json_output_path("reels_creator_report")
    txt_yolu = txt_output_path("reels_creator_report")
    subtitle_outputs = subtitle_outputs or []

    payload = {
        "source_video": str(video_path),
        "source_subtitle": str(subtitle_path),
        "ideas_json": str(ideas_path),
        "output_root": str(output_root) if output_root else "",
        "generated_videos": [str(item) for item in outputs],
        "ideas_used": ideas,
        "burn_subtitles": burn_subtitles,
        "generated_subtitle_files": [str(item) for item in subtitle_outputs],
    }

    with open(json_yolu, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = [
        "=== REEL / SHORTS OLUSTURUCU RAPORU ===",
        f"Kaynak Video: {video_path}",
        f"Kaynak Altyazi: {subtitle_path}",
        f"Reel Plani JSON: {ideas_path}",
        f"Cikti Klasoru: {output_root if output_root else REELS_RENDER_DIR}",
        f"Altyazi Gomulu: {'Evet' if burn_subtitles else 'Hayir'}",
        "",
        "OLUSTURULAN DOSYALAR",
        "-" * 60,
    ]
    for item in outputs:
        lines.append(f"- {item}")

    if subtitle_outputs:
        lines.extend(["", "URETILEN KLIP ALTYAZILARI", "-" * 60])
        for item in subtitle_outputs:
            lines.append(f"- {item}")

    lines.extend(["", "KULLANILAN FIKIRLER", "-" * 60])
    for idea in ideas:
        lines.append(f"#{idea.get('rank')} | {idea.get('concept', '')}")
        for segment in idea.get("segments", []):
            lines.append(f"  - {segment.get('start')} --> {segment.get('end')}")

    txt_yolu.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return json_yolu, txt_yolu


def create(video_path: Path, subtitle_path: Path, ideas_path: Path, ideas: list[dict], burn_subtitles: bool = False) -> Optional[dict]:
    if not ffmpeg_available():
        logger.error("FFmpeg/ffprobe bulunamadi. Reel/Shorts olusturmak icin FFmpeg kurup PATH'e eklemelisin.")
        return None

    video_meta = probe_video(video_path)
    video_duration_seconds = float(video_meta.get("duration_seconds") or 0.0)
    if video_duration_seconds <= 0:
        logger.error("❌ Kaynak video suresi okunamadi.")
        return None

    outputs = []
    subtitle_outputs: list[Path] = []
    used_ideas: list[dict] = []
    output_root = REELS_RENDER_DIR / _safe_name(video_path.stem)
    output_root.mkdir(parents=True, exist_ok=True)
    video_dikey = _is_vertical_from_meta(video_meta)
    source_subtitle_path = subtitle_path
    source_subtitle_blocks = parse_srt_blocks(read_srt_file(source_subtitle_path))
    render_source_video = video_path
    render_source_meta = video_meta

    if video_dikey:
        _notify("✅ Kaynak video 9:16 formatinda. Ek reframing uygulanmayacak.")
    else:
        _notify("⚠️ Secilen video 9:16 degil. Ortak vertical master kullanilarak hizlandirilacak.")
        render_source_video, render_source_meta = _ensure_vertical_master(video_path, output_root, video_meta)
        video_dikey = True

    for idea in ideas:
        reel_no = int(idea.get("rank", 0) or 0)
        concept = idea.get("concept", f"Reel {reel_no}")
        _notify(f"🎬 Reel {reel_no} isleniyor: {concept}")

        validated_idea = _normalize_segments_for_video(idea, video_duration_seconds)
        if not validated_idea:
            logger.warning(f"Reel {reel_no} atlandi; video suresi icinde gecerli segment kalmadi.")
            continue

        output_video = output_root / f"Reel_{reel_no:02d}.mp4"

        if burn_subtitles:
            with tempfile.TemporaryDirectory(prefix=f"reel_{reel_no:02d}_subtitle_") as temp_dir:
                temp_srt = Path(temp_dir) / f"Reel_{reel_no:02d}.srt"
                clip_subtitle_path = _build_clip_subtitle_file(source_subtitle_blocks, validated_idea, temp_srt)
                if not clip_subtitle_path:
                    logger.warning(f"Reel {reel_no} atlandi; altyazi kesilemedi.")
                    continue
                _notify(f"📝 Reel {reel_no} altyazisi kesildi.")
                render_concat_segments(
                    input_video=render_source_video,
                    output_video=output_video,
                    segments=validated_idea.get("segments", []),
                    force_vertical=not video_dikey,
                    subtitle_path=clip_subtitle_path,
                    input_video_meta=render_source_meta,
                )
                _notify(f"🔥 Reel {reel_no} altyazi gomuldu.")
        else:
            output_srt = output_root / f"Reel_{reel_no:02d}.srt"
            clip_subtitle_path = _build_clip_subtitle_file(source_subtitle_blocks, validated_idea, output_srt)
            if not clip_subtitle_path:
                logger.warning(f"Reel {reel_no} atlandi; altyazi kesilemedi.")
                continue
            subtitle_outputs.append(clip_subtitle_path)
            _notify(f"📝 Reel {reel_no} altyazisi hazir: {clip_subtitle_path.name}")
            render_concat_segments(
                input_video=render_source_video,
                output_video=output_video,
                segments=validated_idea.get("segments", []),
                force_vertical=not video_dikey,
                subtitle_path=None,
                input_video_meta=render_source_meta,
            )

        outputs.append(output_video)
        used_ideas.append(validated_idea)
        _notify(f"✅ Reel {reel_no} hazir: {output_video.name}")

    if not outputs:
        logger.error("❌ Hicbir Reel/Shorts videosu uretilemedi.")
        return None

    json_yolu, txt_yolu = save_report(
        video_path,
        source_subtitle_path,
        ideas_path,
        used_ideas,
        outputs,
        burn_subtitles=burn_subtitles,
        subtitle_outputs=subtitle_outputs,
        output_root=output_root,
    )
    return {
        "data": {
            "source_video": str(video_path),
            "render_source_video": str(render_source_video),
            "source_subtitle": str(source_subtitle_path),
            "ideas_json": str(ideas_path),
            "generated_videos": [str(item) for item in outputs],
            "burn_subtitles": burn_subtitles,
            "output_root": str(output_root),
        },
        "json_path": json_yolu,
        "txt_path": txt_yolu,
        "video_paths": outputs,
        "subtitle_paths": subtitle_outputs,
    }


def run():
    print("\n" + "=" * 60)
    print("REEL / SHORTS OLUSTURUCU")
    print("=" * 60)

    video_path = _select_source_video()
    if not video_path:
        return

    subtitle_path = _select_reel_subtitle()
    if not subtitle_path:
        return

    ideas_path, ideas_payload = _load_reel_plan()
    if not ideas_path or not ideas_payload:
        return

    ideas = ideas_payload.get("ideas") or ideas_payload.get("reel_candidates") or []
    if not ideas:
        return logger.error("❌ Reel plani bulundu ama islenecek fikir yok.")

    burn_subtitles = prompt_burn_subtitles()

    sonuc = create(video_path, subtitle_path, ideas_path, ideas, burn_subtitles=burn_subtitles)
    if not sonuc:
        return logger.error("❌ Reel/Shorts videolari olusturulamadi.")

    print("\n🎉 Reel/Shorts olusturma tamamlandi.")
    for item in sonuc.get("video_paths", []):
        print(f"- {item}")


def run_automatic(video_path: Path, ideas: Optional[list] = None, burn_subtitles: bool = False) -> Optional[dict]:
    ideas_payload = load_latest_reels_data() if ideas is None else {"ideas": ideas}
    ideas_list = []
    if isinstance(ideas_payload, dict):
        ideas_list = ideas_payload.get("ideas") or ideas_payload.get("reel_candidates") or []
    if not ideas_payload or not ideas_list:
        logger.warning("Reel/Shorts Olusturucu atlandi; kullanilabilir fikir bulunamadi.")
        return None
    subtitle_path = find_subtitle_file(DEFAULT_REEL_SUBTITLE_NAME) or _resolve_dynamic_subtitle_path(video_path, auto_generate=True)
    if not subtitle_path:
        logger.warning("Reel/Shorts Olusturucu atlandi; kullanilabilir altyazi bulunamadi.")
        return None
    ideas_path = json_output_path("reels_ideas")
    return create(video_path, subtitle_path, ideas_path, ideas_list, burn_subtitles=burn_subtitles)

