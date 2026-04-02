import json
import math
import re
from pathlib import Path
from typing import Optional, Tuple

from moduller._module_alias import load_numbered_module
from moduller.llm_manager import (
    CentralLLM,
    get_module_recommended_llm_config,
    print_module_llm_choice_summary,
    prompt_module_llm_plan,
    select_llm,
)
from moduller.logger import get_logger
from moduller.output_paths import stem_json_output_path, txt_output_path
from moduller.social_media_utils import (
    build_broll_summary,
    build_critic_summary,
    build_metadata_summary,
    load_related_json,
    select_metadata_language,
    select_primary_srt,
)
from moduller.srt_utils import parse_srt_blocks, read_srt_file
from moduller.youtube_llm_profiles import call_with_youtube_profile

_VIDEO_CRITIC_MODULE = load_numbered_module("202_video_critic.py")
extract_json_response = _VIDEO_CRITIC_MODULE.extract_json_response
normalize_whitespace = _VIDEO_CRITIC_MODULE.normalize_whitespace

logger = get_logger("music")

MAX_TRANSCRIPT_CHARS = 26000
ADAPTIVE_TRANSCRIPT_MAX_CHARS = 14000
ADAPTIVE_TRANSCRIPT_TARGET_ANCHORS = 8
ADAPTIVE_TRANSCRIPT_WINDOW_RADIUS = 1
MAX_SEGMENTS = 8
MIN_SEGMENT_SECONDS = 20
LOCAL_ONLY_MAX_SECONDS = 150
LOCAL_ONLY_MAX_BLOCKS = 16
LOCAL_ONLY_MAX_TRANSCRIPT_CHARS = 1800

STOP_WORDS = {
    "ve",
    "ile",
    "ama",
    "fakat",
    "bir",
    "bu",
    "şu",
    "icin",
    "için",
    "gibi",
    "çok",
    "daha",
    "sonra",
    "yine",
    "kadar",
    "olarak",
    "that",
    "this",
    "with",
    "from",
    "into",
    "your",
    "what",
    "when",
    "where",
    "have",
    "about",
    "will",
    "would",
    "there",
}

NEGATIVE_TONE_WORDS = {
    "şok",
    "kriz",
    "sorun",
    "risk",
    "tehlike",
    "hata",
    "kayıp",
    "kayip",
    "çöküş",
    "cokus",
    "fail",
    "problem",
    "mistake",
    "crisis",
    "danger",
    "loss",
}

POSITIVE_TONE_WORDS = {
    "çözüm",
    "cozum",
    "fırsat",
    "firsat",
    "umut",
    "başarı",
    "basari",
    "kazanç",
    "kazanc",
    "avantaj",
    "benefit",
    "solution",
    "opportunity",
    "success",
    "win",
}

EXPLAINER_WORDS = {
    "nasıl",
    "nasil",
    "neden",
    "what",
    "why",
    "how",
    "adım",
    "adim",
    "yöntem",
    "yontem",
    "detay",
    "örnek",
    "ornek",
    "explain",
    "guide",
}


def _parse_timecode(value: str) -> int:
    text = str(value or "").strip().replace(",", ".")
    if not text:
        return 0

    if re.match(r"^\d{2}:\d{2}$", text):
        minutes, seconds = text.split(":")
        return int(minutes) * 60 + int(seconds)

    match = re.match(r"(?:(\d+):)?(\d{2}):(\d{2})(?:\.\d+)?$", text)
    if not match:
        return 0

    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    return (hours * 3600) + (minutes * 60) + seconds


def _seconds_to_mmss(value: int) -> str:
    total_seconds = max(0, int(value))
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


def _parse_timing_line(timing_line: str) -> tuple[int, int]:
    if "-->" not in str(timing_line):
        return 0, 0
    start_raw, end_raw = [part.strip() for part in str(timing_line).split("-->", 1)]
    return _parse_timecode(start_raw), _parse_timecode(end_raw)


def _extract_keywords(text: str, max_items: int = 5) -> list[str]:
    candidates = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿĞğİıŞşÇçÖöÜü0-9]{4,}", text or "")
    results = []
    seen = set()
    for item in candidates:
        cleaned = normalize_whitespace(item)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in STOP_WORDS or key in seen:
            continue
        seen.add(key)
        results.append(cleaned)
        if len(results) >= max_items:
            break
    return results


def _timed_blocks_from_srt(girdi_dosyasi: Path) -> list[dict]:
    blocks = parse_srt_blocks(read_srt_file(girdi_dosyasi))
    timed_blocks = []
    for block in blocks:
        if not block.is_processable:
            continue
        start_sec, end_sec = _parse_timing_line(block.timing_line or "")
        if end_sec <= start_sec:
            continue
        text = normalize_whitespace(block.text_content)
        if not text:
            continue
        timed_blocks.append(
            {
                "start_sec": start_sec,
                "end_sec": end_sec,
                "start": _seconds_to_mmss(start_sec),
                "end": _seconds_to_mmss(end_sec),
                "text": text,
            }
        )
    return timed_blocks


def _extract_reference_moments(
    metadata_data: Optional[dict],
    critic_data: Optional[dict],
    broll_data: Optional[list],
) -> list[int]:
    moments: list[int] = []

    secilen = select_metadata_language(metadata_data)
    if secilen:
        for item in secilen.get("chapters", [])[:6]:
            if not isinstance(item, dict):
                continue
            seconds = _parse_timecode(item.get("timestamp", ""))
            if seconds > 0:
                moments.append(seconds)

    if isinstance(critic_data, dict):
        for key in ("timeline_notes", "rewrite_opportunities"):
            for item in critic_data.get(key, [])[:4]:
                if not isinstance(item, dict):
                    continue
                seconds = _parse_timecode(item.get("timestamp", ""))
                if seconds > 0:
                    moments.append(seconds)

    if isinstance(broll_data, list):
        for item in broll_data[:5]:
            if not isinstance(item, dict):
                continue
            seconds = _parse_timecode(item.get("timestamp", ""))
            if seconds > 0:
                moments.append(seconds)

    return moments


def _find_closest_block_index(start_times: list[int], target_seconds: int) -> Optional[int]:
    best_idx = None
    best_distance = None
    for idx, start_sec in enumerate(start_times):
        distance = abs(start_sec - target_seconds)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_idx = idx
    return best_idx


def _build_adaptive_transcript(
    timed_blocks: list[dict],
    metadata_data: Optional[dict] = None,
    critic_data: Optional[dict] = None,
    broll_data: Optional[list] = None,
    max_chars: int = ADAPTIVE_TRANSCRIPT_MAX_CHARS,
) -> str:
    if not timed_blocks:
        return ""

    lines = [f"[{item['start']} - {item['end']}] {item['text']}" for item in timed_blocks]
    full_text = "\n".join(lines)
    if len(full_text) <= max_chars:
        return full_text

    anchor_indices = {0, max(0, len(timed_blocks) - 1)}
    if len(timed_blocks) > 1:
        anchor_indices.add(1)
        anchor_indices.add(max(0, len(timed_blocks) - 2))
    if len(timed_blocks) > 2:
        for idx in range(ADAPTIVE_TRANSCRIPT_TARGET_ANCHORS):
            fraction = idx / max(ADAPTIVE_TRANSCRIPT_TARGET_ANCHORS - 1, 1)
            anchor_indices.add(round((len(timed_blocks) - 1) * fraction))

    start_times = [int(item["start_sec"]) for item in timed_blocks]
    for moment in _extract_reference_moments(metadata_data, critic_data, broll_data):
        closest = _find_closest_block_index(start_times, moment)
        if closest is not None:
            anchor_indices.add(closest)

    ordered_indices = []
    seen = set()
    for idx in sorted(anchor_indices):
        for expanded in range(
            max(0, idx - ADAPTIVE_TRANSCRIPT_WINDOW_RADIUS),
            min(len(timed_blocks), idx + ADAPTIVE_TRANSCRIPT_WINDOW_RADIUS + 1),
        ):
            if expanded in seen:
                continue
            seen.add(expanded)
            ordered_indices.append(expanded)

    selected_lines: list[str] = []
    current_length = 0
    for idx in ordered_indices:
        line = lines[idx]
        candidate_length = current_length + len(line) + (1 if selected_lines else 0)
        if candidate_length > max_chars and current_length >= int(max_chars * 0.72):
            break
        selected_lines.append(line)
        current_length = candidate_length

    if len(selected_lines) < 8:
        return full_text[:max_chars]
    return "\n".join(selected_lines)


def _estimate_track_count(total_duration_seconds: int) -> int:
    if total_duration_seconds <= 90:
        return 1
    if total_duration_seconds <= 210:
        return 2
    if total_duration_seconds <= 420:
        return 3
    if total_duration_seconds <= 720:
        return 4
    if total_duration_seconds <= 1200:
        return 5
    return 6


def _count_hint_text(total_duration_seconds: int) -> str:
    estimate = _estimate_track_count(total_duration_seconds)
    lower = max(1, estimate - 1)
    upper = min(MAX_SEGMENTS, estimate + 1)
    return f"{lower}-{upper}"


def _segment_slice_blocks(timed_blocks: list[dict], segment_count: int) -> list[dict]:
    if not timed_blocks:
        return []

    total_duration_seconds = timed_blocks[-1]["end_sec"]
    if total_duration_seconds <= 0:
        return []

    segment_count = max(1, min(MAX_SEGMENTS, segment_count))
    slice_duration = max(MIN_SEGMENT_SECONDS, math.ceil(total_duration_seconds / segment_count))
    slices = []

    for index in range(segment_count):
        start_sec = min(total_duration_seconds, index * slice_duration)
        end_sec = total_duration_seconds if index == segment_count - 1 else min(total_duration_seconds, (index + 1) * slice_duration)
        if end_sec <= start_sec:
            continue

        related = [
            item for item in timed_blocks
            if item["start_sec"] < end_sec and item["end_sec"] > start_sec
        ]
        text = " ".join(item["text"] for item in related).strip()
        slices.append(
            {
                "start_sec": start_sec,
                "end_sec": end_sec,
                "start": _seconds_to_mmss(start_sec),
                "end": _seconds_to_mmss(end_sec),
                "text": text,
            }
        )
    return slices


def _detect_segment_profile(text: str, position: int, total_segments: int) -> dict:
    lowered = (text or "").casefold()
    keywords = ", ".join(_extract_keywords(text, max_items=4)) or "the video topic"

    if position == 0:
        return {
            "music_type_tr": "Merak uyandiran sinematik giris",
            "why_tr": "Acilista izleyiciyi videoya sokmak ve ilk saniyelerde merak duygusu kurmak icin hafif ama dikkat cekici bir katman gerekir.",
            "intensity": "low_to_medium",
            "mood_en": f"subtle cinematic tension with curiosity-building pulse around {keywords}",
        }

    if position == total_segments - 1:
        positive = any(word in lowered for word in POSITIVE_TONE_WORDS)
        if positive:
            return {
                "music_type_tr": "Rahatlatan ve sonuc hissi veren kapanis muzigi",
                "why_tr": "Final bolumunde anlatilan sonucu toparlamak ve videoyu tatmin hissiyle kapatmak icin daha yumusak ama umutlu bir muzik uygundur.",
                "intensity": "low",
                "mood_en": f"warm resolving documentary underscore with a subtle uplifting finish around {keywords}",
            }
        return {
            "music_type_tr": "Dusuk yogunluklu dusunduren kapanis muzigi",
            "why_tr": "Kapanista bilgiyi sindirmek ve yorum birakma/izlemeye devam etme istegi yaratmak icin kontrollu bir final tonu gerekir.",
            "intensity": "low",
            "mood_en": f"reflective minimalist documentary underscore with a restrained ending around {keywords}",
        }

    if any(word in lowered for word in NEGATIVE_TONE_WORDS):
        return {
            "music_type_tr": "Gerilimli ama arka planda kalan dramatik muzik",
            "why_tr": "Bu bolumde risk, sorun veya sarsici bilgi one cikiyor; muzigin tansiyonu hafifce yukseltmesi izleyici dikkatini toplar.",
            "intensity": "medium",
            "mood_en": f"dramatic investigative tension underscore with pulsing rhythm around {keywords}",
        }

    if any(word in lowered for word in POSITIVE_TONE_WORDS):
        return {
            "music_type_tr": "Yukselen ama abartisiz motive edici muzik",
            "why_tr": "Bu kisimda cozum, avantaj veya umut hissi var; muzik hafif yukselerek anlatinin odagini desteklemeli.",
            "intensity": "medium",
            "mood_en": f"inspiring modern documentary background music with gentle lift around {keywords}",
        }

    if any(word in lowered for word in EXPLAINER_WORDS):
        return {
            "music_type_tr": "Temiz ve dikkat dagitmayan aciklayici arka plan muzigi",
            "why_tr": "Bilgi aktarimi on planda oldugu icin konusmayi bozmayan, ritmi sabit bir muzik daha dogru olur.",
            "intensity": "low",
            "mood_en": f"clean minimal explainer underscore with light rhythmic motion around {keywords}",
        }

    return {
        "music_type_tr": "Dengeli belgesel tarzi background muzik",
        "why_tr": "Akis bolumunde tekdüzeligi kirarken anlatim alanini daraltmayan orta yogunlukta bir katman yeterlidir.",
        "intensity": "low_to_medium",
        "mood_en": f"balanced cinematic documentary underscore with restrained momentum around {keywords}",
    }


def _ensure_prompt_quality(prompt_en: str, duration_seconds: int, fallback_mood: str) -> str:
    prompt = normalize_whitespace(prompt_en)
    if not prompt:
        prompt = f"Create {fallback_mood}"

    lower = prompt.casefold()
    extras = []
    if "instrumental" not in lower:
        extras.append("instrumental only")
    if "no vocals" not in lower and "without vocals" not in lower:
        extras.append("no vocals")
    if "background" not in lower and "underscore" not in lower:
        extras.append("background underscore")
    if "voice-over" not in lower and "voice over" not in lower:
        extras.append("voice-over friendly mix")
    if "clean mix" not in lower:
        extras.append("clean mix with space for narration")
    if "seconds" not in lower and "duration" not in lower:
        extras.append(f"duration around {duration_seconds} seconds")
    if "loop" not in lower:
        extras.append("smooth intro and outro, easy to trim or loop")

    if extras:
        prompt = f"{prompt.rstrip('.').rstrip(',')}, {', '.join(extras)}."
    elif not prompt.endswith("."):
        prompt += "."
    return prompt


def _fallback_plan(
    timed_blocks: list[dict],
    total_duration_seconds: int,
    strategy_note: str = "",
) -> dict:
    segment_count = _estimate_track_count(total_duration_seconds)
    slices = _segment_slice_blocks(timed_blocks, segment_count)
    segments = []

    for index, item in enumerate(slices):
        profile = _detect_segment_profile(item["text"], index, len(slices))
        duration_seconds = max(MIN_SEGMENT_SECONDS, item["end_sec"] - item["start_sec"])
        segments.append(
            {
                "start": item["start"],
                "end": item["end"],
                "music_type_tr": profile["music_type_tr"],
                "why_tr": profile["why_tr"],
                "intensity": profile["intensity"],
                "english_prompt": _ensure_prompt_quality(
                    (
                        f"Create {profile['mood_en']}, subtle modern trailer-style texture, soft percussion, "
                        f"light ambient pads, emotional control, suitable for a YouTube documentary segment."
                    ),
                    duration_seconds,
                    profile["mood_en"],
                ),
            }
        )

    return {
        "total_duration": _seconds_to_mmss(total_duration_seconds),
        "recommended_track_count": len(segments),
        "overall_strategy_tr": (
            "Muzik degisimlerini asiri siklastirmadan videonun acilis, gelisme ve final akisina gore "
            "katmanli bir plan kuruldu. Her prompt arka planda kalacak, anlatimi bastirmayacak ve "
            "Gemini veya Suno/Sona benzeri ureticilerde kullanilabilecek seviyede detaylandirildi."
        )
        + (f" {strategy_note}" if strategy_note else ""),
        "segments": segments,
    }


def _normalize_segment(item: dict, fallback: dict) -> dict:
    start_sec = _parse_timecode(item.get("start", fallback["start"]))
    end_sec = _parse_timecode(item.get("end", fallback["end"]))
    fallback_start_sec = _parse_timecode(fallback["start"])
    fallback_end_sec = _parse_timecode(fallback["end"])

    if end_sec <= start_sec:
        start_sec = fallback_start_sec
        end_sec = fallback_end_sec

    duration_seconds = max(MIN_SEGMENT_SECONDS, end_sec - start_sec)
    profile = _detect_segment_profile(fallback.get("text", ""), 0, 1)

    return {
        "start": _seconds_to_mmss(start_sec),
        "end": _seconds_to_mmss(end_sec),
        "music_type_tr": normalize_whitespace(item.get("music_type_tr", "")) or fallback.get("music_type_tr", profile["music_type_tr"]),
        "why_tr": normalize_whitespace(item.get("why_tr", "")) or fallback.get("why_tr", profile["why_tr"]),
        "intensity": normalize_whitespace(item.get("intensity", "")) or fallback.get("intensity", profile["intensity"]),
        "english_prompt": _ensure_prompt_quality(
            item.get("english_prompt", "") or fallback.get("english_prompt", ""),
            duration_seconds,
            profile["mood_en"],
        ),
    }


def normalize_plan(data: dict, timed_blocks: list[dict], total_duration_seconds: int, strategy_note: str = "") -> dict:
    fallback = _fallback_plan(timed_blocks, total_duration_seconds, strategy_note=strategy_note)
    if not isinstance(data, dict):
        return fallback

    slices = _segment_slice_blocks(
        timed_blocks,
        int(data.get("recommended_track_count") or len(data.get("segments", [])) or fallback["recommended_track_count"]),
    ) or _segment_slice_blocks(timed_blocks, fallback["recommended_track_count"])

    raw_segments = data.get("segments", [])
    if not isinstance(raw_segments, list) or not raw_segments:
        return fallback

    normalized_segments = []
    for index, item in enumerate(raw_segments[:MAX_SEGMENTS]):
        if not isinstance(item, dict):
            continue
        fallback_plan_item = fallback["segments"][min(index, len(fallback["segments"]) - 1)]
        fallback_slice_item = slices[min(index, len(slices) - 1)] if slices else {}
        fallback_item = {
            "start": fallback_slice_item.get("start", fallback_plan_item["start"]),
            "end": fallback_slice_item.get("end", fallback_plan_item["end"]),
            "text": fallback_slice_item.get("text", ""),
            "music_type_tr": fallback_plan_item.get("music_type_tr", ""),
            "why_tr": fallback_plan_item.get("why_tr", ""),
            "intensity": fallback_plan_item.get("intensity", ""),
            "english_prompt": fallback_plan_item.get("english_prompt", ""),
        }
        normalized_segments.append(_normalize_segment(item, fallback_item))

    if not normalized_segments:
        return fallback

    normalized_segments.sort(key=lambda item: _parse_timecode(item["start"]))
    return {
        "total_duration": _seconds_to_mmss(total_duration_seconds),
        "recommended_track_count": len(normalized_segments),
        "overall_strategy_tr": normalize_whitespace(data.get("overall_strategy_tr", "")) or fallback["overall_strategy_tr"],
        "segments": normalized_segments,
    }


def _build_strict_json_retry_prompt(prompt: str) -> str:
    return (
        f"{prompt}\n\n"
        "JSON ZORLAMASI:\n"
        "- Yalnizca tek bir gecerli JSON nesnesi dondur.\n"
        "- Markdown, aciklama, on soz, analiz veya kod blogu ekleme.\n"
        "- JSON disinda hicbir sey yazma.\n"
    ).strip()


def _should_use_local_only_plan(timed_blocks: list[dict], total_duration_seconds: int, transcript_text: str) -> tuple[bool, str]:
    if total_duration_seconds <= LOCAL_ONLY_MAX_SECONDS:
        return True, f"Video suresi {_seconds_to_mmss(total_duration_seconds)} oldugu icin hizli local muzik plani yeterli goruldu."
    if len(timed_blocks) <= LOCAL_ONLY_MAX_BLOCKS:
        return True, f"Zamanli blok sayisi {len(timed_blocks)} oldugu icin local fallback plana gecildi."
    if len(transcript_text) <= LOCAL_ONLY_MAX_TRANSCRIPT_CHARS:
        return True, f"Temsili transcript boyutu {len(transcript_text)} karakter oldugu icin local muzik plani secildi."
    return False, ""


def _build_prompt(
    timed_transcript: str,
    total_duration_seconds: int,
    metadata_summary: str = "",
    critic_summary: str = "",
    broll_summary: str = "",
) -> str:
    return f"""
Sen bir YouTube muzik supervizoru, trailer editoru ve background score stratejistisin.

Gorevin:
Asagidaki transcripti ve paketleme sinyallerini kullanarak videonun akis mantigini anla.
Videonun uzunluguna ve konu gecislerine gore arka planda kac farkli muzik bolumu gerektigini kendin belirle.

KRITIK KURALLAR:
- Sadece gecerli JSON dondur.
- JSON disinda hicbir aciklama yazma.
- overall_strategy_tr, music_type_tr ve why_tr alanlari Turkce olsun.
- english_prompt alani Ingilizce olsun.
- Muzik sadece background score olarak dusunulsun.
- Vokalli sarki onermemelisin.
- Gereksiz yere her 10 saniyede bir muzik degistirme.
- Ayni videoda mantikli sayida segment kullan.
- Promptlar Gemini veya Suno/Sona benzeri muzik ureticilerinde calisabilecek kadar net olsun.
- Her promptta mood, tempo, instrumentation ve narration-friendly karakter net hissedilsin.

JSON SEMASI:
{{
  "overall_strategy_tr": "",
  "recommended_track_count": 0,
  "segments": [
    {{
      "start": "00:00",
      "end": "00:45",
      "music_type_tr": "",
      "why_tr": "",
      "intensity": "low",
      "english_prompt": ""
    }}
  ]
}}

Video toplam suresi: {_seconds_to_mmss(total_duration_seconds)}
Mantikli muzik segment araligi: {_count_hint_text(total_duration_seconds)}

METADATA OZETI:
{metadata_summary or "YouTube metadata verisi yok."}

AI CRITIC OZETI:
{critic_summary or "Video Elestirmeni verisi yok."}

B-ROLL OZETI:
{broll_summary or "B-roll verisi yok."}

Temsili Altyazi:
{timed_transcript[:MAX_TRANSCRIPT_CHARS]}
""".strip()


def analyze(
    girdi_dosyasi: Path,
    llm: CentralLLM,
    metadata_data: Optional[dict] = None,
    critic_data: Optional[dict] = None,
    broll_data: Optional[list] = None,
) -> Optional[dict]:
    timed_blocks = _timed_blocks_from_srt(girdi_dosyasi)
    if not timed_blocks:
        logger.error("Gecerli zaman bilgisi olan altyazi bloklari bulunamadi.")
        return None

    if metadata_data is None:
        metadata_data = load_related_json(girdi_dosyasi, "_metadata.json")
    if critic_data is None:
        critic_data = load_related_json(girdi_dosyasi, "_video_critic.json")
    if broll_data is None:
        broll_data = load_related_json(girdi_dosyasi, "_B_roll_fikirleri.json")

    total_duration_seconds = timed_blocks[-1]["end_sec"]
    timed_transcript = _build_adaptive_transcript(
        timed_blocks,
        metadata_data=metadata_data,
        critic_data=critic_data,
        broll_data=broll_data if isinstance(broll_data, list) else None,
        max_chars=ADAPTIVE_TRANSCRIPT_MAX_CHARS,
    )
    metadata_summary = build_metadata_summary(metadata_data)
    critic_summary = build_critic_summary(critic_data)
    broll_summary = build_broll_summary(broll_data if isinstance(broll_data, list) else None)

    local_only, local_reason = _should_use_local_only_plan(timed_blocks, total_duration_seconds, timed_transcript)
    if local_only:
        logger.info(f"Muzik modulu local-only hizli moda gecti: {local_reason}")
        local_plan = _fallback_plan(timed_blocks, total_duration_seconds, strategy_note=local_reason)
        local_plan["generation_mode"] = "local_only"
        local_plan["generation_note"] = local_reason
        return local_plan

    prompt = _build_prompt(
        timed_transcript,
        total_duration_seconds,
        metadata_summary=metadata_summary,
        critic_summary=critic_summary,
        broll_summary=broll_summary,
    )
    logger.info("Arka plan muzik plani ve promptlari uretiliyor...")

    parsed = None
    try:
        parsed = extract_json_response(
            call_with_youtube_profile(llm, prompt, profile="creative_ideation"),
            logger_override=logger,
        )
    except Exception as exc:
        logger.warning(f"Muzik prompt LLM cagrisinda hata: {exc}")
    strategy_note = ""
    if not isinstance(parsed, dict) or not isinstance(parsed.get("segments"), list) or not parsed.get("segments"):
        logger.warning("Muzik prompt creative_ideation cevabi parse edilemedi veya bos geldi. strict_json retry deneniyor.")
        try:
            parsed = extract_json_response(
                call_with_youtube_profile(llm, _build_strict_json_retry_prompt(prompt), profile="strict_json"),
                logger_override=logger,
            )
            strategy_note = "Ilk LLM cevabi parse edilemedigi icin strict_json retry kullanildi."
        except Exception as exc:
            logger.warning(f"Muzik prompt strict_json retry basarisiz oldu: {exc}")
            parsed = None

    normalized = normalize_plan(parsed, timed_blocks, total_duration_seconds, strategy_note=strategy_note)
    if strategy_note and isinstance(parsed, dict):
        generation_mode = "strict_json_retry"
    elif strategy_note:
        generation_mode = "fallback_after_retry"
    elif isinstance(parsed, dict):
        generation_mode = "llm"
    else:
        generation_mode = "fallback"
    normalized["generation_mode"] = generation_mode
    normalized["generation_note"] = (
        strategy_note
        or ("LLM sonucu gecerli sekilde normalize edildi." if isinstance(parsed, dict) else "LLM sonucu yerine local fallback plan kullanildi.")
    )
    return normalized


def build_report_text(girdi_stem: str, data: dict, model_adi: str) -> str:
    lines = [
        f"=== {girdi_stem} ICIN MUZIK PROMPT RAPORU ===",
        f"Kullanilan Model: {model_adi}",
        f"Toplam Sure: {data.get('total_duration', '')}",
        f"Onerilen Muzik Segment Sayisi: {data.get('recommended_track_count', 0)}",
        f"Uretim Modu: {data.get('generation_mode', '')}",
        f"Not: {data.get('generation_note', '')}",
        "",
        "GENEL STRATEJI",
        "-" * 60,
        data.get("overall_strategy_tr", ""),
    ]

    for index, item in enumerate(data.get("segments", []), start=1):
        lines.extend(
            [
                "",
                f"SEGMENT {index}",
                "-" * 60,
                f"Zaman: {item.get('start', '')} - {item.get('end', '')}",
                f"Muzik Tipi: {item.get('music_type_tr', '')}",
                f"Neden: {item.get('why_tr', '')}",
                f"Yogunluk: {item.get('intensity', '')}",
                "English Prompt:",
                item.get("english_prompt", ""),
            ]
        )

    return "\n".join(lines).strip() + "\n"


def save_reports(girdi_dosyasi: Path, data: dict, model_adi: str) -> Tuple[Path, Path]:
    json_yolu = stem_json_output_path(girdi_dosyasi.stem, "_music_prompts.json", group="youtube")
    txt_yolu = txt_output_path("music_prompts")

    with open(json_yolu, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)

    txt_yolu.write_text(build_report_text(girdi_dosyasi.stem, data, model_adi), encoding="utf-8")
    logger.info(f"Muzik prompt JSON kaydedildi: {json_yolu.name}")
    logger.info(f"Muzik prompt TXT kaydedildi: {txt_yolu.name}")
    return json_yolu, txt_yolu


def _select_srt() -> Optional[Path]:
    return select_primary_srt(logger, "Muzik Prompt Uretici")


def run() -> None:
    print("\n" + "=" * 60)
    print("MUZIK PROMPT OLUSTURUCU")
    print("=" * 60)

    girdi = _select_srt()
    if not girdi:
        return

    use_recommended = prompt_module_llm_plan("205", needs_main=True)
    if use_recommended:
        saglayici, model_adi = get_module_recommended_llm_config("205", "main")
        print_module_llm_choice_summary("205", {"main": (saglayici, model_adi)})
    else:
        saglayici, model_adi = select_llm("main")
    llm = CentralLLM(provider=saglayici, model_name=model_adi)

    data = analyze(
        girdi,
        llm,
        metadata_data=load_related_json(girdi, "_metadata.json"),
        critic_data=load_related_json(girdi, "_video_critic.json"),
        broll_data=load_related_json(girdi, "_B_roll_fikirleri.json"),
    )
    if not data:
        logger.error("❌ Muzik prompt plani olusturulamadi.")
        return

    save_reports(girdi, data, model_adi)
    logger.info("🎉 Muzik prompt olusturma islemi tamamlandi.")


def run_automatic(
    girdi_dosyasi: Path,
    llm: CentralLLM,
    metadata_data: Optional[dict] = None,
    critic_data: Optional[dict] = None,
    broll_data: Optional[list] = None,
) -> Optional[dict]:
    logger.info(f"🔄 OTOMASYON: {girdi_dosyasi.name} icin muzik promptlari uretiliyor...")
    data = analyze(
        girdi_dosyasi,
        llm,
        metadata_data=metadata_data,
        critic_data=critic_data,
        broll_data=broll_data,
    )
    if not data:
        logger.error("❌ Otomatik muzik prompt uretimi basarisiz oldu.")
        return None

    json_yolu, txt_yolu = save_reports(girdi_dosyasi, data, llm.model_name)
    return {
        "data": data,
        "json_path": json_yolu,
        "txt_path": txt_yolu,
    }

