# moduller/altyazi_olusturucu.py
import os
import time
from moduller.config import INPUTS_DIR, OUTPUTS_DIR
from moduller.logger import get_logger
from moduller.runtime_utils import format_elapsed
from moduller.subtitle_output_utils import (
    relocate_known_subtitle_intermediates,
    subtitle_intermediate_output_path,
)
from moduller.transcriber import WhisperMotor, resolve_shorts_word_limit
from pathlib import Path

logger = get_logger("subtitle")

STANDARD_SUBTITLE_NAME = "subtitle_raw_tr.srt"
SHORTS_SUBTITLE_NAME = "subtitle_raw_shorts.srt"
WHISPER_EN_SUBTITLE_NAME = "subtitle_raw_en.srt"
SUPPORTED_MEDIA_PATTERNS = ("*.mp4", "*.mp3", "*.wav")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _shorts_should_be_generated_automatically() -> bool:
    return _env_bool("WHISPER_AUTO_GENERATE_SHORTS", False)


def _english_should_be_generated_automatically() -> bool:
    return _env_bool("WHISPER_AUTO_GENERATE_ENGLISH", False)


def _list_media_files() -> list[Path]:
    media_files: list[Path] = []
    for pattern in SUPPORTED_MEDIA_PATTERNS:
        media_files.extend(INPUTS_DIR.glob(pattern))
        media_files.extend(OUTPUTS_DIR.rglob(pattern))

    unique_files: list[Path] = []
    seen = set()
    for media_file in media_files:
        key = str(media_file.resolve()).casefold() if media_file.exists() else str(media_file).casefold()
        if key in seen:
            continue
        seen.add(key)
        unique_files.append(media_file)

    return sorted(
        unique_files,
        key=lambda path: (
            0 if INPUTS_DIR in path.parents else 1,
            path.parent.name.casefold(),
            path.name.casefold(),
        ),
    )

def run():
    started_at = time.perf_counter()
    shorts_word_limit = resolve_shorts_word_limit()
    print("\n" + "="*50)
    print("📝 AŞAMA 1: Altyazı Oluşturucu")
    print("="*50)

    # 1. TÜM MEDYA DOSYALARINI TOPLA
    video_files = _list_media_files()

    if not video_files:
        logger.error("❌ Hiç video veya ses dosyası bulunamadı!")
        return

    print("\n📂 Dosya seçin:")
    for idx, video in enumerate(video_files, start=1):
        print(f"  [{idx}] {video.parent.name} / {video.name}")

    try:
        secim = int(input("👉 Seçiminiz: ")) - 1
        secilen_video = video_files[secim]
    except:
        logger.error("❌ Geçersiz seçim!")
        return

    relocate_known_subtitle_intermediates()
    output_standard = subtitle_intermediate_output_path(STANDARD_SUBTITLE_NAME)
    output_shorts = subtitle_intermediate_output_path(SHORTS_SUBTITLE_NAME)
    output_english = subtitle_intermediate_output_path(WHISPER_EN_SUBTITLE_NAME)
    generate_shorts = _shorts_should_be_generated_automatically()
    generate_english = _english_should_be_generated_automatically()

    logger.info("Standart Türkçe altyazı oluşturuluyor...")
    WhisperMotor.generate_standard_transcript(str(secilen_video), str(output_standard))

    if generate_shorts:
        logger.info("Reel/Shorts için Türkçe altyazı oluşturuluyor...")
        WhisperMotor.generate_dynamic_transcript(
            str(secilen_video),
            str(output_shorts),
            kelime_siniri=shorts_word_limit,
        )
    else:
        logger.info("Shorts Türkçe altyazı env geregi kapali. Lazy modda birakildi.")

    if generate_english:
        logger.info("Whisper ile dogrudan İngilizce altyazı oluşturuluyor...")
        WhisperMotor.generate_english_translation_transcript(str(secilen_video), str(output_english))
    else:
        logger.info("Whisper İngilizce altyazı env geregi kapali.")

    elapsed = format_elapsed(time.perf_counter() - started_at)
    logger.info(f"✅ Tamamlandı: {output_standard.name}")
    logger.info(f"⏱️ Modul 1 toplam sure: {elapsed}")

def run_automatic(video_yolu: Path) -> Path:
    """Tam otomasyon icin standart TR SRT uretir; shorts TR ve Whisper EN env'e gore opsiyoneldir."""
    started_at = time.perf_counter()
    logger.info(f"🔄 OTOMASYON: {video_yolu.name} için altyazılar çıkartılıyor...")
    
    # Çıktı yolları
    relocate_known_subtitle_intermediates()
    output_standart = subtitle_intermediate_output_path(STANDARD_SUBTITLE_NAME)
    output_shorts = subtitle_intermediate_output_path(SHORTS_SUBTITLE_NAME)
    output_english = subtitle_intermediate_output_path(WHISPER_EN_SUBTITLE_NAME)
    shorts_word_limit = resolve_shorts_word_limit()
    generate_shorts = _shorts_should_be_generated_automatically()
    generate_english = _english_should_be_generated_automatically()

    logger.info("-> Standart Türkçe altyazı üretiliyor...")
    WhisperMotor.generate_standard_transcript(str(video_yolu), str(output_standart))
    if generate_shorts:
        logger.info("-> Dinamik (Shorts) Türkçe altyazı üretiliyor...")
        WhisperMotor.generate_dynamic_transcript(
            str(video_yolu),
            str(output_shorts),
            kelime_siniri=shorts_word_limit,
        )
    else:
        logger.info(
            "-> Dinamik (Shorts) altyazı lazy moda bırakıldı. "
            "Gerektiğinde ilgili modül otomatik üretecek."
        )
    if generate_english:
        logger.info("-> Whisper ile dogrudan İngilizce altyazı üretiliyor...")
        WhisperMotor.generate_english_translation_transcript(str(video_yolu), str(output_english))
    else:
        logger.info("-> Whisper İngilizce altyazı env geregi kapali.")
    
    # Gramer düzeltici standart altyazı üzerinden ilerleyeceği için onun yolunu döndürüyoruz
    elapsed = format_elapsed(time.perf_counter() - started_at)
    logger.info(f"⏱️ Modul 1 toplam sure: {elapsed}")
    return output_standart

