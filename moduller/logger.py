# moduller/logger.py
import logging
import os
import sys
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_LEVEL = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
LOG_FILE_PATH = LOG_DIR / "automation.log"

_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | [%(name)s] -> %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_STREAM_TARGET = sys.stdout
if hasattr(_STREAM_TARGET, "reconfigure"):
    try:
        _STREAM_TARGET.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
_STREAM_HANDLER = logging.StreamHandler(_STREAM_TARGET)
_STREAM_HANDLER.setFormatter(_FORMATTER)

_FILE_HANDLER = RotatingFileHandler(
    LOG_FILE_PATH,
    maxBytes=2_000_000,
    backupCount=3,
    encoding="utf-8",
)
_FILE_HANDLER.setFormatter(_FORMATTER)


@lru_cache(maxsize=1)
def _logger_alias_map() -> dict[str, str]:
    from moduller.module_registry import MODULE_REGISTRY

    alias_map: dict[str, str] = {}

    def register(alias: str, canonical: str) -> None:
        if alias:
            alias_map[str(alias)] = canonical

    for entry in MODULE_REGISTRY:
        canonical = f"{entry.number} | {entry.title}"
        aliases = {
            entry.key,
            entry.number,
            entry.title,
            entry.module_path,
            entry.module_path.rsplit(".", 1)[-1],
        }
        for alias in aliases:
            register(alias, canonical)

    special_aliases = {
        "full_automation": "Otomasyon | Coklu Secim Pipeline",
        "moduller.full_automation": "Otomasyon | Coklu Secim Pipeline",
        "AnaMenu": "Ana Menu",
        "__main__": "Ana Menu",
    }
    for alias, canonical in special_aliases.items():
        register(alias, canonical)

    legacy_aliases = {
        "Modul_1": "subtitle",
        "Modul_2_Gramer": "grammar",
        "Modul_3_Ceviri": "translation",
        "Modul_4_Reels_Fikir_Uretici": "reels",
        "Modul_5_Video_Description": "description",
        "Modul_6_BRoll_Onerici": "broll",
        "Modul_7_Video_Critic": "critic",
        "Modul_8_Hook_Rewriter": "hook",
        "Modul_9_Trim_Suggester": "trim",
        "Modul_10_Full_Automation": "full_automation",
        "Modul_13_Analitik_Geri_Bildirim": "feedback",
        "Modul_14_Carousel_Olusturucu": "carousel",
        "Modul_15_IG_Metadata": "ig_metadata",
        "Modul_16_Story_Planlayici": "story",
        "Modul_17_Reel_Shorts_Olusturucu": "reels_render",
        "Modul_18_Thumbnail_Uretici": "thumbnail_main",
        "Modul_20_YouTube_Draft_Upload": "youtube_uploader",
        "Modul_21_Video_Title": "title",
        "Modul_22_Muzik_Prompt": "music",
        "AutomaticBrollDownloader": "automatic_broll_downloader",
        "PremiereProXmlIntegration": "premiere_xml",
        "TopicSelectionEngine": "topic",
    }
    for alias, target in legacy_aliases.items():
        register(alias, alias_map.get(target, target))

    return alias_map


def _resolve_logger_name(module_name: str) -> str:
    return _logger_alias_map().get(str(module_name), str(module_name))


def get_logger(module_name: str) -> logging.Logger:
    """Tum sistem icin ortak console ve file logger dondurur."""
    logger = logging.getLogger(_resolve_logger_name(module_name))

    if not logger.handlers:
        logger.setLevel(LOG_LEVEL)
        logger.addHandler(_STREAM_HANDLER)
        logger.addHandler(_FILE_HANDLER)
        logger.propagate = False

    return logger
