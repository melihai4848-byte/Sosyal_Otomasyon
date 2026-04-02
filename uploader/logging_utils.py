"""Logging helpers for the YouTube draft uploader."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from uploader.models import UploaderConfig


def configure_logging(config: UploaderConfig) -> logging.Logger:
    """Configure uploader logging once and return the base logger."""
    logger = logging.getLogger("YouTubeDraftUploader")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = RotatingFileHandler(
        config.log_file,
        maxBytes=2_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger
