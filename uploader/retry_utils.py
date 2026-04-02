"""Retry helpers for transient uploader operations."""

from __future__ import annotations

import socket
import time
from typing import Callable, TypeVar

from uploader.errors import RetryableApiError
from uploader.models import RetrySettings

T = TypeVar("T")


def _is_retryable_http_error(exc: Exception) -> bool:
    status_code = getattr(getattr(exc, "resp", None), "status", None)
    return status_code in {408, 409, 429, 500, 502, 503, 504}


def is_retryable_exception(exc: Exception) -> bool:
    """Return whether the exception is a retry candidate."""
    if isinstance(exc, RetryableApiError):
        return True
    if isinstance(exc, (TimeoutError, ConnectionError, socket.timeout)):
        return True
    return _is_retryable_http_error(exc)


def _delay_for_attempt(retry: RetrySettings, attempt: int) -> int:
    raw = retry.initial_delay_seconds * (2 ** max(attempt - 1, 0))
    return min(raw, retry.max_delay_seconds)


def execute_with_retry(
    operation: Callable[[], T],
    *,
    description: str,
    retry: RetrySettings,
    logger,
) -> T:
    """Execute an operation with exponential backoff."""
    last_error: Exception | None = None
    for attempt in range(1, retry.max_attempts + 1):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= retry.max_attempts or not is_retryable_exception(exc):
                raise
            delay_seconds = _delay_for_attempt(retry, attempt)
            logger.warning(
                f"{description} gecici hata verdi. {delay_seconds} saniye sonra tekrar denenecek. "
                f"Deneme {attempt}/{retry.max_attempts}. Hata: {exc}"
            )
            time.sleep(delay_seconds)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Retry mekanizmasi beklenmedik sekilde sonlandi.")
