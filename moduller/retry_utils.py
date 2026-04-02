import sys
import time
from typing import Callable, TypeVar

import requests

T = TypeVar("T")

RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
RETRYABLE_MESSAGE_HINTS = (
    "rate limit",
    "too many requests",
    "temporarily unavailable",
    "service unavailable",
    "connection reset",
    "connection aborted",
    "connection broken",
    "timed out",
    "timeout",
    "try again later",
    "quota",
    "remote end closed connection",
)


def _response_from_exception(exc: Exception):
    return getattr(exc, "response", None)


def _status_code_from_exception(exc: Exception) -> int | None:
    response = _response_from_exception(exc)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def retry_after_seconds(exc: Exception) -> int | None:
    response = _response_from_exception(exc)
    headers = getattr(response, "headers", {}) or {}
    raw = headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(1, int(float(str(raw).strip())))
    except (TypeError, ValueError):
        return None


def is_retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True

    status_code = _status_code_from_exception(exc)
    if status_code in RETRYABLE_STATUS_CODES:
        return True

    if isinstance(exc, requests.exceptions.RequestException):
        return status_code in RETRYABLE_STATUS_CODES or status_code is None

    message = str(exc).lower()
    return any(hint in message for hint in RETRYABLE_MESSAGE_HINTS)


def _sleep_with_countdown(delay_seconds: int, description: str, logger) -> None:
    if delay_seconds <= 0:
        logger.info(f"{description} tekrar deneniyor...")
        return

    is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    if not is_tty:
        time.sleep(delay_seconds)
        logger.info(f"{description} tekrar deneniyor...")
        return

    countdown_template = "{description} tekrar denemesine {remaining} saniye kaldi..."
    last_rendered = ""

    for remaining in range(delay_seconds, 0, -1):
        last_rendered = countdown_template.format(description=description, remaining=remaining)
        sys.stdout.write("\r" + last_rendered.ljust(120))
        sys.stdout.flush()
        time.sleep(1)

    sys.stdout.write("\r" + (" " * max(len(last_rendered), 120)) + "\r")
    sys.stdout.flush()
    logger.info(f"{description} tekrar deneniyor...")


def retry_with_backoff(
    action: Callable[[], T],
    description: str,
    logger,
    max_attempts: int = 4,
    base_delay_seconds: int = 15,
    max_delay_seconds: int = 60,
) -> T:
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return action()
        except Exception as exc:
            last_error = exc
            retryable = is_retryable_exception(exc)
            if attempt >= max_attempts or not retryable:
                raise

            hinted_delay = retry_after_seconds(exc)
            delay_seconds = hinted_delay if hinted_delay is not None else min(
                max_delay_seconds,
                base_delay_seconds * attempt,
            )
            logger.warning(
                f"{description} gecici hata verdi ({attempt}/{max_attempts}): {exc}. "
                f"{delay_seconds} saniye sonra tekrar denenecek."
            )
            _sleep_with_countdown(delay_seconds, description, logger)

    if last_error:
        raise last_error
    raise RuntimeError(f"{description} tekrar denemelere ragmen calistirilamadi.")
