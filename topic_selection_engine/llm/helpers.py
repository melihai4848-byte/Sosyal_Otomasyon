from __future__ import annotations

import json
import re
from typing import Any, Optional, Tuple

from moduller.llm_manager import CentralLLM, get_default_llm_config
from moduller.logger import get_logger
from moduller.youtube_llm_profiles import call_with_youtube_profile

logger = get_logger("topic_llm")


def extract_json_response(llm_response: str, logger_override=None):
    active_logger = logger_override or logger
    if not llm_response:
        return None

    try:
        response = re.sub(r"```json\s*|```", "", str(llm_response)).strip()

        try:
            return json.loads(response)
        except Exception:
            pass

        start_positions = [i for i, ch in enumerate(response) if ch == "{"]
        for start in start_positions:
            depth = 0
            in_string = False
            escape = False

            for idx in range(start, len(response)):
                ch = response[idx]

                if escape:
                    escape = False
                    continue

                if ch == "\\":
                    escape = True
                    continue

                if ch == '"':
                    in_string = not in_string
                    continue

                if not in_string:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            return json.loads(response[start:idx + 1])
    except Exception as exc:
        active_logger.error(f"Topic LLM JSON ayrıştırma hatası: {exc}")

    active_logger.error("Topic LLM cevabından geçerli JSON çıkarılamadı.")
    return None


def resolve_topic_llm_config(settings) -> Tuple[Optional[str], Optional[str]]:
    llm_settings = getattr(settings, "llm", None)
    if not llm_settings or not getattr(llm_settings, "enabled", False):
        return None, None

    provider = str(getattr(llm_settings, "provider", "") or "").strip().upper()
    model_name = str(getattr(llm_settings, "model_name", "") or "").strip()
    if provider in {"", "AUTO", "AUTO_SMART"} or not model_name:
        provider, model_name = get_default_llm_config("smart")
    return provider, model_name


def build_topic_llm(settings) -> tuple[Optional[CentralLLM], dict]:
    provider, model_name = resolve_topic_llm_config(settings)
    if not provider or not model_name:
        return None, {"enabled": False, "provider": "", "model_name": ""}

    return (
        CentralLLM(provider=provider, model_name=model_name),
        {
            "enabled": True,
            "provider": provider,
            "model_name": model_name,
        },
    )


def call_topic_llm_json(
    llm: CentralLLM,
    prompt: str,
    profile: str = "analytic_json",
    logger_override=None,
    retries: int = 2,
) -> Any:
    active_logger = logger_override or logger
    last_error: Optional[Exception] = None

    for attempt in range(1, max(retries, 1) + 1):
        try:
            response = call_with_youtube_profile(llm, prompt, profile=profile)
            parsed = extract_json_response(response, logger_override=active_logger)
            if parsed is not None:
                return parsed
        except Exception as exc:
            last_error = exc
            active_logger.warning(f"Topic LLM çağrısı başarısız oldu ({attempt}/{retries}): {exc}")

    if last_error:
        active_logger.warning(f"Topic LLM fallback devreye girdi: {last_error}")
    return None
