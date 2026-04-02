from __future__ import annotations

from typing import Optional

from moduller.llm_manager import CentralLLM


_PROFILE_DEFAULTS = {
    "creative_ideation": {
        "timeout": 180,
        "max_retries": 1,
        "temperature": 0.85,
        "top_p": 0.92,
        "top_k": 60,
        "repeat_penalty": 1.05,
        "num_ctx": 16384,
        "keep_alive": "25m",
    },
    "creative_ranker": {
        "timeout": 150,
        "max_retries": 1,
        "temperature": 0.22,
        "top_p": 0.82,
        "top_k": 30,
        "repeat_penalty": 1.08,
        "num_ctx": 16384,
        "keep_alive": "20m",
    },
    "analytic_json": {
        "timeout": 180,
        "max_retries": 2,
        "temperature": 0.18,
        "top_p": 0.80,
        "top_k": 30,
        "repeat_penalty": 1.08,
        "num_ctx": 16384,
        "keep_alive": "20m",
    },
    "strict_json": {
        "timeout": 150,
        "max_retries": 2,
        "temperature": 0.10,
        "top_p": 0.75,
        "top_k": 20,
        "repeat_penalty": 1.10,
        "num_ctx": 12288,
        "keep_alive": "20m",
    },
    "description_json": {
        "timeout": 210,
        "max_retries": 2,
        "temperature": 0.12,
        "top_p": 0.78,
        "top_k": 24,
        "repeat_penalty": 1.08,
        "num_ctx": 16384,
        "keep_alive": "25m",
    },
}


def _profile_config(profile: str) -> dict:
    return dict(_PROFILE_DEFAULTS.get(profile, _PROFILE_DEFAULTS["strict_json"]))


def call_with_youtube_profile(
    llm: CentralLLM,
    prompt: str,
    profile: str = "strict_json",
    timeout: Optional[int] = None,
    max_retries: Optional[int] = None,
) -> str:
    config = _profile_config(profile)
    effective_timeout = timeout if timeout is not None else config["timeout"]
    effective_retries = max_retries if max_retries is not None else config["max_retries"]

    ollama_options = None
    ollama_keep_alive = None
    if llm.provider == "OLLAMA":
        ollama_options = {
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "top_k": config["top_k"],
            "repeat_penalty": config["repeat_penalty"],
            "num_ctx": config["num_ctx"],
        }
        ollama_keep_alive = config["keep_alive"]

    return llm.uret(
        prompt,
        timeout=effective_timeout,
        max_retries=effective_retries,
        ollama_options=ollama_options,
        ollama_keep_alive=ollama_keep_alive,
    )
