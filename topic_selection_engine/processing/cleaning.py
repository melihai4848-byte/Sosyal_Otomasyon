from __future__ import annotations

import re
from typing import Iterable, List

from topic_selection_engine.models import RawSignal


NOISE_PATTERNS = [
    r"http\S+",
    r"\bsubscribe\b",
    r"\blike and comment\b",
    r"\s+",
]


def _normalize_text(text: str) -> str:
    out = text or ""
    for pattern in NOISE_PATTERNS:
        replacement = " " if pattern == r"\s+" else ""
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    return out.strip()


def clean_signals(signals: Iterable[RawSignal]) -> List[RawSignal]:
    cleaned: List[RawSignal] = []
    for signal in signals:
        text = _normalize_text(f"{signal.title} {signal.text}")
        if hasattr(signal, "model_copy"):
            cleaned.append(signal.model_copy(update={"text": text}))
        else:
            cleaned.append(signal.copy(update={"text": text}))
    return cleaned


def filter_relevant_signals(signals: Iterable[RawSignal], niche_keywords: List[str]) -> List[RawSignal]:
    if not niche_keywords:
        return list(signals)

    filtered: List[RawSignal] = []
    normalized_keywords = [keyword.lower().strip() for keyword in niche_keywords if keyword.strip()]
    for signal in signals:
        blob = f"{signal.title} {signal.text}".lower()
        if any(keyword in blob for keyword in normalized_keywords):
            filtered.append(signal)
    return filtered


def deduplicate_signals(signals: Iterable[RawSignal]) -> List[RawSignal]:
    seen = set()
    deduped: List[RawSignal] = []
    for signal in signals:
        text_head = re.sub(r"\s+", " ", signal.text.lower().strip())[:180]
        key = (signal.source, signal.title.lower().strip(), text_head)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(signal)
    return deduped
