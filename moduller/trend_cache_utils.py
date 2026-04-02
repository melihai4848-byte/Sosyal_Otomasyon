import json
from pathlib import Path
from typing import Optional, Tuple

from moduller.logger import get_logger
from moduller.output_paths import json_output_path, txt_output_path
from topic_selection_engine.live_sources import extract_live_keywords, summarize_live_bundle

logger = get_logger("trend_cache")

TREND_JSON_PATH = json_output_path("live_trends")
TREND_TXT_PATH = txt_output_path("live_trends")


def _make_json_serializable(data: dict) -> dict:
    payload = dict(data or {})
    signals = payload.get("signals", [])
    serializable_signals = []
    for signal in signals:
        if hasattr(signal, "model_dump"):
            serializable_signals.append(signal.model_dump())
        elif hasattr(signal, "dict"):
            serializable_signals.append(signal.dict())
        elif isinstance(signal, dict):
            serializable_signals.append(signal)
    payload["signals"] = serializable_signals
    return payload


def extract_trend_keywords(trend_data: Optional[dict], max_items: int = 12) -> list:
    return extract_live_keywords(trend_data or {}, max_items=max_items)


def build_trend_summary(trend_data: Optional[dict], max_keywords: int = 10) -> str:
    if not isinstance(trend_data, dict):
        return "Canli trend verisi yok."
    return summarize_live_bundle(trend_data, max_keywords=max_keywords)


def load_latest_trend_data() -> Optional[dict]:
    if not TREND_JSON_PATH.exists():
        return None
    try:
        return json.loads(TREND_JSON_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Canli trend raporu okunamadi: {exc}")
        return None


def save_trend_reports(data: dict) -> Tuple[Path, Path]:
    serializable = _make_json_serializable(data)

    with open(TREND_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)

    lines = [
        "=== CANLI TREND VE VERI RAPORU ===",
        "",
        f"Cekim Zamani: {serializable.get('fetched_at', '')}",
        f"Kullanilan Kaynaklar: {', '.join(serializable.get('sources_used', [])) or 'Yok'}",
        "",
        "TREND OZETI",
        "-" * 60,
        build_trend_summary(serializable),
        "",
        "TOP ANAHTAR KELIMELER",
        "-" * 60,
    ]

    for keyword in extract_trend_keywords(serializable, max_items=15):
        lines.append(f"- {keyword}")

    lines.extend([
        "",
        "VIRAL BASLIKLAR / KONULAR",
        "-" * 60,
    ])
    for topic in serializable.get("viral_topics", [])[:12]:
        lines.append(f"- {topic}")

    lines.extend([
        "",
        "NOTLAR",
        "-" * 60,
    ])
    for note in serializable.get("notes", []):
        lines.append(f"- {note}")

    TREND_TXT_PATH.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    logger.info(f"Canli trend cache guncellendi: {TREND_TXT_PATH}")
    return TREND_JSON_PATH, TREND_TXT_PATH
