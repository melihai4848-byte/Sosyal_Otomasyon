from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from topic_selection_engine.config import SourceSettings
from topic_selection_engine.models import RawSignal


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

    text = str(value or "").strip()
    if not text:
        return datetime.now(timezone.utc)

    for parser in (
        lambda v: datetime.fromisoformat(v.replace("Z", "+00:00")),
        lambda v: datetime.fromtimestamp(float(v), tz=timezone.utc),
    ):
        try:
            parsed = parser(text)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except Exception:
            continue

    return datetime.now(timezone.utc)


def _ensure_list_payload(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("items", "data", "rows", "records"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [payload]
        return []

    if suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if isinstance(item, dict):
                rows.append(item)
        return rows

    if suffix == ".csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]

    return []


def _build_text(record: Dict[str, Any], text_fields: List[str]) -> str:
    parts: List[str] = []
    for field in text_fields:
        value = record.get(field)
        if isinstance(value, list):
            parts.extend(str(item) for item in value if str(item).strip())
        elif value is not None and str(value).strip():
            parts.append(str(value))
    return "\n".join(parts).strip()


def load_signals_from_paths(source_name: str, settings: SourceSettings, limit: int) -> List[RawSignal]:
    signals: List[RawSignal] = []
    for raw_path in settings.input_paths:
        path = Path(raw_path)
        if not path.exists() or not path.is_file():
            continue

        for index, record in enumerate(_ensure_list_payload(path), start=1):
            title = str(record.get(settings.title_field, "")).strip()
            text = _build_text(record, settings.text_fields)
            if not title and not text:
                continue

            metadata = {}
            for key in settings.metadata_fields:
                if key in record:
                    metadata[key] = record[key]

            signals.append(
                RawSignal(
                    source=source_name,
                    source_id=str(record.get(settings.id_field, f"{path.stem}-{index}")),
                    title=title or text[:120],
                    text=text,
                    url=str(record.get(settings.url_field, "")),
                    published_at=_parse_datetime(record.get(settings.date_field)),
                    metadata=metadata,
                )
            )
            if len(signals) >= limit:
                return signals

    return signals
