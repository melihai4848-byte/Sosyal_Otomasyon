"""CLI entrypoint for the YouTube draft uploader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from uploader.config import load_uploader_config
from uploader.engine import DraftUploadEngine
from uploader.errors import UploaderError
from uploader.logging_utils import configure_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YouTube Draft Upload Engine")
    parser.add_argument("--config", help="Opsiyonel uploader config JSON yolu.")
    parser.add_argument("--folder", help="Tek bir hazir video klasoru yukle.")
    parser.add_argument(
        "--batch",
        nargs="?",
        const="__CONFIG__",
        help="Input altindaki tum hazir klasorleri yukle. Istersen ozel kok klasor de verebilirsin.",
    )
    parser.add_argument(
        "--watch",
        nargs="?",
        const="__CONFIG__",
        help="Watch mode. Istersen izlenecek kok klasoru ver.",
    )
    parser.add_argument("--dry-run", action="store_true", help="API cagrisi yapmadan plan dogrulama yap.")
    parser.add_argument("--force", action="store_true", help="Ayni klasor icin tekrar islem yapmaya zorla.")
    return parser


def _resolve_optional_root(raw_value: str | None, config_root: Path) -> Path:
    if not raw_value or raw_value == "__CONFIG__":
        return config_root
    return Path(raw_value).resolve()


def _summarize(outcomes) -> dict:
    payload = {
        "total": len(outcomes),
        "uploaded": 0,
        "failed": 0,
        "pending": 0,
        "items": [],
    }
    for outcome in outcomes:
        if outcome.status == "uploaded":
            payload["uploaded"] += 1
        elif outcome.status == "failed":
            payload["failed"] += 1
        else:
            payload["pending"] += 1
        payload["items"].append(
            {
                "job_dir": str(outcome.job_dir),
                "status": outcome.status,
                "video_id": outcome.video_id,
                "video_url": outcome.video_url,
                "warnings": outcome.warnings,
                "errors": outcome.errors,
            }
        )
    return payload


def run_from_args(argv: Iterable[str] | None = None) -> int:
    """Execute the uploader CLI."""
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        config = load_uploader_config(args.config)
        logger = configure_logging(config)
        engine = DraftUploadEngine(config, logger)

        if args.folder:
            outcome = engine.process_folder(Path(args.folder), dry_run=args.dry_run, force=args.force)
            print(json.dumps(_summarize([outcome]), ensure_ascii=False, indent=2))
            return 0 if outcome.status != "failed" else 1

        if args.batch is not None:
            root_dir = _resolve_optional_root(args.batch, config.input_root)
            outcomes = engine.process_batch(root_dir, dry_run=args.dry_run, force=args.force)
            print(json.dumps(_summarize(outcomes), ensure_ascii=False, indent=2))
            return 0 if not any(item.status == "failed" for item in outcomes) else 1

        if args.watch is not None:
            root_dir = _resolve_optional_root(args.watch, config.input_root)
            engine.watch(root_dir, dry_run=args.dry_run, force=args.force)
            return 0

        parser.error("En az bir mod secmelisin: --folder, --batch veya --watch")
        return 2
    except UploaderError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(run_from_args())
