from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from moduller.config import OUTPUTS_DIR
from moduller.logger import get_logger
from moduller.output_paths import GROUP_DIR_NAMES, ROOT_JSON_CACHE_DIR, output_group_dir
from moduller.project_paths import LOGS_DIR, STATE_DIR, UPLOADER_DIR

logger = get_logger("output_cleaner")
AYIRICI = "=" * 60
UPLOAD_RECEIPT_NAME = "youtube_upload_result.json"
PROFILE_OPTIONS: list[tuple[str, str, str]] = [
    ("1", "full", "Tam temizlik"),
    ("2", "subtitle", "Sadece 100_Altyazi"),
    ("3", "youtube", "Sadece 200_YouTube"),
    ("4", "instagram", "Sadece 300_Instagram"),
    ("5", "research", "Sadece 400_Arastirma_Sonuclari"),
    ("6", "tools", "Sadece 500_Araclar_Sonuclari"),
    ("7", "cache_only", "Sadece JSON cache klasorleri"),
    ("8", "runtime_only", "Sadece workspace runtime"),
]
PROFILE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "full": {
        "output_groups": list(GROUP_DIR_NAMES.keys()),
        "clear_root_json_cache": True,
        "clear_state": True,
        "clear_logs": True,
        "clear_upload_receipts": True,
        "label": "Tam temizlik",
    },
    "subtitle": {
        "output_groups": ["subtitle"],
        "clear_root_json_cache": False,
        "clear_state": False,
        "clear_logs": False,
        "clear_upload_receipts": False,
        "label": "100_Altyazi",
    },
    "youtube": {
        "output_groups": ["youtube"],
        "clear_root_json_cache": False,
        "clear_state": False,
        "clear_logs": False,
        "clear_upload_receipts": False,
        "label": "200_YouTube",
    },
    "instagram": {
        "output_groups": ["instagram"],
        "clear_root_json_cache": False,
        "clear_state": False,
        "clear_logs": False,
        "clear_upload_receipts": False,
        "label": "300_Instagram",
    },
    "research": {
        "output_groups": ["research"],
        "clear_root_json_cache": False,
        "clear_state": False,
        "clear_logs": False,
        "clear_upload_receipts": False,
        "label": "400_Arastirma_Sonuclari",
    },
    "tools": {
        "output_groups": ["tools"],
        "clear_root_json_cache": False,
        "clear_state": False,
        "clear_logs": False,
        "clear_upload_receipts": False,
        "label": "500_Araclar_Sonuclari",
    },
    "cache_only": {
        "output_groups": [],
        "clear_root_json_cache": True,
        "clear_group_json_caches": True,
        "clear_state": False,
        "clear_logs": False,
        "clear_upload_receipts": False,
        "label": "JSON cache",
    },
    "runtime_only": {
        "output_groups": [],
        "clear_root_json_cache": False,
        "clear_state": True,
        "clear_logs": True,
        "clear_upload_receipts": True,
        "label": "Runtime dosyalari",
    },
}


def _select_cleanup_profile() -> str:
    print("\n" + AYIRICI)
    print("CIKTI TEMIZLEYICI")
    print(AYIRICI)
    print("Temizlik profili secin:")
    for option, profile_key, label in PROFILE_OPTIONS:
        print(f"[{option}] {label} ({profile_key})")
    print(AYIRICI)
    raw = input("👉 Profil (bos = tam temizlik, 0 = iptal): ").strip()
    if raw == "0":
        return ""
    if not raw:
        return "full"
    for option, profile_key, _label in PROFILE_OPTIONS:
        if raw == option or raw.lower() == profile_key:
            return profile_key
    logger.warning("Gecersiz profil secimi yapildi. Tam temizlik kullanilacak.")
    return "full"


def _confirm_cleanup(profile_key: str) -> bool:
    profile = PROFILE_DEFINITIONS[profile_key]
    output_groups = profile.get("output_groups", [])
    clear_group_json_caches = bool(profile.get("clear_group_json_caches", False))

    print("\n" + AYIRICI)
    print("TEMIZLIK ONAYI")
    print(AYIRICI)
    print(f"Profil: {profile.get('label', profile_key)}")
    print("")
    if output_groups:
        print("- Temizlenecek output gruplari:")
        for group in output_groups:
            print(f"  * {GROUP_DIR_NAMES[group]}")
    if clear_group_json_caches:
        print("- Eski grup _json_cache klasorleri")
    if profile.get("clear_root_json_cache"):
        print("- Kok _json_cache klasoru")
    if profile.get("clear_state"):
        print("- workspace/state altindaki uploader state ve lock dosyalari")
    if profile.get("clear_logs"):
        print("- workspace/logs altindaki yerel log dosyalari")
    if profile.get("clear_upload_receipts"):
        print("- workspace/uploader altindaki youtube_upload_result.json dosyalari")
    print("")
    print("Korunanlar: workspace/00_Inputs, oauth/token dosyalari, kullanici kaynak videolari,")
    print("upload klasorlerinin asil icerigi ve diger repo dosyalari.")
    print(AYIRICI)
    yanit = input("👉 Temizlik baslasin mi? [E/h]: ").strip().lower()
    return yanit in {"", "e", "evet", "y", "yes"}


def _remove_path(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_dir():
        shutil.rmtree(path)
        return 1
    path.unlink()
    return 1


def _clear_directory_contents(path: Path) -> tuple[int, list[str], list[str]]:
    cleaned = 0
    warnings: list[str] = []
    cleaned_items: list[str] = []

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return cleaned, warnings, cleaned_items

    for child in list(path.iterdir()):
        try:
            cleaned += _remove_path(child)
            cleaned_items.append(str(child))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{child}: {exc}")

    path.mkdir(parents=True, exist_ok=True)
    return cleaned, warnings, cleaned_items


def _truncate_active_log_file(path: Path) -> bool:
    resolved = str(path.resolve())
    logger_candidates = [
        logger,
        logging.getLogger("Ana Menu"),
        logging.getLogger("YouTubeDraftUploader"),
    ]

    for current_logger in logger_candidates:
        for handler in current_logger.handlers:
            base_filename = getattr(handler, "baseFilename", "")
            if not base_filename:
                continue
            try:
                if str(Path(base_filename).resolve()) != resolved:
                    continue
            except OSError:
                continue

            handler.acquire()
            try:
                stream = getattr(handler, "stream", None)
                if stream is None and hasattr(handler, "_open"):
                    stream = handler._open()
                    handler.stream = stream
                if stream is None:
                    return False
                handler.flush()
                stream.seek(0)
                stream.truncate()
                return True
            finally:
                handler.release()

    return False


def _clear_log_outputs(log_dir: Path) -> tuple[int, list[str], list[str]]:
    cleaned = 0
    warnings: list[str] = []
    cleaned_items: list[str] = []

    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
        return cleaned, warnings, cleaned_items

    for child in list(log_dir.iterdir()):
        try:
            if child.is_dir():
                cleaned += _remove_path(child)
                cleaned_items.append(str(child))
                continue

            if _truncate_active_log_file(child):
                cleaned += 1
                cleaned_items.append(f"{child} (truncate)")
                continue

            cleaned += _remove_path(child)
            cleaned_items.append(str(child))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{child}: {exc}")

    log_dir.mkdir(parents=True, exist_ok=True)
    return cleaned, warnings, cleaned_items


def _clear_upload_receipts(*roots: Path) -> tuple[int, list[str], list[str]]:
    cleaned = 0
    warnings: list[str] = []
    cleaned_items: list[str] = []

    for root in roots:
        if not root.exists():
            continue
        for receipt_path in root.rglob(UPLOAD_RECEIPT_NAME):
            try:
                cleaned += _remove_path(receipt_path)
                cleaned_items.append(str(receipt_path))
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"{receipt_path}: {exc}")

    return cleaned, warnings, cleaned_items


def _group_output_path(group: str) -> Path:
    return OUTPUTS_DIR / GROUP_DIR_NAMES[group]


def _group_json_cache_path(group: str) -> Path:
    return _group_output_path(group) / "_json_cache"


def _ensure_output_structure(profile_key: str) -> None:
    profile = PROFILE_DEFINITIONS[profile_key]
    target_groups = list(profile.get("output_groups", []))

    if profile_key == "full":
        target_groups = list(GROUP_DIR_NAMES.keys())

    for group in target_groups:
        output_group_dir(group)

    if profile.get("clear_group_json_caches"):
        for group in GROUP_DIR_NAMES:
            output_group_dir(group)

    if profile.get("clear_root_json_cache"):
        ROOT_JSON_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _run_cleanup_plan(profile_key: str) -> tuple[dict[str, int], list[str], dict[str, list[str]]]:
    profile = PROFILE_DEFINITIONS[profile_key]
    warnings: list[str] = []
    details = {
        "outputs": 0,
        "json_caches": 0,
        "root_json_cache": 0,
        "state": 0,
        "logs": 0,
        "upload_receipts": 0,
    }
    cleaned_items = {
        "outputs": [],
        "json_caches": [],
        "root_json_cache": [],
        "state": [],
        "logs": [],
        "upload_receipts": [],
    }

    for group in profile.get("output_groups", []):
        cleaned, current_warnings, current_items = _clear_directory_contents(_group_output_path(group))
        details["outputs"] += cleaned
        warnings.extend(current_warnings)
        cleaned_items["outputs"].extend(current_items)

    if profile.get("clear_group_json_caches"):
        for group in GROUP_DIR_NAMES:
            cleaned, current_warnings, current_items = _clear_directory_contents(_group_json_cache_path(group))
            details["json_caches"] += cleaned
            warnings.extend(current_warnings)
            cleaned_items["json_caches"].extend(current_items)

    if profile.get("clear_root_json_cache"):
        cleaned, current_warnings, current_items = _clear_directory_contents(ROOT_JSON_CACHE_DIR)
        details["root_json_cache"] += cleaned
        warnings.extend(current_warnings)
        cleaned_items["root_json_cache"].extend(current_items)

    if profile.get("clear_state"):
        cleaned, current_warnings, current_items = _clear_directory_contents(STATE_DIR)
        details["state"] += cleaned
        warnings.extend(current_warnings)
        cleaned_items["state"].extend(current_items)

    if profile.get("clear_logs"):
        cleaned, current_warnings, current_items = _clear_log_outputs(LOGS_DIR)
        details["logs"] += cleaned
        warnings.extend(current_warnings)
        cleaned_items["logs"].extend(current_items)

    if profile.get("clear_upload_receipts"):
        cleaned, current_warnings, current_items = _clear_upload_receipts(
            UPLOADER_DIR / "input",
            UPLOADER_DIR / "success",
            UPLOADER_DIR / "failed",
        )
        details["upload_receipts"] += cleaned
        warnings.extend(current_warnings)
        cleaned_items["upload_receipts"].extend(current_items)

    _ensure_output_structure(profile_key)

    return details, warnings, cleaned_items


def run() -> dict:
    profile_key = _select_cleanup_profile()
    if not profile_key:
        print("Temizlik iptal edildi.")
        return {
            "status": "cancelled",
            "cleaned_items": 0,
            "warnings": [],
        }

    if not _confirm_cleanup(profile_key):
        print("Temizlik iptal edildi.")
        return {
            "status": "cancelled",
            "cleaned_items": 0,
            "warnings": [],
            "profile": profile_key,
        }

    logger.info(
        f"Cikti temizligi baslatildi. profil={profile_key}"
    )

    details, warnings, cleaned_items = _run_cleanup_plan(profile_key)
    cleaned_total = sum(details.values())

    print("\n" + AYIRICI)
    print("TEMIZLIK OZETI")
    print(AYIRICI)
    print(f"Profil:                    {PROFILE_DEFINITIONS[profile_key].get('label', profile_key)}")
    print(f"Output temizlenen oge:     {details['outputs']}")
    print(f"Legacy JSON cache:         {details['json_caches']}")
    print(f"Kok _json_cache:           {details['root_json_cache']}")
    print(f"state temizlenen oge:      {details['state']}")
    print(f"logs temizlenen oge:       {details['logs']}")
    print(f"upload receipt temizligi:  {details['upload_receipts']}")
    print(f"Toplam temizlenen oge:     {cleaned_total}")

    flat_cleaned_items = [
        item
        for key in ("outputs", "json_caches", "root_json_cache", "state", "logs", "upload_receipts")
        for item in cleaned_items[key]
    ]
    if flat_cleaned_items:
        print("")
        print("Temizlenen ogeler:")
        for item in flat_cleaned_items:
            print(f"- {item}")

    if warnings:
        print("")
        print("Uyarilar:")
        for item in warnings[:10]:
            print(f"- {item}")
        if len(warnings) > 10:
            print(f"- ... {len(warnings) - 10} ek uyari daha var")
    print(AYIRICI)

    logger.info(
        f"Cikti temizligi tamamlandi. profil={profile_key}, temizlenen toplam oge={cleaned_total}"
    )
    if warnings:
        logger.warning(f"Temizlik sirasinda {len(warnings)} uyari olustu.")

    return {
        "status": "completed_with_warnings" if warnings else "completed",
        "cleaned_items": cleaned_total,
        "warnings": warnings,
        "profile": profile_key,
        "details": {
            **details,
        },
        "cleaned_paths": flat_cleaned_items,
    }
