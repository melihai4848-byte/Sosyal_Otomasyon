# main.py
import ctypes
import os
import platform
import subprocess
import sys
import time
import traceback
from importlib import import_module, reload
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from moduller.config import reload_project_env
from moduller.logger import get_logger
from moduller.module_registry import MENU_GROUPS, ModuleEntry, format_menu_label, iter_group_modules
from moduller.output_paths import cleanup_hidden_outputs
from moduller.runtime_utils import format_elapsed

main_logger = get_logger("AnaMenu")
AYIRICI = "-" * 72
YOUTUBE_UPLOADER_CLI_FLAGS = {
    "-h",
    "--help",
    "--folder",
    "--batch",
    "--watch",
    "--dry-run",
    "--force",
    "--config",
}
GROUP_MENU_CODES = {
    "subtitle": "100",
    "youtube": "200",
    "instagram": "300",
    "research": "400",
    "tools": "500",
}
_SYSTEM_SUMMARY_PRINTED = False


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _run_powershell_text(command: str) -> str:
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
    except Exception:
        return ""
    return (completed.stdout or "").strip()


def _format_gb_from_bytes(value: int | float | None) -> str:
    if not value:
        return "Bilinmiyor"
    try:
        gb_value = float(value) / (1024 ** 3)
    except Exception:
        return "Bilinmiyor"
    return f"{gb_value:.1f} GB"


def _format_gb_from_mb(value: int | float | None) -> str:
    if not value:
        return "Bilinmiyor"
    try:
        gb_value = float(value) / 1024
    except Exception:
        return "Bilinmiyor"
    return f"{gb_value:.1f} GB"


def _get_total_ram_bytes() -> int:
    try:
        memory_status = _MemoryStatusEx()
        memory_status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status)):
            return int(memory_status.ullTotalPhys)
    except Exception:
        pass
    return 0


def _get_cpu_name() -> str:
    cpu_name = _run_powershell_text("(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)")
    if cpu_name:
        return " ".join(cpu_name.split())
    return platform.processor() or "Bilinmiyor"


def _get_gpu_summaries() -> list[str]:
    nvidia_raw = ""
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
        nvidia_raw = (completed.stdout or "").strip()
    except Exception:
        nvidia_raw = ""

    nvidia_results: list[str] = []
    nvidia_seen = set()
    for line in nvidia_raw.splitlines():
        cleaned = str(line).strip()
        if not cleaned or "," not in cleaned:
            continue
        name_raw, memory_raw = [part.strip() for part in cleaned.split(",", 1)]
        name = " ".join(name_raw.split()).strip()
        if not name:
            continue
        vram_text = _format_gb_from_mb(memory_raw)
        label = f"{name} ({vram_text} VRAM)" if vram_text != "Bilinmiyor" else name
        key = name.casefold()
        if key in nvidia_seen:
            continue
        nvidia_seen.add(key)
        nvidia_results.append(label)

    raw = _run_powershell_text(
        "Get-CimInstance Win32_VideoController | ForEach-Object { \"$($_.Name)||$($_.AdapterRAM)\" }"
    )
    if not raw:
        return nvidia_results
    seen = set()
    results: list[str] = []
    for line in raw.splitlines():
        cleaned = str(line).strip()
        if not cleaned:
            continue
        name_raw, _, vram_raw = cleaned.partition("||")
        name = " ".join(name_raw.split()).strip()
        if not name:
            continue
        if "nvidia" in name.casefold() and nvidia_results:
            continue
        vram_text = _format_gb_from_bytes(vram_raw.strip())
        label = f"{name} ({vram_text} VRAM)" if vram_text != "Bilinmiyor" else name
        key = label.casefold()
        if key in seen:
            continue
        seen.add(key)
        results.append(label)
    return nvidia_results + results


def print_system_summary() -> None:
    cpu_name = _get_cpu_name()
    cpu_threads = os.cpu_count() or 0
    total_ram = _format_gb_from_bytes(_get_total_ram_bytes())
    gpu_summaries = _get_gpu_summaries()
    gpu_text = ", ".join(gpu_summaries) if gpu_summaries else "GPU bilgisi bulunamadi"

    print(AYIRICI)
    print("SISTEM OZETI")
    print(AYIRICI)
    print(f"CPU: {cpu_name}")
    print(f"CPU Thread: {cpu_threads or 'Bilinmiyor'}")
    print(f"RAM: {total_ram}")
    print(f"GPU: {gpu_text}")
    print(AYIRICI)
    print("")


def _load_runtime_module(modul_yolu: str):
    son_parca = str(modul_yolu or "").split(".")[-1]
    if son_parca and son_parca[0].isdigit():
        hedef = Path(__file__).resolve().parent / "moduller" / f"{son_parca}.py"
        spec = spec_from_file_location(modul_yolu, hedef)
        if spec is None or spec.loader is None:
            raise ImportError(f"Modul yuklenemedi: {modul_yolu}")
        modul = module_from_spec(spec)
        sys.modules[modul_yolu] = modul
        spec.loader.exec_module(modul)
        return modul
    if modul_yolu in sys.modules:
        return reload(sys.modules[modul_yolu])
    return import_module(modul_yolu)


def print_terminal_block(baslik: str, emoji: str = "🚀", detay: str = ""):
    print("\n" * 2, end="")
    print(AYIRICI)
    print(f"{emoji} {baslik}")
    if detay:
        print(detay)
    print(AYIRICI)
    print("\n" * 2, end="")


def print_module_result_summary(module_result):
    if not isinstance(module_result, dict):
        return

    provider_summary = module_result.get("provider_summary")
    if not isinstance(provider_summary, dict) or not provider_summary:
        return

    print(AYIRICI)
    print("PROVIDER OZETI")
    print(AYIRICI)
    for provider_name, provider_data in provider_summary.items():
        if not isinstance(provider_data, dict):
            continue
        print(
            f"{provider_name}: secildi={provider_data.get('chosen_count', 0)}, "
            f"indirilen_dosya={provider_data.get('downloaded_file_count', 0)}, "
            f"aday={provider_data.get('candidate_count', 0)}, hata={provider_data.get('error_count', 0)}"
        )
    print(AYIRICI)


def run_module(modul_yolu: str, acilis_mesaji: str, fonksiyon_adi: str = "run", *args, **kwargs):
    print_terminal_block("MODUL BASLATILIYOR", "🚀")
    if acilis_mesaji:
        main_logger.info(acilis_mesaji)

    started_at = time.perf_counter()
    module_result = None
    try:
        reload_project_env(override=True)
        modul = _load_runtime_module(modul_yolu)
        calistir_fonksiyonu = getattr(modul, fonksiyon_adi)
        module_result = calistir_fonksiyonu(*args, **kwargs)
        print_terminal_block("MODUL TAMAMLANDI", "🎉", f"{modul_yolu} islemi tamamlandi.")
    except Exception as e:
        print_terminal_block("MODUL HATA ILE DURDU", "❌", f"{modul_yolu} calistirilirken hata olustu: {e}")
        main_logger.error(f"{modul_yolu} çalıştırılırken hata oluştu: {e}")
        main_logger.error(traceback.format_exc())
    finally:
        elapsed = format_elapsed(time.perf_counter() - started_at)
        main_logger.info(f"⏱️ {modul_yolu} modul suresi: {elapsed}")
        print_module_result_summary(module_result)
        cleanup_hidden_outputs()


def prompt_continue_or_exit() -> bool:
    print("\n" + AYIRICI)
    print("Uretime devam etmek istiyor musunuz?")
    print("[1] Ana menuye don ve devam et")
    print("[0] Cikis")
    print(AYIRICI)
    while True:
        secim = input("👉 Seciminiz (1 veya 0): ").strip()
        if secim == "1":
            return True
        if secim == "0":
            return False
        print("Lutfen sadece 1 veya 0 girin.")


def _run_entry(entry: ModuleEntry):
    run_module(
        entry.manual_module_path or entry.module_path,
        entry.manual_launch_message or entry.launch_message,
        entry.manual_run_function or entry.run_function,
    )


def _run_entries(entries: list[ModuleEntry]):
    if not entries:
        return

    if len(entries) == 1:
        _run_entry(entries[0])
        return

    if all(entry.pipeline_enabled and entry.group in {"subtitle", "youtube", "instagram"} for entry in entries):
        run_module(
            "moduller.full_automation",
            "",
            "run",
            preselected_selection=",".join(entry.number for entry in entries),
        )
        return

    for entry in entries:
        _run_entry(entry)


def _get_group_menu_entries() -> list[tuple[str, str, str, list[ModuleEntry]]]:
    groups: list[tuple[str, str, str, list[ModuleEntry]]] = []
    for group_key, group_title in MENU_GROUPS:
        entries = iter_group_modules(group_key)
        if entries:
            groups.append((GROUP_MENU_CODES.get(group_key, group_key), group_key, group_title, entries))
    return groups


def print_group_menu(group_entries: list[tuple[str, str, str, list[ModuleEntry]]]):
    print("\n" + "=" * 50)
    print("🎬 SOSYAL MEDYA OTOMASYON MERKEZİ 🎬")
    print("🎬 (Powered by Hectoor) 🎬")
    print("=" * 50)
    for option, _group_key, group_title, entries in group_entries:
        numara_araligi = entries[0].number if len(entries) == 1 else f"{entries[0].number}-{entries[-1].number}"
        print(f"[{option}] {group_title} ({len(entries)} modul, {numara_araligi})")
    print("")
    print("Ipucu: Once grup sec. Bos birakirsan 1xx + 2xx + 3xx pipeline'i calisir.")
    print("[0] 👋 Çıkış 👋")
    print("=" * 50)


def print_group_modules(group_title: str, entries: list[ModuleEntry]):
    print("\n" + "=" * 50)
    print(group_title)
    print("=" * 50)
    for entry in entries:
        print(f"[{entry.number}] {format_menu_label(entry)} - {entry.description}")
    print("")
    print("Ipucu: Tekli icin bir numara yaz. Ayni gruptan coklu icin virgulle yaz. Bos birakirsan bu gruptaki tum moduller calisir.")
    print("[0] Geri don")
    print("=" * 50)


def _resolve_group_module_selection(selection: str, entries: list[ModuleEntry]) -> list[ModuleEntry] | None:
    secilenler = [parca.strip() for parca in selection.split(",") if parca.strip()]
    if not secilenler:
        return None

    mevcutlar = {entry.number: entry for entry in entries}
    sonuc: list[ModuleEntry] = []
    for numara in secilenler:
        entry = mevcutlar.get(numara)
        if entry is None:
            return None
        sonuc.append(entry)
    return sonuc


def main():
    global _SYSTEM_SUMMARY_PRINTED
    cleanup_hidden_outputs()
    if not _SYSTEM_SUMMARY_PRINTED:
        print_system_summary()
        _SYSTEM_SUMMARY_PRINTED = True
    while True:
        grup_menusu = _get_group_menu_entries()
        grup_haritasi = {option: (group_key, group_title, entries) for option, group_key, group_title, entries in grup_menusu}
        print_group_menu(grup_menusu)
        secim = input("\nLutfen once bir grup secin: ").strip()

        if secim == "0":
            main_logger.info("Sistem kapatılıyor. İyi çalışmalar!")
            sys.exit(0)

        if not secim:
            run_module(
                "moduller.full_automation",
                "",
                "run",
                preselected_selection="",
            )
            if not prompt_continue_or_exit():
                main_logger.info("Sistem kapatılıyor. İyi çalışmalar!")
                sys.exit(0)
            continue

        grup_bilgisi = grup_haritasi.get(secim)
        if grup_bilgisi is None:
            main_logger.warning("Gecersiz secim! Lutfen once gecerli bir grup numarasi girin.")
            continue

        _group_key, group_title, group_entries = grup_bilgisi
        while True:
            print_group_modules(group_title, group_entries)
            modul_secimi = input("\nBu gruptan calistirmak istediginiz modulu secin: ").strip()

            if not modul_secimi:
                _run_entries(group_entries)
                if not prompt_continue_or_exit():
                    main_logger.info("Sistem kapatılıyor. İyi çalışmalar!")
                    sys.exit(0)
                break

            if modul_secimi.lower() == "b" or modul_secimi == "0":
                break

            secilen_moduller = _resolve_group_module_selection(modul_secimi, group_entries)
            if not secilen_moduller:
                main_logger.warning("Gecersiz secim! Lutfen sadece bu gruptaki modul numaralarini girin.")
                continue

            _run_entries(secilen_moduller)
            if not prompt_continue_or_exit():
                main_logger.info("Sistem kapatılıyor. İyi çalışmalar!")
                sys.exit(0)
            break


def should_switch_to_cli_mode(argv: list[str]) -> bool:
    return any(arg.split("=")[0] in YOUTUBE_UPLOADER_CLI_FLAGS for arg in argv)


if __name__ == "__main__":
    try:
        if should_switch_to_cli_mode(sys.argv[1:]):
            main_cli = getattr(_load_runtime_module("moduller.502_youtube_draft_upload_engine"), "main_cli")
            sys.exit(main_cli(sys.argv[1:]))
        main()
    except KeyboardInterrupt:
        main_logger.info("\nKullanıcı tarafından zorla durduruldu.")
        sys.exit(0)
