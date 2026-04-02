import hashlib
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
REQUIREMENTS_STAMP = VENV_DIR / ".requirements.sha256"


def _print(message: str) -> None:
    print(message, flush=True)


def _is_running_inside_project_venv() -> bool:
    try:
        return Path(sys.executable).resolve() == VENV_PYTHON.resolve()
    except Exception:
        return False


def _run(command: list[str], *, cwd: Path | None = None) -> int:
    completed = subprocess.run(command, cwd=str(cwd or PROJECT_ROOT), check=False)
    return int(completed.returncode)


def _run_or_raise(command: list[str], *, cwd: Path | None = None, step_name: str = "") -> None:
    exit_code = _run(command, cwd=cwd)
    if exit_code != 0:
        label = step_name or "Komut"
        raise RuntimeError(f"{label} basarisiz oldu (kod={exit_code}).")


def _find_bootstrap_python() -> list[str]:
    candidates = [
        ["py", "-3.11"],
        ["py", "-3.10"],
        ["py"],
        ["python"],
    ]
    for candidate in candidates:
        try:
            completed = subprocess.run(
                [*candidate, "--version"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except Exception:
            continue
        if completed.returncode == 0:
            return candidate
    raise RuntimeError("Python bulunamadi. Lutfen Python 3.10+ kur.")


def _ensure_venv_exists() -> None:
    if VENV_PYTHON.exists():
        return
    bootstrap_python = _find_bootstrap_python()
    _print("[Launcher] .venv bulunamadi, olusturuluyor...")
    _run_or_raise(
        [*bootstrap_python, "-m", "venv", str(VENV_DIR)],
        step_name="Sanal ortam olusturma",
    )


def _ensure_venv_tools() -> None:
    _print("[Launcher] pip/setuptools/wheel kontrol ediliyor...")
    _run_or_raise(
        [str(VENV_PYTHON), "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"],
        step_name="Pip araclari kurulumu",
    )


def _requirements_hash() -> str:
    if not REQUIREMENTS_FILE.exists():
        return ""
    payload = REQUIREMENTS_FILE.read_bytes()
    return hashlib.sha256(payload).hexdigest()


def _normalize_requirement_name(requirement_line: str) -> str:
    cleaned = str(requirement_line or "").strip()
    if not cleaned or cleaned.startswith("#"):
        return ""
    if ";" in cleaned:
        cleaned = cleaned.split(";", 1)[0].strip()
    for marker in ("==", ">=", "<=", "~=", "!=", ">", "<"):
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0].strip()
            break
    if "[" in cleaned:
        cleaned = cleaned.split("[", 1)[0].strip()
    return cleaned.replace("_", "-").casefold()


def _required_packages() -> list[str]:
    packages: list[str] = []
    if not REQUIREMENTS_FILE.exists():
        return packages
    for line in REQUIREMENTS_FILE.read_text(encoding="utf-8").splitlines():
        package_name = _normalize_requirement_name(line)
        if package_name:
            packages.append(package_name)
    return packages


def _installed_packages() -> set[str]:
    installed = set()
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name") or ""
        normalized = name.replace("_", "-").casefold().strip()
        if normalized:
            installed.add(normalized)
    return installed


def _missing_required_packages() -> list[str]:
    installed = _installed_packages()
    return [package for package in _required_packages() if package not in installed]


def _requirements_are_current() -> bool:
    expected_hash = _requirements_hash()
    if not expected_hash:
        return True
    if not REQUIREMENTS_STAMP.exists():
        return False
    try:
        saved_hash = REQUIREMENTS_STAMP.read_text(encoding="utf-8").strip()
    except Exception:
        return False
    return saved_hash == expected_hash


def _write_requirements_stamp() -> None:
    REQUIREMENTS_STAMP.write_text(_requirements_hash(), encoding="utf-8")


def _ensure_requirements_installed() -> None:
    missing_packages = _missing_required_packages()
    if not missing_packages and _requirements_are_current():
        _print("[Launcher] Requirement kontrolu tamam, eksik paket yok.")
        return

    if missing_packages:
        _print(f"[Launcher] Eksik paketler bulundu: {', '.join(missing_packages)}")
    else:
        _print("[Launcher] requirements.txt degismis, bagimliliklar guncelleniyor...")

    _run_or_raise(
        [str(VENV_PYTHON), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)],
        step_name="Requirement kurulumu",
    )
    _write_requirements_stamp()


def _relaunch_inside_venv() -> None:
    command = [str(VENV_PYTHON), str(PROJECT_ROOT / "run_automation.py"), "--venv-runner"]
    exit_code = _run(command)
    raise SystemExit(exit_code)


def _launch_main() -> int:
    _print("[Launcher] Otomasyon baslatiliyor...")
    return _run([str(VENV_PYTHON), str(PROJECT_ROOT / "main.py")])


def main() -> int:
    os.chdir(PROJECT_ROOT)

    if "--venv-runner" not in sys.argv:
        _ensure_venv_exists()
        _relaunch_inside_venv()

    if not _is_running_inside_project_venv():
        raise RuntimeError("Launcher proje .venv icinde yeniden baslatilamadi.")

    _ensure_venv_tools()
    _ensure_requirements_installed()
    return _launch_main()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        _print("\n[Launcher] Kullanici tarafindan durduruldu.")
        raise SystemExit(130)
    except Exception as exc:
        _print(f"\n[Launcher] Hata: {exc}")
        raise SystemExit(1)
