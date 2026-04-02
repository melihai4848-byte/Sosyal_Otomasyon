import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache

from moduller.logger import get_logger

logger = get_logger("HardwareProfile")


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


def _cpu_name() -> str:
    cpu_name = _run_powershell_text("(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)")
    if cpu_name:
        return " ".join(cpu_name.split())
    return platform.processor() or ""


def _has_nvidia_cuda() -> bool:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False
    try:
        completed = subprocess.run(
            [nvidia_smi, "-L"],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
    except Exception:
        return False
    return completed.returncode == 0 and bool((completed.stdout or "").strip())


def _normalized_machine() -> str:
    candidates = [
        platform.machine(),
        os.getenv("PROCESSOR_ARCHITECTURE", ""),
        os.getenv("PROCESSOR_ARCHITEW6432", ""),
    ]
    for candidate in candidates:
        normalized = str(candidate or "").strip().lower()
        if normalized:
            return normalized
    return ""


@dataclass(frozen=True)
class DetectedHardware:
    system: str
    machine: str
    cpu_name: str
    has_cuda_gpu: bool
    is_windows: bool
    is_arm64: bool
    is_windows_arm64: bool
    is_probably_qualcomm: bool
    can_try_npu_stack: bool
    recommended_profile: str


@lru_cache(maxsize=1)
def detect_hardware() -> DetectedHardware:
    system = str(platform.system() or "").strip().lower()
    machine = _normalized_machine()
    cpu_name = _cpu_name()
    cpu_name_lower = cpu_name.casefold()
    is_windows = system == "windows"
    is_arm64 = machine in {"arm64", "aarch64"}
    is_windows_arm64 = is_windows and is_arm64
    has_cuda_gpu = _has_nvidia_cuda()
    is_probably_qualcomm = "snapdragon" in cpu_name_lower or "qualcomm" in cpu_name_lower
    can_try_npu_stack = is_windows_arm64

    if has_cuda_gpu:
        recommended_profile = "desktop_cuda"
    elif can_try_npu_stack:
        recommended_profile = "surface_arm_npu_ready"
    else:
        recommended_profile = "desktop_cpu"

    hardware = DetectedHardware(
        system=system or "unknown",
        machine=machine or "unknown",
        cpu_name=cpu_name or "unknown",
        has_cuda_gpu=has_cuda_gpu,
        is_windows=is_windows,
        is_arm64=is_arm64,
        is_windows_arm64=is_windows_arm64,
        is_probably_qualcomm=is_probably_qualcomm,
        can_try_npu_stack=can_try_npu_stack,
        recommended_profile=recommended_profile,
    )

    logger.info(
        "Donanim profili algilandi: "
        f"profile={hardware.recommended_profile} | system={hardware.system} | machine={hardware.machine} | "
        f"cpu={hardware.cpu_name} | cuda={hardware.has_cuda_gpu} | npu_adayi={hardware.can_try_npu_stack}"
    )
    return hardware
