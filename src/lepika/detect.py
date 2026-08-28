"""Detect what machine we're on and what's already installed."""

from __future__ import annotations

import platform as _platform
import shutil
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from lepika import proc
from lepika.errors import FriendlyError

OsName = Literal["macos", "linux", "windows"]
Gpu = Literal["apple", "nvidia", "none"]
RunFn = Callable[..., Any]
UrlOpenFn = Callable[..., Any]

OLLAMA_URL = "http://127.0.0.1:11434"


@dataclass(frozen=True)
class SystemInfo:
    os: OsName
    arch: str
    gpu: Gpu
    ram_gb: float
    has_docker: bool
    has_ollama: bool
    ollama_running: bool


def detect_os(system: str | None = None) -> OsName:
    s = (system or _platform.system()).lower()
    if s == "darwin":
        return "macos"
    if s == "linux":
        return "linux"
    if s == "windows":
        return "windows"
    raise FriendlyError(
        f"Unsupported operating system: {s}",
        "LePika supports macOS, Linux, and Windows.",
    )


def detect_gpu(
    os_name: OsName,
    arch: str,
    which: Callable[[str], str | None] = shutil.which,
) -> Gpu:
    if os_name == "macos" and arch == "arm64":
        return "apple"
    if which("nvidia-smi") is not None:
        return "nvidia"
    return "none"


def _global_memory_status() -> tuple[bool, int]:  # pragma: no cover - Windows only
    """Call GlobalMemoryStatusEx, returning (succeeded, total physical bytes)."""
    import ctypes

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MemoryStatusEx()
    stat.dwLength = ctypes.sizeof(MemoryStatusEx)
    ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
    return bool(ok), int(stat.ullTotalPhys)


def _windows_ram_gb(status: Callable[[], tuple[bool, int]] | None = None) -> float:
    # GlobalMemoryStatusEx signals failure through its BOOL return, leaving the
    # struct untouched. Ignoring it reported a 0 GB machine, which silently
    # filtered the curated list down to nothing.
    reader = status if status is not None else _global_memory_status
    ok, total_bytes = reader()
    if not ok:
        raise FriendlyError(
            "Could not detect system memory.",
            "Re-run `lepika`; if this persists, file an issue with `lepika doctor` output.",
        )
    return float(total_bytes) / 2**30


def detect_ram_gb(
    os_name: OsName,
    run: RunFn = proc.run_logged,
    meminfo: Path = Path("/proc/meminfo"),
) -> float:
    if os_name == "macos":
        # A pure read of the machine: nothing changed, so nothing is logged.
        result = run(["sysctl", "-n", "hw.memsize"], log=False)
        return float(int(result.stdout.strip())) / 2**30
    if os_name == "linux":
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                return float(kb) / 2**20
        return 0.0
    return _windows_ram_gb()


def api_up(url: str, timeout: float = 1.0, urlopen: UrlOpenFn | None = None) -> bool:
    opener: UrlOpenFn = urlopen if urlopen is not None else urllib.request.urlopen
    try:
        opener(f"{url}/api/version", timeout=timeout)
    except Exception:
        return False
    return True


def detect(
    which: Callable[[str], str | None] = shutil.which,
    run: RunFn = proc.run_logged,
    urlopen: UrlOpenFn | None = None,
) -> SystemInfo:
    os_name = detect_os()
    arch = _platform.machine().lower()
    return SystemInfo(
        os=os_name,
        arch=arch,
        gpu=detect_gpu(os_name, arch, which),
        ram_gb=detect_ram_gb(os_name, run),
        has_docker=which("docker") is not None,
        has_ollama=which("ollama") is not None,
        ollama_running=api_up(OLLAMA_URL, urlopen=urlopen),
    )


_GPU_LABEL = {"apple": "Apple Metal GPU", "nvidia": "NVIDIA GPU", "none": "CPU only"}
_OS_LABEL = {"macos": "macOS", "linux": "Linux", "windows": "Windows"}


def plan_sentence(info: SystemInfo) -> str:
    speed_note = "" if info.gpu != "none" else " (no GPU found — models will run slowly)"
    return (
        f"Detected {_OS_LABEL[info.os]} on {info.arch} with {info.ram_gb:.0f} GB RAM "
        f"and {_GPU_LABEL[info.gpu]}{speed_note} — Express mode: native Ollama + "
        f"OpenWebUI, no Docker needed."
    )
