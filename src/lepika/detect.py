"""Detect what machine we're on and what's already installed."""

from __future__ import annotations

import platform as _platform
from dataclasses import dataclass
from typing import Literal

from lepika import config, health
from lepika.errors import ISSUES_URL, FriendlyError
from lepika.system import System

OsName = Literal["macos", "linux", "windows"]
Gpu = Literal["apple", "nvidia", "none"]


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


def detect_gpu(os_name: OsName, arch: str, *, sys: System) -> Gpu:
    if os_name == "macos" and arch == "arm64":
        return "apple"
    if sys.which("nvidia-smi") is not None:
        return "nvidia"
    return "none"


_NVIDIA_SMI_MEMORY = ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]


def gpu_memory_gb(info: SystemInfo, *, sys: System) -> float:
    """Memory a model can be loaded into on the GPU, in GiB; 0.0 without one.

    Asked by the quant picker alone, never by `detect()`: on NVIDIA it is a
    subprocess (rule 6), and a wedged driver makes `nvidia-smi` hang — hence the
    timeout, which comes back as a FriendlyError and reads here as "unknown".
    Apple Silicon shares RAM with the GPU, and Metal caps a process's working set
    at two thirds of it below 36 GB and three quarters above — the number Ollama
    itself budgets with. GiB like RAM, so a card sold as 16 GB reads as 16; model
    files are sized in decimal GB, and comparing those against a GiB budget errs
    7 % on the safe side.
    """
    if info.gpu == "apple":
        return info.ram_gb * (0.75 if info.ram_gb >= 36 else 2 / 3)
    if info.gpu != "nvidia":
        return 0.0
    try:
        output = str(sys.run(_NVIDIA_SMI_MEMORY, check=False, log=False, timeout=15).stdout)
    except FriendlyError:
        return 0.0
    total_mib = 0
    for line in output.splitlines():
        try:
            total_mib += int(float(line.strip()))
        except ValueError:
            continue
    return total_mib / 1024


_DOCKER_MEMORY = ["docker", "info", "--format", "{{.MemTotal}}"]


def docker_memory_gb(*, sys: System) -> float:
    """Memory Docker's containers can use, in GiB; 0.0 when Docker will not say.

    On Linux that is the machine's own RAM. Docker Desktop (macOS, Windows) runs
    containers in a VM sized in its settings — often half the machine — and that is
    all Server mode's engine container can load a model into. A subprocess with a
    timeout, like `nvidia-smi`: a stopped daemon must read as "unknown", not hang.
    """
    try:
        output = str(sys.run(_DOCKER_MEMORY, check=False, log=False, timeout=15).stdout).strip()
    except FriendlyError:
        return 0.0
    return int(output) / 2**30 if output.isdecimal() else 0.0


def detect_ram_gb(*, sys: System) -> float:
    total_bytes = sys.ram_bytes()
    if total_bytes is None:
        # Guessing 0 GB would silently filter the curated list down to nothing.
        raise FriendlyError(
            "Could not detect system memory.",
            # Not `lepika doctor`: it detects memory as well, and would fail the same way.
            f"Re-run the command; if it keeps happening, report it at {ISSUES_URL}.",
        )
    return float(total_bytes) / 2**30


def detect(*, sys: System) -> SystemInfo:
    """The machine as LePika needs it, read without starting a process (rule 6)."""
    os_name = detect_os()
    arch = _platform.machine().lower()
    return SystemInfo(
        os=os_name,
        arch=arch,
        gpu=detect_gpu(os_name, arch, sys=sys),
        ram_gb=detect_ram_gb(sys=sys),
        has_docker=sys.which("docker") is not None,
        has_ollama=sys.which("ollama") is not None,
        ollama_running=health.api_up(config.DEFAULT_ENGINE_URL, sys=sys),
    )


_GPU_LABEL = {"apple": "Apple Metal GPU", "nvidia": "NVIDIA GPU", "none": "CPU only"}
_OS_LABEL = {"macos": "macOS", "linux": "Linux", "windows": "Windows"}


def plan_sentence(info: SystemInfo, mode: str = "express", engine: str = "Ollama") -> str:
    speed_note = "" if info.gpu != "none" else " (no GPU found — models will run slowly)"
    hardware = (
        f"Detected {_OS_LABEL[info.os]} on {info.arch} with {info.ram_gb:.0f} GB RAM "
        f"and {_GPU_LABEL[info.gpu]}{speed_note}"
    )
    if mode == "server":
        return f"{hardware} — Server mode: docker compose stack (OpenWebUI + {engine})."
    return f"{hardware} — Express mode: native Ollama + OpenWebUI, no Docker needed."
