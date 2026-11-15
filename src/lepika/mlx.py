"""Ollama's MLX bundle, and where Ollama keeps its files."""

from __future__ import annotations

import os
import re
import shlex
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from lepika import log, proc
from lepika.detect import SystemInfo
from lepika.errors import FriendlyError

RunFn = Callable[..., Any]
WhichFn = Callable[[str], str | None]
CallFn = Callable[[list[str]], int]

MLX_BUNDLE = {
    "linux": "https://ollama.com/download/ollama-linux-amd64-mlx.tar.zst",
    "windows": "https://ollama.com/download/ollama-windows-amd64-mlx.zip",
}
_CUDA_VERSION = re.compile(r"CUDA Version:\s*(\d+)")


def ollama_install_dir(os_name: str, which: WhichFn = shutil.which) -> Path | None:
    """Where Ollama's `lib/ollama` lives: derived from the binary, as install.sh does.

    Linux: `/usr/local/bin/ollama` → `/usr/local` (`OLLAMA_INSTALL_DIR=$(dirname $BINDIR)`).
    Windows: `ollama.exe` sits next to `lib\\ollama` in `%LOCALAPPDATA%\\Programs\\Ollama`.
    """
    exe = which("ollama")
    if exe is None:
        return None
    binary = Path(exe).resolve()
    return binary.parent if os_name == "windows" else binary.parent.parent


# Where the official Linux install script points the systemd service's store — a
# system path, and frequently a different filesystem from the user's $HOME.
_SERVICE_STORE = Path("/usr/share/ollama/.ollama")


def ollama_store(
    environ: Mapping[str, str] | None = None,
    exists: Callable[[Path], bool] = lambda p: p.is_dir(),
) -> Path:
    """Where Ollama keeps the models it serves — the disk an import actually fills.

    Not the same disk as `~/.lepika` on a stock Linux service install, which is why
    the import's disk check asks both.
    """
    env = environ if environ is not None else os.environ
    configured = env.get("OLLAMA_MODELS", "")
    if configured:
        return Path(configured)
    if exists(_SERVICE_STORE):
        return _SERVICE_STORE
    return Path.home() / ".ollama"


def mlx_present(install_dir: Path) -> bool:
    return (install_dir / "lib" / "ollama" / "mlx_cuda_v13").is_dir()


def cuda_major(run: RunFn = proc.run_logged) -> int:
    """The driver's CUDA major from `nvidia-smi`'s header; 0 when there is none."""
    try:
        # Bounded: a wedged driver makes `nvidia-smi` hang forever, and this is the
        # last question asked before a multi-gigabyte download. The timeout comes back
        # as a FriendlyError, which reads here as "no CUDA" — the refusal below.
        output = str(run(["nvidia-smi"], check=False, log=False, timeout=15).stdout)
    except FriendlyError:
        return 0
    match = _CUDA_VERSION.search(output)
    return int(match.group(1)) if match else 0


def ensure_mlx(
    info: SystemInfo,
    which: WhichFn = shutil.which,
    run: RunFn = proc.run_logged,
    call: CallFn = proc.call,
    writable: Callable[[Path], bool] = lambda p: os.access(p, os.W_OK),
) -> None:
    """Make sure Ollama can run an imported (safetensors) model on this machine.

    macOS arm64 builds carry the MLX runner; Linux and Windows get it from a
    separate official bundle that neither install.sh nor OllamaSetup.exe installs.
    Only ever called on the import path — a GGUF user never downloads a gigabyte of
    CUDA libraries. No restart: the runner is a per-load subprocess that searches
    `lib/ollama` when it starts.
    """
    if info.os == "macos":
        return
    url = MLX_BUNDLE[info.os]
    install_dir = ollama_install_dir(info.os, which)
    if install_dir is None:
        # An engine can be answering while `which` still misses it: OllamaSetup.exe
        # puts the binary on the PATH of shells opened after it ran, not this one.
        # That is a stale PATH, not a broken install, and it has its own fix.
        raise FriendlyError(
            "Ollama is installed but not on your PATH yet.",
            "Close this terminal, open a new one, and try again.",
        )
    if not (install_dir / "lib" / "ollama").is_dir():
        # Name the directory that was probed: "not a standard install" is only
        # actionable if the user can see which path LePika looked at.
        raise FriendlyError(
            f"Ollama's MLX engine bundle is missing and {install_dir} is not a standard "
            "Ollama install.",
            f"Extract {url} over your Ollama install, then try again.",
        )
    # Where tar writes, and so what `writable` has to probe below.
    runners = install_dir / "lib" / "ollama"
    if mlx_present(install_dir):
        return
    major = cuda_major(run)
    if major < 13:
        raise FriendlyError(
            "Ollama's MLX engine needs an NVIDIA driver with CUDA 13 or newer "
            f"(yours reports {major or 'none'}).",
            "Update the driver from https://www.nvidia.com/drivers, then try again.",
        )
    if info.os == "linux":
        if which("zstd") is None:
            raise FriendlyError(
                "zstd is needed to unpack Ollama's MLX engine bundle.",
                "Install it (apt-get install zstd · dnf install zstd · pacman -S zstd) "
                "and try again.",
            )
        # The other two ends of the pipeline. A minimal container image has neither,
        # and `sh -c` would report only the exit status of the last stage — tar's.
        if which("curl") is None or which("tar") is None:
            raise FriendlyError(
                "Installing Ollama's MLX engine bundle needs curl and tar.",
                "Install them with your package manager (e.g. `sudo apt install curl tar`) "
                "and try again.",
            )
        # `lib/ollama`, not the install root: that is the directory tar writes into,
        # and it is the one whose permissions decide whether sudo is needed.
        sudo = "" if writable(runners) else "sudo "
        # Streamed, not captured: exactly how install.sh is run, so a sudo prompt shows —
        # and so a gigabyte of download can draw a progress bar (`-S` keeps curl's own
        # errors visible behind it).
        script = (
            f"curl -fSL --progress-bar {url} | zstd -d | "
            f"{sudo}tar -xf - -C {shlex.quote(str(install_dir))}"
        )
        cmd = ["sh", "-c", script]
    else:
        # PowerShell quoting: inside single quotes only `'` is special, and doubling
        # it escapes it — so C:\Users\O'Brien survives as 'C:\Users\O''Brien'.
        dest = str(install_dir).replace("'", "''")
        script = (
            # Windows PowerShell 5.1 renders a progress bar per chunk, which costs
            # more than the download for a 1 GB file; -UseBasicParsing keeps it off
            # the Internet Explorer engine, which may not be configured at all.
            "$ProgressPreference = 'SilentlyContinue'; "
            "$t = Join-Path $env:TEMP 'ollama-mlx.zip'; "
            f"Invoke-WebRequest -UseBasicParsing -Uri '{url}' -OutFile $t; "
            f"Expand-Archive -Path $t -DestinationPath '{dest}' -Force; Remove-Item $t"
        )
        cmd = ["powershell", "-NoProfile", "-Command", script]
    # Safe to hand a shell: the script and the URL are constants, and the one
    # interpolated value — a path from `which`, never user input — is quoted for the
    # shell it goes to (`shlex.quote` for sh, `'` doubled for PowerShell).
    code = call(cmd)
    logger = log.get_logger()
    if code != 0 or not mlx_present(install_dir):
        logger.warning("engine.mlx_install", dir=str(install_dir), result="failed")
        raise FriendlyError(
            "Installing Ollama's MLX engine bundle failed.",
            f"Extract {url} over {install_dir} yourself, then try again.",
        )
    logger.info("engine.mlx_install", dir=str(install_dir), result="success")
