"""Express mode: native Ollama + OpenWebUI via uv. No Docker anywhere."""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Callable
from typing import Any

from ezai import detect, proc
from ezai.detect import SystemInfo
from ezai.errors import FriendlyError
from ezai.models import ModelRef
from ezai.paths import logs_dir

RunFn = Callable[..., Any]
WhichFn = Callable[[str], str | None]
PopenFn = Callable[..., Any]
SleepFn = Callable[[float], None]
CallFn = Callable[[list[str]], int]

# Windows: DETACHED_PROCESS (0x8) | CREATE_NEW_PROCESS_GROUP (0x200)
_WINDOWS_DETACH_FLAGS = 0x00000208


def _detach_kwargs(os_name: str) -> dict[str, Any]:
    if os_name == "windows":
        return {"creationflags": _WINDOWS_DETACH_FLAGS}
    return {"start_new_session": True}


def install_ollama(
    info: SystemInfo,
    run: RunFn = proc.run_logged,
    which: WhichFn = shutil.which,
    call: CallFn = subprocess.call,
) -> None:
    if info.os == "macos":
        if which("brew") is not None:
            run(["brew", "install", "ollama"])
        else:
            raise FriendlyError(
                "Ollama is not installed and Homebrew was not found.",
                "Install Ollama from https://ollama.com/download/mac then run `ezai` again.",
            )
    elif info.os == "linux":
        # Streamed, not captured: the official script may prompt for sudo, and a
        # password prompt hidden behind captured output looks like a hang.
        code = call(["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"])
        if code != 0:
            raise FriendlyError(
                "Ollama installation failed.",
                "See https://ollama.com/download/linux for manual install steps.",
            )
    else:  # windows
        run(
            [
                "winget",
                "install",
                "--id",
                "Ollama.Ollama",
                "-e",
                "--accept-package-agreements",
                "--accept-source-agreements",
            ]
        )


def start_ollama(os_name: str, popen: PopenFn = subprocess.Popen) -> None:
    log = (logs_dir() / "ollama.log").open("ab")
    try:
        popen(["ollama", "serve"], stdout=log, stderr=log, **_detach_kwargs(os_name))
    except FileNotFoundError as exc:
        log.close()
        raise FriendlyError(
            "Ollama is installed but not on your PATH yet.",
            "Close this terminal, open a new one, and run `ezai` again.",
        ) from exc


def wait_for(
    predicate: Callable[[], bool],
    seconds: int,
    what: str,
    sleep: SleepFn = time.sleep,
) -> None:
    for _ in range(seconds):
        if predicate():
            return
        sleep(1)
    raise FriendlyError(
        f"{what} did not become ready within {seconds}s.",
        f"Check the logs in {logs_dir()} and run `ezai doctor`.",
    )


def ensure_ollama(
    info: SystemInfo,
    run: RunFn = proc.run_logged,
    which: WhichFn = shutil.which,
    popen: PopenFn = subprocess.Popen,
    api_up: Callable[..., bool] = detect.api_up,
    sleep: SleepFn = time.sleep,
    call: CallFn = subprocess.call,
) -> None:
    if not info.has_ollama:
        install_ollama(info, run=run, which=which, call=call)
    if not api_up(detect.OLLAMA_URL):
        start_ollama(info.os, popen=popen)
        wait_for(lambda: api_up(detect.OLLAMA_URL), 30, "Ollama API", sleep=sleep)


def pull_model(ref: ModelRef, call: CallFn = subprocess.call) -> None:
    # Streams ollama's own progress bar to the terminal on purpose.
    code = call(["ollama", "pull", ref.raw])
    if code != 0:
        raise FriendlyError(
            f"Failed to pull model '{ref.raw}'.",
            "Check the model name/URL — e.g. qwen3:8b or hf.co/<org>/<repo>-GGUF — "
            "and your internet connection.",
        )
