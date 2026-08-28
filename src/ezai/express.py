"""Express mode: native Ollama + OpenWebUI via uv. No Docker anywhere."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

from ezai import detect, proc
from ezai.config import Config
from ezai.detect import SystemInfo
from ezai.errors import FriendlyError
from ezai.models import ModelRef
from ezai.paths import logs_dir, pid_file

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


def webui_url(port: int) -> str:
    return f"http://localhost:{port}"


def webui_up(port: int, urlopen: Callable[..., Any] | None = None) -> bool:
    opener: Callable[..., Any] = urlopen if urlopen is not None else urllib.request.urlopen
    try:
        opener(f"http://127.0.0.1:{port}/health", timeout=1.0)
    except Exception:
        return False
    return True


def install_openwebui(run: RunFn = proc.run_logged) -> None:
    # `uv tool install` is idempotent: a second run is a no-op upgrade check.
    run(["uv", "tool", "install", "--python", "3.11", "open-webui"])


def start_openwebui(
    port: int,
    engine_url: str,
    popen: PopenFn = subprocess.Popen,
    environ: Mapping[str, str] | None = None,
) -> int:
    env = dict(environ if environ is not None else os.environ)
    env["OLLAMA_BASE_URL"] = engine_url
    log = (logs_dir() / "openwebui.log").open("ab")
    try:
        proc_handle = popen(
            [
                "uv",
                "tool",
                "run",
                "--from",
                "open-webui",
                "open-webui",
                "serve",
                "--port",
                str(port),
            ],
            env=env,
            stdout=log,
            stderr=log,
            **_detach_kwargs("windows" if os.name == "nt" else "posix"),
        )
    except FileNotFoundError as exc:
        log.close()
        raise FriendlyError(
            "uv is required to run OpenWebUI but was not found.",
            "Install uv: https://docs.astral.sh/uv/getting-started/installation/",
        ) from exc
    pid = int(proc_handle.pid)
    pid_file("openwebui").write_text(str(pid))
    return pid


def stop_openwebui(
    os_name: str,
    run: RunFn = proc.run_logged,
    kill: Callable[[int, int], None] = os.kill,
) -> bool:
    pf = pid_file("openwebui")
    if not pf.exists():
        return False
    try:
        pid = int(pf.read_text().strip())
    except ValueError:
        # Truncated or garbage pid file: nothing to stop, just clear the stale file.
        pf.unlink(missing_ok=True)
        return False
    if pid <= 0:
        # `os.kill(0, SIGTERM)` signals our whole process group, and negatives
        # signal a group too — neither is ever what a stale pid file meant.
        pf.unlink(missing_ok=True)
        return False
    try:
        if os_name == "windows":
            run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False)
        else:
            kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        # Gone, or recycled into a process we don't own (a truncated pid can
        # parse to a live foreign pid) — either way, drop the stale file.
        pass
    pf.unlink(missing_ok=True)
    return True


def ensure_openwebui(
    cfg: Config,
    run: RunFn = proc.run_logged,
    popen: PopenFn = subprocess.Popen,
    up: Callable[..., bool] = webui_up,
    sleep: SleepFn = time.sleep,
) -> None:
    # Probe first: an already-healthy OpenWebUI must cost nothing — no resolver
    # round-trip, and `ezai` keeps working offline.
    if up(cfg.webui_port):
        return
    install_openwebui(run=run)
    start_openwebui(cfg.webui_port, cfg.engine_url, popen=popen)
    wait_for(lambda: up(cfg.webui_port), 180, "OpenWebUI", sleep=sleep)
