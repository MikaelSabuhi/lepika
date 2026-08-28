"""Express mode: native Ollama + OpenWebUI via uv. No Docker anywhere."""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

from lepika import detect, proc
from lepika.config import Config, config_path
from lepika.detect import SystemInfo
from lepika.errors import FriendlyError
from lepika.models import ModelRef
from lepika.paths import logs_dir, pid_file

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
                "Install Ollama from https://ollama.com/download/mac then run `lepika` again.",
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
            "Close this terminal, open a new one, and run `lepika` again.",
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
        f"Check the logs in {logs_dir()} and run `lepika doctor`.",
    )


def ensure_ollama(
    info: SystemInfo,
    run: RunFn = proc.run_logged,
    which: WhichFn = shutil.which,
    popen: PopenFn = subprocess.Popen,
    api_up: Callable[..., bool] = detect.api_up,
    sleep: SleepFn = time.sleep,
    call: CallFn = subprocess.call,
    url: str = detect.OLLAMA_URL,
) -> None:
    if not info.has_ollama:
        install_ollama(info, run=run, which=which, call=call)
    if not api_up(url):
        start_ollama(info.os, popen=popen)
        wait_for(lambda: api_up(url), 30, "Ollama API", sleep=sleep)


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
    # `uv tool install` is idempotent: with open-webui already installed it is a
    # no-op that does NOT upgrade — upgrading is `lepika update`'s job.
    run(["uv", "tool", "install", "--python", "3.11", "open-webui"])


# Only defined on Windows, so it is looked up rather than referenced.
_WINDOWS_EXCLUSIVE_ADDR = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)

# A server can hold a port on the wildcard address or on loopback alone, and on
# macOS a loopback bind succeeds straight over an existing 0.0.0.0 listener. Both
# are probed, because either one is a real conflict.
# B104 suppressed: these are bind probes, closed immediately without listening.
# Detecting a conflict on the wildcard address requires naming it.
_PROBE_ADDRESSES = ("0.0.0.0", "127.0.0.1")  # nosec B104


def _bind_address(address: str, port: int) -> bool:
    """Try to bind one address, reporting whether the port was available there."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if os.name == "nt" and _WINDOWS_EXCLUSIVE_ADDR is not None:
            # Windows' SO_REUSEADDR lets two sockets own the same port outright,
            # which would hide every conflict. SO_EXCLUSIVEADDRUSE asks for the
            # exclusivity that POSIX gives by default.
            sock.setsockopt(socket.SOL_SOCKET, _WINDOWS_EXCLUSIVE_ADDR, 1)
        else:
            # SO_REUSEADDR matches what the OpenWebUI server itself sets, so a port
            # left in TIME_WAIT by the previous run reads as free here exactly as
            # it would there.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((address, port))
        except OSError:
            return False
    return True


def _bind_once(
    port: int,
    bind_address: Callable[[str, int], bool] = _bind_address,
) -> bool:
    """Is `port` bindable on every address a server might already be holding?"""
    # all() short-circuits: one taken address is enough to call the port busy.
    return all(bind_address(address, port) for address in _PROBE_ADDRESSES)


def port_free(
    port: int,
    bind: Callable[[int], bool] = _bind_once,
    sleep: SleepFn = time.sleep,
    attempts: int = 3,
) -> bool:
    """Can a server bind this port, or is another application already on it?

    Retried, because `lepika update` asks this moments after stopping the server
    that held the port: an OS that has not finished releasing the listening
    socket would otherwise be reported as a port conflict. A free port answers on
    the first try, so only the failing case pays for the wait.
    """
    for attempt in range(attempts):
        if bind(port):
            return True
        if attempt < attempts - 1:
            sleep(0.5)
    return False


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
    port: int | None = None,
    up: Callable[..., bool] | None = None,
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
    if port is not None:
        up_fn = up if up is not None else webui_up
        if not up_fn(port):
            # Nothing is answering on our port, so the recorded pid is not our
            # OpenWebUI — after a reboot the OS will have handed that number to an
            # unrelated process. A stale pid file is never a licence to signal it.
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
    bind_check: Callable[[int], bool] | None = None,
) -> None:
    # Probe first: an already-healthy OpenWebUI must cost nothing — no resolver
    # round-trip, and `lepika` keeps working offline.
    if up(cfg.webui_port):
        return
    # The port is not answering /health, but something else may still own it.
    # Caught here it is one clear sentence; caught later it is a 180s wait that
    # ends in "OpenWebUI did not become ready".
    free = bind_check if bind_check is not None else port_free
    if not free(cfg.webui_port):
        raise FriendlyError(
            f"Port {cfg.webui_port} is in use by another application.",
            f"Change webui_port in {config_path()} and run `lepika up` again.",
        )
    install_openwebui(run=run)
    start_openwebui(cfg.webui_port, cfg.engine_url, popen=popen)
    wait_for(lambda: up(cfg.webui_port), 180, "OpenWebUI", sleep=sleep)


def wait_until_down(
    port: int,
    up: Callable[..., bool],
    attempts: int = 30,
    sleep: SleepFn = time.sleep,
) -> None:
    """Block until nothing answers on `port`.

    A SIGTERMed OpenWebUI keeps serving /health for a moment while it shuts down.
    Probing straight after the signal sees that corpse, calls it healthy, and skips
    the restart entirely — so the wait is what makes a restart a restart.

    Counted in probes, not seconds: each pass also pays `webui_up`'s own connection
    timeout, so a run of 30 is up to about a minute of wall time rather than 30s.
    """
    for _ in range(attempts):
        if not up(port):
            return
        sleep(1)
    raise FriendlyError(
        f"OpenWebUI on port {port} is still answering after {attempts} shutdown checks.",
        f"Stop whatever is listening on port {port}, then run `lepika update` again.",
    )


def restart_openwebui(
    cfg: Config,
    os_name: str,
    run: RunFn = proc.run_logged,
    popen: PopenFn = subprocess.Popen,
    up: Callable[..., bool] | None = None,
    sleep: SleepFn = time.sleep,
    kill: Callable[[int, int], None] = os.kill,
    bind_check: Callable[[int], bool] | None = None,
) -> None:
    """Stop OpenWebUI, wait for it to really be gone, then start it again."""
    up_fn = up if up is not None else webui_up
    stop_openwebui(os_name, run=run, kill=kill, port=cfg.webui_port, up=up_fn)
    wait_until_down(cfg.webui_port, up=up_fn, sleep=sleep)
    ensure_openwebui(cfg, run=run, popen=popen, up=up_fn, sleep=sleep, bind_check=bind_check)


def start_stack(
    info: SystemInfo,
    cfg: Config,
    after_engine: Callable[[], None] | None = None,
) -> str:
    """Bring the stack up and return the URL to open.

    The single source of truth for the ordering both `lepika up` and the wizard
    depend on: engine first, then whatever the caller needs the engine for (the
    wizard pulls a model here), then the UI that talks to it. `lepika up` passes no
    hook; the wizard passes its pull.
    """
    ensure_ollama(info, url=cfg.engine_url)
    if after_engine is not None:
        after_engine()
    ensure_openwebui(cfg)
    return webui_url(cfg.webui_port)
