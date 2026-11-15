"""The native OpenWebUI: install, secret, start, find, stop, restart."""

from __future__ import annotations

import json
import os
import re
import secrets
import signal
import subprocess
import time
from collections.abc import Callable, Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

from lepika import fsio, health, ports, proc
from lepika.config import Config, config_path
from lepika.errors import FriendlyError
from lepika.paths import logs_dir, openwebui_data_dir, pid_file
from lepika.system import _detach_kwargs

RunFn = Callable[..., Any]
PopenFn = Callable[..., Any]
SleepFn = Callable[[float], None]

# The interpreter OpenWebUI is installed on AND run on — both, or `uv tool run`
# builds an ephemeral environment on the system Python instead of reusing the
# installed one. On a box whose default is newer than OpenWebUI's dependencies
# (pyarrow has no 3.14 wheel) that means a source build that never finishes.
OPENWEBUI_PYTHON = "3.11"


_OPENWEBUI_VERSION = re.compile(r"^open-webui v(\S+)", re.MULTILINE)


def install_openwebui(run: RunFn = proc.run_logged) -> None:
    # `uv tool install` is idempotent: with open-webui already installed it is a
    # no-op that does NOT upgrade — upgrading is `lepika update`'s job.
    run(["uv", "tool", "install", "--python", OPENWEBUI_PYTHON, "open-webui"])


def _webui_secret(data_dir: Path) -> str:
    """Read — or create, once — the secret OpenWebUI signs its sessions with.

    Unset, OpenWebUI writes a `.webui_secret_key` into whatever directory `lepika
    up` happened to run from, under the ambient umask. Stable, because a fresh
    secret on every start would sign every user out on `lepika update`.
    """
    path = data_dir / "secret_key"
    try:
        existing = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        # Missing, unreadable, or not text at all: a corrupted secret is worth
        # exactly one re-signed session, never a traceback (rule 2).
        existing = ""
    if existing:
        return existing
    secret = secrets.token_urlsafe(32)
    fsio.write_private(path, secret + "\n")
    return secret


def start_openwebui(
    port: int,
    engine_url: str,
    popen: PopenFn = subprocess.Popen,
    environ: Mapping[str, str] | None = None,
    engine_key: str = "",
) -> int:
    env = dict(environ if environ is not None else os.environ)
    env["OLLAMA_BASE_URL"] = engine_url
    # LePika owns the wiring: OpenWebUI otherwise persists whatever its admin panel
    # saved on first run and ignores the env from then on, so `lepika connect` would
    # move the engine everywhere except in the UI.
    env["ENABLE_PERSISTENT_CONFIG"] = "false"
    if engine_key:
        # OpenWebUI keys its engines by index; ours is the only one.
        env["OLLAMA_API_CONFIGS"] = json.dumps({"0": {"key": engine_key}})
    # Chats, users, and uploads default to a directory inside the uv tool venv:
    # shared by every LEPIKA_HOME, outside ~/.lepika, and rewritten by the
    # `uv tool upgrade` behind `lepika update`.
    data_dir = openwebui_data_dir()
    env["DATA_DIR"] = str(data_dir)
    # Never logged, never in argv.
    env["WEBUI_SECRET_KEY"] = _webui_secret(data_dir)
    log_file = (logs_dir() / "openwebui.log").open("ab")
    try:
        proc_handle = popen(
            [
                "uv",
                "tool",
                "run",
                # Same interpreter as the install, so this reuses that
                # environment instead of building an ephemeral one.
                "--python",
                OPENWEBUI_PYTHON,
                "--from",
                "open-webui",
                "open-webui",
                "serve",
                # OpenWebUI binds 0.0.0.0 by default; nothing of ours reaches the
                # network until `lepika expose` (Server mode) says so.
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            env=env,
            stdout=log_file,
            stderr=log_file,
            **_detach_kwargs("windows" if os.name == "nt" else "posix"),
        )
    except FileNotFoundError as exc:
        log_file.close()
        raise FriendlyError(
            "uv is required to run OpenWebUI but was not found.",
            "Install uv: https://docs.astral.sh/uv/getting-started/installation/",
        ) from exc
    pid = int(proc_handle.pid)
    pid_file("openwebui").write_text(str(pid))
    return pid


def _is_webui_process(pid: int, os_name: str, run: RunFn) -> bool:
    """Is this pid an OpenWebUI of ours that has merely stopped answering?

    The port probe alone cannot tell a wedged UI from a recycled pid, and the two
    want opposite things: one has to be signalled, the other must never be. The
    command line separates them — `start_openwebui` launches `open-webui serve`, so
    the name is in the argv whatever uv resolved it to. Windows has no equivalent
    (`tasklist` lists images, not argv), so there the port stays the only evidence.
    """
    if os_name == "windows":
        return False
    # log=False: a pure read, and a pid that is simply gone is not a failure.
    listed = run(["ps", "-o", "args=", "-p", str(pid)], check=False, log=False)
    return "open-webui" in str(listed.stdout)


def _find_openwebui(port: int, os_name: str, run: RunFn) -> list[int]:
    """Every pid whose command line says it is an OpenWebUI of ours serving `port`.

    The counterpart to `_is_webui_process`, for when there is no recorded pid to
    check at all. `ensure_openwebui` returns early on a port that already answers,
    so a `lepika up` which adopted a UI that was already running writes no pid file
    — and `lepika down`, with only that file to go on, reported "Nothing was
    running" while `lepika status` went on showing the UI up. `start_openwebui`'s
    argv names `open-webui serve` and the port, so a process carrying both is a UI
    of ours whichever run started it: the same evidence rule 7 already trusts for a
    hung pid, read the other way round. Windows has no argv to read (`tasklist`
    lists images), so there the pid file stays the only handle.
    """
    if os_name == "windows":
        return []
    # log=False: a pure read, and a machine that runs no OpenWebUI is not a failure.
    listed = run(["ps", "-eo", "pid=,args="], check=False, log=False)
    found: list[int] = []
    for line in str(listed.stdout).splitlines():
        pid_text, _, args = line.strip().partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            # A header line, or a row `ps` wrapped: neither names a pid to signal.
            continue
        words = args.split()
        if "serve" not in words or not any("open-webui" in word for word in words):
            continue
        # `--port` and its value are separate words, so the port is matched whole:
        # a substring test would read `--port 30000` as our `--port 3000`.
        if ("--port", str(port)) not in pairwise(words):
            continue
        found.append(pid)
    return found


def _stop_recorded_openwebui(
    os_name: str,
    run: RunFn,
    kill: Callable[[int, int], None],
    port: int | None,
    up: Callable[..., bool] | None,
) -> bool:
    """Stop the OpenWebUI named by the pid file, if that pid is vouched for."""
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
        up_fn = up if up is not None else health.webui_up
        if not up_fn(port) and not _is_webui_process(pid, os_name, run):
            # Nothing answers on our port and no process by that pid names an
            # open-webui, so the recorded pid is not our UI — after a reboot the OS
            # will have handed that number to an unrelated process. A stale pid file
            # is never a licence to signal it (rule 7).
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


def stop_openwebui(
    os_name: str,
    run: RunFn = proc.run_logged,
    kill: Callable[[int, int], None] = os.kill,
    port: int | None = None,
    up: Callable[..., bool] | None = None,
) -> bool:
    """Stop our OpenWebUI, by pid file first and by command line second.

    The pid file is the cheap answer and stays the preferred one — it costs no
    subprocess, and a healthy port confirms it outright. But it is written only by
    the run that actually started the server, so it is missing exactly when a
    `lepika up` found the UI already answering and left it alone. Falling back to
    the process list is what keeps `lepika down` honest across those runs: without
    it, a UI nothing recorded can never be stopped again, and every `down` says
    "Nothing was running" while the port keeps serving.
    """
    if _stop_recorded_openwebui(os_name, run, kill, port, up):
        return True
    if port is None:
        # No port, no command line to match: the pid file was the only handle there is.
        return False
    stopped = False
    for pid in _find_openwebui(port, os_name, run):
        try:
            kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            # Listed a moment ago, but gone — or another user's — by the time we
            # signalled it. Neither is a process this run managed to stop.
            continue
        stopped = True
    return stopped


def ensure_openwebui(
    cfg: Config,
    run: RunFn = proc.run_logged,
    popen: PopenFn = subprocess.Popen,
    up: Callable[..., bool] = health.webui_up,
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
    free = bind_check if bind_check is not None else ports.port_free
    if not free(cfg.webui_port):
        raise FriendlyError(
            f"Port {cfg.webui_port} is in use by another application.",
            f"Change webui_port in {config_path()} and run `lepika up` again.",
        )
    install_openwebui(run=run)
    start_openwebui(cfg.webui_port, cfg.engine_url, popen=popen, engine_key=cfg.engine_key)
    health.wait_for(lambda: up(cfg.webui_port), 180, "OpenWebUI", sleep=sleep)


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
        # `lepika connect` restarts the UI too, so the hint names no single caller.
        f"Stop whatever is listening on port {port}, then try again.",
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
    up_fn = up if up is not None else health.webui_up
    stop_openwebui(os_name, run=run, kill=kill, port=cfg.webui_port, up=up_fn)
    wait_until_down(cfg.webui_port, up=up_fn, sleep=sleep)
    ensure_openwebui(cfg, run=run, popen=popen, up=up_fn, sleep=sleep, bind_check=bind_check)


def openwebui_version(run: RunFn = proc.run_logged) -> str | None:
    """The installed OpenWebUI, as `uv tool list` names it."""
    try:
        result = run(["uv", "tool", "list"], check=False, log=False)
    except FriendlyError:
        return None
    found = _OPENWEBUI_VERSION.search(str(result.stdout or ""))
    return found.group(1) if found else None
