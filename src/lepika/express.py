"""Express mode: native Ollama + OpenWebUI via uv. No Docker anywhere."""

from __future__ import annotations

import json
import os
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import time
import urllib.request
from collections.abc import Callable, Mapping
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any

from lepika import detect, fsio, log, proc
from lepika.backend import UpdateStep
from lepika.config import DEFAULT_CONTEXT_LENGTH, DEFAULT_ENGINE_URL, Config, config_path
from lepika.detect import SystemInfo
from lepika.errors import FriendlyError
from lepika.paths import lepika_home, logs_dir, openwebui_data_dir, pid_file
from lepika.system import _bind_address, _detach_kwargs, _effective_uid

RunFn = Callable[..., Any]
WhichFn = Callable[[str], str | None]
PopenFn = Callable[..., Any]
SleepFn = Callable[[float], None]
CallFn = Callable[[list[str]], int]

# The interpreter OpenWebUI is installed on AND run on — both, or `uv tool run`
# builds an ephemeral environment on the system Python instead of reusing the
# installed one. On a box whose default is newer than OpenWebUI's dependencies
# (pyarrow has no 3.14 wheel) that means a source build that never finishes.
OPENWEBUI_PYTHON = "3.11"

# The official Linux installer, which both installs and upgrades in place.
OLLAMA_INSTALL_SCRIPT = "curl -fsSL https://ollama.com/install.sh | sh"

# winget's APPINSTALLER_CLI_ERROR_UPDATE_NOT_APPLICABLE ("No applicable update found"):
# `winget upgrade` exits with it when Ollama is already the latest version. Python
# reports it unsigned on Windows; the signed 32-bit form is accepted as well.
_WINGET_OK = (0, 0x8A15002B, 0x8A15002B - 2**32)

# systemctl(1) uses LSB codes: `is-active` 0 active, 3 not active, 4 no such unit;
# `is-enabled` 0 enabled, non-zero (1) disabled or masked, 4 no such unit. Anything
# else — systemctl on PATH with no systemd running it — is no answer at all.
_IS_ACTIVE_OK = (0, 3, 4)
_IS_ENABLED_OK = (0, 1, 4)
# What `is-enabled` prints (systemctl(1), "is-enabled output"). systemctl also folds
# errors such as a refused bus into exit 1, printing no state: then nothing was read.
_UNIT_FILE_STATES = frozenset(
    {
        "enabled",
        "enabled-runtime",
        "linked",
        "linked-runtime",
        "alias",
        "masked",
        "masked-runtime",
        "static",
        "indirect",
        "disabled",
        "generated",
        "transient",
        "bad",
        "not-found",
    }
)

_OLLAMA_VERSION = re.compile(r"version is (\S+)")
_OPENWEBUI_VERSION = re.compile(r"^open-webui v(\S+)", re.MULTILINE)


def _systemctl(*args: str, euid: int | None = None) -> list[str]:
    """`systemctl` changing a system unit: through `sudo` unless LePika already is root.

    A server whose root has a password may have no `sudo` at all — the Ollama install
    script supports that — and root needs none. `euid` stands in for the real one.
    """
    uid = euid if euid is not None else _effective_uid()
    return ["systemctl", *args] if uid == 0 else ["sudo", "systemctl", *args]


def install_ollama(
    info: SystemInfo,
    run: RunFn = proc.run_logged,
    which: WhichFn = shutil.which,
    call: CallFn = proc.call,
    euid: int | None = None,
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
        code = call(["sh", "-c", OLLAMA_INSTALL_SCRIPT])
        # The binary on PATH, not the exit code, is what says the install worked.
        # The script runs under `set -e`, and everything past the point where the
        # engine lands — its own comment calls it optional — is the systemd setup
        # LePika undoes on the next line anyway. On a box whose systemd is wedged
        # ("Reload daemon failed: Connection reset by peer") `systemctl enable`
        # aborts the script with a perfectly good engine already installed.
        if code != 0:
            if which("ollama") is None:
                raise FriendlyError(
                    "Ollama installation failed.",
                    "See https://ollama.com/download/linux for manual install steps.",
                )
            log.get_logger().warning("ollama.install.optional_step_failed", code=code)
        # The script enables + starts a systemd ollama.service. LePika manages its
        # own `ollama serve` on every OS, so left enabled the unit crash-loops
        # against ours on port 11434 and, after a reboot, wins the port with an
        # empty model store (/usr/share/ollama/.ollama vs ~/.ollama). A distro
        # without systemd has nothing to disable. Streamed for the same reason as the
        # script; the outcome is deliberately ignored — a unit left running is a
        # nuisance, not a reason to fail an install whose engine is in place.
        if which("systemctl") is not None:
            try:
                call(_systemctl("disable", "--now", "ollama.service", euid=euid))
            except (FriendlyError, OSError):
                log.get_logger().warning("ollama.install.unit_disable_failed")
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


def ollama_env(context_length: int, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment `ollama serve` starts with: the shell's, plus the context length.

    Ollama has no config file; `OLLAMA_CONTEXT_LENGTH` is how its default context is
    set, and it defaults to 4096 — one pasted document over the limit on any model.
    A value already in the shell wins over config.toml, as `HF_TOKEN` does in Server
    mode: what the user just exported is the freshest intent they have.
    """
    env = dict(os.environ if environ is None else environ)
    env.setdefault("OLLAMA_CONTEXT_LENGTH", str(context_length))
    return env


def start_ollama(
    os_name: str,
    popen: PopenFn = subprocess.Popen,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    environ: Mapping[str, str] | None = None,
) -> None:
    env = ollama_env(context_length, environ)
    log_file = (logs_dir() / "ollama.log").open("ab")
    try:
        proc_handle = popen(
            ["ollama", "serve"],
            stdout=log_file,
            stderr=log_file,
            env=env,
            **_detach_kwargs(os_name),
        )
    except FileNotFoundError as exc:
        log_file.close()
        raise FriendlyError(
            "Ollama is installed but not on your PATH yet.",
            "Close this terminal, open a new one, and run `lepika` again.",
        ) from exc
    # Recorded so a mode switch can tell the engine LePika started from one the
    # machine already ran: only a pid we wrote down is ever ours to stop.
    pid_file("ollama").write_text(str(int(proc_handle.pid)))
    # And what it was started with: Ollama has no way to ask an idle engine its
    # context length, and `lepika down` leaves this one running, so the next `up`
    # can only tell a changed setting from this record.
    _context_record().write_text(env["OLLAMA_CONTEXT_LENGTH"])


def _context_record() -> Path:
    return lepika_home() / "ollama.context"


def _is_ollama_process(pid: int, os_name: str, run: RunFn) -> bool:
    """Does this pid belong to an Ollama, or has the number been recycled?

    `ollama.pid` outlives `lepika down`, which leaves the engine running on purpose,
    so it is the one pid file that routinely survives a reboot. By then the OS may
    have handed the number to a stranger while its own Ollama service answers on
    11434 — the API probe alone would approve signalling that stranger.
    """
    # log=False: a pure read, and a pid that is simply gone is not a failure.
    if os_name == "windows":
        listed = run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], check=False, log=False
        )
        return "ollama" in str(listed.stdout).lower()
    listed = run(["ps", "-p", str(pid), "-o", "comm="], check=False, log=False)
    # `comm=` is a bare name on Linux and a full path on macOS; both end in the binary.
    return PurePosixPath(str(listed.stdout).strip()).name.startswith("ollama")


def stop_ollama(
    os_name: str,
    url: str,
    run: RunFn = proc.run_logged,
    kill: Callable[[int, int], None] = os.kill,
    api_up: Callable[..., bool] = detect.api_up,
    key: str = "",
    attempts: int = 30,
    sleep: SleepFn = time.sleep,
    rerun: str = "lepika --mode server",
) -> bool:
    """Stop the native Ollama LePika started — never one the machine already ran.

    Not part of `stop`: `lepika down` leaves the engine up on purpose, because it is
    a shared service. A mode switch has to reclaim port 11434, and `ensure_ollama`
    restarts it for a changed context length; `rerun` is the command each retries with.

    Three things must agree before anything is signalled: a pid file we wrote, an
    engine answering on `url`, and a process by that pid that really is an Ollama.
    """
    pf = pid_file("ollama")
    if not pf.exists():
        # Homebrew, systemd, or the tray app started it: not ours, not our business.
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
    if not api_up(url, key=key) or not _is_ollama_process(pid, os_name, run):
        # Nothing is answering, or the pid is not an Ollama: either way the recorded
        # number is not our engine, and a stale pid file is never a licence to signal
        # it (rule 7).
        pf.unlink(missing_ok=True)
        return False
    try:
        if os_name == "windows":
            run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False)
        else:
            kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        # Gone, or recycled into a process we don't own (a truncated pid can
        # parse to a live foreign pid) — the wait below decides what that meant.
        pass
    # Signalled is not stopped: Ollama unloads its models on the way out, and the
    # caller's next move is a Server stack that wants port 11434. Counted in probes
    # like `wait_until_down`, because each pass also pays `api_up`'s own timeout.
    for _ in range(attempts):
        if not api_up(url, key=key):
            pf.unlink(missing_ok=True)
            return True
        sleep(1)
    # The pid file stays: it is the only record of which engine was ours, and a
    # retry that has lost it can never stop this process at all.
    raise FriendlyError(
        f"Ollama at {url} is still answering after {attempts} shutdown checks.",
        f'Stop it yourself (`pkill -f "ollama serve"`), then run `{rerun}` again.',
    )


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
        # `lepika logs`, not a raw path: in Server mode the cause is in the
        # container's logs, which no file under ~/.lepika/logs holds.
        "Run `lepika logs` to see why, then `lepika doctor`.",
    )


def ensure_ollama(
    info: SystemInfo,
    *,
    context_length: int,
    run: RunFn = proc.run_logged,
    which: WhichFn = shutil.which,
    popen: PopenFn = subprocess.Popen,
    api_up: Callable[..., bool] = detect.api_up,
    sleep: SleepFn = time.sleep,
    call: CallFn = proc.call,
    url: str = DEFAULT_ENGINE_URL,
    kill: Callable[[int, int], None] = os.kill,
    environ: Mapping[str, str] | None = None,
) -> None:
    if not info.has_ollama:
        install_ollama(info, run=run, which=which, call=call)
    if api_up(url):
        # Only an engine we started is ours to restart; any other keeps its own
        # setting and `context_note` says where it lives. A missing record is an
        # engine from before LePika set one at all: Ollama's 4096, not the config.
        wanted = ollama_env(context_length, environ)["OLLAMA_CONTEXT_LENGTH"]
        if not _ours(info.os, run) or _launched_context() == wanted:
            return
        log.get_logger().info("engine.restart", reason="context_length", wanted=wanted)
        stop_ollama(info.os, url, run=run, kill=kill, api_up=api_up, sleep=sleep, rerun="lepika up")
    start_ollama(info.os, popen=popen, context_length=context_length, environ=environ)
    wait_for(lambda: api_up(url), 30, "Ollama API", sleep=sleep)


def _launched_context() -> str | None:
    record = _context_record()
    return record.read_text().strip() if record.exists() else None


def _ours(os_name: str, run: RunFn) -> bool:
    """Did LePika start the Ollama that is running? A pid we wrote, still an Ollama."""
    pf = pid_file("ollama")
    if not pf.exists():
        return False
    try:
        pid = int(pf.read_text().strip())
    except ValueError:
        return False
    return pid > 0 and _is_ollama_process(pid, os_name, run)


def context_note(info: SystemInfo, cfg: Config, run: RunFn = proc.run_logged) -> str | None:
    """One line for `lepika up` when the context length in config.toml cannot apply.

    `ensure_ollama` adopts an engine that already answers instead of starting one,
    so `OLLAMA_CONTEXT_LENGTH` never reaches it: the tray app on Windows, brew
    services, a `systemd` unit, or a user's own `ollama serve`. Ollama exposes no
    way to read an idle engine's context length, so this cannot check — it says
    where the setting lives on that engine. Quiet for the engine LePika itself
    started, which got the value at launch, and for a remote engine, which was
    never ours to configure.
    """
    if not (cfg.mode == "express" and cfg.engine_managed and info.ollama_running):
        return None
    if _ours(info.os, run):
        return None
    if info.os == "windows":
        how = (
            "Ollama's tray icon → Settings → Context length, or a user environment "
            f"variable OLLAMA_CONTEXT_LENGTH={cfg.context_length} and restart Ollama"
        )
    else:
        how = f"quit Ollama and run `lepika up` again to start it with {cfg.context_length}"
    return (
        "Ollama was already running, so it keeps its own context length "
        f"(Ollama's default is 4096 tokens). To use {cfg.context_length}: {how}."
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
    run(["uv", "tool", "install", "--python", OPENWEBUI_PYTHON, "open-webui"])


# A server can hold a port on the wildcard address or on loopback alone, and on
# macOS a loopback bind succeeds straight over an existing 0.0.0.0 listener. Both
# are probed, because either one is a real conflict.
# B104 suppressed: these are bind probes, closed immediately without listening.
# Detecting a conflict on the wildcard address requires naming it.
_PROBE_ADDRESSES = ("0.0.0.0", "127.0.0.1")  # nosec B104


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
        up_fn = up if up is not None else webui_up
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
    start_openwebui(cfg.webui_port, cfg.engine_url, popen=popen, engine_key=cfg.engine_key)
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
    up_fn = up if up is not None else webui_up
    stop_openwebui(os_name, run=run, kill=kill, port=cfg.webui_port, up=up_fn)
    wait_until_down(cfg.webui_port, up=up_fn, sleep=sleep)
    ensure_openwebui(cfg, run=run, popen=popen, up=up_fn, sleep=sleep, bind_check=bind_check)


def check_remote_engine(cfg: Config, api_up: Callable[..., bool] = detect.api_up) -> None:
    """Confirm an engine we do not manage is answering — the only thing we may do to it."""
    if api_up(cfg.engine_url, key=cfg.engine_key):
        return
    raise FriendlyError(
        f"The engine at {cfg.engine_url} is not answering.",
        "Make sure that machine is up (and `lepika expose` is on there), or run "
        "`lepika connect --local` to go back to a local engine.",
    )


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


def import_allowed(cfg: Config, info: SystemInfo) -> bool:
    """Can this machine import a full-weight repo into Ollama? (rule 10, Express)

    Imports run on Ollama's MLX engine, which is built into the macOS arm64 build.
    On Linux/Windows it is a separate amd64-only CUDA 13 bundle that `ensure_mlx`
    installs on the import path (so an NVIDIA Jetson could never have it, and neither
    could an Intel Mac with an NVIDIA card — there is no macOS bundle at all).
    Express only, and only for an engine that is ours: the weights have to land on
    the engine's machine.
    """
    if not (cfg.mode == "express" and cfg.engine_managed):
        return False
    if info.gpu == "apple":
        return True
    return info.os != "macos" and info.gpu == "nvidia" and info.arch in ("x86_64", "amd64")


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
    if cfg.engine_managed:
        ensure_ollama(info, url=cfg.engine_url, context_length=cfg.context_length)
    else:
        # Someone else runs this engine: never install or start anything for it.
        check_remote_engine(cfg, api_up=detect.api_up)
    # Logged once the engine is really up, like Server: a pre-flight that refused
    # started no stack, and a `stack.up` above its own failure line reads as one that did.
    log.get_logger().info("stack.up", mode="express", engine_managed=cfg.engine_managed)
    if after_engine is not None:
        after_engine()
    ensure_openwebui(cfg)
    return webui_url(cfg.webui_port)


def stop(info: SystemInfo, cfg: Config) -> bool:
    """`lepika down` in Express mode: stop OpenWebUI; Ollama stays as a shared service."""
    # The port is what proves the recorded pid is still our OpenWebUI.
    stopped = stop_openwebui(info.os, port=cfg.webui_port)
    log.get_logger().info("stack.down", mode="express", stopped=stopped)
    return stopped


def ollama_binary_version(run: RunFn = proc.run_logged) -> str | None:
    """The version of the `ollama` binary on PATH, whether or not an engine is running.

    With an engine up it prints `ollama version is X`; without one, a warning ending
    `client version is X`. Both name the binary, which is the one this asks about.
    """
    try:
        result = run(["ollama", "--version"], check=False, log=False)
    except FriendlyError:
        return None
    found = _OLLAMA_VERSION.findall(f"{result.stdout or ''}\n{result.stderr or ''}")
    return str(found[-1]) if found else None


def openwebui_version(run: RunFn = proc.run_logged) -> str | None:
    """The installed OpenWebUI, as `uv tool list` names it."""
    try:
        result = run(["uv", "tool", "list"], check=False, log=False)
    except FriendlyError:
        return None
    found = _OPENWEBUI_VERSION.search(str(result.stdout or ""))
    return found.group(1) if found else None


def _unit_state(run: RunFn, which: WhichFn) -> tuple[bool, bool] | None:
    """(enabled, active) of `ollama.service`, or None when systemd cannot say.

    Unknown is not "disabled": restoring from a guess would run `sudo systemctl` on a
    unit whose state nobody read.
    """
    if which("systemctl") is None:
        return None
    unit = "ollama.service"
    # Not --quiet: the printed state is what tells "disabled" from "could not ask".
    enabled = run(["systemctl", "is-enabled", unit], check=False, log=False, ok=_IS_ENABLED_OK)
    active = run(
        ["systemctl", "is-active", "--quiet", unit], check=False, log=False, ok=_IS_ACTIVE_OK
    )
    read_enabled = enabled.returncode == 0 or (
        enabled.returncode in _IS_ENABLED_OK
        and str(enabled.stdout or "").strip() in _UNIT_FILE_STATES
    )
    if not read_enabled or active.returncode not in _IS_ACTIVE_OK:
        log.get_logger().warning(
            "ollama.update.unit_state_unknown",
            is_enabled=enabled.returncode,
            is_active=active.returncode,
        )
        return None
    return enabled.returncode == 0, active.returncode == 0


def _restore_unit(state: tuple[bool, bool] | None, call: CallFn, euid: int | None = None) -> bool:
    """Put `ollama.service` back the way it was before the install script touched it.

    The script enables and starts the unit every time it runs. A unit the user ran
    before LePika stays exactly as they had it; one LePika disabled is disabled again.
    Returns whether the unit is back in its recorded state. Never raises: it runs
    while the installer's own error may be on its way out, and must not replace it.
    """
    if state is None:
        return True
    enabled, active = state
    if enabled and active:
        return True
    if enabled:
        verb = ["stop"]
    elif active:
        verb = ["disable"]
    else:
        # `--now` is the stop: one verb, so one sudo prompt.
        verb = ["disable", "--now"]
    # Streamed, like the script: sudo may prompt. A unit left running is a nuisance,
    # not a broken engine, so a restore that fails is reported, not raised.
    try:
        return call(_systemctl(*verb, "ollama.service", euid=euid)) == 0
    except (FriendlyError, OSError) as exc:
        log.get_logger().warning("ollama.update.unit_restore_failed", error=str(exc))
        return False


def upgrade_ollama(
    info: SystemInfo,
    *,
    run: RunFn,
    which: WhichFn,
    call: CallFn,
) -> None:
    """Upgrade the native Ollama in place, each OS through its own package manager.

    On Linux the install script re-enables and starts `ollama.service`; the caller
    puts the unit back (`_restore_unit`) whether or not the script succeeded.
    """
    if info.os == "macos":
        # No Homebrew means the Ollama.app installer, which updates itself.
        if which("brew") is None:
            return
        # Asked first, so the upgrade itself can be checked. `outdated` names the
        # formula only when there is something to do — and then exits 1, as it does
        # for any outdated formula named on its command line: an answer, not a failure.
        outdated = run(["brew", "outdated", "--quiet", "ollama"], check=False, log=False, ok=(0, 1))
        if "ollama" in str(outdated.stdout or "").split():
            run(["brew", "upgrade", "ollama"], check=True)
    elif info.os == "linux":
        # Streamed, not captured: the script may prompt for sudo.
        if call(["sh", "-c", OLLAMA_INSTALL_SCRIPT]) != 0:
            raise FriendlyError(
                "Ollama upgrade failed.",
                "See https://ollama.com/download/linux for manual install steps.",
            )
    else:  # windows
        result = run(
            [
                "winget",
                "upgrade",
                "--id",
                "Ollama.Ollama",
                "-e",
                "--accept-package-agreements",
                "--accept-source-agreements",
            ],
            check=False,
            # "No applicable update" is an answer: a current Ollama logs no failure.
            ok=_WINGET_OK,
        )
        if result.returncode not in _WINGET_OK:
            raise FriendlyError(
                "Ollama upgrade failed.",
                "Run `winget upgrade --id Ollama.Ollama -e` to see why, then `lepika update`.",
            )


def _changed(before: str | None, after: str | None) -> str:
    return f"{before} → {after}" if before and after else after or ""


def update(
    info: SystemInfo,
    cfg: Config,
    run: RunFn | None = None,
    which: WhichFn | None = None,
    call: CallFn | None = None,
    popen: PopenFn | None = None,
    api_up: Callable[..., bool] | None = None,
    sleep: SleepFn | None = None,
    kill: Callable[[int, int], None] | None = None,
    running_version: Callable[..., str] | None = None,
    euid: int | None = None,
) -> list[UpdateStep]:
    """`lepika update` in Express mode: upgrade the engine (if ours) and OpenWebUI.

    Resolved at call time, not bound as defaults, so the injected callables stay
    patchable from the module — the same reason the helpers are called by name.
    """
    run_fn = run if run is not None else proc.run_logged
    popen_fn = popen if popen is not None else subprocess.Popen
    sleep_fn = sleep if sleep is not None else time.sleep
    kill_fn = kill if kill is not None else os.kill
    steps: list[UpdateStep] = []
    if cfg.engine_managed:
        which_fn = which if which is not None else shutil.which
        # Read before the install script runs: it re-enables and starts the unit.
        systemd = info.os == "linux" and which_fn("systemctl") is not None
        unit = _unit_state(run_fn, which_fn) if systemd else None
        step, restored = _update_ollama(
            info,
            cfg,
            run=run_fn,
            which=which_fn,
            call=call if call is not None else proc.call,
            popen=popen_fn,
            api_up=api_up if api_up is not None else detect.api_up,
            sleep=sleep_fn,
            kill=kill_fn,
            running_version=running_version,
            unit=unit,
            euid=euid,
        )
        steps.append(step)
        if systemd and unit is None:
            # Left alone rather than restored from a guess: wrongly disabling a
            # user's own service is the worse mistake. But a unit the script just
            # enabled and started can contend for port 11434, so the user hears of it.
            steps.append(
                UpdateStep(
                    "Ollama service",
                    "note",
                    "could not read ollama.service's state, so it was left as the installer "
                    "set it — if Ollama misbehaves, run "
                    "`systemctl disable --now ollama.service` as root",
                )
            )
        elif not restored:
            steps.append(
                UpdateStep(
                    "Ollama service",
                    "note",
                    "could not put ollama.service back as it was — if Ollama misbehaves, "
                    "run `systemctl disable --now ollama.service` as root",
                )
            )
    before = openwebui_version(run_fn)
    # Pinned like the install and the run: an upgrade that resolved a different
    # interpreter would move the tool env out from under `uv tool run --python`,
    # which then builds an ephemeral one instead of reusing it. check=True: uv exits
    # 0 with "Nothing to upgrade" when current, so a nonzero exit is a real failure.
    run_fn(["uv", "tool", "upgrade", "--python", OPENWEBUI_PYTHON, "open-webui"], check=True)
    after = openwebui_version(run_fn)
    if before is not None and before == after:
        # Unchanged: a running UI is not interrupted for nothing.
        steps.append(UpdateStep("OpenWebUI", "current", after))
        return steps
    # A restart, not a stop-then-probe: the upgraded build only takes effect once
    # the old server is really gone. An unknown version restarts too.
    restart_openwebui(cfg, info.os, run=run_fn, popen=popen_fn, sleep=sleep_fn, kill=kill_fn)
    steps.append(UpdateStep("OpenWebUI", "upgraded", _changed(before, after)))
    return steps


def _update_ollama(
    info: SystemInfo,
    cfg: Config,
    *,
    run: RunFn,
    which: WhichFn,
    call: CallFn,
    popen: PopenFn,
    api_up: Callable[..., bool],
    sleep: SleepFn,
    kill: Callable[[int, int], None],
    running_version: Callable[..., str] | None,
    unit: tuple[bool, bool] | None,
    euid: int | None,
) -> tuple[UpdateStep, bool]:
    """The Ollama step, and whether `ollama.service` is back as `unit` recorded it.

    `unit` is the unit's (enabled, active) as read before the upgrade (`_unit_state`);
    None leaves it as the install script sets it — there is none, or its state could
    not be read.
    """
    if running_version is None:
        # Imported here: engine.py is the HTTP layer, and express has no other
        # reason to depend on it.
        from lepika import engine

        running_version = engine.version
    version_of = running_version

    def running() -> str | None:
        """What the engine at the configured URL serves now, or None if it cannot say."""
        if not api_up(cfg.engine_url, key=cfg.engine_key):
            return None
        try:
            return version_of(cfg.engine_url, key=cfg.engine_key)
        except FriendlyError:
            # It answered a moment ago and not now, or not with JSON: unknown, and
            # not a reason to leave OpenWebUI un-upgraded.
            return None

    # Decided before anything changes: the upgrade may stop the process, and then
    # the pid file can no longer be vouched for by a running Ollama.
    ours_and_up = api_up(cfg.engine_url, key=cfg.engine_key) and _ours(info.os, run)
    before = ollama_binary_version(run)
    failure: FriendlyError | None = None
    try:
        upgrade_ollama(info, run=run, which=which, call=call)
    except FriendlyError as exc:
        failure = exc
    finally:
        # Before anything below starts an engine: a unit the script started holds
        # port 11434. `_restore_unit` never raises, so it cannot mask `failure`.
        restored = _restore_unit(unit, call, euid)
    after = ollama_binary_version(run)
    if failure is not None:
        # On Linux the binary, not the script's exit code, says whether the upgrade
        # happened: the script aborts on its optional systemd steps with the new
        # engine already in place — the same rule `install_ollama` follows.
        if info.os != "linux" or after == before:
            raise failure
        log.get_logger().warning("ollama.update.optional_step_failed")
    if ours_and_up:
        start = False
        if not api_up(cfg.engine_url, key=cfg.engine_key):
            if _ours(info.os, run):
                # Still an Ollama, just not answering (busy loading a model misses a
                # 1s probe). A second `ollama serve` would lose the port and overwrite
                # the pid file, the only record of which engine is ours.
                installed = f"installed {after}; " if after else ""
                return UpdateStep(
                    "Ollama",
                    "stale",
                    f"{installed}the running engine is not answering — "
                    "run `lepika update` again to restart it",
                ), restored
            # Really gone — the installer stopped it (winget replaces a running
            # Ollama): ours, so the command that stopped it brings it back.
            log.get_logger().info("engine.restart", reason="update", stopped_by="installer")
            start = True
        else:
            now = running()
            if now and after and now != after:
                # The old binary keeps serving until its process is replaced.
                log.get_logger().info("engine.restart", reason="update", running=now, wanted=after)
                # Started again only once the stop is confirmed: a refused stop leaves
                # the old engine on the port, and a second one would fight it for it.
                start = stop_ollama(
                    info.os,
                    cfg.engine_url,
                    run=run,
                    kill=kill,
                    api_up=api_up,
                    key=cfg.engine_key,
                    sleep=sleep,
                    rerun="lepika update",
                )
        if start:
            start_ollama(info.os, popen=popen, context_length=cfg.context_length)
            wait_for(
                lambda: api_up(cfg.engine_url, key=cfg.engine_key), 30, "Ollama API", sleep=sleep
            )
    serving = running()
    if serving and after and serving != after:
        # Ours when its stop was refused; otherwise whoever started it restarts it.
        return UpdateStep(
            "Ollama", "stale", f"running {serving}, installed {after} — restart Ollama to load it"
        ), restored
    if before != after:
        return UpdateStep("Ollama", "upgraded", _changed(before, after)), restored
    return UpdateStep("Ollama", "current", after or ""), restored


def logs(lines: int) -> list[tuple[str, str]]:
    """`lepika logs` in Express mode: the tail of every file under ~/.lepika/logs."""
    out: list[tuple[str, str]] = []
    for log_file in sorted(logs_dir().glob("*.log")):
        content = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        out.append((log_file.name, "\n".join(content[-lines:])))
    return out
