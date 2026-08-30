"""Express mode: native Ollama + OpenWebUI via uv. No Docker anywhere."""

from __future__ import annotations

import json
import os
import re
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import time
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from lepika import detect, log, proc
from lepika.config import Config, config_path
from lepika.detect import SystemInfo
from lepika.errors import FriendlyError
from lepika.paths import logs_dir, openwebui_data_dir, pid_file

RunFn = Callable[..., Any]
WhichFn = Callable[[str], str | None]
PopenFn = Callable[..., Any]
SleepFn = Callable[[float], None]
CallFn = Callable[[list[str]], int]

# Windows: DETACHED_PROCESS (0x8) | CREATE_NEW_PROCESS_GROUP (0x200)
_WINDOWS_DETACH_FLAGS = 0x00000208

# The interpreter OpenWebUI is installed on AND run on — both, or `uv tool run`
# builds an ephemeral environment on the system Python instead of reusing the
# installed one. On a box whose default is newer than OpenWebUI's dependencies
# (pyarrow has no 3.14 wheel) that means a source build that never finishes.
OPENWEBUI_PYTHON = "3.11"


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
    log_file = (logs_dir() / "ollama.log").open("ab")
    try:
        proc_handle = popen(
            ["ollama", "serve"], stdout=log_file, stderr=log_file, **_detach_kwargs(os_name)
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
) -> bool:
    """Stop the native Ollama LePika started — never one the machine already ran.

    Not part of `stop`: `lepika down` leaves the engine up on purpose, because it is
    a shared service. A mode switch is the one case that has to reclaim port 11434.

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
        'Stop it yourself (`pkill -f "ollama serve"`), then run `lepika --mode server` again.',
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
    tmp = path.with_suffix(".tmp")
    # Private from the first byte, exactly as `config.save` writes the API key:
    # the mode is set when the file is created, not chmod'ed once the secret is
    # already on disk, and fchmod also tightens a stale tmp file that an
    # interrupted run left world-readable.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    if hasattr(os, "fchmod"):  # POSIX only; Windows ignores mode bits entirely
        os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(secret + "\n")
    os.replace(tmp, path)
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
    call: CallFn = subprocess.call,
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
        ensure_ollama(info, url=cfg.engine_url)
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


def update(
    info: SystemInfo,
    cfg: Config,
    run: RunFn | None = None,
    which: WhichFn | None = None,
) -> None:
    """`lepika update` in Express mode: upgrade the engine (if ours) and OpenWebUI.

    Resolved at call time, not bound as defaults, so the injected callables stay
    patchable from the module — the same reason `install_ollama` is called by name.
    """
    run_fn = run if run is not None else proc.run_logged
    which_fn = which if which is not None else shutil.which
    if cfg.engine_managed:
        if info.os == "macos":
            # No Homebrew means the Ollama.app installer, which updates itself.
            if which_fn("brew") is not None:
                # check=False: brew exits nonzero when ollama is already up to date.
                run_fn(["brew", "upgrade", "ollama"], check=False)
        elif info.os == "linux":
            # Re-running the official script upgrades in place. Reused rather than
            # restated: it must stream, because it may prompt for sudo.
            install_ollama(info)
        else:
            # check=False: winget exits nonzero when no upgrade is available.
            run_fn(["winget", "upgrade", "--id", "Ollama.Ollama", "-e"], check=False)
    # Pinned like the install and the run: an upgrade that resolved a different
    # interpreter would move the tool env out from under `uv tool run --python`,
    # which then builds an ephemeral one instead of reusing it.
    # check=False: uv exits nonzero when open-webui is already the latest version.
    run_fn(["uv", "tool", "upgrade", "--python", OPENWEBUI_PYTHON, "open-webui"], check=False)
    # A restart, not a stop-then-probe: the upgraded build only takes effect once
    # the old server is really gone.
    restart_openwebui(cfg, info.os)


def logs(lines: int) -> list[tuple[str, str]]:
    """`lepika logs` in Express mode: the tail of every file under ~/.lepika/logs."""
    out: list[tuple[str, str]] = []
    for log_file in sorted(logs_dir().glob("*.log")):
        content = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        out.append((log_file.name, "\n".join(content[-lines:])))
    return out
