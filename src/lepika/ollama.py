"""The native Ollama: install, upgrade, its systemd unit, and supervising the process."""

from __future__ import annotations

import re
import signal
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from lepika import engine, health, log
from lepika.backend import UpdateStep, changed
from lepika.config import DEFAULT_CONTEXT_LENGTH, DEFAULT_ENGINE_URL, Config
from lepika.detect import Machine
from lepika.errors import CommandNotFound, FriendlyError
from lepika.paths import lepika_home, logs_dir, pid_file
from lepika.system import System

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


def _systemctl(*args: str, sys: System) -> list[str]:
    """`systemctl` changing a system unit: through `sudo` unless LePika already is root.

    A server whose root has a password may have no `sudo` at all — the Ollama install
    script supports that — and root needs none. Only a uid of exactly 0 skips it: a
    platform with no uid at all gets `sudo`, as it always has.
    """
    return ["systemctl", *args] if sys.euid() == 0 else ["sudo", "systemctl", *args]


def install_ollama(info: Machine, *, sys: System) -> None:
    if info.os == "macos":
        if sys.which("brew") is not None:
            sys.run(["brew", "install", "ollama"])
        else:
            raise FriendlyError(
                "Ollama is not installed and Homebrew was not found.",
                "Install Ollama from https://ollama.com/download/mac then run `lepika` again.",
            )
    elif info.os == "linux":
        # Streamed, not captured: the official script may prompt for sudo, and a
        # password prompt hidden behind captured output looks like a hang.
        code = sys.call(["sh", "-c", OLLAMA_INSTALL_SCRIPT])
        # The binary on PATH, not the exit code, is what says the install worked.
        # The script runs under `set -e`, and everything past the point where the
        # engine lands — its own comment calls it optional — is the systemd setup
        # LePika undoes on the next line anyway. On a box whose systemd is wedged
        # ("Reload daemon failed: Connection reset by peer") `systemctl enable`
        # aborts the script with a perfectly good engine already installed.
        if code != 0:
            if sys.which("ollama") is None:
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
        if sys.which("systemctl") is not None:
            try:
                sys.call(_systemctl("disable", "--now", "ollama.service", sys=sys))
            except (FriendlyError, OSError):
                log.get_logger().warning("ollama.install.unit_disable_failed")
    else:  # windows
        sys.run(
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


def ollama_env(context_length: int, environ: Mapping[str, str]) -> dict[str, str]:
    """The environment `ollama serve` starts with: the shell's, plus the context length.

    Ollama has no config file; `OLLAMA_CONTEXT_LENGTH` is how its default context is
    set, and it defaults to 4096 — one pasted document over the limit on any model.
    A value already in the shell wins over config.toml, as `HF_TOKEN` does in Server
    mode: what the user just exported is the freshest intent they have.
    """
    env = dict(environ)
    env.setdefault("OLLAMA_CONTEXT_LENGTH", str(context_length))
    return env


def start_ollama(context_length: int = DEFAULT_CONTEXT_LENGTH, *, sys: System) -> None:
    env = ollama_env(context_length, sys.environ)
    try:
        pid = sys.spawn(["ollama", "serve"], env=env, log=logs_dir() / "ollama.log")
    except CommandNotFound as exc:
        raise FriendlyError(
            "Ollama is installed but not on your PATH yet.",
            "Close this terminal, open a new one, and run `lepika` again.",
        ) from exc
    # Recorded so a mode switch can tell the engine LePika started from one the
    # machine already ran: only a pid we wrote down is ever ours to stop.
    pid_file("ollama").write_text(str(pid))
    # And what it was started with: Ollama has no way to ask an idle engine its
    # context length, and `lepika down` leaves this one running, so the next `up`
    # can only tell a changed setting from this record.
    _context_record().write_text(env["OLLAMA_CONTEXT_LENGTH"])


def _context_record() -> Path:
    return lepika_home() / "ollama.context"


def _is_ollama_process(pid: int, os_name: str, *, sys: System) -> bool:
    """Does this pid belong to an Ollama, or has the number been recycled?

    `ollama.pid` outlives `lepika down`, which leaves the engine running on purpose,
    so it is the one pid file that routinely survives a reboot. By then the OS may
    have handed the number to a stranger while its own Ollama service answers on
    11434 — the API probe alone would approve signalling that stranger.
    """
    # log=False: a pure read, and a pid that is simply gone is not a failure.
    if os_name == "windows":
        listed = sys.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], check=False, log=False
        )
        return "ollama" in str(listed.stdout).lower()
    listed = sys.run(["ps", "-p", str(pid), "-o", "comm="], check=False, log=False)
    # `comm=` is a bare name on Linux and a full path on macOS; both end in the binary.
    return PurePosixPath(str(listed.stdout).strip()).name.startswith("ollama")


def stop_ollama(
    os_name: str,
    url: str,
    key: str = "",
    attempts: int = 30,
    rerun: str = "lepika --mode server",
    *,
    sys: System,
) -> bool:
    """Stop the native Ollama LePika started — never one the machine already ran.

    Not part of `stop`: `lepika down` leaves the engine up on purpose, because it is
    a shared service. A mode switch has to reclaim port 11434, and `ensure_ollama`
    restarts it for a changed context length; `rerun` is the command each retries with.

    Three things must agree before anything is signalled: a pid file we wrote, an
    engine answering on `url`, and a process by that pid that really is an Ollama.
    """

    def answering() -> bool:
        return health.api_up(url, key=key, sys=sys)

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
    if not answering() or not _is_ollama_process(pid, os_name, sys=sys):
        # Nothing is answering, or the pid is not an Ollama: either way the recorded
        # number is not our engine, and a stale pid file is never a licence to signal
        # it (rule 7).
        pf.unlink(missing_ok=True)
        return False
    try:
        if os_name == "windows":
            sys.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False)
        else:
            sys.signal(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        # Gone, or recycled into a process we don't own (a truncated pid can
        # parse to a live foreign pid) — the wait below decides what that meant.
        pass
    # Signalled is not stopped: Ollama unloads its models on the way out, and the
    # caller's next move is a Server stack that wants port 11434. Counted in probes
    # like `wait_until_down`, because each pass also pays `api_up`'s own timeout.
    for _ in range(attempts):
        if not answering():
            pf.unlink(missing_ok=True)
            return True
        sys.sleep(1)
    # The pid file stays: it is the only record of which engine was ours, and a
    # retry that has lost it can never stop this process at all.
    raise FriendlyError(
        f"Ollama at {url} is still answering after {attempts} shutdown checks.",
        f'Stop it yourself (`pkill -f "ollama serve"`), then run `{rerun}` again.',
    )


def ensure_ollama(
    info: Machine,
    *,
    context_length: int,
    url: str = DEFAULT_ENGINE_URL,
    sys: System,
) -> None:
    """Install Ollama if it is missing, and have an engine answering on `url`."""

    def answering() -> bool:
        return health.api_up(url, sys=sys)

    if sys.which("ollama") is None:
        install_ollama(info, sys=sys)
    if answering():
        # Only an engine we started is ours to restart; any other keeps its own
        # setting and `context_note` says where it lives. A missing record is an
        # engine from before LePika set one at all: Ollama's 4096, not the config.
        wanted = ollama_env(context_length, sys.environ)["OLLAMA_CONTEXT_LENGTH"]
        if not _ours(info.os, sys=sys) or _launched_context() == wanted:
            return
        log.get_logger().info("engine.restart", reason="context_length", wanted=wanted)
        stop_ollama(info.os, url, rerun="lepika up", sys=sys)
    start_ollama(context_length=context_length, sys=sys)
    health.wait_for(answering, 30, "Ollama API", sys=sys)


def _launched_context() -> str | None:
    record = _context_record()
    return record.read_text().strip() if record.exists() else None


def _ours(os_name: str, *, sys: System) -> bool:
    """Did LePika start the Ollama that is running? A pid we wrote, still an Ollama."""
    pf = pid_file("ollama")
    if not pf.exists():
        return False
    try:
        pid = int(pf.read_text().strip())
    except ValueError:
        return False
    return pid > 0 and _is_ollama_process(pid, os_name, sys=sys)


def context_note(info: Machine, cfg: Config, *, sys: System) -> str | None:
    """One line for `lepika up` when the context length in config.toml cannot apply.

    `ensure_ollama` adopts an engine that already answers instead of starting one,
    so `OLLAMA_CONTEXT_LENGTH` never reaches it: the tray app on Windows, brew
    services, a `systemd` unit, or a user's own `ollama serve`. Ollama exposes no
    way to read an idle engine's context length, so this cannot check — it says
    where the setting lives on that engine. Quiet for the engine LePika itself
    started, which got the value at launch, and for a remote engine, which was
    never ours to configure. Asked before the stack starts: an engine answering on
    `engine_url` then is one `ensure_ollama` will adopt.
    """
    if not (cfg.mode == "express" and cfg.engine_managed):
        return None
    # No key: a managed engine takes none.
    if not health.api_up(cfg.engine_url, sys=sys):
        return None
    if _ours(info.os, sys=sys):
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


def ollama_binary_version(*, sys: System) -> str | None:
    """The version of the `ollama` binary on PATH, whether or not an engine is running.

    With an engine up it prints `ollama version is X`; without one, a warning ending
    `client version is X`. Both name the binary, which is the one this asks about.
    """
    try:
        result = sys.run(["ollama", "--version"], check=False, log=False)
    except FriendlyError:
        return None
    found = _OLLAMA_VERSION.findall(f"{result.stdout or ''}\n{result.stderr or ''}")
    return str(found[-1]) if found else None


def _unit_state(*, sys: System) -> tuple[bool, bool] | None:
    """(enabled, active) of `ollama.service`, or None when systemd cannot say.

    Unknown is not "disabled": restoring from a guess would run `sudo systemctl` on a
    unit whose state nobody read.
    """
    if sys.which("systemctl") is None:
        return None
    unit = "ollama.service"
    # Not --quiet: the printed state is what tells "disabled" from "could not ask".
    enabled = sys.run(["systemctl", "is-enabled", unit], check=False, log=False, ok=_IS_ENABLED_OK)
    active = sys.run(
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


def _restore_unit(state: tuple[bool, bool] | None, *, sys: System) -> bool:
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
        return sys.call(_systemctl(*verb, "ollama.service", sys=sys)) == 0
    except (FriendlyError, OSError) as exc:
        log.get_logger().warning("ollama.update.unit_restore_failed", error=str(exc))
        return False


def upgrade_ollama(info: Machine, *, sys: System) -> None:
    """Upgrade the native Ollama in place, each OS through its own package manager.

    On Linux the install script re-enables and starts `ollama.service`; the caller
    puts the unit back (`_restore_unit`) whether or not the script succeeded.
    """
    if info.os == "macos":
        # No Homebrew means the Ollama.app installer, which updates itself.
        if sys.which("brew") is None:
            return
        # Asked first, so the upgrade itself can be checked. `outdated` names the
        # formula only when there is something to do — and then exits 1, as it does
        # for any outdated formula named on its command line: an answer, not a failure.
        outdated = sys.run(
            ["brew", "outdated", "--quiet", "ollama"], check=False, log=False, ok=(0, 1)
        )
        if "ollama" in str(outdated.stdout or "").split():
            sys.run(["brew", "upgrade", "ollama"], check=True)
    elif info.os == "linux":
        # Streamed, not captured: the script may prompt for sudo.
        if sys.call(["sh", "-c", OLLAMA_INSTALL_SCRIPT]) != 0:
            raise FriendlyError(
                "Ollama upgrade failed.",
                "See https://ollama.com/download/linux for manual install steps.",
            )
    else:  # windows
        result = sys.run(
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


def update(info: Machine, cfg: Config, *, sys: System) -> list[UpdateStep]:
    """The Ollama steps of `lepika update` in Express mode, for an engine LePika manages."""
    steps: list[UpdateStep] = []
    # Read before the install script runs: it re-enables and starts the unit.
    systemd = info.os == "linux" and sys.which("systemctl") is not None
    unit = _unit_state(sys=sys) if systemd else None
    step, restored = _update_ollama(info, cfg, unit, sys=sys)
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
    return steps


def _update_ollama(
    info: Machine, cfg: Config, unit: tuple[bool, bool] | None, *, sys: System
) -> tuple[UpdateStep, bool]:
    """The Ollama step, and whether `ollama.service` is back as `unit` recorded it.

    `unit` is the unit's (enabled, active) as read before the upgrade (`_unit_state`);
    None leaves it as the install script sets it — there is none, or its state could
    not be read.
    """

    def answering() -> bool:
        return health.api_up(cfg.engine_url, key=cfg.engine_key, sys=sys)

    def running() -> str | None:
        """What the engine at the configured URL serves now, or None if it cannot say."""
        if not answering():
            return None
        try:
            return engine.version(cfg.engine_url, key=cfg.engine_key, sys=sys)
        except FriendlyError:
            # It answered a moment ago and not now, or not with JSON: unknown, and
            # not a reason to leave OpenWebUI un-upgraded.
            return None

    # Decided before anything changes: the upgrade may stop the process, and then
    # the pid file can no longer be vouched for by a running Ollama.
    ours_and_up = answering() and _ours(info.os, sys=sys)
    before = ollama_binary_version(sys=sys)
    failure: FriendlyError | None = None
    try:
        upgrade_ollama(info, sys=sys)
    except FriendlyError as exc:
        failure = exc
    finally:
        # Before anything below starts an engine: a unit the script started holds
        # port 11434. `_restore_unit` never raises, so it cannot mask `failure`.
        restored = _restore_unit(unit, sys=sys)
    after = ollama_binary_version(sys=sys)
    if failure is not None:
        # On Linux the binary, not the script's exit code, says whether the upgrade
        # happened: the script aborts on its optional systemd steps with the new
        # engine already in place — the same rule `install_ollama` follows.
        if info.os != "linux" or after == before:
            raise failure
        log.get_logger().warning("ollama.update.optional_step_failed")
    if ours_and_up:
        start = False
        if not answering():
            if _ours(info.os, sys=sys):
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
                    key=cfg.engine_key,
                    rerun="lepika update",
                    sys=sys,
                )
        if start:
            start_ollama(context_length=cfg.context_length, sys=sys)
            health.wait_for(answering, 30, "Ollama API", sys=sys)
    serving = running()
    if serving and after and serving != after:
        # Ours when its stop was refused; otherwise whoever started it restarts it.
        return UpdateStep(
            "Ollama", "stale", f"running {serving}, installed {after} — restart Ollama to load it"
        ), restored
    if before != after:
        return UpdateStep("Ollama", "upgraded", changed(before, after)), restored
    return UpdateStep("Ollama", "current", after or ""), restored
