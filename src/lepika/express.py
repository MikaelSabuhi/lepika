"""Express mode: native Ollama + OpenWebUI via uv. No Docker anywhere."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Callable
from typing import Any

from lepika import health, log, ollama, openwebui, proc, system
from lepika.backend import UpdateStep
from lepika.config import Config
from lepika.detect import SystemInfo
from lepika.errors import FriendlyError
from lepika.paths import logs_dir

RunFn = Callable[..., Any]
WhichFn = Callable[[str], str | None]
PopenFn = Callable[..., Any]
SleepFn = Callable[[float], None]
CallFn = Callable[[list[str]], int]


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
        ollama.ensure_ollama(info, url=cfg.engine_url, context_length=cfg.context_length)
    else:
        # Someone else runs this engine: never install or start anything for it.
        health.check_remote_engine(cfg, api_up=health.api_up)
    # Logged once the engine is really up, like Server: a pre-flight that refused
    # started no stack, and a `stack.up` above its own failure line reads as one that did.
    log.get_logger().info("stack.up", mode="express", engine_managed=cfg.engine_managed)
    if after_engine is not None:
        after_engine()
    openwebui.ensure_openwebui(cfg)
    return health.webui_url(cfg.webui_port)


def stop(info: SystemInfo, cfg: Config) -> bool:
    """`lepika down` in Express mode: stop OpenWebUI; Ollama stays as a shared service."""
    # The port is what proves the recorded pid is still our OpenWebUI.
    stopped = openwebui.stop_openwebui(info.os, port=cfg.webui_port)
    log.get_logger().info("stack.down", mode="express", stopped=stopped)
    return stopped


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
        unit = ollama._unit_state(run_fn, which_fn) if systemd else None
        step, restored = _update_ollama(
            info,
            cfg,
            run=run_fn,
            which=which_fn,
            call=call if call is not None else proc.call,
            popen=popen_fn,
            api_up=api_up if api_up is not None else health.api_up,
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
    before = openwebui.openwebui_version(run_fn)
    # Pinned like the install and the run: an upgrade that resolved a different
    # interpreter would move the tool env out from under `uv tool run --python`,
    # which then builds an ephemeral one instead of reusing it. check=True: uv exits
    # 0 with "Nothing to upgrade" when current, so a nonzero exit is a real failure.
    run_fn(
        ["uv", "tool", "upgrade", "--python", openwebui.OPENWEBUI_PYTHON, "open-webui"], check=True
    )
    after = openwebui.openwebui_version(run_fn)
    if before is not None and before == after:
        # Unchanged: a running UI is not interrupted for nothing.
        steps.append(UpdateStep("OpenWebUI", "current", after))
        return steps
    # A restart, not a stop-then-probe: the upgraded build only takes effect once
    # the old server is really gone. An unknown version restarts too.
    openwebui.restart_openwebui(
        cfg, info.os, run=run_fn, popen=popen_fn, sleep=sleep_fn, kill=kill_fn
    )
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

        def running_version(url: str, key: str = "") -> str:
            return engine.version(url, key, sys=system.REAL)

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
    ours_and_up = api_up(cfg.engine_url, key=cfg.engine_key) and ollama._ours(info.os, run)
    before = ollama.ollama_binary_version(run)
    failure: FriendlyError | None = None
    try:
        ollama.upgrade_ollama(info, run=run, which=which, call=call)
    except FriendlyError as exc:
        failure = exc
    finally:
        # Before anything below starts an engine: a unit the script started holds
        # port 11434. `_restore_unit` never raises, so it cannot mask `failure`.
        restored = ollama._restore_unit(unit, call, euid)
    after = ollama.ollama_binary_version(run)
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
            if ollama._ours(info.os, run):
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
                start = ollama.stop_ollama(
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
            ollama.start_ollama(info.os, popen=popen, context_length=cfg.context_length)
            health.wait_for(
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
