"""Express mode: native Ollama + OpenWebUI via uv. No Docker anywhere."""

from __future__ import annotations

from collections.abc import Callable

from lepika import health, log, ollama, openwebui
from lepika.config import Config
from lepika.detect import Machine
from lepika.paths import logs_dir
from lepika.system import System
from lepika.ui import UI
from lepika.updates import UpdateStep, changed


def start_stack(
    info: Machine,
    cfg: Config,
    after_engine: Callable[[], None] | None = None,
    *,
    sys: System,
    ui: UI,
) -> str:
    """Bring the stack up and return the URL to open.

    The single source of truth for the ordering both `lepika up` and the wizard
    depend on: engine first, then whatever the caller needs the engine for (the
    wizard pulls a model here), then the UI that talks to it. `lepika up` passes no
    hook; the wizard passes its pull. `ui` is unused: it is taken only because both
    backends share one call site, and Server's start says something through it.
    """
    if cfg.engine_managed:
        ollama.ensure_ollama(info, url=cfg.engine_url, context_length=cfg.context_length, sys=sys)
    else:
        # Someone else runs this engine: never install or start anything for it.
        health.check_remote_engine(cfg, sys=sys)
    # Logged once the engine is really up, like Server: a pre-flight that refused
    # started no stack, and a `stack.up` above its own failure line reads as one that did.
    log.get_logger().info("stack.up", mode="express", engine_managed=cfg.engine_managed)
    if after_engine is not None:
        after_engine()
    openwebui.ensure_openwebui(cfg, sys=sys)
    return health.webui_url(cfg.webui_port)


def stop(info: Machine, cfg: Config, *, sys: System) -> bool:
    """`lepika down` in Express mode: stop OpenWebUI; Ollama stays as a shared service."""
    # The port is what proves the recorded pid is still our OpenWebUI.
    stopped = openwebui.stop_openwebui(info.os, port=cfg.webui_port, sys=sys)
    log.get_logger().info("stack.down", mode="express", stopped=stopped)
    return stopped


def update(info: Machine, cfg: Config, *, sys: System, ui: UI) -> list[UpdateStep]:
    """`lepika update` in Express mode: upgrade the engine (if ours) and OpenWebUI.

    `ui` is unused, as in `start_stack`: Server's update restarts its stack through it.
    """
    steps = ollama.update(info, cfg, sys=sys) if cfg.engine_managed else []
    before = openwebui.openwebui_version(sys=sys)
    # Pinned like the install and the run: an upgrade that resolved a different
    # interpreter would move the tool env out from under `uv tool run --python`,
    # which then builds an ephemeral one instead of reusing it. check=True: uv exits
    # 0 with "Nothing to upgrade" when current, so a nonzero exit is a real failure.
    sys.run(
        ["uv", "tool", "upgrade", "--python", openwebui.OPENWEBUI_PYTHON, "open-webui"], check=True
    )
    after = openwebui.openwebui_version(sys=sys)
    if before is not None and before == after:
        # Unchanged: a running UI is not interrupted for nothing.
        steps.append(UpdateStep("OpenWebUI", "current", after))
        return steps
    # A restart, not a stop-then-probe: the upgraded build only takes effect once
    # the old server is really gone. An unknown version restarts too.
    openwebui.restart_openwebui(cfg, info.os, sys=sys)
    steps.append(UpdateStep("OpenWebUI", "upgraded", changed(before, after)))
    return steps


def logs(lines: int, *, sys: System) -> list[tuple[str, str]]:
    """`lepika logs` in Express mode: the tail of every file under ~/.lepika/logs.

    `sys` is unused: it is taken only because both backends share one call site in
    the CLI, which passes it to whichever backend the config names.
    """
    out: list[tuple[str, str]] = []
    for log_file in sorted(logs_dir().glob("*.log")):
        content = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        out.append((log_file.name, "\n".join(content[-lines:])))
    return out
