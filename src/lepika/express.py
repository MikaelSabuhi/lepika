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


class ExpressBackend:
    """Express mode's lifecycle: native Ollama, OpenWebUI via uv."""

    def __init__(self, *, sys: System, ui: UI) -> None:
        self.sys = sys
        self.ui = ui

    def start(
        self, info: Machine, cfg: Config, after_engine: Callable[[], None] | None = None
    ) -> str:
        """Bring the stack up and return the URL to open.

        The single source of truth for the ordering both `lepika up` and the wizard
        depend on: engine first, then whatever the caller needs the engine for (the
        wizard pulls a model here), then the UI that talks to it. `lepika up` passes no
        hook; the wizard passes its pull.
        """
        if cfg.engine_managed:
            ollama.ensure_ollama(
                info, url=cfg.engine_url, context_length=cfg.context_length, sys=self.sys
            )
        else:
            # Someone else runs this engine: never install or start anything for it.
            health.check_remote_engine(cfg, sys=self.sys)
        # Logged once the engine is really up, like Server: a pre-flight that refused
        # started no stack, and a `stack.up` above its own failure line reads as one that did.
        log.get_logger().info("stack.up", mode="express", engine_managed=cfg.engine_managed)
        if after_engine is not None:
            after_engine()
        openwebui.ensure_openwebui(cfg, sys=self.sys)
        return health.webui_url(cfg.webui_port)

    def stop(self, info: Machine, cfg: Config) -> bool:
        """`lepika down` in Express mode: stop OpenWebUI; Ollama stays as a shared service."""
        # The port is what proves the recorded pid is still our OpenWebUI.
        stopped = openwebui.stop_openwebui(info.os, port=cfg.webui_port, sys=self.sys)
        log.get_logger().info("stack.down", mode="express", stopped=stopped)
        return stopped

    def update(self, info: Machine, cfg: Config) -> list[UpdateStep]:
        """`lepika update` in Express mode: upgrade the engine (if ours) and OpenWebUI."""
        steps = ollama.update(info, cfg, sys=self.sys) if cfg.engine_managed else []
        before = openwebui.openwebui_version(sys=self.sys)
        # Pinned like the install and the run: an upgrade that resolved a different
        # interpreter would move the tool env out from under `uv tool run --python`,
        # which then builds an ephemeral one instead of reusing it. check=True: uv exits
        # 0 with "Nothing to upgrade" when current, so a nonzero exit is a real failure.
        self.sys.run(
            ["uv", "tool", "upgrade", "--python", openwebui.OPENWEBUI_PYTHON, "open-webui"],
            check=True,
        )
        after = openwebui.openwebui_version(sys=self.sys)
        if before is not None and before == after:
            # Unchanged: a running UI is not interrupted for nothing.
            steps.append(UpdateStep("OpenWebUI", "current", after))
            return steps
        # A restart, not a stop-then-probe: the upgraded build only takes effect once
        # the old server is really gone. An unknown version restarts too.
        openwebui.restart_openwebui(cfg, info.os, sys=self.sys)
        steps.append(UpdateStep("OpenWebUI", "upgraded", changed(before, after)))
        return steps

    def logs(self, lines: int) -> list[tuple[str, str]]:
        """`lepika logs` in Express mode: the tail of every file under ~/.lepika/logs."""
        out: list[tuple[str, str]] = []
        for log_file in sorted(logs_dir().glob("*.log")):
            content = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
            out.append((log_file.name, "\n".join(content[-lines:])))
        return out

    def notes(self, info: Machine, cfg: Config) -> list[str]:
        """Before `lepika up` starts anything: whether config.toml's context length can apply."""
        note = ollama.context_note(info, cfg, sys=self.sys)
        return [] if note is None else [note]

    def apply_engine_change(self, info: Machine, cfg: Config) -> bool:
        """Make a running OpenWebUI use the engine just configured; did anything restart?

        OpenWebUI reads the engine URL and key from its environment once, at startup, so
        a UI that is already up keeps talking to the old engine no matter what the config
        says — and `lepika up` would not fix it, because a healthy UI is left alone. This
        is what makes `lepika connect` take effect on a live stack.
        """
        if not health.webui_up(cfg.webui_port, sys=self.sys):
            return False
        openwebui.restart_openwebui(cfg, info.os, sys=self.sys)
        return True

    def ensure_engine(self, info: Machine, cfg: Config) -> None:
        """Have the engine ready to pull a model into: ours started, a remote one checked."""
        if cfg.engine_managed:
            ollama.ensure_ollama(
                info, url=cfg.engine_url, context_length=cfg.context_length, sys=self.sys
            )
        else:
            # A remote engine is someone else's to run: check it, never install for it.
            health.check_remote_engine(cfg, sys=self.sys)

    def release(self, info: Machine, cfg: Config) -> str | None:
        """On the way to Server mode: stop the native Ollama LePika itself started."""
        # `lepika down` leaves Ollama up on purpose, but Server's pre-flight refuses a
        # port 11434 that a native engine holds. Only the one LePika started is stopped
        # — anything else has no pid file — and rule 9 keeps a remote engine untouched,
        # which is what `engine_managed` short-circuits before the call.
        if not cfg.engine_managed:
            return None
        if ollama.stop_ollama(info.os, cfg.engine_url, key=cfg.engine_key, sys=self.sys):
            return "Stopped the Ollama LePika had started."
        return None
