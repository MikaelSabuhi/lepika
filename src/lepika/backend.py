"""The two backends' one contract, and the one place a mode picks its backend."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from lepika.config import Config
from lepika.detect import Machine
from lepika.express import ExpressBackend
from lepika.server import ServerBackend
from lepika.system import System
from lepika.ui import UI
from lepika.updates import UpdateStep


class Backend(Protocol):
    """What every lifecycle command needs from a mode, so each is written once."""

    def start(
        self, info: Machine, cfg: Config, after_engine: Callable[[], None] | None = None
    ) -> str: ...

    def stop(self, info: Machine, cfg: Config) -> bool: ...

    def update(self, info: Machine, cfg: Config) -> list[UpdateStep]: ...

    def logs(self, lines: int) -> list[tuple[str, str]]: ...


def backend_for(cfg: Config, *, sys: System, ui: UI) -> Backend:
    """The backend for this config's mode — the only place the two modes are told apart."""
    if cfg.mode == "server":
        return ServerBackend(sys=sys, ui=ui)
    return ExpressBackend(sys=sys, ui=ui)
