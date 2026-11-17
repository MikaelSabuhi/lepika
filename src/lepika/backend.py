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
    ) -> str:
        """Bring the stack up for `cfg`; the URL to open.

        `after_engine` runs once the engine answers — for vLLM, once `compose up` succeeded.
        """
        ...

    def stop(self, info: Machine, cfg: Config) -> bool: ...

    def update(self, info: Machine, cfg: Config) -> list[UpdateStep]: ...

    def logs(self, lines: int) -> list[tuple[str, str]]: ...

    def notes(self, info: Machine, cfg: Config) -> list[str]:
        """Lines `lepika up` shows before it starts anything, in order. Asked before `start`."""
        ...

    def apply_engine_change(self, info: Machine, cfg: Config) -> bool:
        """Make a running stack use the engine the config now names; did anything restart?"""
        ...

    def ensure_engine(self, info: Machine, cfg: Config) -> None:
        """Have the engine `cfg` names ready to pull a model into."""
        ...

    def release(self, info: Machine, cfg: Config) -> str | None:
        """After `stop`, on the way to the other mode: free what that mode needs.

        A line to show if something was freed.
        """
        ...


def backend_for(cfg: Config, *, sys: System, ui: UI) -> Backend:
    """The backend for this config's mode — the only place the two modes are told apart."""
    if cfg.mode == "server":
        return ServerBackend(sys=sys, ui=ui)
    return ExpressBackend(sys=sys, ui=ui)
