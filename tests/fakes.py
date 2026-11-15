"""Shared test doubles: `FakeSystem`, and the injected subprocess callables."""

from __future__ import annotations

import subprocess
import urllib.request
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

from lepika import proc
from lepika.errors import CommandNotFound, FriendlyError
from lepika.system import Response


class Runner:
    """Stands in for proc.run_logged; canned stdout per leading argv words."""

    def __init__(self, stdout: dict[str, str] | None = None, code: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.stdout = stdout or {}
        self.code = code

    def __call__(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(cmd))
        out = next((v for k, v in self.stdout.items() if " ".join(cmd).startswith(k)), "")
        return subprocess.CompletedProcess(cmd, self.code, stdout=out, stderr="")


class Caller:
    """Stands in for proc.call: records the argv, returns a canned exit code."""

    def __init__(self, code: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.code = code

    def __call__(self, cmd: list[str]) -> int:
        self.calls.append(list(cmd))
        return self.code


class Streamer:
    """Stands in for proc.stream: records argv and kwargs, returns a canned (code, tail)."""

    def __init__(self, code: int = 0, tail: str = "") -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.code = code
        self.tail = tail

    def __call__(self, cmd: list[str], **kwargs: Any) -> tuple[int, str]:
        self.calls.append((list(cmd), dict(kwargs)))
        return self.code, self.tail


def _leading(table: Mapping[str, Any], cmd: Sequence[str], default: Any) -> Any:
    """The answer for the first key `cmd` starts with: the matching rule `Runner` uses."""
    line = " ".join(cmd)
    return next((v for k, v in table.items() if line.startswith(k)), default)


def _no_url(request: urllib.request.Request) -> Response:
    raise AssertionError(f"FakeSystem was not given an answer for {request.full_url}")


class FakeSystem:
    """Implements `lepika.system.System` without touching the machine; records every call.

    `stdout` and `codes` answer by the leading words of the argv, and a program
    named in `missing` is not installed for every launcher alike.
    """

    def __init__(
        self,
        *,
        stdout: dict[str, str] | None = None,
        codes: dict[str, int] | None = None,
        which: dict[str, str] | None = None,
        environ: dict[str, str] | None = None,
        urls: Callable[[urllib.request.Request], Response] = _no_url,
        stream: tuple[int, str] = (0, ""),
        spawn_pid: int = 4242,
        euid: int | None = 1000,
        free: int = 10**12,
        bindable: bool = True,
        missing: set[str] | None = None,
    ) -> None:
        self.stdout = stdout or {}
        self.codes = codes or {}
        self.paths = which or {}
        self.environ: Mapping[str, str] = environ if environ is not None else {}
        self.urls = urls
        self.stream_answer = stream
        self.spawn_pid = spawn_pid
        self.uid = euid
        self.free = free
        self.bindable = bindable
        self.missing = missing or set()
        self.ran: list[list[str]] = []
        self.ran_kwargs: list[dict[str, Any]] = []
        self.called: list[list[str]] = []
        self.streamed: list[tuple[list[str], dict[str, Any]]] = []
        self.spawned: list[tuple[list[str], Mapping[str, str], Path]] = []
        self.signals: list[tuple[int, int]] = []
        self.slept: list[float] = []
        self.requested: list[urllib.request.Request] = []
        self.browsed: list[str] = []

    def _launch(self, cmd: Sequence[str]) -> None:
        if cmd and cmd[0] in self.missing:
            raise CommandNotFound(cmd[0])

    def run(
        self,
        cmd: Sequence[str],
        *,
        check: bool = True,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        log: bool = True,
        ok: Collection[int] = (0,),
    ) -> subprocess.CompletedProcess[str]:
        self.ran.append(list(cmd))
        self.ran_kwargs.append(
            {"check": check, "env": env, "timeout": timeout, "log": log, "ok": ok}
        )
        self._launch(cmd)
        code = _leading(self.codes, cmd, 0)
        if check and code not in ok:
            # The same error `proc.run_logged` raises for an unexpected exit code.
            raise FriendlyError(
                f"Command failed: {' '.join(cmd)}",
                f"Details were logged to {proc._log_path()}",
            )
        out = _leading(self.stdout, cmd, "")
        return subprocess.CompletedProcess(list(cmd), code, stdout=out, stderr="")

    def call(self, cmd: Sequence[str]) -> int:
        self.called.append(list(cmd))
        self._launch(cmd)
        code: int = _leading(self.codes, cmd, 0)
        return code

    def stream(
        self,
        cmd: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
    ) -> tuple[int, str]:
        self.streamed.append((list(cmd), {"env": env, "cwd": cwd}))
        self._launch(cmd)
        return self.stream_answer

    def spawn(self, cmd: Sequence[str], *, env: Mapping[str, str], log: Path) -> int:
        self.spawned.append((list(cmd), env, log))
        self._launch(cmd)
        return self.spawn_pid

    def which(self, name: str) -> str | None:
        return self.paths.get(name)

    def open_url(self, request: urllib.request.Request, *, timeout: float) -> Response:
        self.requested.append(request)
        return self.urls(request)

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)

    def signal(self, pid: int, sig: int) -> None:
        self.signals.append((pid, sig))

    def can_bind(self, address: str, port: int) -> bool:
        return self.bindable

    def open_browser(self, url: str) -> None:
        self.browsed.append(url)

    def euid(self) -> int | None:
        return self.uid

    def disk_free(self, path: Path) -> int:
        return self.free
