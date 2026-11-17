"""Shared test doubles: `FakeSystem` and its HTTP answers, and `FakeUI`."""

from __future__ import annotations

import contextlib
import io
import subprocess
import urllib.request
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from lepika import proc, ui
from lepika.errors import CommandNotFound, FriendlyError
from lepika.system import Response


def _leading(table: Mapping[str, Any], cmd: Sequence[str], default: Any) -> Any:
    """The answer for the first key the command line starts with, in the table's order."""
    line = " ".join(cmd)
    return next((v for k, v in table.items() if line.startswith(k)), default)


class UnansweredURL(BaseException):
    """A test let code open a URL it gave `FakeSystem` no answer for.

    A BaseException for the same reason as the suite's network guard: every opener
    in `src/` sits under an `except Exception` that would read this as `False`.
    """


def _no_url(request: urllib.request.Request) -> Response:
    url = getattr(request, "full_url", request)
    raise UnansweredURL(f"FakeSystem was not given an answer for {url}")


class FakeResponse:
    """What `open_url` hands back: a body read whole or line by line, and a status.

    `body` as bytes is split after each newline when iterated, as an HTTP response
    is; as a list, each item is one line. `fails` is raised once the lines run out —
    a read timeout or a dropped connection, long after the request succeeded.
    """

    def __init__(
        self,
        body: bytes | Sequence[bytes] = b"",
        *,
        status: int = 200,
        fails: BaseException | None = None,
    ) -> None:
        self.lines = list(io.BytesIO(body)) if isinstance(body, bytes) else list(body)
        self.status = status
        self.fails = fails
        self.closed = False

    def read(self) -> bytes:
        if self.fails is not None:
            raise self.fails
        return b"".join(self.lines)

    def __iter__(self) -> Iterator[bytes]:
        yield from self.lines
        if self.fails is not None:
            raise self.fails

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        self.closed = True


Answer = Response | BaseException


def routes(
    table: Mapping[str, Answer | Sequence[Answer]],
) -> Callable[[urllib.request.Request], Response]:
    """An `urls` answer for `FakeSystem`: by the request's full URL.

    A value that is an exception is raised (an `HTTPError`, a refused connection);
    a URL the table does not list raises `UnansweredURL`, which no broad except hides.
    A list answers that URL's successive requests in order, and its last item
    repeats: a server that is down and then up, or answering and then gone.
    """
    served: dict[str, int] = {}

    def answer(request: urllib.request.Request) -> Response:
        if request.full_url not in table:
            _no_url(request)
        found = table[request.full_url]
        if isinstance(found, Sequence):
            index = served.get(request.full_url, 0)
            served[request.full_url] = index + 1
            found = found[min(index, len(found) - 1)]
        if isinstance(found, BaseException):
            raise found
        return found

    return answer


class FakeSystem:
    """Implements `lepika.system.System` without touching the machine; records every call.

    `stdout`, `codes` and `raises` answer by the leading words of the argv: a
    matching `run`, `call` or `stream` is recorded, then raises (a timeout, say).
    A `stdout` list answers that command's successive runs in order, its last item
    repeating: the same question asked before and after an upgrade.
    A program named in `missing` is not installed for every launcher alike.
    `while_streaming` is handed the argv and cwd of each `stream` while it "runs":
    a working directory the caller stages is often gone once the call returns.
    `while_running` is handed the argv of each `run` the same way, before it returns:
    a test can leave behind what the real command would have written.
    `bindable` answers every port probe alike, or, as a callable, per address and
    port. `writable` answers every `can_write` alike, or, as a callable, per path.
    `signal_errors` is raised, after recording, for a signal sent to that pid.
    `route_address` and `host_addresses` are this machine's addresses, as unfiltered
    as the real ones: by default it has no route and its hostname resolves to nothing.
    """

    def __init__(
        self,
        *,
        stdout: dict[str, str | list[str]] | None = None,
        codes: dict[str, int] | None = None,
        which: dict[str, str] | None = None,
        environ: dict[str, str] | None = None,
        urls: Callable[[urllib.request.Request], Response] = _no_url,
        stream: tuple[int, str] = (0, ""),
        spawn_pid: int = 4242,
        euid: int | None = 1000,
        free: int = 10**12,
        bindable: bool | Callable[[str, int], bool] = True,
        writable: bool | Callable[[Path], bool] = True,
        signal_errors: dict[int, BaseException] | None = None,
        missing: set[str] | None = None,
        ram_bytes: int | None = 16 * 2**30,
        raises: dict[str, BaseException] | None = None,
        while_streaming: Callable[[list[str], Path | None], None] | None = None,
        while_running: Callable[[list[str]], None] | None = None,
        route_address: str | None = None,
        host_addresses: Sequence[str] = (),
    ) -> None:
        self.route = route_address
        self.hosts = list(host_addresses)
        self.stdout = stdout or {}
        self._outputs_served: dict[str, int] = {}
        self.while_streaming = while_streaming
        self.while_running = while_running
        self.codes = codes or {}
        self.paths = which or {}
        self.environ: Mapping[str, str] = environ if environ is not None else {}
        self.urls = urls
        self.stream_answer = stream
        self.spawn_pid = spawn_pid
        self.uid = euid
        self.free = free
        self.bindable = bindable
        self.writable = writable
        self.signal_errors = signal_errors or {}
        self.missing = missing or set()
        self.memory = ram_bytes
        self.raises = raises or {}
        self.ran: list[list[str]] = []
        self.ran_kwargs: list[dict[str, Any]] = []
        self.called: list[list[str]] = []
        self.streamed: list[tuple[list[str], dict[str, Any]]] = []
        self.spawned: list[tuple[list[str], Mapping[str, str], Path]] = []
        self.signals: list[tuple[int, int]] = []
        self.bind_probes: list[tuple[str, int]] = []
        self.write_probes: list[Path] = []
        self.slept: list[float] = []
        self.requested: list[urllib.request.Request] = []
        self.request_timeouts: list[float] = []
        self.browsed: list[str] = []

    def _launch(self, cmd: Sequence[str]) -> None:
        if cmd and cmd[0] in self.missing:
            raise CommandNotFound(cmd[0])

    def _start(self, cmd: Sequence[str]) -> None:
        """`_launch`, then whatever `raises` holds for this command."""
        self._launch(cmd)
        error = _leading(self.raises, cmd, None)
        if error is not None:
            raise error

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
            {
                "check": check,
                "env": dict(env) if env is not None else None,
                "timeout": timeout,
                "log": log,
                "ok": ok,
            }
        )
        self._start(cmd)
        if self.while_running is not None:
            self.while_running(list(cmd))
        code = _leading(self.codes, cmd, 0)
        if check and code not in ok:
            # The same error `proc.run_logged` raises for an unexpected exit code.
            raise FriendlyError(
                f"Command failed: {' '.join(cmd)}",
                f"Details were logged to {proc._log_path()}",
            )
        return subprocess.CompletedProcess(list(cmd), code, stdout=self._output(cmd), stderr="")

    def _output(self, cmd: Sequence[str]) -> str:
        line = " ".join(cmd)
        key = next((k for k in self.stdout if line.startswith(k)), None)
        if key is None:
            return ""
        found = self.stdout[key]
        if isinstance(found, str):
            return found
        index = self._outputs_served.get(key, 0)
        self._outputs_served[key] = index + 1
        return found[min(index, len(found) - 1)]

    def call(self, cmd: Sequence[str]) -> int:
        self.called.append(list(cmd))
        self._start(cmd)
        code: int = _leading(self.codes, cmd, 0)
        return code

    def stream(
        self,
        cmd: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
    ) -> tuple[int, str]:
        self.streamed.append(
            (list(cmd), {"env": dict(env) if env is not None else None, "cwd": cwd})
        )
        self._start(cmd)
        if self.while_streaming is not None:
            self.while_streaming(list(cmd), cwd)
        return self.stream_answer

    def spawn(self, cmd: Sequence[str], *, env: Mapping[str, str], log: Path) -> int:
        self.spawned.append((list(cmd), dict(env), log))
        self._launch(cmd)
        return self.spawn_pid

    def which(self, name: str) -> str | None:
        return self.paths.get(name)

    def open_url(self, request: urllib.request.Request, *, timeout: float) -> Response:
        self.requested.append(request)
        self.request_timeouts.append(timeout)
        return self.urls(request)

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)

    def signal(self, pid: int, sig: int) -> None:
        self.signals.append((pid, sig))
        error = self.signal_errors.get(pid)
        if error is not None:
            raise error

    def can_bind(self, address: str, port: int) -> bool:
        self.bind_probes.append((address, port))
        if callable(self.bindable):
            return self.bindable(address, port)
        return self.bindable

    def open_browser(self, url: str) -> None:
        self.browsed.append(url)

    def euid(self) -> int | None:
        return self.uid

    def can_write(self, path: Path) -> bool:
        self.write_probes.append(path)
        if callable(self.writable):
            return self.writable(path)
        return self.writable

    def disk_free(self, path: Path) -> int:
        return self.free

    def ram_bytes(self) -> int | None:
        return self.memory

    def route_address(self) -> str | None:
        return self.route

    def host_addresses(self) -> list[str]:
        return list(self.hosts)


class UnansweredQuestion(BaseException):
    """A test let code ask a question it gave `FakeUI` no answer for.

    A BaseException, not the `EOFError` a terminal-less stdin gives: that would be
    read as "no terminal, take the default", and the test would pass for the wrong
    reason. Nothing in `src/` catches a BaseException.
    """


class FakeUI(ui.UI):
    """The real `UI` with its input side scripted; output still renders through rich.

    `answers` answer `ask` and `confirms` answer `confirm`, in order. A caller's own
    `reader`/`confirm_reader` answers instead, so a reader that asserts on what it was
    handed keeps doing so. `no_terminal` makes every question meet EOF, which the real
    policy in `UI.ask`/`UI.confirm` then handles. Every question is recorded in
    `asked`/`confirmed` with the keywords its reader received; one nobody scripted
    raises `UnansweredQuestion`. `progress` draws nothing: it records each bar's label
    in `progress_labels` and every `(completed, total)` report in `reports`.
    """

    def __init__(
        self,
        *,
        answers: Sequence[str] = (),
        confirms: Sequence[bool] = (),
        reader: ui.Reader | None = None,
        confirm_reader: ui.Reader | None = None,
        no_terminal: bool = False,
    ) -> None:
        self.answers = list(answers)
        self.confirms = list(confirms)
        self.caller_reader = reader
        self.caller_confirm_reader = confirm_reader
        self.no_terminal = no_terminal
        self.asked: list[tuple[str, dict[str, Any]]] = []
        self.confirmed: list[tuple[str, dict[str, Any]]] = []
        self.progress_labels: list[str] = []
        self.reports: list[tuple[int, int]] = []
        super().__init__(reader=self._read, confirm_reader=self._read_confirm)

    def _read(self, prompt: str, **kwargs: Any) -> Any:
        self.asked.append((prompt, kwargs))
        if self.no_terminal:
            raise EOFError("stdin is not a terminal")
        if self.caller_reader is not None:
            return self.caller_reader(prompt, **kwargs)
        if not self.answers:
            raise UnansweredQuestion(f"FakeUI was not given an answer for {prompt!r}")
        return self.answers.pop(0)

    def _read_confirm(self, prompt: str, **kwargs: Any) -> Any:
        self.confirmed.append((prompt, kwargs))
        if self.no_terminal:
            raise EOFError("stdin is not a terminal")
        if self.caller_confirm_reader is not None:
            return self.caller_confirm_reader(prompt, **kwargs)
        if not self.confirms:
            raise UnansweredQuestion(f"FakeUI was not given a yes/no for {prompt!r}")
        return self.confirms.pop(0)

    @contextlib.contextmanager
    def progress(self, label: str) -> Iterator[ui.ProgressFn]:
        self.progress_labels.append(label)

        def report(completed: int, total: int) -> None:
            self.reports.append((completed, total))

        yield report
