"""The `System` boundary: the real effects it wraps, and the fake that stands in for it."""

from __future__ import annotations

import inspect
import io
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import RealNetworkCall
from fakes import FakeSystem, UnansweredURL

from lepika import proc, system
from lepika.errors import CommandNotFound, FriendlyError
from lepika.system import REAL, RealSystem, System

_METHODS = (
    "run",
    "call",
    "stream",
    "spawn",
    "which",
    "open_url",
    "sleep",
    "signal",
    "can_bind",
    "open_browser",
    "euid",
    "disk_free",
    "ram_bytes",
)


def _shape(func: Any) -> list[tuple[str, Any, Any]]:
    # Annotations are strings under `from __future__ import annotations`, and how
    # they print differs between interpreters: names, kinds and defaults are what
    # a caller depends on.
    return [(p.name, p.kind, p.default) for p in inspect.signature(func).parameters.values()]


def test_real_and_fake_system_match_the_protocol_signatures() -> None:
    declared = {
        name
        for name, value in vars(System).items()
        if not name.startswith("_") and (callable(value) or isinstance(value, property))
    } | set(System.__annotations__)
    assert declared == {*_METHODS, "environ"}
    assert isinstance(vars(System)["environ"], property)
    for name in _METHODS:
        expected = _shape(getattr(System, name))
        assert _shape(getattr(RealSystem, name)) == expected, name
        assert _shape(getattr(FakeSystem, name)) == expected, name
    assert isinstance(REAL, System)
    assert isinstance(FakeSystem(), System)


def test_real_system_run_call_and_stream_go_through_proc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[tuple[str, list[str], dict[str, Any]]] = []
    done = subprocess.CompletedProcess(["x"], 3, stdout="out", stderr="")

    def run_logged(cmd: list[str], **kwargs: Any) -> Any:
        seen.append(("run", list(cmd), kwargs))
        return done

    def call(cmd: list[str], **kwargs: Any) -> int:
        seen.append(("call", list(cmd), kwargs))
        return 7

    def stream(cmd: list[str], **kwargs: Any) -> tuple[int, str]:
        seen.append(("stream", list(cmd), kwargs))
        return 1, "tail"

    monkeypatch.setattr(proc, "run_logged", run_logged)
    monkeypatch.setattr(proc, "call", call)
    monkeypatch.setattr(proc, "stream", stream)

    env = {"A": "1"}
    assert (
        REAL.run(["systemctl", "is-active"], check=False, env=env, timeout=5, log=False, ok=(0, 3))
        is done
    )
    assert REAL.call(["sudo", "true"]) == 7
    assert REAL.stream(["hf", "download"], env=env, cwd=tmp_path) == (1, "tail")
    assert seen == [
        (
            "run",
            ["systemctl", "is-active"],
            {"check": False, "env": env, "timeout": 5, "log": False, "ok": (0, 3)},
        ),
        ("call", ["sudo", "true"], {}),
        ("stream", ["hf", "download"], {"env": env, "cwd": tmp_path}),
    ]


class FakePopen:
    """Records how a child was started and the handle it was given."""

    def __init__(self, pid: int = 4321, error: BaseException | None = None) -> None:
        self.pid = pid
        self.error = error
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, cmd: list[str], **kwargs: Any) -> FakePopen:
        self.calls.append((list(cmd), dict(kwargs)))
        if self.error is not None:
            raise self.error
        return self


def _current_detach() -> dict[str, Any]:
    return system._detach_kwargs("windows" if os.name == "nt" else "posix")


def test_spawn_returns_the_pid_and_appends_output_to_the_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log = tmp_path / "ollama.log"
    log.write_bytes(b"earlier run\n")
    popen = FakePopen(pid=4321)
    monkeypatch.setattr(subprocess, "Popen", popen)

    pid = REAL.spawn(["ollama", "serve"], env={"OLLAMA_HOST": "127.0.0.1"}, log=log)

    assert pid == 4321
    cmd, kwargs = popen.calls[0]
    assert cmd == ["ollama", "serve"]
    assert kwargs["env"] == {"OLLAMA_HOST": "127.0.0.1"}
    handle = kwargs["stdout"]
    assert kwargs["stderr"] is handle
    assert isinstance(handle, io.BufferedWriter)
    assert Path(handle.name) == log
    assert handle.mode == "ab"
    detach = _current_detach()
    assert {k: kwargs[k] for k in detach} == detach
    assert set(kwargs) == {"stdout", "stderr", "env", *detach}
    # Appended to, never truncated: the previous run's output is still there.
    assert log.read_bytes() == b"earlier run\n"


def test_spawn_closes_its_own_copy_of_the_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    popen = FakePopen()
    monkeypatch.setattr(subprocess, "Popen", popen)
    REAL.spawn(["ollama", "serve"], env={}, log=tmp_path / "ollama.log")
    assert popen.calls[0][1]["stdout"].closed


def test_spawn_reports_a_missing_program_as_command_not_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    popen = FakePopen(error=FileNotFoundError(2, "No such file or directory", "ollama"))
    monkeypatch.setattr(subprocess, "Popen", popen)
    with pytest.raises(CommandNotFound) as exc:
        REAL.spawn(["ollama", "serve"], env={}, log=tmp_path / "ollama.log")
    assert exc.value.problem == "Command not found: ollama"
    assert exc.value.fix == "Install ollama or run `lepika doctor` for setup help."
    # The handle is closed on the failure path too.
    assert popen.calls[0][1]["stdout"].closed


def test_detach_flags_for_windows_and_posix() -> None:
    # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    assert system._detach_kwargs("windows") == {"creationflags": 0x00000208}
    assert system._detach_kwargs("posix") == {"start_new_session": True}


def test_euid_is_none_where_the_platform_has_none(monkeypatch: pytest.MonkeyPatch) -> None:
    if sys.platform != "win32":
        assert REAL.euid() == os.geteuid()
    monkeypatch.setattr(sys, "platform", "win32")
    assert REAL.euid() is None


def _sysconf(answers: dict[str, int]) -> Any:
    def sysconf(name: str) -> int:
        return answers[name]

    return sysconf


def test_ram_bytes_is_physical_pages_times_page_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    # `raising=False`: Windows has no `os.sysconf`, and the suite runs there too.
    answers = {"SC_PHYS_PAGES": 4 * 2**20, "SC_PAGE_SIZE": 4096}
    monkeypatch.setattr(os, "sysconf", _sysconf(answers), raising=False)
    assert REAL.ram_bytes() == 16 * 2**30


def test_ram_bytes_is_none_when_sysconf_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    for error in (ValueError("unknown name"), OSError("not supported")):

        def failing(name: str, error: Exception = error) -> int:
            raise error

        monkeypatch.setattr(os, "sysconf", failing, raising=False)
        assert REAL.ram_bytes() is None, type(error).__name__


def test_ram_bytes_is_none_for_a_non_positive_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    # sysconf answers -1 for a value the platform does not know, without raising.
    for pages in (-1, 0):
        answers = {"SC_PHYS_PAGES": pages, "SC_PAGE_SIZE": 4096}
        monkeypatch.setattr(os, "sysconf", _sysconf(answers), raising=False)
        assert REAL.ram_bytes() is None, pages


def test_ram_bytes_on_windows_reads_global_memory_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(system, "_global_memory_status", lambda: (True, 16 * 2**30))
    assert REAL.ram_bytes() == 16 * 2**30


def test_ram_bytes_on_windows_is_none_when_the_call_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GlobalMemoryStatusEx signals failure through its BOOL return and leaves the
    struct untouched: reading it anyway reported a 0 GB machine."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(system, "_global_memory_status", lambda: (False, 0))
    assert REAL.ram_bytes() is None


def test_can_bind_is_false_for_a_port_already_taken() -> None:
    # A listening socket with no options, as a server holding the port would be.
    # The probe sets SO_REUSEADDR on POSIX and SO_EXCLUSIVEADDRUSE on Windows; a
    # listener on the same address refuses both on every OS.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        port = holder.getsockname()[1]
        assert REAL.can_bind("127.0.0.1", port) is False


def test_open_url_cannot_reach_the_network_under_the_suite_guard() -> None:
    request = urllib.request.Request("http://example.invalid/api/version")
    with pytest.raises(RealNetworkCall, match=re.escape("example.invalid")):
        REAL.open_url(request, timeout=0.1)


def test_which_sleep_signal_browser_and_disk_free_delegate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[tuple[str, tuple[Any, ...]]] = []

    def record(name: str, answer: Any = None) -> Any:
        def fake(*args: Any) -> Any:
            seen.append((name, args))
            return answer

        return fake

    monkeypatch.setattr(shutil, "which", record("which", "/usr/bin/ollama"))
    monkeypatch.setattr(time, "sleep", record("sleep"))
    monkeypatch.setattr(os, "kill", record("kill"))
    monkeypatch.setattr(webbrowser, "open", record("open", True))
    usage = SimpleNamespace(total=100, used=40, free=60)
    monkeypatch.setattr(shutil, "disk_usage", record("disk_usage", usage))

    assert REAL.which("ollama") == "/usr/bin/ollama"
    REAL.sleep(0.5)
    REAL.signal(123, 15)
    REAL.open_browser("http://localhost:3000")
    assert REAL.disk_free(tmp_path) == 60
    assert seen == [
        ("which", ("ollama",)),
        ("sleep", (0.5,)),
        ("kill", (123, 15)),
        ("open", ("http://localhost:3000",)),
        ("disk_usage", (tmp_path,)),
    ]
    assert REAL.environ is os.environ


def test_fake_system_answers_run_by_leading_argv() -> None:
    fake = FakeSystem(
        stdout={"ollama --version": "ollama version is 0.33.2"},
        codes={"systemctl is-active": 3},
    )
    version = fake.run(["ollama", "--version"])
    assert (version.returncode, version.stdout) == (0, "ollama version is 0.33.2")
    active = fake.run(["systemctl", "is-active", "ollama"], check=False)
    assert (active.returncode, active.stdout) == (3, "")
    assert fake.call(["systemctl", "is-active", "ollama"]) == 3
    assert fake.call(["true"]) == 0


def test_fake_system_honours_check_and_ok_like_the_real_runner() -> None:
    fake = FakeSystem(codes={"systemctl is-active": 3})
    cmd = ["systemctl", "is-active", "ollama"]
    with pytest.raises(FriendlyError) as exc:
        fake.run(cmd)
    assert exc.value.problem == "Command failed: systemctl is-active ollama"
    assert not isinstance(exc.value, CommandNotFound)
    assert fake.run(cmd, ok=(0, 3)).returncode == 3
    assert fake.run(cmd, check=False).returncode == 3


def test_fake_system_reports_a_missing_program_from_every_launcher(tmp_path: Path) -> None:
    fake = FakeSystem(missing={"ollama"})
    launchers = [
        lambda: fake.run(["ollama", "serve"], check=False),
        lambda: fake.call(["ollama", "serve"]),
        lambda: fake.stream(["ollama", "create"]),
        lambda: fake.spawn(["ollama", "serve"], env={}, log=tmp_path / "x.log"),
    ]
    for launch in launchers:
        with pytest.raises(CommandNotFound) as exc:
            launch()
        assert exc.value.problem == "Command not found: ollama"
    # The other programs on the box are still there.
    assert fake.call(["uv", "--version"]) == 0


def test_fake_system_records_every_effect_in_order(tmp_path: Path) -> None:
    seen: list[str] = []

    def urls(request: urllib.request.Request) -> Any:
        seen.append(request.full_url)
        return "response"

    fake = FakeSystem(
        which={"ollama": "/usr/bin/ollama"},
        environ={"HF_TOKEN": "t"},
        urls=urls,
        stream=(1, "boom"),
        spawn_pid=99,
        euid=0,
        free=5,
        bindable=False,
    )
    fake.run(["a", "1"], timeout=3)
    fake.run(["b"], log=False)
    fake.call(["c"])
    fake.call(["d"])
    assert fake.stream(["e"], env={"K": "v"}, cwd=tmp_path) == (1, "boom")
    assert fake.spawn(["f"], env={"E": "1"}, log=tmp_path / "f.log") == 99
    fake.signal(99, 15)
    fake.signal(98, 9)
    fake.sleep(0.5)
    fake.sleep(1.0)
    first = urllib.request.Request("http://x/1")
    second = urllib.request.Request("http://x/2")
    assert fake.open_url(first, timeout=1) == "response"
    assert fake.open_url(second, timeout=1) == "response"
    fake.open_browser("http://localhost:3000")
    fake.open_browser("http://localhost:3001")

    assert fake.ran == [["a", "1"], ["b"]]
    assert fake.ran_kwargs[0]["timeout"] == 3
    assert fake.ran_kwargs[1]["log"] is False
    assert fake.called == [["c"], ["d"]]
    assert fake.streamed == [(["e"], {"env": {"K": "v"}, "cwd": tmp_path})]
    assert fake.spawned == [(["f"], {"E": "1"}, tmp_path / "f.log")]
    assert fake.signals == [(99, 15), (98, 9)]
    assert fake.slept == [0.5, 1.0]
    assert fake.requested == [first, second]
    assert seen == ["http://x/1", "http://x/2"]
    assert fake.browsed == ["http://localhost:3000", "http://localhost:3001"]
    assert fake.which("ollama") == "/usr/bin/ollama"
    assert fake.which("docker") is None
    assert fake.environ == {"HF_TOKEN": "t"}
    assert (fake.euid(), fake.disk_free(tmp_path), fake.can_bind("0.0.0.0", 1)) == (0, 5, False)


def test_fake_system_refuses_an_unanswered_url() -> None:
    fake = FakeSystem()
    with pytest.raises(UnansweredURL, match=re.escape("http://x/unanswered")):
        fake.open_url(urllib.request.Request("http://x/unanswered"), timeout=1)


def test_an_unanswered_url_is_not_swallowed_by_a_broad_except() -> None:
    """Every opener in `src/` turns an `Exception` into `False` or a fallback, so a
    refusal that was one would let a test that forgot to answer a URL pass."""
    fake = FakeSystem()
    request = urllib.request.Request("http://x/api/version")
    with pytest.raises(UnansweredURL):
        try:
            fake.open_url(request, timeout=1)
            result = True
        except Exception:
            result = False
        assert result  # never reached: the refusal must escape the except
