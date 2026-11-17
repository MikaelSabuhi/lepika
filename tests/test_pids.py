from __future__ import annotations

import signal
from pathlib import Path

import pytest
from fakes import FakeSystem

from lepika import paths, pids


def test_record_writes_the_bare_pid(isolated_home: Path) -> None:
    pids.record("openwebui", 4242)
    assert (isolated_home / "openwebui.pid").read_text() == "4242"


def test_recorded_is_none_without_a_file() -> None:
    assert pids.recorded("ollama") is None


def test_recorded_reads_a_good_pid() -> None:
    paths.pid_file("ollama").write_text(" 4242\n")
    assert pids.recorded("ollama") == 4242


@pytest.mark.parametrize("text", ["", "garbage", "42x", "0", "-1", "-4242"])
def test_recorded_refuses_garbage_zero_and_negative_and_deletes_nothing(text: str) -> None:
    pf = paths.pid_file("ollama")
    pf.write_text(text)
    assert pids.recorded("ollama") is None
    assert pf.read_text() == text


def test_forget_removes_the_file(isolated_home: Path) -> None:
    pids.record("ollama", 4242)
    pids.forget("ollama")
    assert not (isolated_home / "ollama.pid").exists()


def test_forget_without_a_file_is_a_no_op(isolated_home: Path) -> None:
    pids.forget("ollama")
    assert not (isolated_home / "ollama.pid").exists()


def test_terminate_on_windows_runs_taskkill_for_the_tree() -> None:
    fake = FakeSystem()
    assert pids.terminate(4242, "windows", sys=fake) is True
    assert fake.ran == [["taskkill", "/PID", "4242", "/T", "/F"]]
    assert fake.ran_kwargs[0]["check"] is False
    assert fake.signals == []


def test_terminate_on_windows_a_refused_taskkill_is_false() -> None:
    fake = FakeSystem(raises={"taskkill": PermissionError()})
    assert pids.terminate(4242, "windows", sys=fake) is False
    assert fake.ran == [["taskkill", "/PID", "4242", "/T", "/F"]]


def test_terminate_elsewhere_sends_sigterm() -> None:
    fake = FakeSystem()
    assert pids.terminate(4242, "linux", sys=fake) is True
    assert fake.signals == [(4242, signal.SIGTERM)]
    assert fake.ran == []


def test_terminate_a_gone_process_is_false() -> None:
    fake = FakeSystem(signal_errors={4242: ProcessLookupError()})
    assert pids.terminate(4242, "darwin", sys=fake) is False
    assert fake.signals == [(4242, signal.SIGTERM)]


def test_terminate_a_foreign_process_is_false() -> None:
    fake = FakeSystem(signal_errors={4242: PermissionError()})
    assert pids.terminate(4242, "linux", sys=fake) is False
    assert fake.signals == [(4242, signal.SIGTERM)]


def test_terminate_lets_any_other_error_through() -> None:
    fake = FakeSystem(signal_errors={4242: OSError("boom")})
    with pytest.raises(OSError, match="boom"):
        pids.terminate(4242, "linux", sys=fake)
