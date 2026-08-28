from __future__ import annotations

import signal
from pathlib import Path
from typing import Any

import pytest

from ezai import express, paths
from ezai.config import Config
from ezai.errors import FriendlyError


class FakeProc:
    pid = 4242


class PopenRecorder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.envs: list[dict[str, str] | None] = []
        self.kwargs: list[dict[str, Any]] = []

    def __call__(self, cmd: list[str], **kwargs: Any) -> FakeProc:
        self.calls.append(list(cmd))
        self.envs.append(kwargs.get("env"))
        self.kwargs.append(dict(kwargs))
        return FakeProc()


def test_start_openwebui_sets_engine_env_and_writes_pidfile(isolated_home: Path) -> None:
    popen = PopenRecorder()
    pid = express.start_openwebui(
        3000, "http://127.0.0.1:11434", popen=popen, environ={"PATH": "/usr/bin"}
    )
    assert pid == 4242
    assert paths.pid_file("openwebui").read_text() == "4242"
    assert popen.envs[0] is not None
    assert popen.envs[0]["OLLAMA_BASE_URL"] == "http://127.0.0.1:11434"
    assert "open-webui" in " ".join(popen.calls[0])
    popen.kwargs[0]["stdout"].close()


def test_start_openwebui_missing_uv_is_friendly(isolated_home: Path) -> None:
    def missing(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError(2, "No such file or directory: 'uv'")

    with pytest.raises(FriendlyError) as exc:
        express.start_openwebui(3000, "http://127.0.0.1:11434", popen=missing)
    assert "uv" in exc.value.problem
    assert "astral.sh/uv" in exc.value.fix
    assert not paths.pid_file("openwebui").exists()


def test_stop_openwebui_kills_pid_and_removes_pidfile(isolated_home: Path) -> None:
    paths.pid_file("openwebui").write_text("4242")
    killed: list[tuple[int, int]] = []
    stopped = express.stop_openwebui("linux", kill=lambda pid, sig: killed.append((pid, sig)))
    assert stopped is True
    assert killed == [(4242, signal.SIGTERM)]
    assert not paths.pid_file("openwebui").exists()


def test_stop_openwebui_no_pidfile_returns_false(isolated_home: Path) -> None:
    assert express.stop_openwebui("linux") is False


def test_ensure_openwebui_noop_when_healthy(isolated_home: Path) -> None:
    """A healthy OpenWebUI must cost nothing: no install round-trip, no start."""

    class NoStart:
        def __call__(self, *a: Any, **k: Any) -> FakeProc:
            raise AssertionError("should not start")

    run_calls: list[list[str]] = []
    express.ensure_openwebui(
        Config(),
        run=lambda cmd, **k: run_calls.append(list(cmd)),
        popen=NoStart(),
        up=lambda port, urlopen=None: True,
        sleep=lambda s: None,
    )
    assert run_calls == []


def test_ensure_openwebui_installs_then_starts_then_waits(isolated_home: Path) -> None:
    events: list[str] = []
    popen = PopenRecorder()

    def record_run(cmd: list[str], **k: Any) -> None:
        events.append("install:" + " ".join(cmd))

    def record_popen(cmd: list[str], **k: Any) -> FakeProc:
        events.append("start:" + " ".join(cmd))
        return popen(cmd, **k)

    # Down until started, up afterwards — so wait_for must run after start.
    def up(port: int, urlopen: Any = None) -> bool:
        events.append("probe")
        return any(e.startswith("start:") for e in events)

    express.ensure_openwebui(
        Config(),
        run=record_run,
        popen=record_popen,
        up=up,
        sleep=lambda s: None,
    )
    kinds = [e.split(":")[0] for e in events]
    assert kinds == ["probe", "install", "start", "probe"]
    assert "uv tool install --python 3.11 open-webui" in events[1]
    assert "--port 3000" in events[2]
    popen.kwargs[0]["stdout"].close()


def test_webui_url_uses_localhost() -> None:
    assert express.webui_url(3000) == "http://localhost:3000"


def test_webui_up_is_false_when_health_check_fails() -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise OSError("connection refused")

    assert express.webui_up(3000, urlopen=boom) is False


def test_webui_up_is_true_when_health_check_succeeds() -> None:
    seen: list[str] = []

    def ok(url: str, **kwargs: Any) -> Any:
        seen.append(url)
        return object()

    assert express.webui_up(3000, urlopen=ok) is True
    assert seen == ["http://127.0.0.1:3000/health"]
