from __future__ import annotations

import json
import signal
import socket
from pathlib import Path
from typing import Any

import pytest

from lepika import express, paths
from lepika.config import Config
from lepika.detect import SystemInfo
from lepika.errors import FriendlyError


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


def test_start_openwebui_makes_env_win_over_the_saved_admin_config(isolated_home: Path) -> None:
    """OpenWebUI persists its admin panel's engine URL and would ignore ours on restart."""
    popen = PopenRecorder()
    express.start_openwebui(3000, "http://gpu-box:11435", popen=popen, environ={})
    assert popen.envs[0] is not None
    assert popen.envs[0]["ENABLE_PERSISTENT_CONFIG"] == "false"
    popen.kwargs[0]["stdout"].close()


def test_start_openwebui_passes_the_engine_key_to_openwebui(isolated_home: Path) -> None:
    popen = PopenRecorder()
    express.start_openwebui(3000, "http://gpu-box:11435", popen=popen, environ={}, engine_key="k")
    assert popen.envs[0] is not None
    assert json.loads(popen.envs[0]["OLLAMA_API_CONFIGS"]) == {"0": {"key": "k"}}
    popen.kwargs[0]["stdout"].close()


def test_start_openwebui_omits_the_key_config_when_there_is_no_key(isolated_home: Path) -> None:
    popen = PopenRecorder()
    express.start_openwebui(3000, "http://127.0.0.1:11434", popen=popen, environ={})
    assert popen.envs[0] is not None
    assert "OLLAMA_API_CONFIGS" not in popen.envs[0]
    popen.kwargs[0]["stdout"].close()


def test_stop_is_the_backend_stop_for_express(monkeypatch: pytest.MonkeyPatch) -> None:
    """`lepika down` in Express stops the UI only — Ollama is a shared service."""
    info = SystemInfo(
        os="linux",
        arch="x86_64",
        gpu="none",
        ram_gb=16.0,
        has_docker=False,
        has_ollama=True,
        ollama_running=True,
    )
    seen: list[tuple[str, int | None]] = []
    monkeypatch.setattr(
        express,
        "stop_openwebui",
        lambda os_name, port=None, **k: bool(seen.append((os_name, port))) or True,
    )
    assert express.stop(info, Config(webui_port=3210)) is True
    assert seen == [("linux", 3210)]


def test_logs_returns_the_tail_of_every_log_file(isolated_home: Path) -> None:
    (paths.logs_dir() / "ollama.log").write_text("one\ntwo\nthree\n")
    (paths.logs_dir() / "openwebui.log").write_text("only\n")
    assert express.logs(2) == [("ollama.log", "two\nthree"), ("openwebui.log", "only")]


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


def test_stop_openwebui_malformed_pidfile_is_cleaned_up(isolated_home: Path) -> None:
    """A truncated or garbage pid file means nothing to stop — never a traceback."""
    pf = paths.pid_file("openwebui")
    pf.write_text("not-a-pid\n")

    def never(pid: int, sig: int) -> None:
        raise AssertionError("should not signal anything")

    assert express.stop_openwebui("linux", kill=never) is False
    assert not pf.exists()


@pytest.mark.parametrize("content", ["0", "-5\n"])
def test_stop_openwebui_nonpositive_pid_is_never_signalled(
    isolated_home: Path, content: str
) -> None:
    """`os.kill(0, ...)` signals the whole process group; 0 and negatives never reach kill."""
    pf = paths.pid_file("openwebui")
    pf.write_text(content)

    def never(pid: int, sig: int) -> None:
        raise AssertionError("should not signal anything")

    assert express.stop_openwebui("linux", kill=never) is False
    assert not pf.exists()


def test_stop_openwebui_permission_error_is_not_fatal(isolated_home: Path) -> None:
    """A truncated pid can parse to a live foreign pid — refusing it is not a crash."""
    pf = paths.pid_file("openwebui")
    pf.write_text("1")

    def denied(pid: int, sig: int) -> None:
        raise PermissionError(1, "Operation not permitted")

    assert express.stop_openwebui("linux", kill=denied) is True
    assert not pf.exists()


def test_stop_openwebui_stale_pidfile_is_never_signalled(isolated_home: Path) -> None:
    """After a reboot the recorded pid can belong to a stranger — the port decides."""
    pf = paths.pid_file("openwebui")
    pf.write_text("4242")

    def never(pid: int, sig: int) -> None:
        raise AssertionError("must not signal a process we cannot confirm is ours")

    stopped = express.stop_openwebui("linux", kill=never, port=3000, up=lambda port, **k: False)
    assert stopped is False
    assert not pf.exists()


def test_stop_openwebui_signals_when_the_port_answers(isolated_home: Path) -> None:
    """The healthy path is unchanged: a live OpenWebUI on our port still gets SIGTERM."""
    paths.pid_file("openwebui").write_text("4242")
    killed: list[tuple[int, int]] = []
    stopped = express.stop_openwebui(
        "linux",
        kill=lambda pid, sig: killed.append((pid, sig)),
        port=3000,
        up=lambda port, **k: True,
    )
    assert stopped is True
    assert killed == [(4242, signal.SIGTERM)]
    assert not paths.pid_file("openwebui").exists()


def test_restart_openwebui_waits_for_the_old_server_to_die_before_starting(
    isolated_home: Path,
) -> None:
    """`lepika update` must not probe the dying server and call it 'already running'."""
    paths.pid_file("openwebui").write_text("4242")
    events: list[str] = []
    popen = PopenRecorder()
    lingering = {"probes": 0}

    def up(port: int, **k: Any) -> bool:
        events.append("probe")
        if "start" in events:
            return True
        # Still answering for a few probes after the SIGTERM, then gone.
        lingering["probes"] += 1
        return lingering["probes"] <= 4

    def record_popen(cmd: list[str], **k: Any) -> FakeProc:
        events.append("start")
        return popen(cmd, **k)

    express.restart_openwebui(
        Config(),
        "linux",
        run=lambda cmd, **k: events.append("install"),
        popen=record_popen,
        up=up,
        sleep=lambda s: None,
        kill=lambda pid, sig: events.append("kill"),
        bind_check=lambda port: True,
    )
    assert "kill" in events
    # 4 lingering "yes" answers had to be outwaited, not believed: the staleness
    # gate, four probes that still saw the dying server, and the one that finally
    # saw it gone — then ensure_openwebui's own probe before it starts anything.
    assert lingering["probes"] == 6
    assert events.index("kill") < events.index("install") < events.index("start")
    popen.kwargs[0]["stdout"].close()


def test_restart_openwebui_raises_when_the_old_server_never_dies(
    isolated_home: Path,
) -> None:
    paths.pid_file("openwebui").write_text("4242")
    with pytest.raises(FriendlyError) as exc:
        express.restart_openwebui(
            Config(webui_port=3210),
            "linux",
            run=lambda cmd, **k: None,
            popen=lambda *a, **k: pytest.fail("must not start a second server"),
            up=lambda port, **k: True,
            sleep=lambda s: None,
            kill=lambda pid, sig: None,
            bind_check=lambda port: True,
        )
    assert "3210" in exc.value.problem or "3210" in exc.value.fix


def test_port_free_sees_a_listener_bound_to_the_wildcard_address() -> None:
    """Measured on Darwin: with SO_REUSEADDR, 127.0.0.1 binds happily over 0.0.0.0.

    Probing only the loopback address therefore called a genuinely taken port free
    and let the install/start run into a conflict it had just declared impossible.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        holder.bind(("0.0.0.0", 0))
        holder.listen(1)
        port = int(holder.getsockname()[1])
        assert express.port_free(port, sleep=lambda s: None) is False


@pytest.mark.parametrize("busy", ["0.0.0.0", "127.0.0.1"])
def test_bind_once_calls_a_port_busy_when_either_address_is_taken(busy: str) -> None:
    def bind(address: str, port: int) -> bool:
        return address != busy

    assert express._bind_once(3000, bind_address=bind) is False


def test_bind_once_probes_both_addresses_before_calling_a_port_free() -> None:
    probed: list[str] = []

    def bind(address: str, port: int) -> bool:
        probed.append(address)
        return True

    assert express._bind_once(3000, bind_address=bind) is True
    assert probed == ["0.0.0.0", "127.0.0.1"]


def test_port_free_retries_before_calling_a_port_busy() -> None:
    """A port we just stopped serving can need a moment to be released."""
    attempts = {"n": 0}

    def bind(port: int) -> bool:
        attempts["n"] += 1
        return attempts["n"] >= 3

    assert express.port_free(3000, bind=bind, sleep=lambda s: None) is True
    assert attempts["n"] == 3


def test_port_free_gives_up_on_a_genuinely_busy_port() -> None:
    assert express.port_free(3000, bind=lambda port: False, sleep=lambda s: None) is False


def test_ensure_openwebui_rejects_a_port_another_app_holds(isolated_home: Path) -> None:
    """A busy port must be named, not hidden behind a 180s 'did not become ready'."""

    def never(*a: Any, **k: Any) -> Any:
        raise AssertionError("must not install or start when the port is taken")

    with pytest.raises(FriendlyError) as exc:
        express.ensure_openwebui(
            Config(webui_port=3210),
            run=never,
            popen=never,
            up=lambda port, **k: False,
            sleep=lambda s: None,
            bind_check=lambda port: False,
        )
    assert "3210" in exc.value.problem
    assert "webui_port" in exc.value.fix


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
        bind_check=lambda port: True,
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
