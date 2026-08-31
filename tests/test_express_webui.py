from __future__ import annotations

import json
import os
import signal
import socket
import stat
from pathlib import Path
from typing import Any

import pytest
from fakes import Runner

from lepika import express, log, paths
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


def test_start_openwebui_pins_the_interpreter(isolated_home: Path) -> None:
    """Unpinned, `uv tool run` builds an ephemeral env on the system Python.

    On a box whose default is 3.14 that means rebuilding wheels from source —
    pyarrow has none for 3.14 — so the launch never finishes and the user only
    sees "OpenWebUI did not become ready".
    """
    popen = PopenRecorder()
    express.start_openwebui(3000, "http://127.0.0.1:11434", popen=popen, environ={})
    argv = popen.calls[0]
    after_run = argv.index("run") + 1
    assert argv[after_run : after_run + 2] == ["--python", express.OPENWEBUI_PYTHON]
    popen.kwargs[0]["stdout"].close()


def test_install_openwebui_pins_the_same_interpreter_as_the_launch() -> None:
    cmds: list[list[str]] = []
    express.install_openwebui(run=lambda cmd, **k: cmds.append(list(cmd)))
    assert cmds == [["uv", "tool", "install", "--python", express.OPENWEBUI_PYTHON, "open-webui"]]
    assert express.OPENWEBUI_PYTHON == "3.11"


def test_start_openwebui_keeps_openwebui_data_under_lepika_home(isolated_home: Path) -> None:
    """Left to itself OpenWebUI stores chats inside the uv tool venv, which
    `lepika update` rewrites — and which no `LEPIKA_HOME` can move."""
    popen = PopenRecorder()
    express.start_openwebui(3000, "http://127.0.0.1:11434", popen=popen, environ={})
    assert popen.envs[0] is not None
    assert popen.envs[0]["DATA_DIR"] == str(isolated_home / "openwebui")
    assert (isolated_home / "openwebui").is_dir()
    popen.kwargs[0]["stdout"].close()


def test_start_openwebui_owns_the_signing_secret(isolated_home: Path) -> None:
    """Unset, OpenWebUI drops a `.webui_secret_key` in whatever directory the
    user happened to run `lepika up` from, with the ambient umask."""
    popen = PopenRecorder()
    express.start_openwebui(3000, "http://127.0.0.1:11434", popen=popen, environ={})
    secret = (isolated_home / "openwebui" / "secret_key").read_text(encoding="utf-8").strip()
    assert secret != ""
    assert popen.envs[0] is not None
    assert popen.envs[0]["WEBUI_SECRET_KEY"] == secret
    popen.kwargs[0]["stdout"].close()


def test_start_openwebui_never_puts_the_secret_in_argv(isolated_home: Path) -> None:
    popen = PopenRecorder()
    express.start_openwebui(3000, "http://127.0.0.1:11434", popen=popen, environ={})
    secret = (isolated_home / "openwebui" / "secret_key").read_text(encoding="utf-8").strip()
    assert secret not in " ".join(popen.calls[0])
    popen.kwargs[0]["stdout"].close()


def test_webui_secret_is_stable_across_restarts(isolated_home: Path) -> None:
    """A fresh secret on every start would sign every user out on `lepika update`."""
    data_dir = paths.openwebui_data_dir()
    first = express._webui_secret(data_dir)
    assert express._webui_secret(data_dir) == first


@pytest.mark.skipif(os.name == "nt", reason="Windows ignores POSIX mode bits")
def test_webui_secret_file_is_private_from_creation(isolated_home: Path) -> None:
    express._webui_secret(paths.openwebui_data_dir())
    mode = (isolated_home / "openwebui" / "secret_key").stat().st_mode
    assert stat.S_IMODE(mode) == 0o600


def test_start_openwebui_replaces_an_unreadable_secret_file(isolated_home: Path) -> None:
    """A corrupted secret_key is a fresh secret, never a UnicodeDecodeError."""
    data_dir = paths.openwebui_data_dir()
    (data_dir / "secret_key").write_bytes(b"\xff\xfe not utf-8")
    popen = PopenRecorder()
    express.start_openwebui(3000, "http://127.0.0.1:11434", popen=popen, environ={})
    secret = (data_dir / "secret_key").read_text(encoding="utf-8").strip()
    assert secret != ""
    assert popen.envs[0] is not None
    assert popen.envs[0]["WEBUI_SECRET_KEY"] == secret
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


def _log_events() -> list[str]:
    log_file = paths.logs_dir() / log.LOG_FILE
    # A run that logged nothing never creates the file, which is itself an answer.
    text = log_file.read_text(encoding="utf-8") if log_file.exists() else ""
    return [json.loads(line)["event"] for line in text.splitlines() if line.strip()]


def test_start_stack_writes_one_stack_up_line(monkeypatch: pytest.MonkeyPatch) -> None:
    """Server mode logs `stack.up`; the log must say the same about Express."""
    info = SystemInfo("linux", "x86_64", "none", 16.0, False, True, True)
    monkeypatch.setattr(express, "ensure_ollama", lambda info, url=None, **k: None)
    monkeypatch.setattr(express, "ensure_openwebui", lambda cfg, **k: None)
    express.start_stack(info, Config())
    assert _log_events().count("stack.up") == 1


def test_start_stack_logs_nothing_when_the_engine_pre_flight_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run that never started leaves the failure line only, exactly as Server does."""
    info = SystemInfo("linux", "x86_64", "none", 16.0, False, True, True)

    def refuse(info: SystemInfo, url: str | None = None, **k: Any) -> None:
        raise FriendlyError("Ollama is already running natively.", "Stop it.")

    monkeypatch.setattr(express, "ensure_ollama", refuse)
    monkeypatch.setattr(express, "ensure_openwebui", lambda cfg, **k: None)
    with pytest.raises(FriendlyError):
        express.start_stack(info, Config())
    assert "stack.up" not in _log_events()


def test_stop_writes_a_stack_down_line(monkeypatch: pytest.MonkeyPatch) -> None:
    info = SystemInfo("linux", "x86_64", "none", 16.0, False, True, True)
    monkeypatch.setattr(express, "stop_openwebui", lambda os_name, port=None, **k: False)
    assert express.stop(info, Config()) is False
    assert "stack.down" in _log_events()


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
    """After a reboot the recorded pid can belong to a stranger — nothing vouches for it."""
    pf = paths.pid_file("openwebui")
    pf.write_text("4242")

    def never(pid: int, sig: int) -> None:
        raise AssertionError("must not signal a process we cannot confirm is ours")

    stopped = express.stop_openwebui(
        "linux", run=Runner(), kill=never, port=3000, up=lambda port, **k: False
    )
    assert stopped is False
    assert not pf.exists()


def test_stop_openwebui_stops_a_hung_but_alive_webui(isolated_home: Path) -> None:
    """A UI wedged mid-request fails /health while still holding the port.

    Walking away from it leaves the port taken, so the next `lepika up` cannot start
    a replacement — and `lepika down` reported success.
    """
    pf = paths.pid_file("openwebui")
    pf.write_text("4242")
    killed: list[tuple[int, int]] = []
    run = Runner(stdout={"ps": "uv tool run --from open-webui open-webui serve --port 3000\n"})
    stopped = express.stop_openwebui(
        "linux",
        run=run,
        kill=lambda pid, sig: killed.append((pid, sig)),
        port=3000,
        up=lambda port, **k: False,
    )
    assert stopped is True
    assert killed == [(4242, signal.SIGTERM)]
    assert run.calls == [["ps", "-o", "args=", "-p", "4242"]]
    assert not pf.exists()


def test_stop_openwebui_leaves_a_recycled_pid_alone(isolated_home: Path) -> None:
    """The command line is what separates our hung UI from a stranger's process."""
    pf = paths.pid_file("openwebui")
    pf.write_text("4242")

    def never(pid: int, sig: int) -> None:
        raise AssertionError("must not signal a process we cannot confirm is ours")

    stopped = express.stop_openwebui(
        "linux",
        run=Runner(stdout={"ps": "/usr/lib/systemd/systemd --user\n"}),
        kill=never,
        port=3000,
        up=lambda port, **k: False,
    )
    assert stopped is False
    assert not pf.exists()


def test_stop_openwebui_reads_the_command_line_without_logging_it(isolated_home: Path) -> None:
    """A pure read, and a pid that is simply gone is not a failure (rule 12)."""
    paths.pid_file("openwebui").write_text("4242")
    seen: list[dict[str, Any]] = []

    def run(cmd: list[str], **kwargs: Any) -> Any:
        seen.append(dict(kwargs))
        return Runner()(cmd, **kwargs)

    express.stop_openwebui(
        "linux", run=run, kill=lambda pid, sig: None, port=3000, up=lambda port, **k: False
    )
    # A stale pid file reads twice: its own command line, then the process list the
    # fallback searches once that pid has vouched for nothing. Both are pure reads.
    assert seen == [{"check": False, "log": False}, {"check": False, "log": False}]


def test_stop_openwebui_on_windows_still_trusts_the_port_alone(isolated_home: Path) -> None:
    """`tasklist` lists images, not argv: nothing there can tell our UI from a stranger."""
    pf = paths.pid_file("openwebui")
    pf.write_text("4242")

    def never(pid: int, sig: int) -> None:
        raise AssertionError("must not signal a process we cannot confirm is ours")

    run = Runner(stdout={"tasklist": '"python.exe","4242","Console","1","94,208 K"\n'})
    stopped = express.stop_openwebui(
        "windows", run=run, kill=never, port=3000, up=lambda port, **k: False
    )
    assert stopped is False
    assert run.calls == []
    assert not pf.exists()


def test_stop_openwebui_signals_when_the_port_answers(isolated_home: Path) -> None:
    """The healthy path is unchanged: a live OpenWebUI on our port still gets SIGTERM."""
    paths.pid_file("openwebui").write_text("4242")
    killed: list[tuple[int, int]] = []
    run = Runner()
    stopped = express.stop_openwebui(
        "linux",
        run=run,
        kill=lambda pid, sig: killed.append((pid, sig)),
        port=3000,
        up=lambda port, **k: True,
    )
    assert stopped is True
    assert killed == [(4242, signal.SIGTERM)]
    # An answering port settles it on its own: the `and` must short-circuit before
    # `ps` runs, or the common path pays a subprocess it never needed.
    assert run.calls == []
    assert not paths.pid_file("openwebui").exists()


def _ps_line(pid: int, port: int) -> str:
    """One `ps -eo pid=,args=` row for an OpenWebUI started the way LePika starts it."""
    return (
        f" {pid} uv tool run --python 3.11 --from open-webui open-webui serve"
        f" --host 127.0.0.1 --port {port}\n"
    )


def test_stop_openwebui_adopts_a_webui_no_pidfile_names(isolated_home: Path) -> None:
    """`lepika up` records no pid when it finds the UI already answering.

    `ensure_openwebui` returns early on a healthy port, so the pid file can be
    absent while the UI keeps serving — and `down` reported "Nothing was running"
    every time while `status` went on showing it up.
    """
    killed: list[tuple[int, int]] = []
    run = Runner(stdout={"ps": _ps_line(900, 3000)})
    stopped = express.stop_openwebui(
        "linux",
        run=run,
        kill=lambda pid, sig: killed.append((pid, sig)),
        port=3000,
        up=lambda port, **k: True,
    )
    assert stopped is True
    assert killed == [(900, signal.SIGTERM)]
    assert run.calls == [["ps", "-eo", "pid=,args="]]


def test_stop_openwebui_adopts_every_process_serving_our_port(isolated_home: Path) -> None:
    """`uv tool run` and the server it wraps both carry the argv; neither may be left."""
    killed: list[int] = []
    run = Runner(
        stdout={
            "ps": _ps_line(900, 3000)
            + " 901 /opt/uv/tools/open-webui/bin/python"
            + " /opt/uv/tools/open-webui/bin/open-webui serve --host 127.0.0.1 --port 3000\n"
        }
    )
    stopped = express.stop_openwebui(
        "linux", run=run, kill=lambda pid, sig: killed.append(pid), port=3000
    )
    assert stopped is True
    assert killed == [900, 901]


def test_stop_openwebui_adopts_nothing_when_no_process_is_ours(isolated_home: Path) -> None:
    """Without a pid file the argv is the only evidence — a stranger provides none."""

    def never(pid: int, sig: int) -> None:
        raise AssertionError("must not signal a process we cannot confirm is ours")

    run = Runner(stdout={"ps": " 900 /usr/lib/systemd/systemd --user\n"})
    assert express.stop_openwebui("linux", run=run, kill=never, port=3000) is False


def test_stop_openwebui_adopts_only_the_webui_on_our_own_port(isolated_home: Path) -> None:
    """A second LEPIKA_HOME runs its own UI on its own port; ours is the one we stop."""

    def never(pid: int, sig: int) -> None:
        raise AssertionError("another port's UI belongs to another LePika")

    run = Runner(stdout={"ps": _ps_line(900, 3001)})
    assert express.stop_openwebui("linux", run=run, kill=never, port=3000) is False


def test_stop_openwebui_matches_the_port_whole(isolated_home: Path) -> None:
    """`--port 30000` starts with `--port 3000`: a substring match would kill it."""

    def never(pid: int, sig: int) -> None:
        raise AssertionError("port 30000 is not port 3000")

    run = Runner(stdout={"ps": _ps_line(900, 30000)})
    assert express.stop_openwebui("linux", run=run, kill=never, port=3000) is False


def test_stop_openwebui_adopts_nothing_on_windows(isolated_home: Path) -> None:
    """`tasklist` lists images, not argv: there is no command line to search there."""

    def never(pid: int, sig: int) -> None:
        raise AssertionError("must not signal a process we cannot confirm is ours")

    run = Runner(stdout={"ps": _ps_line(900, 3000)})
    assert express.stop_openwebui("windows", run=run, kill=never, port=3000) is False
    assert run.calls == []


def test_stop_openwebui_never_searches_without_a_port(isolated_home: Path) -> None:
    """The port is what the search matches on — with none given there is nothing to ask."""
    run = Runner(stdout={"ps": _ps_line(900, 3000)})
    assert express.stop_openwebui("linux", run=run, kill=lambda pid, sig: None) is False
    assert run.calls == []


def test_stop_openwebui_adoption_reads_the_process_list_without_logging_it(
    isolated_home: Path,
) -> None:
    """A pure read of every command line on the machine is never written to the log."""
    seen: list[dict[str, Any]] = []

    def run(cmd: list[str], **kwargs: Any) -> Any:
        seen.append(dict(kwargs))
        return Runner(stdout={"ps": _ps_line(900, 3000)})(cmd, **kwargs)

    express.stop_openwebui("linux", run=run, kill=lambda pid, sig: None, port=3000)
    assert seen == [{"check": False, "log": False}]


def test_stop_openwebui_adoption_survives_a_process_that_just_exited(
    isolated_home: Path,
) -> None:
    """`ps` and the signal are two moments: what it listed may already be gone."""

    def vanished(pid: int, sig: int) -> None:
        raise ProcessLookupError(3, "No such process")

    run = Runner(stdout={"ps": _ps_line(900, 3000)})
    assert express.stop_openwebui("linux", run=run, kill=vanished, port=3000) is False


def test_stop_openwebui_adoption_ignores_rows_with_no_readable_pid(
    isolated_home: Path,
) -> None:
    """`ps` output is parsed, not trusted: a header or a wrapped row is not a pid."""

    def never(pid: int, sig: int) -> None:
        raise AssertionError("must not signal something that is not a pid")

    run = Runner(stdout={"ps": "  PID COMMAND\n" + " open-webui serve --port 3000\n"})
    assert express.stop_openwebui("linux", run=run, kill=never, port=3000) is False


def test_stop_openwebui_prefers_the_pidfile_over_the_search(isolated_home: Path) -> None:
    """The recorded pid is still the first answer; the search is only the fallback."""
    paths.pid_file("openwebui").write_text("4242")
    killed: list[int] = []
    run = Runner(stdout={"ps": _ps_line(900, 3000)})
    stopped = express.stop_openwebui(
        "linux",
        run=run,
        kill=lambda pid, sig: killed.append(pid),
        port=3000,
        up=lambda port, **k: True,
    )
    assert stopped is True
    assert killed == [4242]
    assert run.calls == []


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


def test_start_openwebui_binds_to_localhost_only(isolated_home: Path) -> None:
    """OpenWebUI defaults to 0.0.0.0: Express mode must not put the UI on the network."""
    popen = PopenRecorder()
    express.start_openwebui(3000, "http://127.0.0.1:11434", popen=popen, environ={})
    argv = popen.calls[0]
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    popen.kwargs[0]["stdout"].close()


def test_wait_until_down_hint_does_not_name_a_single_caller() -> None:
    """`lepika connect` restarts the UI too — the hint must not say `lepika update`."""
    with pytest.raises(FriendlyError) as exc:
        express.wait_until_down(3210, up=lambda port, **k: True, attempts=2, sleep=lambda s: None)
    assert exc.value.fix == "Stop whatever is listening on port 3210, then try again."


def test_wait_for_hint_points_at_lepika_logs() -> None:
    """A raw ~/.lepika/logs path is the wrong place in Server mode; `lepika logs` is not."""
    with pytest.raises(FriendlyError) as exc:
        express.wait_for(lambda: False, seconds=1, what="OpenWebUI", sleep=lambda s: None)
    assert exc.value.fix == "Run `lepika logs` to see why, then `lepika doctor`."
