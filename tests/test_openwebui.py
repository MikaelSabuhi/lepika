from __future__ import annotations

import json
import os
import signal
import socket
import stat
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeResponse, FakeSystem, FakeUI, routes

from lepika import health, log, ollama, openwebui, paths, ports, system
from lepika.config import Config
from lepika.detect import Machine
from lepika.errors import FriendlyError
from lepika.express import ExpressBackend
from lepika.system import Response

# What `health.webui_up` asks of the default port.
_HEALTH = "http://127.0.0.1:3000/health"
# What a pure read of the process table hands `run`: unchecked and unlogged.
_READ = {"check": False, "env": None, "timeout": None, "log": False, "ok": (0,)}


def answering(*up: bool, port: int = 3000, **kwargs: Any) -> FakeSystem:
    """A machine whose OpenWebUI on `port` answers /health as `up` says, probe by probe.

    The last answer repeats. Anything else asked of the network is refused.
    """
    url = f"http://127.0.0.1:{port}/health"
    said = [FakeResponse() if yes else OSError("connection refused") for yes in up]
    return FakeSystem(urls=routes({url: said}), **kwargs)


class Recorded(FakeSystem):
    """A FakeSystem that also writes, to one shared list, the order its effects ran in."""

    def __init__(self, events: list[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.events = events

    def run(self, cmd: Sequence[str], **kwargs: Any) -> Any:
        kind = "install" if list(cmd[:3]) == ["uv", "tool", "install"] else "run"
        self.events.append(f"{kind}:" + " ".join(cmd))
        return super().run(cmd, **kwargs)

    def spawn(self, cmd: Sequence[str], *, env: Mapping[str, str], log: Path) -> int:
        self.events.append("start:" + " ".join(cmd))
        return super().spawn(cmd, env=env, log=log)

    def signal(self, pid: int, sig: int) -> None:
        self.events.append("kill")
        super().signal(pid, sig)


def _kinds(events: list[str]) -> list[str]:
    return [e.split(":")[0] for e in events]


def test_start_openwebui_sets_engine_env_and_writes_pidfile(isolated_home: Path) -> None:
    sys = FakeSystem(environ={"PATH": "/usr/bin"})
    pid = openwebui.start_openwebui(3000, "http://127.0.0.1:11434", sys=sys)
    assert pid == 4242
    assert paths.pid_file("openwebui").read_text() == "4242"
    cmd, env, log_path = sys.spawned[0]
    assert env["OLLAMA_BASE_URL"] == "http://127.0.0.1:11434"
    assert env["PATH"] == "/usr/bin"
    assert "open-webui" in " ".join(cmd)
    # Its output goes where `lepika logs` reads it.
    assert log_path == paths.logs_dir() / "openwebui.log"


def test_start_openwebui_makes_env_win_over_the_saved_admin_config(isolated_home: Path) -> None:
    """OpenWebUI persists its admin panel's engine URL and would ignore ours on restart."""
    sys = FakeSystem()
    openwebui.start_openwebui(3000, "http://gpu-box:11435", sys=sys)
    _, env, _ = sys.spawned[0]
    assert env["ENABLE_PERSISTENT_CONFIG"] == "false"


def test_start_openwebui_passes_the_engine_key_to_openwebui(isolated_home: Path) -> None:
    sys = FakeSystem()
    openwebui.start_openwebui(3000, "http://gpu-box:11435", engine_key="k", sys=sys)
    _, env, _ = sys.spawned[0]
    assert json.loads(env["OLLAMA_API_CONFIGS"]) == {"0": {"key": "k"}}


def test_start_openwebui_omits_the_key_config_when_there_is_no_key(isolated_home: Path) -> None:
    sys = FakeSystem()
    openwebui.start_openwebui(3000, "http://127.0.0.1:11434", sys=sys)
    _, env, _ = sys.spawned[0]
    assert "OLLAMA_API_CONFIGS" not in env


def test_start_openwebui_pins_the_interpreter(isolated_home: Path) -> None:
    """Unpinned, `uv tool run` builds an ephemeral env on the system Python.

    On a box whose default is 3.14 that means rebuilding wheels from source —
    pyarrow has none for 3.14 — so the launch never finishes and the user only
    sees "OpenWebUI did not become ready".
    """
    sys = FakeSystem()
    openwebui.start_openwebui(3000, "http://127.0.0.1:11434", sys=sys)
    argv, _, _ = sys.spawned[0]
    after_run = argv.index("run") + 1
    assert argv[after_run : after_run + 2] == ["--python", openwebui.OPENWEBUI_PYTHON]


def test_install_openwebui_pins_the_same_interpreter_as_the_launch() -> None:
    sys = FakeSystem()
    openwebui.install_openwebui(sys=sys)
    assert sys.ran == [
        ["uv", "tool", "install", "--python", openwebui.OPENWEBUI_PYTHON, "open-webui"]
    ]
    assert openwebui.OPENWEBUI_PYTHON == "3.11"


def test_start_openwebui_keeps_openwebui_data_under_lepika_home(isolated_home: Path) -> None:
    """Left to itself OpenWebUI stores chats inside the uv tool venv, which
    `lepika update` rewrites — and which no `LEPIKA_HOME` can move."""
    sys = FakeSystem()
    openwebui.start_openwebui(3000, "http://127.0.0.1:11434", sys=sys)
    _, env, _ = sys.spawned[0]
    assert env["DATA_DIR"] == str(isolated_home / "openwebui")
    assert (isolated_home / "openwebui").is_dir()


def test_start_openwebui_owns_the_signing_secret(isolated_home: Path) -> None:
    """Unset, OpenWebUI drops a `.webui_secret_key` in whatever directory the
    user happened to run `lepika up` from, with the ambient umask."""
    sys = FakeSystem()
    openwebui.start_openwebui(3000, "http://127.0.0.1:11434", sys=sys)
    secret = (isolated_home / "openwebui" / "secret_key").read_text(encoding="utf-8").strip()
    assert secret != ""
    _, env, _ = sys.spawned[0]
    assert env["WEBUI_SECRET_KEY"] == secret


def test_start_openwebui_never_puts_the_secret_in_argv(isolated_home: Path) -> None:
    sys = FakeSystem()
    openwebui.start_openwebui(3000, "http://127.0.0.1:11434", sys=sys)
    secret = (isolated_home / "openwebui" / "secret_key").read_text(encoding="utf-8").strip()
    argv, _, _ = sys.spawned[0]
    assert secret not in " ".join(argv)


def test_webui_secret_is_stable_across_restarts(isolated_home: Path) -> None:
    """A fresh secret on every start would sign every user out on `lepika update`."""
    data_dir = paths.openwebui_data_dir()
    first = openwebui._webui_secret(data_dir)
    assert openwebui._webui_secret(data_dir) == first


@pytest.mark.skipif(os.name == "nt", reason="Windows ignores POSIX mode bits")
def test_webui_secret_file_is_private_from_creation(isolated_home: Path) -> None:
    openwebui._webui_secret(paths.openwebui_data_dir())
    mode = (isolated_home / "openwebui" / "secret_key").stat().st_mode
    assert stat.S_IMODE(mode) == 0o600


def test_start_openwebui_replaces_an_unreadable_secret_file(isolated_home: Path) -> None:
    """A corrupted secret_key is a fresh secret, never a UnicodeDecodeError."""
    data_dir = paths.openwebui_data_dir()
    (data_dir / "secret_key").write_bytes(b"\xff\xfe not utf-8")
    sys = FakeSystem()
    openwebui.start_openwebui(3000, "http://127.0.0.1:11434", sys=sys)
    secret = (data_dir / "secret_key").read_text(encoding="utf-8").strip()
    assert secret != ""
    _, env, _ = sys.spawned[0]
    assert env["WEBUI_SECRET_KEY"] == secret


def test_stop_is_the_backend_stop_for_express(monkeypatch: pytest.MonkeyPatch) -> None:
    """`lepika down` in Express stops the UI only — Ollama is a shared service."""
    info = Machine(
        os="linux",
        arch="x86_64",
        gpu="none",
        ram_gb=16.0,
    )
    seen: list[tuple[str, int | None]] = []
    monkeypatch.setattr(
        openwebui,
        "stop_openwebui",
        lambda os_name, port=None, **k: bool(seen.append((os_name, port))) or True,
    )
    assert ExpressBackend(sys=FakeSystem(), ui=FakeUI()).stop(info, Config(webui_port=3210)) is True
    assert seen == [("linux", 3210)]


def _log_events() -> list[str]:
    log_file = paths.logs_dir() / log.LOG_FILE
    # A run that logged nothing never creates the file, which is itself an answer.
    text = log_file.read_text(encoding="utf-8") if log_file.exists() else ""
    return [json.loads(line)["event"] for line in text.splitlines() if line.strip()]


def test_start_stack_writes_one_stack_up_line(monkeypatch: pytest.MonkeyPatch) -> None:
    """Server mode logs `stack.up`; the log must say the same about Express."""
    info = Machine("linux", "x86_64", "none", 16.0)
    monkeypatch.setattr(ollama, "ensure_ollama", lambda info, url=None, **k: None)
    monkeypatch.setattr(openwebui, "ensure_openwebui", lambda cfg, **k: None)
    ExpressBackend(sys=FakeSystem(), ui=FakeUI()).start(info, Config())
    assert _log_events().count("stack.up") == 1


def test_start_stack_logs_nothing_when_the_engine_pre_flight_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run that never started leaves the failure line only, exactly as Server does."""
    info = Machine("linux", "x86_64", "none", 16.0)

    def refuse(info: Machine, url: str | None = None, **k: Any) -> None:
        raise FriendlyError("Ollama is already running natively.", "Stop it.")

    monkeypatch.setattr(ollama, "ensure_ollama", refuse)
    monkeypatch.setattr(openwebui, "ensure_openwebui", lambda cfg, **k: None)
    with pytest.raises(FriendlyError):
        ExpressBackend(sys=FakeSystem(), ui=FakeUI()).start(info, Config())
    assert "stack.up" not in _log_events()


def test_stop_writes_a_stack_down_line(monkeypatch: pytest.MonkeyPatch) -> None:
    info = Machine("linux", "x86_64", "none", 16.0)
    monkeypatch.setattr(openwebui, "stop_openwebui", lambda os_name, port=None, **k: False)
    assert ExpressBackend(sys=FakeSystem(), ui=FakeUI()).stop(info, Config()) is False
    assert "stack.down" in _log_events()


def test_logs_returns_the_tail_of_every_log_file(isolated_home: Path) -> None:
    (paths.logs_dir() / "ollama.log").write_text("one\ntwo\nthree\n")
    (paths.logs_dir() / "openwebui.log").write_text("only\n")
    assert ExpressBackend(sys=FakeSystem(), ui=FakeUI()).logs(2) == [
        ("ollama.log", "two\nthree"),
        ("openwebui.log", "only"),
    ]


def test_start_openwebui_missing_uv_is_friendly(isolated_home: Path) -> None:
    with pytest.raises(FriendlyError) as exc:
        openwebui.start_openwebui(3000, "http://127.0.0.1:11434", sys=FakeSystem(missing={"uv"}))
    assert "uv" in exc.value.problem
    assert "astral.sh/uv" in exc.value.fix
    assert not paths.pid_file("openwebui").exists()


def test_stop_openwebui_kills_pid_and_removes_pidfile(isolated_home: Path) -> None:
    paths.pid_file("openwebui").write_text("4242")
    sys = FakeSystem()
    stopped = openwebui.stop_openwebui("linux", sys=sys)
    assert stopped is True
    assert sys.signals == [(4242, signal.SIGTERM)]
    assert not paths.pid_file("openwebui").exists()


def test_stop_openwebui_no_pidfile_returns_false(isolated_home: Path) -> None:
    assert openwebui.stop_openwebui("linux", sys=FakeSystem()) is False


def test_stop_openwebui_malformed_pidfile_is_cleaned_up(isolated_home: Path) -> None:
    """A truncated or garbage pid file means nothing to stop — never a traceback."""
    pf = paths.pid_file("openwebui")
    pf.write_text("not-a-pid\n")
    sys = FakeSystem()
    assert openwebui.stop_openwebui("linux", sys=sys) is False
    assert sys.signals == []
    assert not pf.exists()


@pytest.mark.parametrize("content", ["0", "-5\n"])
def test_stop_openwebui_nonpositive_pid_is_never_signalled(
    isolated_home: Path, content: str
) -> None:
    """`os.kill(0, ...)` signals the whole process group; 0 and negatives never reach kill."""
    pf = paths.pid_file("openwebui")
    pf.write_text(content)
    sys = FakeSystem()
    assert openwebui.stop_openwebui("linux", sys=sys) is False
    assert sys.signals == []
    assert not pf.exists()


def test_stop_openwebui_permission_error_is_not_fatal(isolated_home: Path) -> None:
    """A truncated pid can parse to a live foreign pid — refusing it is not a crash."""
    pf = paths.pid_file("openwebui")
    pf.write_text("1")
    sys = FakeSystem(signal_errors={1: PermissionError(1, "Operation not permitted")})
    assert openwebui.stop_openwebui("linux", sys=sys) is True
    assert not pf.exists()


def test_stop_openwebui_stale_pidfile_is_never_signalled(isolated_home: Path) -> None:
    """After a reboot the recorded pid can belong to a stranger — nothing vouches for it."""
    pf = paths.pid_file("openwebui")
    pf.write_text("4242")
    sys = answering(False)
    stopped = openwebui.stop_openwebui("linux", port=3000, sys=sys)
    assert stopped is False
    assert sys.signals == []
    assert not pf.exists()


def test_stop_openwebui_stops_a_hung_but_alive_webui(isolated_home: Path) -> None:
    """A UI wedged mid-request fails /health while still holding the port.

    Walking away from it leaves the port taken, so the next `lepika up` cannot start
    a replacement — and `lepika down` reported success.
    """
    pf = paths.pid_file("openwebui")
    pf.write_text("4242")
    sys = answering(
        False, stdout={"ps": "uv tool run --from open-webui open-webui serve --port 3000\n"}
    )
    stopped = openwebui.stop_openwebui("linux", port=3000, sys=sys)
    assert stopped is True
    assert sys.signals == [(4242, signal.SIGTERM)]
    assert sys.ran == [["ps", "-o", "args=", "-p", "4242"]]
    assert not pf.exists()


def test_stop_openwebui_leaves_a_recycled_pid_alone(isolated_home: Path) -> None:
    """The command line is what separates our hung UI from a stranger's process."""
    pf = paths.pid_file("openwebui")
    pf.write_text("4242")
    sys = answering(False, stdout={"ps": "/usr/lib/systemd/systemd --user\n"})
    stopped = openwebui.stop_openwebui("linux", port=3000, sys=sys)
    assert stopped is False
    assert sys.signals == []
    assert not pf.exists()


def test_stop_openwebui_reads_the_command_line_without_logging_it(isolated_home: Path) -> None:
    """A pure read, and a pid that is simply gone is not a failure (rule 12)."""
    paths.pid_file("openwebui").write_text("4242")
    sys = answering(False)
    openwebui.stop_openwebui("linux", port=3000, sys=sys)
    # A stale pid file reads twice: its own command line, then the process list the
    # fallback searches once that pid has vouched for nothing. Both are pure reads.
    assert sys.ran_kwargs == [_READ, _READ]


def test_stop_openwebui_on_windows_still_trusts_the_port_alone(isolated_home: Path) -> None:
    """`tasklist` lists images, not argv: nothing there can tell our UI from a stranger."""
    pf = paths.pid_file("openwebui")
    pf.write_text("4242")
    sys = answering(False, stdout={"tasklist": '"python.exe","4242","Console","1","94,208 K"\n'})
    stopped = openwebui.stop_openwebui("windows", port=3000, sys=sys)
    assert stopped is False
    assert sys.signals == []
    assert sys.ran == []
    assert not pf.exists()


def test_stop_openwebui_signals_when_the_port_answers(isolated_home: Path) -> None:
    """The healthy path is unchanged: a live OpenWebUI on our port still gets SIGTERM."""
    paths.pid_file("openwebui").write_text("4242")
    sys = answering(True)
    stopped = openwebui.stop_openwebui("linux", port=3000, sys=sys)
    assert stopped is True
    assert sys.signals == [(4242, signal.SIGTERM)]
    # An answering port settles it on its own: the `and` must short-circuit before
    # `ps` runs, or the common path pays a subprocess it never needed.
    assert sys.ran == []
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
    sys = answering(True, stdout={"ps": _ps_line(900, 3000)})
    stopped = openwebui.stop_openwebui("linux", port=3000, sys=sys)
    assert stopped is True
    assert sys.signals == [(900, signal.SIGTERM)]
    assert sys.ran == [["ps", "-eo", "pid=,args="]]


def test_stop_openwebui_adopts_every_process_serving_our_port(isolated_home: Path) -> None:
    """`uv tool run` and the server it wraps both carry the argv; neither may be left."""
    sys = FakeSystem(
        stdout={
            "ps": _ps_line(900, 3000)
            + " 901 /opt/uv/tools/open-webui/bin/python"
            + " /opt/uv/tools/open-webui/bin/open-webui serve --host 127.0.0.1 --port 3000\n"
        }
    )
    stopped = openwebui.stop_openwebui("linux", port=3000, sys=sys)
    assert stopped is True
    assert [pid for pid, _ in sys.signals] == [900, 901]


def test_stop_openwebui_adopts_nothing_when_no_process_is_ours(isolated_home: Path) -> None:
    """Without a pid file the argv is the only evidence — a stranger provides none."""
    sys = FakeSystem(stdout={"ps": " 900 /usr/lib/systemd/systemd --user\n"})
    assert openwebui.stop_openwebui("linux", port=3000, sys=sys) is False
    assert sys.signals == []


def test_stop_openwebui_adopts_only_the_webui_on_our_own_port(isolated_home: Path) -> None:
    """A second LEPIKA_HOME runs its own UI on its own port; ours is the one we stop."""
    sys = FakeSystem(stdout={"ps": _ps_line(900, 3001)})
    assert openwebui.stop_openwebui("linux", port=3000, sys=sys) is False
    assert sys.signals == []


def test_stop_openwebui_matches_the_port_whole(isolated_home: Path) -> None:
    """`--port 30000` starts with `--port 3000`: a substring match would kill it."""
    sys = FakeSystem(stdout={"ps": _ps_line(900, 30000)})
    assert openwebui.stop_openwebui("linux", port=3000, sys=sys) is False
    assert sys.signals == []


def test_stop_openwebui_adopts_nothing_on_windows(isolated_home: Path) -> None:
    """`tasklist` lists images, not argv: there is no command line to search there."""
    sys = FakeSystem(stdout={"ps": _ps_line(900, 3000)})
    assert openwebui.stop_openwebui("windows", port=3000, sys=sys) is False
    assert sys.signals == []
    assert sys.ran == []


def test_stop_openwebui_never_searches_without_a_port(isolated_home: Path) -> None:
    """The port is what the search matches on — with none given there is nothing to ask."""
    sys = FakeSystem(stdout={"ps": _ps_line(900, 3000)})
    assert openwebui.stop_openwebui("linux", sys=sys) is False
    assert sys.ran == []


def test_stop_openwebui_adoption_reads_the_process_list_without_logging_it(
    isolated_home: Path,
) -> None:
    """A pure read of every command line on the machine is never written to the log."""
    sys = FakeSystem(stdout={"ps": _ps_line(900, 3000)})
    openwebui.stop_openwebui("linux", port=3000, sys=sys)
    assert sys.ran_kwargs == [_READ]


def test_stop_openwebui_adoption_survives_a_process_that_just_exited(
    isolated_home: Path,
) -> None:
    """`ps` and the signal are two moments: what it listed may already be gone."""
    sys = FakeSystem(
        stdout={"ps": _ps_line(900, 3000)},
        signal_errors={900: ProcessLookupError(3, "No such process")},
    )
    assert openwebui.stop_openwebui("linux", port=3000, sys=sys) is False


def test_stop_openwebui_adoption_ignores_rows_with_no_readable_pid(
    isolated_home: Path,
) -> None:
    """`ps` output is parsed, not trusted: a header or a wrapped row is not a pid."""
    sys = FakeSystem(stdout={"ps": "  PID COMMAND\n" + " open-webui serve --port 3000\n"})
    assert openwebui.stop_openwebui("linux", port=3000, sys=sys) is False
    assert sys.signals == []


def test_stop_openwebui_prefers_the_pidfile_over_the_search(isolated_home: Path) -> None:
    """The recorded pid is still the first answer; the search is only the fallback."""
    paths.pid_file("openwebui").write_text("4242")
    sys = answering(True, stdout={"ps": _ps_line(900, 3000)})
    stopped = openwebui.stop_openwebui("linux", port=3000, sys=sys)
    assert stopped is True
    assert [pid for pid, _ in sys.signals] == [4242]
    assert sys.ran == []


def test_restart_openwebui_waits_for_the_old_server_to_die_before_starting(
    isolated_home: Path,
) -> None:
    """`lepika update` must not probe the dying server and call it 'already running'."""
    paths.pid_file("openwebui").write_text("4242")
    events: list[str] = []
    lingering = {"probes": 0}

    def up(request: urllib.request.Request) -> Response:
        assert request.full_url == _HEALTH
        events.append("probe")
        if "start" in _kinds(events):
            return FakeResponse()
        # Still answering for a few probes after the SIGTERM, then gone.
        lingering["probes"] += 1
        if lingering["probes"] <= 4:
            return FakeResponse()
        raise OSError("connection refused")

    openwebui.restart_openwebui(Config(), "linux", sys=Recorded(events, urls=up))
    kinds = _kinds(events)
    assert "kill" in kinds
    # 4 lingering "yes" answers had to be outwaited, not believed: the staleness
    # gate, four probes that still saw the dying server, and the one that finally
    # saw it gone — then ensure_openwebui's own probe before it starts anything.
    assert lingering["probes"] == 6
    assert kinds.index("kill") < kinds.index("install") < kinds.index("start")


def test_restart_openwebui_raises_when_the_old_server_never_dies(
    isolated_home: Path,
) -> None:
    paths.pid_file("openwebui").write_text("4242")
    sys = answering(True, port=3210)
    with pytest.raises(FriendlyError) as exc:
        openwebui.restart_openwebui(Config(webui_port=3210), "linux", sys=sys)
    assert "3210" in exc.value.problem or "3210" in exc.value.fix
    # Never a second server next to the one that would not die.
    assert sys.spawned == []


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
        # The real bind probe, on a fake clock: what is tested is that the
        # wildcard address is asked, which only a real listener can answer.
        sys = FakeSystem(bindable=system.REAL.can_bind)
        assert ports.port_free(port, sys=sys) is False
        assert ("0.0.0.0", port) in sys.bind_probes


@pytest.mark.parametrize("busy", ["0.0.0.0", "127.0.0.1"])
def test_bind_once_calls_a_port_busy_when_either_address_is_taken(busy: str) -> None:
    sys = FakeSystem(bindable=lambda address, port: address != busy)
    assert ports._bind_once(3000, sys=sys) is False


def test_bind_once_probes_both_addresses_before_calling_a_port_free() -> None:
    sys = FakeSystem(bindable=True)
    assert ports._bind_once(3000, sys=sys) is True
    assert [address for address, _ in sys.bind_probes] == ["0.0.0.0", "127.0.0.1"]


def test_port_free_retries_before_calling_a_port_busy() -> None:
    """A port we just stopped serving can need a moment to be released."""
    # Each attempt probes both addresses while the port is free; the first two
    # attempts are refused on the first address, the third succeeds on both.
    attempts = {"n": 0}

    def bind(address: str, port: int) -> bool:
        if address == "0.0.0.0":
            attempts["n"] += 1
            return attempts["n"] >= 3
        return True

    sys = FakeSystem(bindable=bind)
    assert ports.port_free(3000, sys=sys) is True
    assert attempts["n"] == 3
    assert sys.slept == [0.5, 0.5]


def test_port_free_gives_up_on_a_genuinely_busy_port() -> None:
    assert ports.port_free(3000, sys=FakeSystem(bindable=False)) is False


def test_ensure_openwebui_rejects_a_port_another_app_holds(isolated_home: Path) -> None:
    """A busy port must be named, not hidden behind a 180s 'did not become ready'."""
    sys = answering(False, port=3210, bindable=False)
    with pytest.raises(FriendlyError) as exc:
        openwebui.ensure_openwebui(Config(webui_port=3210), sys=sys)
    assert "3210" in exc.value.problem
    assert "webui_port" in exc.value.fix
    # Neither installed nor started when the port is taken.
    assert sys.ran == []
    assert sys.spawned == []


def test_ensure_openwebui_noop_when_healthy(isolated_home: Path) -> None:
    """A healthy OpenWebUI must cost nothing: no install round-trip, no start."""
    sys = answering(True)
    openwebui.ensure_openwebui(Config(), sys=sys)
    assert sys.ran == []
    assert sys.spawned == []


def test_ensure_openwebui_installs_then_starts_then_waits(isolated_home: Path) -> None:
    events: list[str] = []

    # Down until started, up afterwards — so wait_for must run after start.
    def up(request: urllib.request.Request) -> Response:
        assert request.full_url == _HEALTH
        events.append("probe")
        if any(e.startswith("start:") for e in events):
            return FakeResponse()
        raise OSError("connection refused")

    openwebui.ensure_openwebui(Config(), sys=Recorded(events, urls=up))
    assert _kinds(events) == ["probe", "install", "start", "probe"]
    assert "uv tool install --python 3.11 open-webui" in events[1]
    assert "--port 3000" in events[2]


def test_webui_url_uses_localhost() -> None:
    assert health.webui_url(3000) == "http://localhost:3000"


def test_webui_up_is_false_when_health_check_fails() -> None:
    sys = FakeSystem(urls=routes({_HEALTH: OSError("connection refused")}))
    assert health.webui_up(3000, sys=sys) is False


def test_webui_up_is_true_when_health_check_succeeds() -> None:
    sys = FakeSystem(urls=routes({_HEALTH: FakeResponse()}))
    assert health.webui_up(3000, sys=sys) is True
    assert [r.full_url for r in sys.requested] == ["http://127.0.0.1:3000/health"]
    assert sys.request_timeouts == [1.0]


def test_start_openwebui_binds_to_localhost_only(isolated_home: Path) -> None:
    """OpenWebUI defaults to 0.0.0.0: Express mode must not put the UI on the network."""
    sys = FakeSystem()
    openwebui.start_openwebui(3000, "http://127.0.0.1:11434", sys=sys)
    argv, _, _ = sys.spawned[0]
    assert argv[argv.index("--host") + 1] == "127.0.0.1"


def test_wait_until_down_hint_does_not_name_a_single_caller() -> None:
    """`lepika connect` restarts the UI too — the hint must not say `lepika update`."""
    with pytest.raises(FriendlyError) as exc:
        openwebui.wait_until_down(3210, attempts=2, sys=answering(True, port=3210))
    assert exc.value.fix == "Stop whatever is listening on port 3210, then try again."


def test_wait_for_hint_points_at_lepika_logs() -> None:
    """A raw ~/.lepika/logs path is the wrong place in Server mode; `lepika logs` is not."""
    with pytest.raises(FriendlyError) as exc:
        health.wait_for(lambda: False, seconds=1, what="OpenWebUI", sys=FakeSystem())
    assert exc.value.fix == "Run `lepika logs` to see why, then `lepika doctor`."
