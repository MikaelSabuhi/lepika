from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fakes import Caller

from lepika import config, detect, express, paths
from lepika.backend import UpdateStep
from lepika.errors import FriendlyError


def _info(os_name: str) -> detect.SystemInfo:
    return detect.SystemInfo(
        os=os_name,
        arch="x86_64",
        gpu="nvidia",
        ram_gb=32.0,
        has_docker=False,
        has_ollama=True,
        ollama_running=True,
    )


class Seq:
    """Stands in for proc.run_logged: canned results per leading argv words.

    A list of outputs is consumed in order (the last one repeats), because `update`
    asks the same question — "which version?" — before and after an upgrade.
    """

    def __init__(self, answers: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.answers = answers or {}

    def __call__(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append((list(cmd), dict(kwargs)))
        joined = " ".join(cmd)
        for prefix, answer in self.answers.items():
            if joined.startswith(prefix):
                if isinstance(answer, BaseException):
                    raise answer
                if isinstance(answer, list):
                    answer = answer.pop(0) if len(answer) > 1 else answer[0]
                if isinstance(answer, tuple):
                    code, out = answer
                else:
                    code, out = 0, answer
                # As run_logged: a code outside `ok` fails a checked call.
                if kwargs.get("check", True) and code not in kwargs.get("ok", (0,)):
                    raise FriendlyError(f"Command failed: {joined}", "see the log")
                return subprocess.CompletedProcess(cmd, code, stdout=out, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    def argvs(self) -> list[list[str]]:
        return [c for c, _ in self.calls]


def _version(v: str) -> str:
    return f"ollama version is {v}\n"


def _which(*present: str) -> Callable[[str], str | None]:
    return lambda name: f"/usr/bin/{name}" if name in present else None


@pytest.fixture()
def restarts(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []
    monkeypatch.setattr(express, "restart_openwebui", lambda cfg, os_name, **k: seen.append("ui"))
    monkeypatch.setattr(
        express, "stop_ollama", lambda os_name, url, **k: seen.append("stop") or True
    )
    monkeypatch.setattr(express, "start_ollama", lambda os_name, **k: seen.append("start"))
    monkeypatch.setattr(express, "wait_for", lambda pred, secs, what, **k: None)
    return seen


def _answers(values: list[Any]) -> Callable[..., Any]:
    """Each call takes the next value (the last one repeats); an exception is raised."""

    def answer(*args: Any, **kwargs: Any) -> Any:
        value = values.pop(0) if len(values) > 1 else values[0]
        if isinstance(value, BaseException):
            raise value
        return value

    return answer


def _update(
    info: detect.SystemInfo,
    run: Seq,
    *,
    cfg: config.Config | None = None,
    which: Callable[[str], str | None] | None = None,
    call: Caller | None = None,
    running: str | list[Any] | None = None,
    up: list[bool] | None = None,
) -> list[UpdateStep]:
    """`running` is what the engine reports (None: nothing answers); `up` overrides
    the answers to `api_up` in order, for an engine that comes and goes."""
    versions = running if isinstance(running, list) else [running or ""]
    return express.update(
        info,
        cfg or config.Config(),
        run=run,
        which=which or _which("systemctl", "brew"),
        call=call or Caller(),
        popen=lambda *a, **k: pytest.fail("popen must go through start_ollama"),
        api_up=_answers(up if up is not None else [running is not None]),
        sleep=lambda s: None,
        kill=lambda pid, sig: pytest.fail("kill must go through stop_ollama"),
        running_version=_answers(versions),
    )


def _linux_run(enabled: int, active: int, versions: list[str] | None = None) -> Seq:
    return Seq(
        {
            "systemctl is-enabled": (enabled, ""),
            "systemctl is-active": (active, ""),
            "ollama --version": [_version(v) for v in (versions or ["0.34.0"])],
        }
    )


def test_update_leaves_an_enabled_systemd_unit_enabled(restarts: list[str]) -> None:
    call = Caller()
    _update(_info("linux"), _linux_run(0, 0), call=call, running="0.34.0")
    assert any("install.sh" in " ".join(c) for c in call.calls)
    assert not any("disable" in c or "stop" in c for c in call.calls)


def test_update_disables_again_a_unit_that_was_disabled(restarts: list[str]) -> None:
    call = Caller()
    _update(_info("linux"), _linux_run(1, 3), call=call, running="0.34.0")
    restores = [c for c in call.calls if "systemctl" in c]
    # `--now` is what stops it: one verb, one sudo prompt.
    assert restores == [["sudo", "systemctl", "disable", "--now", "ollama.service"]]


def test_update_restores_only_what_the_script_changed(restarts: list[str]) -> None:
    call = Caller()
    _update(_info("linux"), _linux_run(0, 3), call=call, running="0.34.0")
    assert [c for c in call.calls if "systemctl" in c] == [
        ["sudo", "systemctl", "stop", "ollama.service"]
    ]


def test_update_restores_the_unit_even_when_the_script_fails(restarts: list[str]) -> None:
    call = Caller(code=1)
    with pytest.raises(FriendlyError) as exc:
        _update(_info("linux"), _linux_run(1, 3, ["0.33.2"]), call=call, running="0.33.2")
    assert exc.value.problem == "Ollama upgrade failed."
    assert ["sudo", "systemctl", "disable", "--now", "ollama.service"] in call.calls
    assert "ui" not in restarts


def test_update_accepts_a_failed_optional_step_once_the_binary_moved(
    restarts: list[str],
) -> None:
    """The script aborts after the binary lands when systemd itself is wedged."""
    steps = _update(
        _info("linux"), _linux_run(0, 0, ["0.33.2", "0.34.0"]), call=Caller(code=1), running=None
    )
    assert steps[0] == UpdateStep("Ollama", "upgraded", "0.33.2 → 0.34.0")


def test_update_without_systemctl_touches_no_unit(restarts: list[str]) -> None:
    call = Caller()
    run = _linux_run(1, 3)
    _update(_info("linux"), run, which=_which(), call=call, running="0.34.0")
    assert any("install.sh" in " ".join(c) for c in call.calls)
    assert not any("systemctl" in " ".join(c) for c in call.calls + run.argvs())


def _pid_file() -> None:
    paths.pid_file("ollama").write_text("4242")


def test_update_restarts_the_engine_lepika_started_onto_the_new_binary(
    restarts: list[str], isolated_home: Path
) -> None:
    _pid_file()
    run = _linux_run(0, 0, ["0.33.2", "0.34.0"])
    run.answers["ps -p 4242"] = "ollama\n"
    steps = _update(_info("linux"), run, running=["0.33.2", "0.34.0"])
    assert restarts == ["stop", "start", "ui"]
    assert steps[0] == UpdateStep("Ollama", "upgraded", "0.33.2 → 0.34.0")


def test_update_does_not_start_a_second_engine_when_the_stop_was_refused(
    restarts: list[str], isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pid_file()
    run = _linux_run(0, 0, ["0.33.2", "0.34.0"])
    run.answers["ps -p 4242"] = "ollama\n"
    monkeypatch.setattr(
        express, "stop_ollama", lambda os_name, url, **k: bool(restarts.append("stop"))
    )
    steps = _update(_info("linux"), run, running="0.33.2")
    assert "stop" in restarts and "start" not in restarts
    assert steps[0].outcome == "stale"


def test_update_brings_back_its_own_engine_when_the_upgrade_stopped_it(
    restarts: list[str], isolated_home: Path
) -> None:
    """winget may stop a running Ollama to replace it: ours is started again."""
    _pid_file()
    run = Seq(
        {
            "winget upgrade": (0, ""),
            "ollama --version": [_version("0.33.2"), _version("0.34.0")],
            "tasklist": '"ollama.exe","4242"\n',
        }
    )
    steps = _update(_info("windows"), run, running="0.34.0", up=[True, False, True])
    assert restarts == ["start", "ui"]
    assert steps[0] == UpdateStep("Ollama", "upgraded", "0.33.2 → 0.34.0")


def test_update_never_starts_an_adopted_engine_the_upgrade_stopped(restarts: list[str]) -> None:
    run = Seq({"winget upgrade": (0, ""), "ollama --version": _version("0.34.0")})
    _update(_info("windows"), run, running="0.34.0", up=[True, False])
    assert "start" not in restarts and "stop" not in restarts


def test_update_survives_an_unreadable_running_version(restarts: list[str]) -> None:
    run = _linux_run(0, 0, ["0.33.2", "0.34.0"])
    unreadable = FriendlyError("Can't reach the engine", "start it")
    steps = _update(_info("linux"), run, running=[unreadable])
    assert steps[0] == UpdateStep("Ollama", "upgraded", "0.33.2 → 0.34.0")
    assert restarts == ["ui"]


def test_update_restores_nothing_when_systemd_does_not_answer(
    restarts: list[str], isolated_home: Path
) -> None:
    """`systemctl` on PATH with no running systemd answers outside its documented codes."""
    call = Caller()
    _update(_info("linux"), _linux_run(1, 1), call=call, running="0.34.0")
    assert any("install.sh" in " ".join(c) for c in call.calls)
    assert not any("systemctl" in c for c in call.calls)
    log_text = (paths.logs_dir() / "lepika.log").read_text()
    assert "ollama.update.unit_state_unknown" in log_text


def test_update_treats_a_missing_unit_as_disabled_and_inactive(restarts: list[str]) -> None:
    call = Caller()
    run = _linux_run(4, 4)
    _update(_info("linux"), run, call=call, running="0.34.0")
    assert [c for c in call.calls if "systemctl" in c] == [
        ["sudo", "systemctl", "disable", "--now", "ollama.service"]
    ]
    # The documented answers are expected, so a disabled unit writes nothing to the log.
    probes = [kw for c, kw in run.calls if c[0] == "systemctl"]
    assert probes and all(4 in kw["ok"] and kw["log"] is False for kw in probes)


def test_update_never_restarts_an_adopted_engine(restarts: list[str]) -> None:
    run = _linux_run(0, 0, ["0.33.2", "0.34.0"])
    steps = _update(_info("linux"), run, running="0.33.2")
    assert "stop" not in restarts and "start" not in restarts
    assert steps[0].component == "Ollama"
    assert steps[0].outcome == "stale"
    assert "0.33.2" in steps[0].detail and "0.34.0" in steps[0].detail


def test_update_reports_current_when_nothing_changed(
    restarts: list[str], isolated_home: Path
) -> None:
    _pid_file()
    run = _linux_run(0, 0, ["0.34.0"])
    run.answers["ps -p 4242"] = "ollama\n"
    steps = _update(_info("linux"), run, running="0.34.0")
    assert steps[0] == UpdateStep("Ollama", "current", "0.34.0")
    assert "stop" not in restarts and "start" not in restarts


def test_update_on_macos_asks_brew_before_upgrading(restarts: list[str]) -> None:
    run = Seq({"brew outdated": "", "ollama --version": _version("0.34.0")})
    _update(_info("macos"), run, running="0.34.0")
    assert ["brew", "upgrade", "ollama"] not in run.argvs()

    run = Seq({"brew outdated": "ollama\n", "ollama --version": _version("0.34.0")})
    _update(_info("macos"), run, running="0.34.0")
    upgrades = [kw for c, kw in run.calls if c == ["brew", "upgrade", "ollama"]]
    assert upgrades == [{"check": True}]


def test_update_on_macos_without_brew_leaves_the_app_to_update_itself(
    restarts: list[str],
) -> None:
    run = Seq({"ollama --version": _version("0.34.0")})
    _update(_info("macos"), run, which=_which(), running="0.34.0")
    assert not any(c[0] == "brew" for c in run.argvs())


def test_update_on_windows_accepts_no_applicable_update(restarts: list[str]) -> None:
    for code in (0x8A15002B, 0x8A15002B - 2**32, 0):
        run = Seq({"winget upgrade": (code, ""), "ollama --version": _version("0.34.0")})
        _update(_info("windows"), run, running="0.34.0")
        assert any(c[:2] == ["winget", "upgrade"] for c in run.argvs())

    run = Seq({"winget upgrade": (1, ""), "ollama --version": _version("0.34.0")})
    with pytest.raises(FriendlyError):
        _update(_info("windows"), run, running="0.34.0")


def test_update_fails_loudly_when_openwebui_cannot_upgrade(restarts: list[str]) -> None:
    run = _linux_run(0, 0)
    run.answers["uv tool upgrade"] = FriendlyError("Command failed: uv", "see the log")
    with pytest.raises(FriendlyError):
        _update(_info("linux"), run, running="0.34.0")
    assert "ui" not in restarts
    upgrade = next(kw for c, kw in run.calls if c[:3] == ["uv", "tool", "upgrade"])
    assert upgrade.get("check", True) is True


def test_update_does_not_restart_an_unchanged_openwebui(restarts: list[str]) -> None:
    run = _linux_run(0, 0)
    run.answers["uv tool list"] = "open-webui v0.6.5\n- open-webui\n"
    steps = _update(_info("linux"), run, running="0.34.0")
    assert "ui" not in restarts
    assert UpdateStep("OpenWebUI", "current", "0.6.5") in steps


def test_update_restarts_an_upgraded_openwebui(restarts: list[str]) -> None:
    run = _linux_run(0, 0)
    run.answers["uv tool list"] = ["open-webui v0.6.5\n", "open-webui v0.6.6\n"]
    steps = _update(_info("linux"), run, running="0.34.0")
    assert restarts == ["ui"]
    assert UpdateStep("OpenWebUI", "upgraded", "0.6.5 → 0.6.6") in steps


def test_update_skips_the_engine_entirely_when_it_is_remote(restarts: list[str]) -> None:
    call = Caller()
    run = Seq()
    cfg = config.Config(engine_managed=False, engine_url="http://gpu-box:11434")
    steps = _update(_info("linux"), run, cfg=cfg, call=call, running="0.34.0")
    assert call.calls == []
    assert not any(c[0] in ("brew", "winget", "ollama", "systemctl") for c in run.argvs())
    assert [s.component for s in steps] == ["OpenWebUI"]


def test_ollama_binary_version_reads_both_spellings() -> None:
    assert express.ollama_binary_version(Seq({"ollama": "ollama version is 0.40.16\n"})) == (
        "0.40.16"
    )
    no_engine = "Warning: could not connect to a running Ollama instance\n"
    no_engine += "Warning: client version is 0.40.16\n"
    assert express.ollama_binary_version(Seq({"ollama": no_engine})) == "0.40.16"
    assert express.ollama_binary_version(Seq({"ollama": ""})) is None
    missing = Seq({"ollama": FriendlyError("Command not found: ollama", "install it")})
    assert express.ollama_binary_version(missing) is None


def test_ollama_binary_version_reads_the_client_line_when_both_are_printed() -> None:
    """A running engine older than the binary prints both: the client line is the binary."""
    both = "ollama version is 0.33.2\nWarning: client version is 0.34.0\n"
    assert express.ollama_binary_version(Seq({"ollama": both})) == "0.34.0"
