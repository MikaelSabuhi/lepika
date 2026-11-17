from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fakes import Answer, FakeResponse, FakeSystem, FakeUI, routes

from lepika import config, detect, express, health, ollama, openwebui, paths
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


def _version(v: str) -> str:
    return f"ollama version is {v}\n"


_VERSION_URL = f"{config.DEFAULT_ENGINE_URL}/api/version"
_DOWN = ConnectionRefusedError(61, "Connection refused")


def _serving(version: str) -> FakeResponse:
    """The engine answering `/api/version`: to a probe, and to a read of its version."""
    return FakeResponse(json.dumps({"version": version}).encode())


def _machine(
    stdout: dict[str, str | list[str]] | None = None,
    *,
    codes: dict[str, int] | None = None,
    which: tuple[str, ...] = ("systemctl", "brew"),
    **kwargs: Any,
) -> FakeSystem:
    """A machine with `which` on PATH. LePika runs as a user, not root, unless told."""
    return FakeSystem(
        stdout=dict(stdout or {}),
        codes=dict(codes or {}),
        which={name: f"/usr/bin/{name}" for name in which},
        **kwargs,
    )


@pytest.fixture()
def restarts(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []
    monkeypatch.setattr(openwebui, "restart_openwebui", lambda cfg, os_name, **k: seen.append("ui"))
    monkeypatch.setattr(
        ollama, "stop_ollama", lambda os_name, url, **k: seen.append("stop") or True
    )
    monkeypatch.setattr(ollama, "start_ollama", lambda **k: seen.append("start"))
    monkeypatch.setattr(health, "wait_for", lambda pred, secs, what, **k: None)
    return seen


def _update(
    info: detect.SystemInfo,
    machine: FakeSystem,
    *,
    engine: Answer | list[Answer] | None = None,
    cfg: config.Config | None = None,
) -> list[UpdateStep]:
    """`engine` answers the engine's `/api/version`, request by request, its last answer
    repeating: "is it answering" and "which version" both ask it, in the order `update`
    does. None: nothing may ask it at all."""
    if engine is not None:
        machine.urls = routes({_VERSION_URL: engine})
    try:
        return express.update(info, cfg or config.Config(), sys=machine, ui=FakeUI())
    finally:
        # Engines start and stop only through `start_ollama` and `stop_ollama`, both
        # doubled by `restarts`: nothing here launches or signals a process itself.
        assert machine.spawned == [], "a start must go through start_ollama"
        assert machine.signals == [], "a stop must go through stop_ollama"


_UNIT_STATE = {0: "enabled\n", 1: "disabled\n", 4: "not-found\n"}


def _linux_run(
    enabled: int,
    active: int,
    versions: list[str] | None = None,
    state: str | None = None,
    *,
    codes: dict[str, int] | None = None,
    **kwargs: Any,
) -> FakeSystem:
    """`state` is what `is-enabled` prints; by default the token its exit code means.
    `codes` adds the exit codes of other commands (the install script's, say)."""
    return _machine(
        {
            "systemctl is-enabled": _UNIT_STATE.get(enabled, "") if state is None else state,
            "ollama --version": [_version(v) for v in (versions or ["0.34.0"])],
        },
        codes={"systemctl is-enabled": enabled, "systemctl is-active": active, **(codes or {})},
        **kwargs,
    )


def _ran_with(machine: FakeSystem) -> list[tuple[list[str], dict[str, Any]]]:
    return list(zip(machine.ran, machine.ran_kwargs, strict=True))


def test_update_leaves_an_enabled_systemd_unit_enabled(restarts: list[str]) -> None:
    machine = _linux_run(0, 0)
    _update(_info("linux"), machine, engine=_serving("0.34.0"))
    assert any("install.sh" in " ".join(c) for c in machine.called)
    assert not any("disable" in c or "stop" in c for c in machine.called)


def test_update_disables_again_a_unit_that_was_disabled(restarts: list[str]) -> None:
    machine = _linux_run(1, 3)
    _update(_info("linux"), machine, engine=_serving("0.34.0"))
    restores = [c for c in machine.called if "systemctl" in c]
    # `--now` is what stops it: one verb, one sudo prompt.
    assert restores == [["sudo", "systemctl", "disable", "--now", "ollama.service"]]


def test_update_restores_only_what_the_script_changed(restarts: list[str]) -> None:
    machine = _linux_run(0, 3)
    _update(_info("linux"), machine, engine=_serving("0.34.0"))
    assert [c for c in machine.called if "systemctl" in c] == [
        ["sudo", "systemctl", "stop", "ollama.service"]
    ]


def test_update_restores_the_unit_even_when_the_script_fails(restarts: list[str]) -> None:
    machine = _linux_run(1, 3, ["0.33.2"], codes={"sh -c": 1})
    with pytest.raises(FriendlyError) as exc:
        _update(_info("linux"), machine, engine=_serving("0.33.2"))
    assert exc.value.problem == "Ollama upgrade failed."
    assert ["sudo", "systemctl", "disable", "--now", "ollama.service"] in machine.called
    assert "ui" not in restarts


_RESTORE_FAILED = UpdateStep(
    "Ollama service",
    "note",
    "could not put ollama.service back as it was — if Ollama misbehaves, run "
    "`systemctl disable --now ollama.service` as root",
)


def test_update_without_sudo_still_upgrades_openwebui_and_reports_the_unit(
    restarts: list[str],
) -> None:
    """A server where root has a password has systemd and no `sudo` at all."""
    machine = _linux_run(1, 3, missing={"sudo"})
    machine.stdout["uv tool list"] = ["open-webui v0.6.5\n", "open-webui v0.6.6\n"]
    steps = _update(_info("linux"), machine, engine=_serving("0.34.0"))
    assert any("install.sh" in " ".join(c) for c in machine.called)
    assert ["sudo", "systemctl", "disable", "--now", "ollama.service"] in machine.called
    assert _service_warning(steps) == _RESTORE_FAILED
    assert UpdateStep("OpenWebUI", "upgraded", "0.6.5 → 0.6.6") in steps
    assert restarts == ["ui"]


def test_update_reports_a_restore_that_exited_non_zero(restarts: list[str]) -> None:
    machine = _linux_run(1, 3, codes={"sudo systemctl": 1})
    steps = _update(_info("linux"), machine, engine=_serving("0.34.0"))
    assert _service_warning(steps) == _RESTORE_FAILED


def test_update_as_root_runs_systemctl_without_sudo(restarts: list[str]) -> None:
    machine = _linux_run(1, 3, euid=0)
    steps = _update(_info("linux"), machine, engine=_serving("0.34.0"))
    assert [c for c in machine.called if "systemctl" in c] == [
        ["systemctl", "disable", "--now", "ollama.service"]
    ]
    assert "Ollama service" not in [s.component for s in steps]


def test_a_failed_restore_never_hides_the_installer_error(restarts: list[str]) -> None:
    """The script failed with the binary unchanged, and the restore could not run either:
    the installer's failure is the one the user needs to see."""
    machine = _linux_run(1, 3, ["0.33.2"], codes={"sh -c": 1}, missing={"sudo"})
    with pytest.raises(FriendlyError) as exc:
        _update(_info("linux"), machine, engine=_serving("0.33.2"))
    assert exc.value.problem == "Ollama upgrade failed."
    assert ["sudo", "systemctl", "disable", "--now", "ollama.service"] in machine.called
    assert "ui" not in restarts


def test_update_accepts_a_failed_optional_step_once_the_binary_moved(
    restarts: list[str],
) -> None:
    """The script aborts after the binary lands when systemd itself is wedged."""
    machine = _linux_run(0, 0, ["0.33.2", "0.34.0"], codes={"sh -c": 1})
    steps = _update(_info("linux"), machine, engine=_DOWN)
    assert steps[0] == UpdateStep("Ollama", "upgraded", "0.33.2 → 0.34.0")


def test_update_without_systemctl_touches_no_unit(restarts: list[str]) -> None:
    machine = _linux_run(1, 3, which=())
    steps = _update(_info("linux"), machine, engine=_serving("0.34.0"))
    assert any("install.sh" in " ".join(c) for c in machine.called)
    assert not any("systemctl" in " ".join(c) for c in machine.called + machine.ran)
    assert "Ollama service" not in [s.component for s in steps]


def _service_warning(steps: list[UpdateStep]) -> UpdateStep:
    """The step that says the unit was left as the installer set it — right after Ollama's."""
    components = [s.component for s in steps]
    assert components[:2] == ["Ollama", "Ollama service"]
    return steps[1]


def test_update_says_nothing_about_the_service_when_its_state_was_read(
    restarts: list[str],
) -> None:
    for enabled, active in ((0, 0), (1, 3)):
        steps = _update(_info("linux"), _linux_run(enabled, active), engine=_serving("0.34.0"))
        assert "Ollama service" not in [s.component for s in steps]


def _pid_file() -> None:
    paths.pid_file("ollama").write_text("4242")


# The engine LePika started serves the old build until it is restarted: the probe
# before the upgrade, the probe after it, then a probe and a version read to compare
# with the binary. The restarted engine answers the last probe and read.
_RESTARTED = [*[_serving("0.33.2")] * 4, _serving("0.34.0")]


def test_update_restarts_the_engine_lepika_started_onto_the_new_binary(
    restarts: list[str], isolated_home: Path
) -> None:
    _pid_file()
    machine = _linux_run(0, 0, ["0.33.2", "0.34.0"])
    machine.stdout["ps -p 4242"] = "ollama\n"
    steps = _update(_info("linux"), machine, engine=_RESTARTED)
    assert restarts == ["stop", "start", "ui"]
    assert steps[0] == UpdateStep("Ollama", "upgraded", "0.33.2 → 0.34.0")


def test_update_does_not_start_a_second_engine_when_the_stop_was_refused(
    restarts: list[str], isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pid_file()
    machine = _linux_run(0, 0, ["0.33.2", "0.34.0"])
    machine.stdout["ps -p 4242"] = "ollama\n"
    monkeypatch.setattr(
        ollama, "stop_ollama", lambda os_name, url, **k: bool(restarts.append("stop"))
    )
    steps = _update(_info("linux"), machine, engine=_serving("0.33.2"))
    assert "stop" in restarts and "start" not in restarts
    assert steps[0].outcome == "stale"


def test_update_brings_back_its_own_engine_when_the_upgrade_stopped_it(
    restarts: list[str], isolated_home: Path
) -> None:
    """winget may stop a running Ollama to replace it: ours is started again."""
    _pid_file()
    machine = _machine(
        {
            "ollama --version": [_version("0.33.2"), _version("0.34.0")],
            # An Ollama before the upgrade; gone after it.
            "tasklist": ['"ollama.exe","4242"\n', "INFO: No tasks are running\n"],
        }
    )
    # Up before the upgrade, down after it, and up again once started.
    engine = [_serving("0.34.0"), _DOWN, _serving("0.34.0")]
    steps = _update(_info("windows"), machine, engine=engine)
    assert restarts == ["start", "ui"]
    assert steps[0] == UpdateStep("Ollama", "upgraded", "0.33.2 → 0.34.0")


def test_update_does_not_start_a_second_engine_when_its_own_is_alive_but_silent(
    restarts: list[str], isolated_home: Path
) -> None:
    """One missed probe is a busy engine, not a stopped one: a second would lose the port
    and overwrite the only record of the first."""
    _pid_file()
    machine = _linux_run(0, 0, ["0.33.2", "0.34.0"])
    machine.stdout["ps -p 4242"] = "ollama\n"
    steps = _update(_info("linux"), machine, engine=[_serving("0.33.2"), _DOWN])
    assert "start" not in restarts and "stop" not in restarts
    assert paths.pid_file("ollama").read_text() == "4242"
    assert steps[0] == UpdateStep(
        "Ollama",
        "stale",
        "installed 0.34.0; the running engine is not answering — "
        "run `lepika update` again to restart it",
    )
    assert restarts == ["ui"]


def test_update_restores_nothing_when_is_enabled_fails_without_a_state(
    restarts: list[str],
) -> None:
    """systemctl folds errors such as a refused bus into exit 1, with no state printed."""
    machine = _linux_run(1, 3, state="")
    steps = _update(_info("linux"), machine, engine=_serving("0.34.0"))
    assert any("install.sh" in " ".join(c) for c in machine.called)
    assert not any("systemctl" in c for c in machine.called)
    assert _service_warning(steps).outcome == "note"


def test_update_never_starts_an_adopted_engine_the_upgrade_stopped(restarts: list[str]) -> None:
    machine = _machine({"ollama --version": _version("0.34.0")})
    _update(_info("windows"), machine, engine=[_serving("0.34.0"), _DOWN])
    assert "start" not in restarts and "stop" not in restarts


def test_update_survives_an_unreadable_running_version(restarts: list[str]) -> None:
    machine = _linux_run(0, 0, ["0.33.2", "0.34.0"])
    # It answers the probe, but not with a version anyone can read.
    steps = _update(_info("linux"), machine, engine=FakeResponse(b"<html>busy</html>"))
    assert steps[0] == UpdateStep("Ollama", "upgraded", "0.33.2 → 0.34.0")
    assert restarts == ["ui"]


def test_update_restores_nothing_when_systemd_does_not_answer(
    restarts: list[str], isolated_home: Path
) -> None:
    """`systemctl` on PATH with no running systemd answers outside its documented codes."""
    machine = _linux_run(1, 1)
    steps = _update(_info("linux"), machine, engine=_serving("0.34.0"))
    assert any("install.sh" in " ".join(c) for c in machine.called)
    assert not any("systemctl" in c for c in machine.called)
    warning = _service_warning(steps)
    assert warning == UpdateStep(
        "Ollama service",
        "note",
        "could not read ollama.service's state, so it was left as the installer set it — "
        "if Ollama misbehaves, run `systemctl disable --now ollama.service` as root",
    )
    log_text = (paths.logs_dir() / "lepika.log").read_text()
    assert "ollama.update.unit_state_unknown" in log_text


def test_update_treats_a_missing_unit_as_disabled_and_inactive(restarts: list[str]) -> None:
    machine = _linux_run(4, 4)
    _update(_info("linux"), machine, engine=_serving("0.34.0"))
    assert [c for c in machine.called if "systemctl" in c] == [
        ["sudo", "systemctl", "disable", "--now", "ollama.service"]
    ]
    # The documented answers are expected, so a disabled unit writes nothing to the log.
    probes = [kw for c, kw in _ran_with(machine) if c[0] == "systemctl"]
    assert probes and all(4 in kw["ok"] and kw["log"] is False for kw in probes)


def test_update_never_restarts_an_adopted_engine(restarts: list[str]) -> None:
    machine = _linux_run(0, 0, ["0.33.2", "0.34.0"])
    steps = _update(_info("linux"), machine, engine=_serving("0.33.2"))
    assert "stop" not in restarts and "start" not in restarts
    assert steps[0].component == "Ollama"
    assert steps[0].outcome == "stale"
    assert "0.33.2" in steps[0].detail and "0.34.0" in steps[0].detail


def test_update_reports_current_when_nothing_changed(
    restarts: list[str], isolated_home: Path
) -> None:
    _pid_file()
    machine = _linux_run(0, 0, ["0.34.0"])
    machine.stdout["ps -p 4242"] = "ollama\n"
    steps = _update(_info("linux"), machine, engine=_serving("0.34.0"))
    assert steps[0] == UpdateStep("Ollama", "current", "0.34.0")
    assert "stop" not in restarts and "start" not in restarts


def test_update_on_macos_asks_brew_before_upgrading(restarts: list[str]) -> None:
    machine = _machine({"ollama --version": _version("0.34.0")})
    _update(_info("macos"), machine, engine=_serving("0.34.0"))
    assert ["brew", "upgrade", "ollama"] not in machine.ran

    # Homebrew exits 1 when a formula named on the command line is outdated.
    machine = _machine(
        {"brew outdated": "ollama\n", "ollama --version": _version("0.34.0")},
        codes={"brew outdated": 1},
    )
    _update(_info("macos"), machine, engine=_serving("0.34.0"))
    upgrades = [kw["check"] for c, kw in _ran_with(machine) if c == ["brew", "upgrade", "ollama"]]
    # Once, and checked: an upgrade that fails is a failure.
    assert upgrades == [True]
    # That answer is expected, so `run_logged` records no failure for it.
    asked = [kw for c, kw in _ran_with(machine) if c[:2] == ["brew", "outdated"]]
    assert asked and all(1 in kw["ok"] for kw in asked)


def test_update_on_macos_without_brew_leaves_the_app_to_update_itself(
    restarts: list[str],
) -> None:
    machine = _machine({"ollama --version": _version("0.34.0")}, which=())
    _update(_info("macos"), machine, engine=_serving("0.34.0"))
    assert not any(c[0] == "brew" for c in machine.ran)


def test_update_on_windows_accepts_no_applicable_update(restarts: list[str]) -> None:
    for code in (0x8A15002B, 0x8A15002B - 2**32, 0):
        machine = _machine({"ollama --version": _version("0.34.0")}, codes={"winget upgrade": code})
        _update(_info("windows"), machine, engine=_serving("0.34.0"))
        assert any(c[:2] == ["winget", "upgrade"] for c in machine.ran)

    machine = _machine({"ollama --version": _version("0.34.0")}, codes={"winget upgrade": 1})
    with pytest.raises(FriendlyError):
        _update(_info("windows"), machine, engine=_serving("0.34.0"))


def test_update_fails_loudly_when_openwebui_cannot_upgrade(restarts: list[str]) -> None:
    failed = FriendlyError("Command failed: uv", "see the log")
    machine = _linux_run(0, 0, raises={"uv tool upgrade": failed})
    with pytest.raises(FriendlyError):
        _update(_info("linux"), machine, engine=_serving("0.34.0"))
    assert "ui" not in restarts
    upgrade = next(kw for c, kw in _ran_with(machine) if c[:3] == ["uv", "tool", "upgrade"])
    assert upgrade["check"] is True


def test_update_does_not_restart_an_unchanged_openwebui(restarts: list[str]) -> None:
    machine = _linux_run(0, 0)
    machine.stdout["uv tool list"] = "open-webui v0.6.5\n- open-webui\n"
    steps = _update(_info("linux"), machine, engine=_serving("0.34.0"))
    assert "ui" not in restarts
    assert UpdateStep("OpenWebUI", "current", "0.6.5") in steps


def test_update_restarts_an_upgraded_openwebui(restarts: list[str]) -> None:
    machine = _linux_run(0, 0)
    machine.stdout["uv tool list"] = ["open-webui v0.6.5\n", "open-webui v0.6.6\n"]
    steps = _update(_info("linux"), machine, engine=_serving("0.34.0"))
    assert restarts == ["ui"]
    assert UpdateStep("OpenWebUI", "upgraded", "0.6.5 → 0.6.6") in steps


def test_update_skips_the_engine_entirely_when_it_is_remote(restarts: list[str]) -> None:
    machine = _machine()
    cfg = config.Config(engine_managed=False, engine_url="http://gpu-box:11434")
    # No engine answers are given: probing any engine at all would fail the test.
    steps = _update(_info("linux"), machine, cfg=cfg)
    assert machine.called == []
    assert not any(c[0] in ("brew", "winget", "ollama", "systemctl") for c in machine.ran)
    assert [s.component for s in steps] == ["OpenWebUI"]


def _binary_version(answer: str | BaseException) -> str | None:
    if isinstance(answer, BaseException):
        return ollama.ollama_binary_version(sys=FakeSystem(raises={"ollama": answer}))
    return ollama.ollama_binary_version(sys=FakeSystem(stdout={"ollama": answer}))


def test_ollama_binary_version_reads_both_spellings() -> None:
    assert _binary_version("ollama version is 0.40.16\n") == "0.40.16"
    no_engine = "Warning: could not connect to a running Ollama instance\n"
    no_engine += "Warning: client version is 0.40.16\n"
    assert _binary_version(no_engine) == "0.40.16"
    assert _binary_version("") is None
    assert _binary_version(FriendlyError("Command not found: ollama", "install it")) is None


def test_ollama_binary_version_reads_the_client_line_when_both_are_printed() -> None:
    """A running engine older than the binary prints both: the client line is the binary."""
    both = "ollama version is 0.33.2\nWarning: client version is 0.34.0\n"
    assert _binary_version(both) == "0.34.0"


def test_update_stops_its_engine_with_the_configured_key(
    restarts: list[str], isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pid_file()
    machine = _linux_run(0, 0, ["0.33.2", "0.34.0"])
    machine.stdout["ps -p 4242"] = "ollama\n"
    keys: list[str] = []
    monkeypatch.setattr(
        ollama, "stop_ollama", lambda os_name, url, **k: keys.append(k["key"]) or True
    )
    _update(_info("linux"), machine, cfg=config.Config(engine_key="k"), engine=_RESTARTED)
    assert keys == ["k"]


def test_update_reports_an_unread_unit_even_when_the_script_failed_after_installing(
    restarts: list[str],
) -> None:
    """A wedged systemd fails both the probe and the script's own systemd step."""
    machine = _linux_run(1, 1, ["0.33.2", "0.34.0"], codes={"sh -c": 1})
    steps = _update(_info("linux"), machine, engine=_DOWN)
    assert steps[0] == UpdateStep("Ollama", "upgraded", "0.33.2 → 0.34.0")
    assert _service_warning(steps).outcome == "note"
