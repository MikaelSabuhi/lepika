from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fakes import Runner
from typer.testing import CliRunner

from lepika import cli, detect, doctor
from lepika.detect import SystemInfo

runner = CliRunner()


def info(ram: float = 32.0, has_ollama: bool = True) -> SystemInfo:
    return SystemInfo(
        os="linux",
        arch="x86_64",
        gpu="nvidia",
        ram_gb=ram,
        has_docker=False,
        has_ollama=has_ollama,
        ollama_running=True,
    )


def test_all_green_when_everything_up(isolated_home: Path) -> None:
    results = doctor.run_checks(
        info(),
        which=lambda n: f"/usr/bin/{n}",
        api_up=lambda url, **k: True,
        webui_up=lambda port, **k: True,
    )
    assert all(r.ok for r in results)


def test_missing_ollama_has_hint(isolated_home: Path) -> None:
    results = doctor.run_checks(
        info(has_ollama=False),
        which=lambda n: "/usr/bin/uv" if n == "uv" else None,
        api_up=lambda url, **k: False,
        webui_up=lambda port, **k: False,
    )
    failed = {r.name: r for r in results if not r.ok}
    assert "Ollama installed" in failed
    assert failed["Ollama installed"].hint


def test_low_ram_is_flagged(isolated_home: Path) -> None:
    results = doctor.run_checks(
        info(ram=4.0),
        which=lambda n: f"/usr/bin/{n}",
        api_up=lambda url, **k: True,
        webui_up=lambda port, **k: True,
    )
    ram_check = next(r for r in results if r.name == doctor.RAM_CHECK)
    assert ram_check.ok is False


def test_checks_probe_the_configured_engine_url(isolated_home: Path) -> None:
    """A remote engine must be diagnosed where it actually lives, not on localhost."""
    from lepika import config

    config.save(config.Config(engine_url="http://gpu-box.local:11434"))
    probed: list[str] = []
    doctor.run_checks(
        info(),
        which=lambda n: f"/usr/bin/{n}",
        api_up=lambda url, **k: bool(probed.append(url)),
        webui_up=lambda port, **k: True,
    )
    assert probed == ["http://gpu-box.local:11434"]


def test_remote_engine_skips_the_local_install_check_and_uses_the_key(isolated_home: Path) -> None:
    """A remote engine is someone else's to install — only whether it answers is ours."""
    from lepika import config

    config.save(
        config.Config(engine_managed=False, engine_url="http://gpu-box:11435", engine_key="k")
    )
    seen: list[str] = []
    results = doctor.run_checks(
        info(has_ollama=False),
        which=lambda n: f"/usr/bin/{n}",
        api_up=lambda url, key="", **k: bool(seen.append(key)) or True,
        webui_up=lambda port, **k: True,
    )
    assert "Ollama installed" not in {r.name for r in results}
    assert seen == ["k"]


def test_a_remote_engine_hint_offers_reconnecting_with_a_new_key(isolated_home: Path) -> None:
    """A rotated key looks exactly like a dead box; the hint has to cover both."""
    from lepika import config

    config.save(config.Config(engine_managed=False, engine_url="http://gpu-box:11435"))
    results = doctor.run_checks(
        info(),
        which=lambda n: f"/usr/bin/{n}",
        api_up=lambda url, **k: False,
        webui_up=lambda port, **k: True,
    )
    hint = next(r for r in results if r.name == "Engine responding").hint
    assert "lepika connect http://gpu-box:11435 --key <key>" in hint
    assert "lepika connect --local" in hint


def test_the_openwebui_hint_points_at_the_log_that_holds_the_cause(isolated_home: Path) -> None:
    results = doctor.run_checks(
        info(),
        which=lambda n: f"/usr/bin/{n}",
        api_up=lambda url, **k: True,
        webui_up=lambda port, **k: False,
    )
    hint = next(r for r in results if r.name == "OpenWebUI responding").hint
    assert "lepika logs" in hint
    assert "openwebui.log" in hint


def test_doctor_command_exits_zero_when_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detect, "detect", lambda **k: info())
    monkeypatch.setattr(
        doctor,
        "run_checks",
        lambda i, **k: [doctor.CheckResult("uv installed", True, "install uv")],
    )
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 0
    assert "uv installed" in result.output


def test_doctor_command_fails_on_core_check_and_prints_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(detect, "detect", lambda **k: info())
    monkeypatch.setattr(
        doctor,
        "run_checks",
        lambda i, **k: [doctor.CheckResult("Engine responding", False, "Run `lepika up`")],
    )
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 1
    assert "Run `lepika up`" in result.output


def test_doctor_command_ram_warning_does_not_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detect, "detect", lambda **k: info(ram=4.0))
    monkeypatch.setattr(
        doctor,
        "run_checks",
        lambda i, **k: [doctor.CheckResult(doctor.RAM_CHECK, False, "4 GB detected")],
    )
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 0
    assert "4 GB detected" in result.output


def test_server_mode_checks_docker_instead_of_native_ollama(isolated_home: Path) -> None:
    """Server mode's prerequisites are Docker's, so its checks are Docker's too."""
    from lepika import config

    config.save(config.Config(mode="server"))
    results = doctor.run_checks(
        info(has_ollama=False),
        which=lambda n: "/usr/bin/docker" if n == "docker" else None,
        api_up=lambda url, **k: True,
        webui_up=lambda port, **k: True,
        run=Runner(stdout={"docker info --format": '{"nvidia": {}}'}),
    )
    names = {r.name for r in results}
    assert {"Docker running", "docker compose available", "NVIDIA GPU visible to Docker"} <= names
    assert "Ollama installed" not in names
    assert "uv installed" not in names
    assert all(r.ok for r in results)


def test_server_mode_never_probes_docker_when_it_is_not_installed(isolated_home: Path) -> None:
    """`docker info` with no docker binary is a FriendlyError, not a red ✗ with a hint."""
    from lepika import config

    config.save(config.Config(mode="server"))
    results = doctor.run_checks(
        info(),
        which=lambda n: None,
        api_up=lambda url, **k: True,
        webui_up=lambda port, **k: True,
        run=lambda cmd, **k: pytest.fail("must not probe a Docker that isn't there"),
    )
    failed = {r.name: r for r in results if not r.ok}
    assert "Docker running" in failed
    assert "lepika --mode express" in failed["Docker running"].hint


def test_a_docker_info_that_times_out_is_a_red_check_not_an_abort(isolated_home: Path) -> None:
    """Docker Desktop mid-start accepts the socket and never answers: still a ✗ with a hint."""
    from lepika import config
    from lepika.errors import FriendlyError

    config.save(config.Config(mode="server"))

    def hang(cmd: list[str], **k: Any) -> Any:
        raise FriendlyError("Command timed out after 20s: docker info", "Try again")

    results = doctor.run_checks(
        info(),
        which=lambda n: "/usr/bin/docker" if n == "docker" else None,
        api_up=lambda url, **k: True,
        webui_up=lambda port, **k: True,
        run=hang,
    )
    failed = {r.name: r for r in results if not r.ok}
    assert "Docker running" in failed
    assert "docker compose available" in failed
