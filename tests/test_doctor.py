from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ezai import cli, detect, doctor
from ezai.detect import SystemInfo

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
    from ezai import config

    config.save(config.Config(engine_url="http://gpu-box.local:11434"))
    probed: list[str] = []
    doctor.run_checks(
        info(),
        which=lambda n: f"/usr/bin/{n}",
        api_up=lambda url, **k: bool(probed.append(url)),
        webui_up=lambda port, **k: True,
    )
    assert probed == ["http://gpu-box.local:11434"]


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
        lambda i, **k: [doctor.CheckResult("Ollama API responding", False, "Run `ezai up`")],
    )
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 1
    assert "Run `ezai up`" in result.output


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
