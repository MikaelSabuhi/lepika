from __future__ import annotations

from pathlib import Path

import pytest
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
