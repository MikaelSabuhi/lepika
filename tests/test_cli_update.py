from __future__ import annotations

import subprocess
from typing import Any

import pytest
from typer.testing import CliRunner

from lepika import cli, detect, express, proc

runner = CliRunner()


def test_update_upgrades_engine_and_webui(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    installed: list[detect.SystemInfo] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    info = detect.SystemInfo(
        os="linux",
        arch="x86_64",
        gpu="nvidia",
        ram_gb=32.0,
        has_docker=False,
        has_ollama=True,
        ollama_running=True,
    )
    monkeypatch.setattr(detect, "detect", lambda **k: info)
    monkeypatch.setattr(proc, "run_logged", fake_run)
    monkeypatch.setattr(express, "install_ollama", lambda i, **k: installed.append(i))
    monkeypatch.setattr(express, "webui_up", lambda port, **k: False)
    monkeypatch.setattr(express, "stop_openwebui", lambda os_name, **k: True)
    monkeypatch.setattr(express, "ensure_openwebui", lambda cfg, **k: None)

    result = runner.invoke(cli.app, ["update"])
    assert result.exit_code == 0
    # The install script must stream (it may prompt for sudo), so it goes through
    # express.install_ollama — never through captured run_logged.
    assert installed == [info]
    assert not any("ollama.com/install.sh" in " ".join(c) for c in calls)
    # The pin travels with every `uv tool` call: an upgrade that resolved a
    # different interpreter would leave `uv tool run --python 3.11` with no
    # matching env to reuse, which is the ephemeral-build failure again.
    assert ["uv", "tool", "upgrade", "--python", express.OPENWEBUI_PYTHON, "open-webui"] in calls


def test_update_restarts_openwebui_rather_than_probing_the_dying_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The upgraded build only takes effect if the old server is actually replaced."""
    info = detect.SystemInfo(
        os="linux",
        arch="x86_64",
        gpu="nvidia",
        ram_gb=32.0,
        has_docker=False,
        has_ollama=True,
        ollama_running=True,
    )
    restarts: list[tuple[int, str]] = []
    monkeypatch.setattr(detect, "detect", lambda **k: info)
    monkeypatch.setattr(proc, "run_logged", lambda cmd, **k: None)
    monkeypatch.setattr(express, "install_ollama", lambda i, **k: None)
    monkeypatch.setattr(
        express,
        "restart_openwebui",
        lambda cfg, os_name, **k: restarts.append((cfg.webui_port, os_name)),
    )
    monkeypatch.setattr(
        express, "stop_openwebui", lambda *a, **k: pytest.fail("update must go through restart")
    )

    result = runner.invoke(cli.app, ["update"])
    assert result.exit_code == 0
    assert restarts == [(3000, "linux")]


def test_down_gates_the_signal_on_the_configured_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """`lepika down` must tell `stop_openwebui` which port proves the pid is ours."""
    info = detect.SystemInfo(
        os="linux",
        arch="x86_64",
        gpu="nvidia",
        ram_gb=32.0,
        has_docker=False,
        has_ollama=True,
        ollama_running=True,
    )
    seen: list[int | None] = []
    monkeypatch.setattr(detect, "detect", lambda **k: info)
    monkeypatch.setattr(
        express,
        "stop_openwebui",
        lambda os_name, port=None, **k: bool(seen.append(port)),
    )
    result = runner.invoke(cli.app, ["down"])
    assert result.exit_code == 0
    assert seen == [3000]
