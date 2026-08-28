from __future__ import annotations

import subprocess
from typing import Any

import pytest
from typer.testing import CliRunner

from ezai import cli, detect, express, proc

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
    monkeypatch.setattr(express, "stop_openwebui", lambda os_name, **k: True)
    monkeypatch.setattr(express, "ensure_openwebui", lambda cfg, **k: None)

    result = runner.invoke(cli.app, ["update"])
    assert result.exit_code == 0
    # The install script must stream (it may prompt for sudo), so it goes through
    # express.install_ollama — never through captured run_logged.
    assert installed == [info]
    assert not any("ollama.com/install.sh" in " ".join(c) for c in calls)
    assert ["uv", "tool", "upgrade", "open-webui"] in calls
