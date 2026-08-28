from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ezai import cli, config, detect, express, paths

runner = CliRunner()

INFO = detect.SystemInfo(
    os="linux",
    arch="x86_64",
    gpu="nvidia",
    ram_gb=32.0,
    has_docker=False,
    has_ollama=True,
    ollama_running=True,
)


@pytest.fixture()
def quiet_stack(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    counts = {"ensure_ollama": 0, "ensure_openwebui": 0, "browser": 0, "stop": 0}
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(
        express,
        "ensure_ollama",
        lambda info, **k: counts.__setitem__("ensure_ollama", counts["ensure_ollama"] + 1),
    )
    monkeypatch.setattr(
        express,
        "ensure_openwebui",
        lambda cfg, **k: counts.__setitem__("ensure_openwebui", counts["ensure_openwebui"] + 1),
    )
    monkeypatch.setattr(
        express,
        "stop_openwebui",
        lambda os_name, **k: counts.__setitem__("stop", counts["stop"] + 1) or True,
    )
    monkeypatch.setattr(
        cli,
        "_open_browser",
        lambda url: counts.__setitem__("browser", counts["browser"] + 1),
    )
    return counts


def test_up_starts_stack_and_opens_browser(quiet_stack: dict[str, int]) -> None:
    result = runner.invoke(cli.app, ["up"])
    assert result.exit_code == 0
    assert quiet_stack["ensure_ollama"] == 1
    assert quiet_stack["ensure_openwebui"] == 1
    assert quiet_stack["browser"] == 1
    assert "http://localhost:3000" in result.output


def test_down_stops_webui(quiet_stack: dict[str, int]) -> None:
    result = runner.invoke(cli.app, ["down"])
    assert result.exit_code == 0
    assert quiet_stack["stop"] == 1


def test_status_reports_services(monkeypatch: pytest.MonkeyPatch, isolated_home: Path) -> None:
    monkeypatch.setattr(detect, "api_up", lambda url, **k: True)
    monkeypatch.setattr(express, "webui_up", lambda port, **k: False)
    config.save(config.Config(model="qwen3:8b"))
    result = runner.invoke(cli.app, ["status"])
    assert result.exit_code == 0
    assert "qwen3:8b" in result.output


def test_logs_prints_tail(isolated_home: Path) -> None:
    (paths.logs_dir() / "ezai.log").write_text("line-one\nline-two\n")
    result = runner.invoke(cli.app, ["logs"])
    assert result.exit_code == 0
    assert "line-two" in result.output
