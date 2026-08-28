from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from ezai import cli, config, detect, express, proc

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
def fake_engine(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    pulled: list[str] = []
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(express, "ensure_ollama", lambda info, **k: None)
    monkeypatch.setattr(express, "pull_model", lambda ref, **k: pulled.append(ref.raw))
    return pulled


def test_model_add_with_ref_pulls_and_saves(fake_engine: list[str], isolated_home: Path) -> None:
    result = runner.invoke(cli.app, ["model", "add", "qwen3:8b"])
    assert result.exit_code == 0
    assert fake_engine == ["qwen3:8b"]
    assert config.load().model == "qwen3:8b"


def test_model_add_rejects_full_weight_repo(fake_engine: list[str], isolated_home: Path) -> None:
    result = runner.invoke(cli.app, ["model", "add", "meta-llama/Llama-3.3-70B"])
    assert result.exit_code != 0
    assert fake_engine == []


def test_model_list_shows_ollama_output(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert cmd == ["ollama", "list"]
        return subprocess.CompletedProcess(cmd, 0, stdout="NAME  SIZE\nqwen3:8b  5GB\n", stderr="")

    monkeypatch.setattr(proc, "run_logged", fake_run)
    result = runner.invoke(cli.app, ["model", "list"])
    assert result.exit_code == 0
    assert "qwen3:8b" in result.output


def test_model_list_empty_suggests_model_add(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(proc, "run_logged", fake_run)
    result = runner.invoke(cli.app, ["model", "list"])
    assert result.exit_code == 0
    assert "ezai model add" in result.output


def test_model_list_failure_points_at_doctor(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="connection refused")

    monkeypatch.setattr(proc, "run_logged", fake_run)
    result = runner.invoke(cli.app, ["model", "list"])
    assert "ezai doctor" in result.output
    assert "No models yet" not in result.output


def test_model_rm_invokes_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(proc, "run_logged", fake_run)
    result = runner.invoke(cli.app, ["model", "rm", "qwen3:8b"])
    assert result.exit_code == 0
    assert ["ollama", "rm", "qwen3:8b"] in calls
