from __future__ import annotations

import shutil
import subprocess
from typing import Any

import fakes
import pytest
from fakes import Backends, FakeBackend, FakeSystem
from typer.testing import CliRunner

from lepika import cli, detect, health, ollama, openwebui, proc
from lepika.updates import UpdateStep

runner = CliRunner()
LINUX = detect.Machine("linux", "x86_64", "nvidia", 32.0)


@pytest.fixture(autouse=True)
def no_binaries_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI hands `update` the real system, whose `which` is resolved when it runs:
    faked here, a Linux host with systemd takes the same branch as every other host."""
    monkeypatch.setattr(shutil, "which", lambda name: None)


def test_update_upgrades_engine_and_webui(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    installed: list[detect.Machine] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    info = detect.Machine(
        os="linux",
        arch="x86_64",
        gpu="nvidia",
        ram_gb=32.0,
    )
    monkeypatch.setattr(proc, "run_logged", fake_run)
    monkeypatch.setattr(ollama, "upgrade_ollama", lambda i, **k: installed.append(i))
    monkeypatch.setattr(health, "api_up", lambda url, **k: False)
    monkeypatch.setattr(health, "webui_up", lambda port, **k: False)
    monkeypatch.setattr(openwebui, "stop_openwebui", lambda os_name, **k: True)
    monkeypatch.setattr(openwebui, "ensure_openwebui", lambda cfg, **k: None)

    # The CLI used to reach `proc.run_logged` through the real system; the fake system
    # hands every command it runs to the same recorder.
    machine = FakeSystem(while_running=fake_run)
    result = runner.invoke(cli.app, ["update"], obj=fakes.deps(sys=machine, machine=info))
    assert result.exit_code == 0
    # The install script must stream (it may prompt for sudo), so it goes through
    # ollama.upgrade_ollama — never through captured run_logged.
    assert installed == [info]
    assert not any("ollama.com/install.sh" in " ".join(c) for c in calls)
    # The pin travels with every `uv tool` call: an upgrade that resolved a
    # different interpreter would leave `uv tool run --python 3.11` with no
    # matching env to reuse, which is the ephemeral-build failure again.
    assert ["uv", "tool", "upgrade", "--python", openwebui.OPENWEBUI_PYTHON, "open-webui"] in calls


def test_update_restarts_openwebui_rather_than_probing_the_dying_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The upgraded build only takes effect if the old server is actually replaced."""
    info = detect.Machine(
        os="linux",
        arch="x86_64",
        gpu="nvidia",
        ram_gb=32.0,
    )
    restarts: list[tuple[int, str]] = []
    monkeypatch.setattr(
        proc, "run_logged", lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, "", "")
    )
    monkeypatch.setattr(ollama, "upgrade_ollama", lambda i, **k: None)
    monkeypatch.setattr(health, "api_up", lambda url, **k: False)
    monkeypatch.setattr(
        openwebui,
        "restart_openwebui",
        lambda cfg, os_name, **k: restarts.append((cfg.webui_port, os_name)),
    )
    monkeypatch.setattr(
        openwebui, "stop_openwebui", lambda *a, **k: pytest.fail("update must go through restart")
    )

    result = runner.invoke(cli.app, ["update"], obj=fakes.deps(machine=info))
    assert result.exit_code == 0
    assert restarts == [(3000, "linux")]


def test_update_prints_what_happened(monkeypatch: pytest.MonkeyPatch) -> None:
    steps = [
        UpdateStep("Ollama", "upgraded", "0.33.2 → 0.34.0"),
        UpdateStep("OpenWebUI", "current", "0.6.5"),
        UpdateStep("Ollama", "stale", "running 0.33.2, installed 0.34.0 — restart Ollama"),
        UpdateStep("Container images", "current"),
        UpdateStep("Ollama service", "note", "could not put ollama.service back as it was"),
    ]
    deps = fakes.deps(
        machine=LINUX,
        backends=Backends(express=FakeBackend(on_update=lambda info, cfg, **k: steps)),
    )
    result = runner.invoke(cli.app, ["update"], obj=deps)
    assert result.exit_code == 0, result.output
    assert "✓ Ollama: upgraded (0.33.2 → 0.34.0)" in result.output
    assert "✓ OpenWebUI: up to date (0.6.5)" in result.output
    assert "! Ollama: running 0.33.2, installed 0.34.0 — restart Ollama" in result.output
    assert "✓ Container images: up to date\n" in result.output
    assert "! Ollama service: could not put ollama.service back as it was" in result.output
    assert "Everything is up to date" not in result.output


def test_down_gates_the_signal_on_the_configured_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """`lepika down` must tell `stop_openwebui` which port proves the pid is ours."""
    info = detect.Machine(
        os="linux",
        arch="x86_64",
        gpu="nvidia",
        ram_gb=32.0,
    )
    seen: list[int | None] = []
    monkeypatch.setattr(
        openwebui,
        "stop_openwebui",
        lambda os_name, port=None, **k: bool(seen.append(port)),
    )
    result = runner.invoke(cli.app, ["down"], obj=fakes.deps(machine=info))
    assert result.exit_code == 0
    assert seen == [3000]
