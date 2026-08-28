from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from ezai import cli, config, detect, express, models, wizard
from ezai.errors import FriendlyError

runner = CliRunner()

INFO = detect.SystemInfo(
    os="macos",
    arch="arm64",
    gpu="apple",
    ram_gb=16.0,
    has_docker=False,
    has_ollama=True,
    ollama_running=True,
)

CURATED = [
    models.CuratedModel(name="Small", ref="llama3.2:3b", min_ram_gb=6),
    models.CuratedModel(name="Huge", ref="llama3.3:70b", min_ram_gb=48),
]


def test_choose_model_by_number_picks_fitting_curated() -> None:
    ref = wizard.choose_model(INFO, ask=lambda *a, **k: "1", curated=CURATED)
    assert ref.raw == "llama3.2:3b"
    assert ref.kind == "ollama"


def test_choose_model_free_form() -> None:
    ref = wizard.choose_model(
        INFO, ask=lambda *a, **k: "hf.co/unsloth/gemma-3-4b-it-GGUF", curated=CURATED
    )
    assert ref.kind == "hf_gguf"


def test_choose_model_rejects_hf_repo_with_gguf_hint() -> None:
    with pytest.raises(FriendlyError) as exc:
        wizard.choose_model(
            INFO, ask=lambda *a, **k: "meta-llama/Llama-3.3-70B-Instruct", curated=CURATED
        )
    assert "GGUF" in exc.value.fix


def test_choose_model_non_decimal_digit_is_a_model_ref_not_a_crash() -> None:
    """`str.isdigit()` is true for "²", which `int()` refuses — that must not traceback."""
    ref = wizard.choose_model(INFO, ask=lambda *a, **k: "²", curated=CURATED)
    assert ref.raw == "²"
    assert ref.kind == "ollama"


def test_dry_run_writes_config_and_executes_nothing(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(models, "load_curated", lambda **k: CURATED)
    monkeypatch.setattr(wizard, "_ask", lambda *a, **k: "1")

    def never(name: str) -> Any:
        def stub(*a: Any, **k: Any) -> Any:
            raise AssertionError(f"--dry-run must not call {name}")

        return stub

    # A dry run installs nothing, downloads nothing, starts nothing, opens nothing:
    # if the early return ever moves, these fail loudly instead of running for real.
    monkeypatch.setattr(express, "ensure_ollama", never("express.ensure_ollama"))
    monkeypatch.setattr(express, "pull_model", never("express.pull_model"))
    monkeypatch.setattr(express, "ensure_openwebui", never("express.ensure_openwebui"))
    monkeypatch.setattr(cli, "_open_browser", never("cli._open_browser"))

    result = runner.invoke(cli.app, ["--dry-run"])
    assert result.exit_code == 0
    assert "would:" in result.output
    assert config.load().model == "llama3.2:3b"
