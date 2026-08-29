from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from lepika import cli, config, detect, engine, express, models, wizard
from lepika.errors import FriendlyError

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
    ref = wizard.choose_model(INFO, config.Config(), ask=lambda *a, **k: "1", curated=CURATED)
    assert ref.raw == "llama3.2:3b"
    assert ref.kind == "ollama"


def test_choose_model_free_form() -> None:
    ref = wizard.choose_model(
        INFO,
        config.Config(),
        ask=lambda *a, **k: "hf.co/unsloth/gemma-3-4b-it-GGUF",
        curated=CURATED,
    )
    assert ref.kind == "hf_gguf"


def test_choose_model_rejects_hf_repo_with_gguf_hint() -> None:
    with pytest.raises(FriendlyError) as exc:
        wizard.choose_model(
            INFO,
            config.Config(),
            ask=lambda *a, **k: "meta-llama/Llama-3.3-70B-Instruct",
            curated=CURATED,
        )
    assert "GGUF" in exc.value.fix


def test_choose_model_non_decimal_digit_is_a_model_ref_not_a_crash() -> None:
    """`str.isdigit()` is true for "²", which `int()` refuses — that must not traceback."""
    ref = wizard.choose_model(INFO, config.Config(), ask=lambda *a, **k: "²", curated=CURATED)
    assert ref.raw == "²"
    assert ref.kind == "ollama"


THREE = [
    models.CuratedModel(name="A", ref="qwen3:0.6b", min_ram_gb=2),
    models.CuratedModel(name="B", ref="llama3.2:3b", min_ram_gb=6),
    models.CuratedModel(name="C", ref="gemma3:4b", min_ram_gb=6),
]

TINY = detect.SystemInfo(
    os="macos",
    arch="arm64",
    gpu="apple",
    ram_gb=1.0,
    has_docker=False,
    has_ollama=True,
    ollama_running=True,
)


def test_choose_model_explains_an_empty_curated_list(capsys: pytest.CaptureFixture[str]) -> None:
    """An empty table is a dead end; say why it's empty and what still works."""
    ref = wizard.choose_model(
        TINY, config.Config(), ask=lambda *a, **k: "qwen3:0.6b", curated=CURATED
    )
    assert ref.raw == "qwen3:0.6b"
    out = capsys.readouterr().out
    assert "Nothing in the curated list" in out
    assert "qwen3:0.6b" in out


def test_choose_model_reprompts_once_on_an_out_of_range_number(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A 99 against 3 rows is a mistyped pick, not a model called 99."""
    answers = iter(["99", "3"])
    ref = wizard.choose_model(
        INFO, config.Config(), ask=lambda *a, **k: next(answers), curated=THREE
    )
    assert ref.raw == "gemma3:4b"
    assert "only 3" in capsys.readouterr().out


def test_choose_model_second_bad_number_falls_through_as_a_model_ref() -> None:
    """The re-prompt is bounded: one explanation, then the flow moves on."""
    answers = iter(["99", "77"])
    ref = wizard.choose_model(
        INFO, config.Config(), ask=lambda *a, **k: next(answers), curated=THREE
    )
    assert ref.raw == "77"


def test_run_wizard_orders_engine_then_pull_then_ui_then_browser(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """The real path's ordering is the whole product: pin it, faking every callee."""
    events: list[str] = []
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(models, "load_curated", lambda **k: CURATED)
    monkeypatch.setattr(wizard, "_ask", lambda *a, **k: "1")
    monkeypatch.setattr(express, "ensure_ollama", lambda info, **k: events.append("engine"))
    monkeypatch.setattr(engine, "pull_model", lambda url, ref, **k: events.append("pull"))
    monkeypatch.setattr(express, "ensure_openwebui", lambda cfg, **k: events.append("ui"))
    monkeypatch.setattr(cli, "_open_browser", lambda url: events.append("browser"))

    result = runner.invoke(cli.app, [])
    assert result.exit_code == 0
    assert events == ["engine", "pull", "ui", "browser"]
    assert config.load().model == "llama3.2:3b"
    assert "http://localhost:3000" in result.output


def test_wizard_keeps_the_old_model_when_the_pull_fails(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """Saving before the pull recorded a model the machine does not have."""
    config.save(config.Config(model="previous:model"))
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(models, "load_curated", lambda **k: CURATED)
    monkeypatch.setattr(wizard, "_ask", lambda *a, **k: "1")
    monkeypatch.setattr(express, "ensure_ollama", lambda info, **k: None)

    def boom(url: str, ref: Any, **k: Any) -> None:
        raise FriendlyError("Failed to pull model.", "Check the name.")

    monkeypatch.setattr(engine, "pull_model", boom)
    monkeypatch.setattr(
        express, "ensure_openwebui", lambda cfg, **k: pytest.fail("no UI after a failed pull")
    )

    with pytest.raises(FriendlyError):
        wizard.run_wizard()
    assert config.load().model == "previous:model"


def test_dry_run_writes_nothing_and_executes_nothing(
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
    monkeypatch.setattr(engine, "pull_model", never("engine.pull_model"))
    monkeypatch.setattr(express, "ensure_openwebui", never("express.ensure_openwebui"))
    monkeypatch.setattr(cli, "_open_browser", never("cli._open_browser"))

    result = runner.invoke(cli.app, ["--dry-run"])
    assert result.exit_code == 0
    assert "would: pull model llama3.2:3b" in result.output
    # "would:" means would: a dry run leaves no config behind either.
    assert not config.config_path().exists()
    assert not (isolated_home / "stack").exists()


def test_choose_model_prompt_lists_all_three_model_ref_shapes() -> None:
    """Two of three shapes reads as "vLLM refs are not accepted here"."""
    prompts: list[str] = []
    wizard.choose_model(
        INFO,
        config.Config(),
        ask=lambda prompt, **k: bool(prompts.append(prompt)) or "1",
        curated=CURATED,
    )
    assert "qwen3:8b" in prompts[0]
    assert "hf.co/<org>/<repo>-GGUF" in prompts[0]
    assert "· <org>/<repo>" in prompts[0]
