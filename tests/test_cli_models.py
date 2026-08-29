from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from lepika import cli, config, detect, engine, express
from lepika.errors import FriendlyError

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
    monkeypatch.setattr(engine, "pull_model", lambda url, ref, **k: pulled.append(ref.raw))
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


def test_model_add_never_installs_an_engine_someone_else_runs(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    config.save(config.Config(engine_managed=False, engine_url="http://gpu-box:11435"))
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(detect, "api_up", lambda url, **k: True)
    monkeypatch.setattr(express, "ensure_ollama", lambda *a, **k: pytest.fail("must not install"))
    monkeypatch.setattr(engine, "pull_model", lambda url, ref, **k: None)
    result = runner.invoke(cli.app, ["model", "add", "qwen3:8b"])
    assert result.exit_code == 0
    assert config.load().model == "qwen3:8b"


def test_model_list_shows_a_table_with_the_default_marked(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    config.save(config.Config(model="qwen3:8b"))
    monkeypatch.setattr(
        engine,
        "list_models",
        lambda url, **k: [("qwen3:8b", 5_000_000_000), ("gemma3:4b", 3_300_000_000)],
    )
    result = runner.invoke(cli.app, ["model", "list"])
    assert result.exit_code == 0
    assert "qwen3:8b" in result.output
    assert "(default)" in result.output
    assert "5.0 GB" in result.output


def test_model_list_empty_suggests_model_add(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine, "list_models", lambda url, **k: [])
    result = runner.invoke(cli.app, ["model", "list"])
    assert result.exit_code == 0
    assert "lepika model add" in result.output


def test_model_list_uses_the_configured_engine_and_key(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    config.save(config.Config(engine_url="http://gpu-box:11435", engine_key="k"))
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(
        engine, "list_models", lambda url, key="", **k: seen.append((url, key)) or []
    )
    runner.invoke(cli.app, ["model", "list"])
    assert seen == [("http://gpu-box:11435", "k")]


def test_model_list_failure_is_friendly(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(url: str, **k: Any) -> list[tuple[str, int]]:
        raise FriendlyError("Could not reach the engine at http://x.", "Run `lepika doctor`.")

    monkeypatch.setattr(engine, "list_models", boom)
    result = runner.invoke(cli.app, ["model", "list"])
    assert result.exit_code != 0


def test_model_rm_deletes_through_the_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    deleted: list[str] = []
    monkeypatch.setattr(engine, "delete_model", lambda url, name, **k: deleted.append(name))
    result = runner.invoke(cli.app, ["model", "rm", "qwen3:8b"])
    assert result.exit_code == 0
    assert deleted == ["qwen3:8b"]


def test_model_rm_of_the_default_clears_it_from_the_config(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """A config still naming a deleted model makes `status` and `up` lie about it."""
    config.save(config.Config(model="qwen3:8b"))
    monkeypatch.setattr(engine, "delete_model", lambda url, name, **k: None)
    result = runner.invoke(cli.app, ["model", "rm", "qwen3:8b"])
    assert result.exit_code == 0, result.output
    assert config.load().model == ""
    assert "default model" in result.output


def test_model_rm_of_another_model_leaves_the_default_alone(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    config.save(config.Config(model="qwen3:8b"))
    monkeypatch.setattr(engine, "delete_model", lambda url, name, **k: None)
    result = runner.invoke(cli.app, ["model", "rm", "llama3.2:3b"])
    assert result.exit_code == 0, result.output
    assert config.load().model == "qwen3:8b"
    assert "default model" not in result.output


def test_model_add_help_lists_all_three_model_ref_shapes() -> None:
    """Listing two of three reads as "a full-weight repo is not accepted here"."""
    result = runner.invoke(cli.app, ["model", "add", "--help"])
    assert result.exit_code == 0
    assert "qwen3:8b" in result.output
    assert "<org>/<repo>" in result.output
    # The bare `<org>/<repo>` shape is the one that was missing; only it says vLLM.
    assert "(vLLM)" in result.output


def test_model_add_in_server_mode_brings_the_engine_container_up_first(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """A container engine has to be running before anything can be pulled into it."""
    from lepika import server

    config.save(config.Config(mode="server"))
    events: list[str] = []
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(server, "start_stack", lambda info, cfg, **k: events.append("stack") or "")
    monkeypatch.setattr(express, "ensure_ollama", lambda *a, **k: pytest.fail("no native install"))
    monkeypatch.setattr(engine, "pull_model", lambda url, ref, **k: events.append("pull"))
    result = runner.invoke(cli.app, ["model", "add", "qwen3:8b"])
    assert result.exit_code == 0, result.output
    assert events == ["stack", "pull"]


def test_model_add_in_server_mode_still_only_checks_a_remote_engine(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    from lepika import server

    config.save(config.Config(mode="server", engine_managed=False, engine_url="http://gpu-box:1"))
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(detect, "api_up", lambda url, **k: True)
    monkeypatch.setattr(server, "start_stack", lambda *a, **k: pytest.fail("not ours to start"))
    monkeypatch.setattr(engine, "pull_model", lambda url, ref, **k: None)
    result = runner.invoke(cli.app, ["model", "add", "qwen3:8b"])
    assert result.exit_code == 0, result.output
