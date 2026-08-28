from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lepika import cli, config, detect, express, paths

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


def test_up_hints_at_adding_a_model_when_none_is_configured(
    quiet_stack: dict[str, int], isolated_home: Path
) -> None:
    """A running UI with no model is a dead end unless `up` says what to do next."""
    result = runner.invoke(cli.app, ["up"])
    assert result.exit_code == 0
    assert "lepika model add" in result.output


def test_up_stays_quiet_when_a_model_is_already_set(
    quiet_stack: dict[str, int], isolated_home: Path
) -> None:
    config.save(config.Config(model="qwen3:8b"))
    result = runner.invoke(cli.app, ["up"])
    assert result.exit_code == 0
    assert "No model yet" not in result.output


def test_down_stops_webui(quiet_stack: dict[str, int]) -> None:
    result = runner.invoke(cli.app, ["down"])
    assert result.exit_code == 0
    assert quiet_stack["stop"] == 1
    assert "Stopped." in result.output


def test_down_says_so_when_nothing_was_running(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(express, "stop_openwebui", lambda os_name, **k: False)
    result = runner.invoke(cli.app, ["down"])
    assert result.exit_code == 0
    assert "Nothing was running." in result.output


def test_status_reports_services(monkeypatch: pytest.MonkeyPatch, isolated_home: Path) -> None:
    monkeypatch.setattr(detect, "api_up", lambda url, **k: True)
    monkeypatch.setattr(express, "webui_up", lambda port, **k: False)
    config.save(config.Config(model="qwen3:8b"))
    result = runner.invoke(cli.app, ["status"])
    assert result.exit_code == 0
    assert "qwen3:8b" in result.output


def test_status_probes_the_configured_engine_url(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """`engine_url` is a real setting, not decoration: status must honour it."""
    probed: list[str] = []
    monkeypatch.setattr(detect, "api_up", lambda url, **k: bool(probed.append(url)))
    monkeypatch.setattr(express, "webui_up", lambda port, **k: False)
    config.save(config.Config(engine_url="http://gpu-box.local:11434"))
    result = runner.invoke(cli.app, ["status"])
    assert result.exit_code == 0
    assert probed == ["http://gpu-box.local:11434"]


def test_status_shows_the_engine_url_and_says_when_it_is_remote(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """With a remote engine, "Engine: up" alone hides which machine answered."""
    monkeypatch.setattr(detect, "api_up", lambda url, **k: True)
    monkeypatch.setattr(express, "webui_up", lambda port, **k: True)
    config.save(config.Config(engine_managed=False, engine_url="http://gpu-box:11435"))
    result = runner.invoke(cli.app, ["status"])
    assert result.exit_code == 0
    assert "http://gpu-box:11435" in result.output
    assert "remote" in result.output


def test_status_probes_a_keyed_engine_with_its_key(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """A keyed engine rejects an unauthenticated probe, so status would call it down."""
    seen: list[str] = []
    monkeypatch.setattr(detect, "api_up", lambda url, key="", **k: bool(seen.append(key)) or True)
    monkeypatch.setattr(express, "webui_up", lambda port, **k: True)
    config.save(
        config.Config(engine_managed=False, engine_url="http://gpu-box:11435", engine_key="k")
    )
    runner.invoke(cli.app, ["status"])
    assert seen == ["k"]


def test_up_points_the_engine_check_at_the_configured_url(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    urls: list[str | None] = []
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(express, "ensure_ollama", lambda info, url=None, **k: urls.append(url))
    monkeypatch.setattr(express, "ensure_openwebui", lambda cfg, **k: None)
    monkeypatch.setattr(cli, "_open_browser", lambda url: None)
    config.save(config.Config(engine_url="http://gpu-box.local:11434"))
    result = runner.invoke(cli.app, ["up"])
    assert result.exit_code == 0
    assert urls == ["http://gpu-box.local:11434"]


def test_logs_prints_tail(isolated_home: Path) -> None:
    (paths.logs_dir() / "lepika.log").write_text("line-one\nline-two\n")
    result = runner.invoke(cli.app, ["logs"])
    assert result.exit_code == 0
    assert "line-two" in result.output


def test_logs_says_so_when_there_is_nothing_to_show(isolated_home: Path) -> None:
    """Silence looks like a broken command; an empty log dir should say it's empty."""
    result = runner.invoke(cli.app, ["logs"])
    assert result.exit_code == 0
    assert "no logs yet" in result.output


def test_logs_rejects_a_nonpositive_line_count(isolated_home: Path) -> None:
    """`content[-0:]` is the whole file, so --lines 0 dumped everything."""
    (paths.logs_dir() / "lepika.log").write_text("line-one\nline-two\n")
    result = runner.invoke(cli.app, ["logs", "--lines", "0"])
    assert result.exit_code != 0
    assert "line-one" not in result.output
