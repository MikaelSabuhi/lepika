from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lepika import cli, config, detect, express, log, paths
from lepika.errors import FriendlyError

runner = CliRunner()


def test_connect_stores_url_key_and_marks_engine_remote(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    monkeypatch.setattr(detect, "api_up", lambda url, **k: True)
    monkeypatch.setattr(express, "webui_up", lambda port, **k: False)
    result = runner.invoke(cli.app, ["connect", "http://gpu-box:11435", "--key", "s3cret"])
    assert result.exit_code == 0
    cfg = config.load()
    assert cfg.engine_url == "http://gpu-box:11435"
    assert cfg.engine_key == "s3cret"
    assert cfg.engine_managed is False
    assert "s3cret" not in result.output


def test_connect_strips_a_trailing_slash(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    monkeypatch.setattr(detect, "api_up", lambda url, **k: True)
    monkeypatch.setattr(express, "webui_up", lambda port, **k: False)
    runner.invoke(cli.app, ["connect", "http://gpu-box:11435/"])
    assert config.load().engine_url == "http://gpu-box:11435"


def test_connect_rejects_a_url_without_a_scheme(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """`gpu-box:11435` reads like a host:port but urllib treats it as a scheme."""
    monkeypatch.setattr(detect, "api_up", lambda url, **k: pytest.fail("must not probe"))
    result = runner.invoke(cli.app, ["connect", "gpu-box:11435"])
    assert result.exit_code != 0
    assert config.load().engine_managed is True


def test_connect_refuses_an_engine_that_does_not_answer(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    monkeypatch.setattr(detect, "api_up", lambda url, **k: False)
    result = runner.invoke(cli.app, ["connect", "http://gpu-box:11435"])
    assert result.exit_code != 0
    assert config.load().engine_managed is True


def test_connect_probes_with_the_key(monkeypatch: pytest.MonkeyPatch, isolated_home: Path) -> None:
    seen: list[str] = []
    monkeypatch.setattr(detect, "api_up", lambda url, key="", **k: bool(seen.append(key)) or True)
    monkeypatch.setattr(express, "webui_up", lambda port, **k: False)
    runner.invoke(cli.app, ["connect", "http://gpu-box:11435", "--key", "k"])
    assert seen == ["k"]


INFO = detect.SystemInfo("linux", "x86_64", "nvidia", 32.0, False, True, True)


def test_connect_restarts_a_running_webui_at_the_new_engine(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """A running OpenWebUI keeps the engine it started with: `lepika up` would be a no-op."""
    monkeypatch.setattr(detect, "api_up", lambda url, **k: True)
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(express, "webui_up", lambda port, **k: True)
    restarts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        express,
        "restart_openwebui",
        lambda cfg, os_name, **k: restarts.append((cfg.engine_url, os_name)),
    )
    result = runner.invoke(cli.app, ["connect", "http://gpu-box:11435"])
    assert result.exit_code == 0
    assert restarts == [("http://gpu-box:11435", "linux")]
    assert "restarted" in result.output


def test_connect_only_says_up_when_no_webui_is_running(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    monkeypatch.setattr(detect, "api_up", lambda url, **k: True)
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(express, "webui_up", lambda port, **k: False)
    monkeypatch.setattr(
        express, "restart_openwebui", lambda *a, **k: pytest.fail("nothing to restart")
    )
    result = runner.invoke(cli.app, ["connect", "http://gpu-box:11435"])
    assert result.exit_code == 0
    assert "lepika up" in result.output


def test_connect_local_restarts_a_running_webui_too(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    config.save(config.Config(engine_managed=False, engine_url="http://gpu-box:11435"))
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(express, "webui_up", lambda port, **k: True)
    restarts: list[str] = []
    monkeypatch.setattr(
        express, "restart_openwebui", lambda cfg, os_name, **k: restarts.append(cfg.engine_url)
    )
    result = runner.invoke(cli.app, ["connect", "--local"])
    assert result.exit_code == 0
    assert restarts == [config.DEFAULT_ENGINE_URL]
    assert "restarted" in result.output


def test_connect_local_is_logged(monkeypatch: pytest.MonkeyPatch, isolated_home: Path) -> None:
    """Changing the engine is an action, and every action leaves one line."""
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(express, "webui_up", lambda port, **k: False)
    runner.invoke(cli.app, ["connect", "--local"])
    written = (paths.logs_dir() / log.LOG_FILE).read_text()
    assert "engine.connect" in written
    assert config.DEFAULT_ENGINE_URL in written


def test_connect_local_restores_the_managed_engine(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    config.save(
        config.Config(engine_managed=False, engine_url="http://gpu-box:11435", engine_key="k")
    )
    monkeypatch.setattr(express, "webui_up", lambda port, **k: False)
    result = runner.invoke(cli.app, ["connect", "--local"])
    assert result.exit_code == 0
    cfg = config.load()
    assert cfg.engine_managed is True
    assert cfg.engine_url == config.DEFAULT_ENGINE_URL
    assert cfg.engine_key == ""


def test_connect_without_url_or_local_is_a_usage_error(isolated_home: Path) -> None:
    result = runner.invoke(cli.app, ["connect"])
    assert result.exit_code != 0


def test_start_stack_never_installs_a_remote_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    info = detect.SystemInfo("linux", "x86_64", "nvidia", 32.0, False, False, False)
    cfg = config.Config(engine_managed=False, engine_url="http://gpu-box:11435", engine_key="k")
    monkeypatch.setattr(express, "ensure_ollama", lambda *a, **k: pytest.fail("must not install"))
    monkeypatch.setattr(express, "ensure_openwebui", lambda cfg, **k: None)
    probed: list[tuple[str, str]] = []
    monkeypatch.setattr(
        detect, "api_up", lambda url, key="", **k: bool(probed.append((url, key))) or True
    )
    express.start_stack(info, cfg)
    assert probed == [("http://gpu-box:11435", "k")]


def test_start_stack_explains_an_unreachable_remote_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    info = detect.SystemInfo("linux", "x86_64", "nvidia", 32.0, False, False, False)
    cfg = config.Config(engine_managed=False, engine_url="http://gpu-box:11435")
    monkeypatch.setattr(detect, "api_up", lambda url, **k: False)
    with pytest.raises(FriendlyError) as exc:
        express.start_stack(info, cfg)
    assert "gpu-box" in exc.value.problem
    assert "lepika connect --local" in exc.value.fix


def test_connect_in_server_mode_reconciles_the_stack_instead_of_restarting_express(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """`compose up -d` is the Server-mode reconciler; the Express restart has no stack to touch."""
    from lepika import server

    config.save(config.Config(mode="server"))
    monkeypatch.setattr(detect, "api_up", lambda url, **k: True)
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    started: list[str] = []
    monkeypatch.setattr(
        server, "start_stack", lambda info, cfg, **k: started.append(cfg.engine_url) or ""
    )
    monkeypatch.setattr(
        express, "restart_openwebui", lambda *a, **k: pytest.fail("express restart in server mode")
    )
    result = runner.invoke(cli.app, ["connect", "http://gpu-box:11435"])
    assert result.exit_code == 0, result.output
    assert started == ["http://gpu-box:11435"]
