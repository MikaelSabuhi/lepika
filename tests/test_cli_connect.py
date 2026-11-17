from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fakes import FakeSystem, FakeUI
from typer.testing import CliRunner

from lepika import cli, config, detect, health, log, ollama, openwebui, paths, system
from lepika.errors import FriendlyError
from lepika.express import ExpressBackend
from lepika.server import ServerBackend

runner = CliRunner()
INFO = detect.Machine("linux", "x86_64", "nvidia", 32.0)


def test_connect_stores_url_key_and_marks_engine_remote(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    monkeypatch.setattr(health, "api_up", lambda url, **k: True)
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(health, "webui_up", lambda port, **k: False)
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
    monkeypatch.setattr(health, "api_up", lambda url, **k: True)
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(health, "webui_up", lambda port, **k: False)
    runner.invoke(cli.app, ["connect", "http://gpu-box:11435/"])
    assert config.load().engine_url == "http://gpu-box:11435"


def test_connect_rejects_a_url_without_a_scheme(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """`gpu-box:11435` reads like a host:port but urllib treats it as a scheme."""
    monkeypatch.setattr(health, "api_up", lambda url, **k: pytest.fail("must not probe"))
    result = runner.invoke(cli.app, ["connect", "gpu-box:11435"])
    assert result.exit_code != 0
    assert config.load().engine_managed is True


def test_connect_refuses_an_engine_that_does_not_answer(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    monkeypatch.setattr(health, "api_up", lambda url, **k: False)
    result = runner.invoke(cli.app, ["connect", "http://gpu-box:11435"])
    assert result.exit_code != 0
    assert config.load().engine_managed is True


def test_connect_probes_with_the_key(monkeypatch: pytest.MonkeyPatch, isolated_home: Path) -> None:
    seen: list[str] = []
    monkeypatch.setattr(health, "api_up", lambda url, key="", **k: bool(seen.append(key)) or True)
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(health, "webui_up", lambda port, **k: False)
    runner.invoke(cli.app, ["connect", "http://gpu-box:11435", "--key", "k"])
    assert seen == ["k"]


def test_connect_restarts_a_running_webui_at_the_new_engine(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """A running OpenWebUI keeps the engine it started with: `lepika up` would be a no-op."""
    monkeypatch.setattr(health, "api_up", lambda url, **k: True)
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(health, "webui_up", lambda port, **k: True)
    restarts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        openwebui,
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
    monkeypatch.setattr(health, "api_up", lambda url, **k: True)
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(health, "webui_up", lambda port, **k: False)
    monkeypatch.setattr(
        openwebui, "restart_openwebui", lambda *a, **k: pytest.fail("nothing to restart")
    )
    result = runner.invoke(cli.app, ["connect", "http://gpu-box:11435"])
    assert result.exit_code == 0
    assert "lepika up" in result.output


def test_connect_local_restarts_a_running_webui_too(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    config.save(config.Config(engine_managed=False, engine_url="http://gpu-box:11435"))
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(health, "webui_up", lambda port, **k: True)
    restarts: list[str] = []
    monkeypatch.setattr(
        openwebui, "restart_openwebui", lambda cfg, os_name, **k: restarts.append(cfg.engine_url)
    )
    result = runner.invoke(cli.app, ["connect", "--local"])
    assert result.exit_code == 0
    assert restarts == [config.DEFAULT_ENGINE_URL]
    assert "restarted" in result.output


def test_connect_local_is_logged(monkeypatch: pytest.MonkeyPatch, isolated_home: Path) -> None:
    """Changing the engine is an action, and every action leaves one line."""
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(health, "webui_up", lambda port, **k: False)
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
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(health, "webui_up", lambda port, **k: False)
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
    info = detect.Machine("linux", "x86_64", "nvidia", 32.0)
    cfg = config.Config(engine_managed=False, engine_url="http://gpu-box:11435", engine_key="k")
    monkeypatch.setattr(ollama, "ensure_ollama", lambda *a, **k: pytest.fail("must not install"))
    monkeypatch.setattr(openwebui, "ensure_openwebui", lambda cfg, **k: None)
    probed: list[tuple[str, str]] = []
    monkeypatch.setattr(
        health, "api_up", lambda url, key="", **k: bool(probed.append((url, key))) or True
    )
    ExpressBackend(sys=FakeSystem(), ui=FakeUI()).start(info, cfg)
    assert probed == [("http://gpu-box:11435", "k")]


def test_start_stack_explains_an_unreachable_remote_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    info = detect.Machine("linux", "x86_64", "nvidia", 32.0)
    cfg = config.Config(engine_managed=False, engine_url="http://gpu-box:11435")
    monkeypatch.setattr(health, "api_up", lambda url, **k: False)
    with pytest.raises(FriendlyError) as exc:
        ExpressBackend(sys=FakeSystem(), ui=FakeUI()).start(info, cfg)
    assert "gpu-box" in exc.value.problem
    assert "lepika connect --local" in exc.value.fix


def test_connect_in_server_mode_reconciles_the_stack_instead_of_restarting_express(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """`compose up -d` is the Server-mode reconciler; the Express restart has no stack to touch."""
    config.save(config.Config(mode="server"))
    monkeypatch.setattr(health, "api_up", lambda url, **k: True)
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(system, "REAL", FakeSystem())
    started: list[str] = []
    monkeypatch.setattr(
        ServerBackend, "start", lambda self, info, cfg, **k: started.append(cfg.engine_url) or ""
    )
    monkeypatch.setattr(
        openwebui,
        "restart_openwebui",
        lambda *a, **k: pytest.fail("express restart in server mode"),
    )
    result = runner.invoke(cli.app, ["connect", "http://gpu-box:11435"])
    assert result.exit_code == 0, result.output
    assert started == ["http://gpu-box:11435"]


def test_connect_with_a_key_refuses_while_this_machine_is_exposed(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """The Caddy in front of us forwards our own key, so a keyed remote only ever 401s."""
    config.save(config.Config(mode="server", exposed=True))
    monkeypatch.setattr(health, "api_up", lambda url, **k: pytest.fail("must not probe"))
    result = runner.invoke(cli.app, ["connect", "http://gpu-box:11435", "--key", "k"])
    assert result.exit_code != 0
    assert isinstance(result.exception, FriendlyError)
    assert "http://gpu-box:11435" in result.exception.problem
    assert "lepika expose --off" in result.exception.fix
    assert config.load().engine_managed is True


def test_connect_without_a_key_is_fine_while_exposed(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """Only a keyed remote is the problem: the proxy relays an open engine happily."""
    config.save(config.Config(mode="server", exposed=True))
    monkeypatch.setattr(health, "api_up", lambda url, **k: True)
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(system, "REAL", FakeSystem())
    monkeypatch.setattr(ServerBackend, "start", lambda self, info, cfg, **k: "")
    result = runner.invoke(cli.app, ["connect", "http://gpu-box:11435"])
    assert result.exit_code == 0, result.output
    assert config.load().engine_url == "http://gpu-box:11435"


def test_connect_local_is_unaffected_by_exposure(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    config.save(config.Config(mode="server", exposed=True, engine_managed=False, engine_key="k"))
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(system, "REAL", FakeSystem())
    monkeypatch.setattr(ServerBackend, "start", lambda self, info, cfg, **k: "")
    result = runner.invoke(cli.app, ["connect", "--local"])
    assert result.exit_code == 0, result.output
    assert config.load().engine_managed is True


def _logged(event: str) -> bool:
    written = paths.logs_dir() / log.LOG_FILE
    return written.exists() and event in written.read_text()


def _server_start(monkeypatch: pytest.MonkeyPatch, start: Any) -> None:
    """Server mode, where `start` stands in for the `compose up` that switches the stack."""
    config.save(config.Config(mode="server"))
    monkeypatch.setattr(health, "api_up", lambda url, **k: True)
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(system, "REAL", FakeSystem())
    monkeypatch.setattr(ServerBackend, "start", start)


def _compose_fails(self: ServerBackend, info: detect.Machine, cfg: config.Config, **k: Any) -> str:
    raise FriendlyError("`docker compose up` failed.", "Run `lepika logs`.")


def test_connect_saves_the_engine_once_the_stack_has_switched_to_it(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    seen: list[tuple[str, str, bool]] = []

    def start(self: ServerBackend, info: detect.Machine, cfg: config.Config, **k: Any) -> str:
        seen.append((cfg.engine_url, config.load().engine_url, _logged("engine.connect")))
        return ""

    _server_start(monkeypatch, start)
    result = runner.invoke(cli.app, ["connect", "http://gpu-box:11435"])
    assert result.exit_code == 0, result.output
    # The stack is switched to the new engine while the file still names the old one.
    assert seen == [("http://gpu-box:11435", config.DEFAULT_ENGINE_URL, False)]
    assert config.load().engine_url == "http://gpu-box:11435"
    assert _logged("engine.connect")


def test_a_failed_switch_leaves_the_config_and_the_log_without_the_new_engine(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """A stack that never reached the new engine must not be described as using it."""
    _server_start(monkeypatch, _compose_fails)
    result = runner.invoke(cli.app, ["connect", "http://gpu-box:11435"])
    assert isinstance(result.exception, FriendlyError)
    assert config.load() == config.Config(mode="server")
    assert not _logged("engine.connect")
    assert "Connected" not in result.output


def test_connect_local_is_saved_even_when_the_switch_fails(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """Going back is a withdrawal: the next `lepika up` must still use the local engine."""
    _server_start(monkeypatch, _compose_fails)
    config.save(config.Config(mode="server", engine_managed=False, engine_url="http://box:1"))
    result = runner.invoke(cli.app, ["connect", "--local"])
    assert isinstance(result.exception, FriendlyError)
    assert config.load() == config.Config(mode="server")
    assert _logged("engine.connect")
