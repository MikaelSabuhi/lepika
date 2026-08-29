from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lepika import cli, config, detect, paths, server
from lepika.errors import FriendlyError

runner = CliRunner()
DOCKER = detect.SystemInfo("linux", "x86_64", "nvidia", 64.0, True, False, False)


@pytest.fixture()
def exposed_box(monkeypatch: pytest.MonkeyPatch, isolated_home: Path) -> list[str]:
    starts: list[str] = []
    config.save(config.Config(mode="server", model="qwen3:8b"))
    monkeypatch.setattr(detect, "detect", lambda **k: DOCKER)
    monkeypatch.setattr(server, "start_stack", lambda info, cfg, **k: starts.append("start") or "u")
    monkeypatch.setattr(server, "lan_ip", lambda **k: "192.168.1.20")
    return starts


def test_expose_turns_on_generates_a_key_and_prints_the_connect_line(
    exposed_box: list[str],
) -> None:
    result = runner.invoke(cli.app, ["expose"])
    assert result.exit_code == 0, result.output
    cfg = config.load()
    assert cfg.exposed is True
    key = server.read_env(paths.stack_dir() / ".env")["LEPIKA_API_KEY"]
    assert key and key in result.output
    assert f"lepika connect http://192.168.1.20:11435 --key {key}" in result.output
    assert "http://192.168.1.20:3000" in result.output
    assert exposed_box == ["start"]


def test_expose_show_reprints_without_restarting(exposed_box: list[str]) -> None:
    runner.invoke(cli.app, ["expose"])
    result = runner.invoke(cli.app, ["expose", "--show"])
    assert result.exit_code == 0
    assert "--key" in result.output
    assert exposed_box == ["start"]


def test_expose_rotate_changes_the_key_and_restarts(exposed_box: list[str]) -> None:
    runner.invoke(cli.app, ["expose"])
    before = server.read_env(paths.stack_dir() / ".env")["LEPIKA_API_KEY"]
    result = runner.invoke(cli.app, ["expose", "--rotate"])
    assert result.exit_code == 0
    assert server.read_env(paths.stack_dir() / ".env")["LEPIKA_API_KEY"] != before
    assert exposed_box == ["start", "start"]


def test_expose_off_binds_back_to_localhost(exposed_box: list[str]) -> None:
    runner.invoke(cli.app, ["expose"])
    result = runner.invoke(cli.app, ["expose", "--off"])
    assert result.exit_code == 0
    assert config.load().exposed is False
    assert exposed_box == ["start", "start"]


def test_expose_in_express_mode_is_friendly(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    monkeypatch.setattr(detect, "detect", lambda **k: DOCKER)
    result = runner.invoke(cli.app, ["expose"])
    assert result.exit_code != 0
    assert "Server mode" in str(result.exception)


def test_the_key_never_reaches_the_log(exposed_box: list[str]) -> None:
    runner.invoke(cli.app, ["expose"])
    key = server.read_env(paths.stack_dir() / ".env")["LEPIKA_API_KEY"]
    assert key not in (paths.logs_dir() / "lepika.log").read_text()


def test_show_before_exposing_says_it_is_off(exposed_box: list[str]) -> None:
    result = runner.invoke(cli.app, ["expose", "--show"])
    assert result.exit_code == 0, result.output
    assert "Not exposed yet" in result.output
    assert "--key" in result.output
    assert config.load().exposed is False
    assert exposed_box == []


def test_show_with_rotate_restarts_so_caddy_gets_the_new_key(exposed_box: list[str]) -> None:
    runner.invoke(cli.app, ["expose"])
    before = server.read_env(paths.stack_dir() / ".env")["LEPIKA_API_KEY"]
    result = runner.invoke(cli.app, ["expose", "--show", "--rotate"])
    assert result.exit_code == 0, result.output
    assert server.read_env(paths.stack_dir() / ".env")["LEPIKA_API_KEY"] != before
    assert exposed_box == ["start", "start"]


def test_the_key_is_in_the_env_before_the_stack_starts(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    # `server.start_stack` refuses to start the `expose` profile with an empty key,
    # so `expose` must write the key first — this records what it would have seen.
    seen: list[str] = []
    config.save(config.Config(mode="server"))
    monkeypatch.setattr(detect, "detect", lambda **k: DOCKER)
    monkeypatch.setattr(server, "lan_ip", lambda **k: "192.168.1.20")

    def record_start(info: detect.SystemInfo, cfg: config.Config, **kwargs: object) -> str:
        seen.append(server.read_env(paths.stack_dir() / server.ENV_FILE).get("LEPIKA_API_KEY", ""))
        return "u"

    monkeypatch.setattr(server, "start_stack", record_start)
    result = runner.invoke(cli.app, ["expose"])
    assert result.exit_code == 0, result.output
    assert seen == [server.read_env(paths.stack_dir() / server.ENV_FILE)["LEPIKA_API_KEY"]]
    assert seen[0]


def test_expose_off_does_not_reprint_the_key(exposed_box: list[str]) -> None:
    runner.invoke(cli.app, ["expose"])
    key = server.read_env(paths.stack_dir() / ".env")["LEPIKA_API_KEY"]
    result = runner.invoke(cli.app, ["expose", "--off"])
    assert key not in result.output
    assert "localhost" in result.output


def test_rotate_tells_machines_that_already_connected_to_reconnect(
    exposed_box: list[str],
) -> None:
    """The old key stops working the moment Caddy restarts; silence looks like a break."""
    runner.invoke(cli.app, ["expose"])
    result = runner.invoke(cli.app, ["expose", "--rotate"])
    assert result.exit_code == 0, result.output
    key = server.read_env(paths.stack_dir() / ".env")["LEPIKA_API_KEY"]
    assert "connected before" in result.output
    assert key in result.output


def test_a_plain_expose_does_not_talk_about_reconnecting(exposed_box: list[str]) -> None:
    result = runner.invoke(cli.app, ["expose"])
    assert "connected before" not in result.output


def test_expose_refuses_an_engine_that_needs_its_own_key(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """Caddy forwards the caller's key upstream, where a keyed remote engine rejects it."""
    starts: list[str] = []
    config.save(
        config.Config(
            mode="server", engine_managed=False, engine_url="http://gpu-box:11435", engine_key="k"
        )
    )
    monkeypatch.setattr(server, "start_stack", lambda info, cfg, **k: starts.append("start"))
    result = runner.invoke(cli.app, ["expose"])
    assert result.exit_code != 0
    exc = result.exception
    assert isinstance(exc, FriendlyError)
    assert "gpu-box" in exc.problem
    assert "lepika connect --local" in exc.fix
    assert starts == []
    assert config.load().exposed is False


def test_expose_off_still_works_with_a_keyed_remote_engine(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    starts: list[str] = []
    config.save(
        config.Config(
            mode="server",
            exposed=True,
            engine_managed=False,
            engine_url="http://gpu-box:11435",
            engine_key="k",
        )
    )
    monkeypatch.setattr(detect, "detect", lambda **k: DOCKER)
    monkeypatch.setattr(server, "start_stack", lambda info, cfg, **k: starts.append("start"))
    result = runner.invoke(cli.app, ["expose", "--off"])
    assert result.exit_code == 0, result.output
    assert config.load().exposed is False
    assert starts == ["start"]
