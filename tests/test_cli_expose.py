from __future__ import annotations

from pathlib import Path

import fakes
import pytest
from fakes import Backends, FakeBackend
from typer.testing import CliRunner

from lepika import cli, config, detect, log, paths, server, stack_env
from lepika.errors import FriendlyError

runner = CliRunner()
DOCKER = detect.Machine("linux", "x86_64", "nvidia", 64.0)


class Starts(list[str]):
    """Each start of the stack, and the `Deps` whose backend records them."""

    deps: cli.Deps


@pytest.fixture()
def exposed_box(monkeypatch: pytest.MonkeyPatch, isolated_home: Path) -> Starts:
    starts = Starts()
    config.save(config.Config(mode="server", model="qwen3:8b"))
    starts.deps = fakes.deps(
        machine=DOCKER,
        backends=Backends(
            server=FakeBackend(on_start=lambda info, cfg, **k: starts.append("start") or "u")
        ),
    )
    monkeypatch.setattr(server, "lan_ips", lambda **k: ["192.168.1.20"])
    return starts


def test_expose_turns_on_generates_a_key_and_prints_the_connect_line(
    exposed_box: Starts,
) -> None:
    result = runner.invoke(cli.app, ["expose"], obj=exposed_box.deps)
    assert result.exit_code == 0, result.output
    cfg = config.load()
    assert cfg.exposed is True
    key = stack_env.read_env(paths.stack_dir() / ".env")["LEPIKA_API_KEY"]
    assert key and key in result.output
    assert f"lepika connect http://192.168.1.20:11435 --key {key}" in result.output
    assert "http://192.168.1.20:3000" in result.output
    assert exposed_box == ["start"]


def test_expose_show_reprints_without_restarting(exposed_box: Starts) -> None:
    runner.invoke(cli.app, ["expose"], obj=exposed_box.deps)
    result = runner.invoke(cli.app, ["expose", "--show"], obj=exposed_box.deps)
    assert result.exit_code == 0
    assert "--key" in result.output
    assert exposed_box == ["start"]


def test_expose_rotate_changes_the_key_and_restarts(exposed_box: Starts) -> None:
    runner.invoke(cli.app, ["expose"], obj=exposed_box.deps)
    before = stack_env.read_env(paths.stack_dir() / ".env")["LEPIKA_API_KEY"]
    result = runner.invoke(cli.app, ["expose", "--rotate"], obj=exposed_box.deps)
    assert result.exit_code == 0
    assert stack_env.read_env(paths.stack_dir() / ".env")["LEPIKA_API_KEY"] != before
    assert exposed_box == ["start", "start"]


def test_expose_off_binds_back_to_localhost(exposed_box: Starts) -> None:
    runner.invoke(cli.app, ["expose"], obj=exposed_box.deps)
    result = runner.invoke(cli.app, ["expose", "--off"], obj=exposed_box.deps)
    assert result.exit_code == 0
    assert config.load().exposed is False
    assert exposed_box == ["start", "start"]


def test_expose_in_express_mode_is_friendly(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    result = runner.invoke(cli.app, ["expose"], obj=fakes.deps(machine=DOCKER))
    assert result.exit_code != 0
    assert "Server mode" in str(result.exception)


def test_the_key_never_reaches_the_log(exposed_box: Starts) -> None:
    runner.invoke(cli.app, ["expose"], obj=exposed_box.deps)
    key = stack_env.read_env(paths.stack_dir() / ".env")["LEPIKA_API_KEY"]
    assert key not in (paths.logs_dir() / "lepika.log").read_text()


def test_show_before_exposing_says_it_is_off(exposed_box: Starts) -> None:
    result = runner.invoke(cli.app, ["expose", "--show"], obj=exposed_box.deps)
    assert result.exit_code == 0, result.output
    assert "Not exposed yet" in result.output
    assert "--key" in result.output
    assert config.load().exposed is False
    assert exposed_box == []


def test_show_with_rotate_restarts_so_caddy_gets_the_new_key(exposed_box: Starts) -> None:
    runner.invoke(cli.app, ["expose"], obj=exposed_box.deps)
    before = stack_env.read_env(paths.stack_dir() / ".env")["LEPIKA_API_KEY"]
    result = runner.invoke(cli.app, ["expose", "--show", "--rotate"], obj=exposed_box.deps)
    assert result.exit_code == 0, result.output
    assert stack_env.read_env(paths.stack_dir() / ".env")["LEPIKA_API_KEY"] != before
    assert exposed_box == ["start", "start"]


def test_the_key_is_in_the_env_before_the_stack_starts(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    # The Server start refuses to start the `expose` profile with an empty key,
    # so `expose` must write the key first — this records what it would have seen.
    seen: list[str] = []
    config.save(config.Config(mode="server"))
    monkeypatch.setattr(server, "lan_ips", lambda **k: ["192.168.1.20"])

    def record_start(info: detect.Machine, cfg: config.Config, **kwargs: object) -> str:
        seen.append(
            stack_env.read_env(paths.stack_dir() / stack_env.ENV_FILE).get("LEPIKA_API_KEY", "")
        )
        return "u"

    deps = fakes.deps(machine=DOCKER, backends=Backends(server=FakeBackend(on_start=record_start)))
    result = runner.invoke(cli.app, ["expose"], obj=deps)
    assert result.exit_code == 0, result.output
    assert seen == [stack_env.read_env(paths.stack_dir() / stack_env.ENV_FILE)["LEPIKA_API_KEY"]]
    assert seen[0]


def test_expose_off_does_not_reprint_the_key(exposed_box: Starts) -> None:
    runner.invoke(cli.app, ["expose"], obj=exposed_box.deps)
    key = stack_env.read_env(paths.stack_dir() / ".env")["LEPIKA_API_KEY"]
    result = runner.invoke(cli.app, ["expose", "--off"], obj=exposed_box.deps)
    assert key not in result.output
    assert "localhost" in result.output


def test_rotate_tells_machines_that_already_connected_to_reconnect(
    exposed_box: Starts,
) -> None:
    """The old key stops working the moment Caddy restarts; silence looks like a break."""
    runner.invoke(cli.app, ["expose"], obj=exposed_box.deps)
    result = runner.invoke(cli.app, ["expose", "--rotate"], obj=exposed_box.deps)
    assert result.exit_code == 0, result.output
    key = stack_env.read_env(paths.stack_dir() / ".env")["LEPIKA_API_KEY"]
    assert "connected before" in result.output
    assert key in result.output


def test_a_plain_expose_does_not_talk_about_reconnecting(exposed_box: Starts) -> None:
    result = runner.invoke(cli.app, ["expose"], obj=exposed_box.deps)
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
    deps = fakes.deps(
        backends=Backends(
            server=FakeBackend(on_start=lambda info, cfg, **k: starts.append("start"))
        )
    )
    result = runner.invoke(cli.app, ["expose"], obj=deps)
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
    deps = fakes.deps(
        machine=DOCKER,
        backends=Backends(
            server=FakeBackend(on_start=lambda info, cfg, **k: starts.append("start"))
        ),
    )
    result = runner.invoke(cli.app, ["expose", "--off"], obj=deps)
    assert result.exit_code == 0, result.output
    assert config.load().exposed is False
    assert starts == ["start"]


def test_expose_lists_the_other_addresses_this_box_answers_on(
    exposed_box: Starts, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One NIC is the common case, but a box on Wi-Fi and Ethernet answers on both."""
    monkeypatch.setattr(server, "lan_ips", lambda **k: ["192.168.1.20", "10.0.0.5"])
    result = runner.invoke(cli.app, ["expose"], obj=exposed_box.deps)
    assert result.exit_code == 0, result.output
    assert "http://192.168.1.20:3000" in result.output
    assert "Also reachable on: http://10.0.0.5:3000" in result.output


def test_expose_with_one_address_prints_no_extra_line(exposed_box: Starts) -> None:
    result = runner.invoke(cli.app, ["expose"], obj=exposed_box.deps)
    assert result.exit_code == 0, result.output
    assert "Also reachable on" not in result.output


def test_expose_says_so_when_no_address_resolves(
    exposed_box: Starts, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline or locked down: the placeholder is words, and no probe is dialled twice."""
    monkeypatch.setattr(server, "lan_ips", lambda **k: [])
    monkeypatch.setattr(server, "lan_ip", lambda **k: pytest.fail("the route was already probed"))
    result = runner.invoke(cli.app, ["expose"], obj=exposed_box.deps)
    assert result.exit_code == 0, result.output
    assert server.NO_ROUTE in result.output
    assert "Also reachable on" not in result.output


def _logged(event: str) -> bool:
    written = paths.logs_dir() / log.LOG_FILE
    return written.exists() and event in written.read_text()


@pytest.mark.parametrize("args", [["expose"], ["expose", "--rotate"]])
def test_exposed_is_saved_once_the_stack_started_with_the_proxy(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path, args: list[str]
) -> None:
    seen: list[tuple[bool, bool, bool]] = []
    config.save(config.Config(mode="server"))
    monkeypatch.setattr(server, "lan_ips", lambda **k: ["192.168.1.20"])

    def start(info: detect.Machine, cfg: config.Config, **k: object) -> str:
        seen.append((cfg.exposed, config.load().exposed, _logged("expose.on")))
        return "u"

    deps = fakes.deps(machine=DOCKER, backends=Backends(server=FakeBackend(on_start=start)))
    result = runner.invoke(cli.app, args, obj=deps)
    assert result.exit_code == 0, result.output
    # Started with the proxy while the file still said it was off.
    assert seen == [(True, False, False)]
    assert config.load().exposed is True
    assert _logged("expose.on")


@pytest.mark.parametrize("args", [["expose"], ["expose", "--rotate"]])
def test_a_failed_expose_leaves_the_machine_saved_as_not_exposed(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path, args: list[str]
) -> None:
    """A proxy that never started must not be what the next `lepika up` starts."""
    config.save(config.Config(mode="server"))

    def compose_fails(info: detect.Machine, cfg: config.Config, **k: object) -> str:
        raise FriendlyError("`docker compose up` failed.", "Run `lepika logs`.")

    deps = fakes.deps(machine=DOCKER, backends=Backends(server=FakeBackend(on_start=compose_fails)))
    result = runner.invoke(cli.app, args, obj=deps)
    assert isinstance(result.exception, FriendlyError)
    assert config.load() == config.Config(mode="server")
    assert not _logged("expose.on")
    assert "Exposed" not in result.output


def test_expose_off_is_saved_even_when_the_restart_fails(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """Closing the port is a withdrawal: a failed restart must not reopen it next `up`."""
    config.save(config.Config(mode="server", exposed=True))
    seen: list[bool] = []

    def compose_fails(info: detect.Machine, cfg: config.Config, **k: object) -> str:
        seen.append(config.load().exposed)
        raise FriendlyError("`docker compose up` failed.", "Run `lepika logs`.")

    deps = fakes.deps(machine=DOCKER, backends=Backends(server=FakeBackend(on_start=compose_fails)))
    result = runner.invoke(cli.app, ["expose", "--off"], obj=deps)
    assert isinstance(result.exception, FriendlyError)
    assert seen == [False]
    assert config.load().exposed is False
    assert not _logged("expose.off")
