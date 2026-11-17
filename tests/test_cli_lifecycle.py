from __future__ import annotations

from pathlib import Path

import fakes
import pytest
from fakes import FakeResponse, FakeSystem, routes
from typer.testing import CliRunner

from lepika import cli, config, detect, health, ollama, openwebui, paths

runner = CliRunner()

INFO = detect.Machine(
    os="linux",
    arch="x86_64",
    gpu="nvidia",
    ram_gb=32.0,
)


def engine_running(url: str = config.DEFAULT_ENGINE_URL) -> FakeSystem:
    """Ollama installed, and an engine already answering at `url`."""
    return FakeSystem(
        which={"ollama": "/usr/local/bin/ollama"},
        urls=routes({f"{url}/api/version": FakeResponse(b"{}")}),
    )


class Counts(dict[str, int]):
    """What `up` and `down` reached, by name, and the `Deps` they reach it through."""

    def __init__(self, deps: cli.Deps) -> None:
        super().__init__(ensure_ollama=0, ensure_openwebui=0, browser=0, stop=0)
        self.deps = deps


@pytest.fixture()
def quiet_stack(monkeypatch: pytest.MonkeyPatch) -> Counts:
    counts = Counts(fakes.deps(sys=engine_running(), machine=INFO))
    monkeypatch.setattr(
        ollama,
        "ensure_ollama",
        lambda info, **k: counts.__setitem__("ensure_ollama", counts["ensure_ollama"] + 1),
    )
    monkeypatch.setattr(
        openwebui,
        "ensure_openwebui",
        lambda cfg, **k: counts.__setitem__("ensure_openwebui", counts["ensure_openwebui"] + 1),
    )
    monkeypatch.setattr(
        openwebui,
        "stop_openwebui",
        lambda os_name, **k: counts.__setitem__("stop", counts["stop"] + 1) or True,
    )
    monkeypatch.setattr(
        cli,
        "_open_browser",
        lambda url: counts.__setitem__("browser", counts["browser"] + 1),
    )
    return counts


def test_up_starts_stack_and_opens_browser(quiet_stack: Counts) -> None:
    result = runner.invoke(cli.app, ["up"], obj=quiet_stack.deps)
    assert result.exit_code == 0
    assert quiet_stack["ensure_ollama"] == 1
    assert quiet_stack["ensure_openwebui"] == 1
    assert quiet_stack["browser"] == 1
    assert "http://localhost:3000" in result.output


def test_up_hints_at_adding_a_model_when_none_is_configured(
    quiet_stack: Counts, isolated_home: Path
) -> None:
    """A running UI with no model is a dead end unless `up` says what to do next."""
    result = runner.invoke(cli.app, ["up"], obj=quiet_stack.deps)
    assert result.exit_code == 0
    assert "lepika model add" in result.output


def test_up_stays_quiet_when_a_model_is_already_set(
    quiet_stack: Counts, isolated_home: Path
) -> None:
    config.save(config.Config(model="qwen3:8b"))
    result = runner.invoke(cli.app, ["up"], obj=quiet_stack.deps)
    assert result.exit_code == 0
    assert "No model yet" not in result.output


def test_down_stops_webui(quiet_stack: Counts) -> None:
    result = runner.invoke(cli.app, ["down"], obj=quiet_stack.deps)
    assert result.exit_code == 0
    assert quiet_stack["stop"] == 1
    assert "Stopped." in result.output


def test_down_says_so_when_nothing_was_running(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(openwebui, "stop_openwebui", lambda os_name, **k: False)
    result = runner.invoke(cli.app, ["down"], obj=fakes.deps(machine=INFO))
    assert result.exit_code == 0
    assert "Nothing was running." in result.output


def test_status_reports_services(monkeypatch: pytest.MonkeyPatch, isolated_home: Path) -> None:
    monkeypatch.setattr(health, "api_up", lambda url, **k: True)
    monkeypatch.setattr(health, "webui_up", lambda port, **k: False)
    config.save(config.Config(model="qwen3:8b"))
    result = runner.invoke(cli.app, ["status"], obj=fakes.deps())
    assert result.exit_code == 0
    assert "qwen3:8b" in result.output


def test_status_probes_the_configured_engine_url(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """`engine_url` is a real setting, not decoration: status must honour it."""
    probed: list[str] = []
    monkeypatch.setattr(health, "api_up", lambda url, **k: bool(probed.append(url)))
    monkeypatch.setattr(health, "webui_up", lambda port, **k: False)
    config.save(config.Config(engine_url="http://gpu-box.local:11434"))
    result = runner.invoke(cli.app, ["status"], obj=fakes.deps())
    assert result.exit_code == 0
    assert probed == ["http://gpu-box.local:11434"]


def test_status_shows_the_engine_url_and_says_when_it_is_remote(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """With a remote engine, "Engine: up" alone hides which machine answered."""
    monkeypatch.setattr(health, "api_up", lambda url, **k: True)
    monkeypatch.setattr(health, "webui_up", lambda port, **k: True)
    config.save(config.Config(engine_managed=False, engine_url="http://gpu-box:11435"))
    result = runner.invoke(cli.app, ["status"], obj=fakes.deps())
    assert result.exit_code == 0
    assert "http://gpu-box:11435" in result.output
    assert "remote" in result.output


def test_status_probes_a_keyed_engine_with_its_key(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """A keyed engine rejects an unauthenticated probe, so status would call it down."""
    seen: list[str] = []
    monkeypatch.setattr(health, "api_up", lambda url, key="", **k: bool(seen.append(key)) or True)
    monkeypatch.setattr(health, "webui_up", lambda port, **k: True)
    config.save(
        config.Config(engine_managed=False, engine_url="http://gpu-box:11435", engine_key="k")
    )
    runner.invoke(cli.app, ["status"], obj=fakes.deps())
    assert seen == ["k"]


def test_up_points_the_engine_check_at_the_configured_url(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    urls: list[str | None] = []
    monkeypatch.setattr(ollama, "ensure_ollama", lambda info, url=None, **k: urls.append(url))
    monkeypatch.setattr(openwebui, "ensure_openwebui", lambda cfg, **k: None)
    monkeypatch.setattr(cli, "_open_browser", lambda url: None)
    config.save(config.Config(engine_url="http://gpu-box.local:11434"))
    result = runner.invoke(
        cli.app,
        ["up"],
        obj=fakes.deps(sys=engine_running("http://gpu-box.local:11434"), machine=INFO),
    )
    assert result.exit_code == 0
    assert urls == ["http://gpu-box.local:11434"]


def test_logs_prints_tail(isolated_home: Path) -> None:
    (paths.logs_dir() / "lepika.log").write_text("line-one\nline-two\n")
    result = runner.invoke(cli.app, ["logs"], obj=fakes.deps())
    assert result.exit_code == 0
    assert "line-two" in result.output


def test_logs_says_so_when_there_is_nothing_to_show(isolated_home: Path) -> None:
    """Silence looks like a broken command; an empty log dir should say it's empty."""
    result = runner.invoke(cli.app, ["logs"], obj=fakes.deps())
    assert result.exit_code == 0
    assert "no logs yet" in result.output


def test_logs_rejects_a_nonpositive_line_count(isolated_home: Path) -> None:
    """`content[-0:]` is the whole file, so --lines 0 dumped everything."""
    (paths.logs_dir() / "lepika.log").write_text("line-one\nline-two\n")
    result = runner.invoke(cli.app, ["logs", "--lines", "0"], obj=fakes.deps())
    assert result.exit_code != 0
    assert "line-one" not in result.output


def test_status_shows_exposure_in_server_mode(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    monkeypatch.setattr(health, "api_up", lambda url, **k: True)
    monkeypatch.setattr(health, "webui_up", lambda port, **k: True)
    deps = fakes.deps()
    config.save(config.Config(mode="server", exposed=True, api_port=12345))
    out = runner.invoke(cli.app, ["status"], obj=deps).output
    assert "Exposed" in out
    assert "12345" in out
    config.save(config.Config(mode="server", exposed=False))
    out = runner.invoke(cli.app, ["status"], obj=deps).output
    assert "Exposed" in out
    assert "11435" not in out


def test_status_has_no_exposure_row_in_express_mode(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """Exposure is a Server-mode feature; a row that can only ever say no is noise."""
    monkeypatch.setattr(health, "api_up", lambda url, **k: True)
    monkeypatch.setattr(health, "webui_up", lambda port, **k: True)
    config.save(config.Config(mode="express"))
    assert "Exposed" not in runner.invoke(cli.app, ["status"], obj=fakes.deps()).output


def test_up_says_how_to_set_the_context_length_on_an_adopted_engine(
    quiet_stack: Counts, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ollama, "context_note", lambda info, cfg, **k: "set it in Settings")
    result = runner.invoke(cli.app, ["up"], obj=quiet_stack.deps)
    assert result.exit_code == 0
    assert "set it in Settings" in result.output


def test_up_stays_quiet_about_context_length_when_there_is_nothing_to_say(
    quiet_stack: Counts, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ollama, "context_note", lambda info, cfg, **k: None)
    result = runner.invoke(cli.app, ["up"], obj=quiet_stack.deps)
    assert result.exit_code == 0
    assert "context" not in result.output.lower()


def test_up_notes_the_context_length_of_the_engine_the_config_names(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """The engine `lepika up` adopts is the one at `engine_url`; whether the note applies
    is asked there, not of the default port, and before the stack starts."""
    order: list[str] = []
    engine = FakeSystem(
        which={"ollama": "/usr/local/bin/ollama"},
        urls=routes(
            {
                "http://127.0.0.1:11500/api/version": FakeResponse(b"{}"),
                "http://127.0.0.1:11434/api/version": OSError("connection refused"),
            }
        ),
    )
    monkeypatch.setattr(
        ollama,
        "ensure_ollama",
        lambda info, **k: order.append(f"start after {[r.full_url for r in engine.requested]}"),
    )
    monkeypatch.setattr(openwebui, "ensure_openwebui", lambda cfg, **k: None)
    monkeypatch.setattr(cli, "_open_browser", lambda url: None)
    config.save(config.Config(engine_url="http://127.0.0.1:11500"))
    # The machine `engine` describes: Linux on x86_64, no GPU, 16 GB.
    found = detect.Machine("linux", "x86_64", "none", 16.0)
    result = runner.invoke(cli.app, ["up"], obj=fakes.deps(sys=engine, machine=found))
    assert result.exit_code == 0, result.output
    assert "Ollama was already running" in result.output
    assert [r.full_url for r in engine.requested] == ["http://127.0.0.1:11500/api/version"]
    assert len(order) == 1 and "11500" in order[0]
