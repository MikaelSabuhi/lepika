"""vLLM profile: full-weight HF repos on Server mode, Linux + NVIDIA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fakes import Caller, Runner
from typer.testing import CliRunner

from lepika import cli, config, detect, doctor, engine, express, models, paths, server, wizard
from lepika.errors import FriendlyError

runner = CliRunner()
LINUX_NVIDIA = detect.SystemInfo("linux", "x86_64", "nvidia", 64.0, True, False, False)
MAC = detect.SystemInfo("macos", "arm64", "apple", 32.0, True, False, False)
REPO = "meta-llama/Llama-3.1-8B-Instruct"


def test_uses_vllm_only_for_full_weight_repos() -> None:
    assert models.uses_vllm(REPO) is True
    assert models.uses_vllm("qwen3:8b") is False
    assert models.uses_vllm("hf.co/unsloth/gemma-3-4b-it-GGUF") is False
    assert models.uses_vllm("") is False


def test_vllm_allowed_needs_server_linux_nvidia_and_our_own_engine() -> None:
    assert server.vllm_allowed(config.Config(mode="server"), LINUX_NVIDIA) is True
    assert server.vllm_allowed(config.Config(mode="express"), LINUX_NVIDIA) is False
    assert server.vllm_allowed(config.Config(mode="server"), MAC) is False
    # `lepika connect` pointed us at someone else's Ollama: we start no vLLM at all,
    # so accepting a repo here would save a model the stack never serves.
    assert (
        server.vllm_allowed(config.Config(mode="server", engine_managed=False), LINUX_NVIDIA)
        is False
    )


def test_vllm_active_is_the_one_predicate_for_a_running_vllm() -> None:
    assert server.vllm_active(config.Config(mode="server", model=REPO)) is True
    assert server.vllm_active(config.Config(mode="server", model="qwen3:8b")) is False
    assert (
        server.vllm_active(config.Config(mode="server", model=REPO, engine_managed=False)) is False
    )
    # There is no vLLM container in Express mode, so a leftover repo ref (a switch
    # that failed halfway) must not make status/doctor/model list report one.
    assert server.vllm_active(config.Config(mode="express", model=REPO)) is False


CPU_LINUX = detect.SystemInfo("linux", "x86_64", "none", 64.0, True, False, False)


def test_validate_accepts_hf_repo_where_vllm_or_an_import_runs_and_rejects_elsewhere() -> None:
    ref = models.parse_model_ref(REPO)
    assert wizard._validate(ref, config.Config(mode="server"), LINUX_NVIDIA) == ref
    # Express on Apple Silicon: Ollama imports the weights (rule 10).
    assert wizard._validate(ref, config.Config(mode="express"), MAC) == ref
    # Express on NVIDIA joins it with the MLX bundle installer, not before.
    for refused in (LINUX_NVIDIA, CPU_LINUX):
        with pytest.raises(FriendlyError) as exc:
            wizard._validate(ref, config.Config(mode="express"), refused)
        assert "GGUF" in exc.value.fix
        assert "Apple Silicon" in exc.value.problem
        # The MLX bundle PR restores this; promising it now would be a dead end.
        assert "NVIDIA GPU on Linux/Windows" not in exc.value.problem
    with pytest.raises(FriendlyError):
        wizard._validate(ref, config.Config(mode="server"), MAC)
    with pytest.raises(FriendlyError) as remote:
        wizard._validate(ref, config.Config(mode="server", engine_managed=False), LINUX_NVIDIA)
    assert "engine's machine" in remote.value.problem


def test_profiles_and_env_switch_to_vllm() -> None:
    cfg = config.Config(mode="server", model=REPO)
    assert server.profiles(cfg) == ["vllm"]
    values = server.env_values(cfg, LINUX_NVIDIA, existing={})
    assert values["VLLM_MODEL"] == REPO
    assert values["ENABLE_OPENAI_API"] == "true"
    assert values["OPENAI_API_BASE_URL"] == "http://vllm:8000/v1"
    assert values["OPENAI_API_KEY"] == "none"
    assert values["LEPIKA_UPSTREAM"] == "vllm:8000"
    # The ollama service is stopped, so the UI offers vLLM's model and nothing else.
    assert values["OLLAMA_BASE_URL"] == "http://ollama:11434"


def test_vllm_up_probes_health() -> None:
    seen: list[Any] = []

    def opener(req: Any, timeout: float = 0) -> Any:
        seen.append(req.full_url)
        return object()

    assert engine.vllm_up("http://127.0.0.1:8000", urlopen=opener) is True
    assert seen == ["http://127.0.0.1:8000/health"]


def test_vllm_up_is_false_when_nothing_answers() -> None:
    def opener(req: Any, timeout: float = 0) -> Any:
        raise OSError("connection refused")

    assert engine.vllm_up("http://127.0.0.1:8000", urlopen=opener) is False


def test_start_stack_waits_for_vllm_not_ollama(isolated_home: Path) -> None:
    cfg = config.Config(mode="server", model=REPO)
    probes: list[str] = []
    server.start_stack(
        LINUX_NVIDIA,
        cfg,
        run=Runner({"docker info": '{"nvidia": {}}'}),
        call=Caller(),
        api_up=lambda url, **k: bool(probes.append("ollama")) or True,
        vllm_up=lambda url, **k: bool(probes.append("vllm")) or True,
        up=lambda port, **k: True,
        sleep=lambda s: None,
    )
    assert probes == ["vllm"]


def test_start_stack_vllm_without_toolkit_is_friendly(isolated_home: Path) -> None:
    with pytest.raises(FriendlyError) as exc:
        server.start_stack(
            LINUX_NVIDIA,
            config.Config(mode="server", model=REPO),
            run=Runner({"docker info": "{}"}),
            call=Caller(),
        )
    assert "Container Toolkit" in exc.value.fix


def test_hf_token_prompt_writes_once_and_skips_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    env = tmp_path / ".env"
    asked: list[str] = []
    server.hf_token_prompt(env, ask=lambda *a, **k: asked.append("?") or "hf_abc")
    assert server.read_env(env)["HF_TOKEN"] == "hf_abc"
    server.hf_token_prompt(env, ask=lambda *a, **k: pytest.fail("asked twice"))
    assert asked == ["?"]


def test_hf_token_prompt_skips_when_the_shell_has_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_from_shell")
    server.hf_token_prompt(tmp_path / ".env", ask=lambda *a, **k: pytest.fail("asked anyway"))


def test_hf_token_prompt_accepts_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    env = tmp_path / ".env"
    server.hf_token_prompt(env, ask=lambda *a, **k: "")
    assert server.read_env(env).get("HF_TOKEN", "") == ""


def test_model_add_hf_repo_in_server_mode_saves_without_pulling(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """vLLM downloads on start, so there is nothing to pull — the config change is the action."""
    config.save(config.Config(mode="server"))
    monkeypatch.setattr(detect, "detect", lambda **k: LINUX_NVIDIA)
    monkeypatch.setattr(engine, "pull_model", lambda *a, **k: pytest.fail("no pull for vLLM"))
    monkeypatch.setattr(server, "hf_token_prompt", lambda env, **k: None)
    monkeypatch.setattr(server, "start_stack", lambda info, cfg, **k: "u")
    monkeypatch.setattr(cli, "_open_browser", lambda url: None)
    result = runner.invoke(cli.app, ["model", "add", REPO])
    assert result.exit_code == 0, result.output
    assert config.load().model == REPO
    assert "lepika up" in result.output or "starting" in result.output.lower()


def test_wizard_saves_the_vllm_model_before_starting_the_stack(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """compose interpolates VLLM_MODEL from the saved config, so it must be saved first."""
    config.save(config.Config(mode="server"))
    seen: list[str] = []
    monkeypatch.setattr(detect, "detect", lambda **k: LINUX_NVIDIA)
    monkeypatch.setattr(models, "load_curated", lambda **k: [])
    monkeypatch.setattr(wizard, "_ask", lambda *a, **k: REPO)
    monkeypatch.setattr(server, "hf_token_prompt", lambda env, **k: seen.append("token"))
    monkeypatch.setattr(engine, "pull_model", lambda *a, **k: pytest.fail("no pull for vLLM"))
    monkeypatch.setattr(cli, "_open_browser", lambda url: None)

    def fake_start(info: Any, cfg: Any, after_engine: Any = None, **k: Any) -> str:
        seen.append(f"start:{config.load().model}")
        if after_engine is not None:
            after_engine()
        return "http://localhost:3000"

    monkeypatch.setattr(server, "start_stack", fake_start)
    wizard.run_wizard(mode="server")
    assert seen == ["token", f"start:{REPO}"]
    assert config.load().model == REPO


def test_dry_run_says_it_would_start_vllm(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    config.save(config.Config(mode="server"))
    monkeypatch.setattr(detect, "detect", lambda **k: LINUX_NVIDIA)
    monkeypatch.setattr(models, "load_curated", lambda **k: [])
    monkeypatch.setattr(wizard, "_ask", lambda *a, **k: REPO)
    result = runner.invoke(cli.app, ["--dry-run", "--mode", "server"])
    assert result.exit_code == 0, result.output
    assert f"would: start vLLM with {REPO}" in result.output
    assert "would: pull model" not in result.output


def test_status_labels_and_probes_vllm(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    config.save(config.Config(mode="server", model=REPO))
    monkeypatch.setattr(detect, "api_up", lambda url, **k: pytest.fail("Ollama is not the engine"))
    monkeypatch.setattr(engine, "vllm_up", lambda url, **k: True)
    monkeypatch.setattr(express, "webui_up", lambda port, **k: True)
    result = runner.invoke(cli.app, ["status"])
    assert result.exit_code == 0, result.output
    assert "Engine (vLLM)" in result.output
    # The engine URL row must name what was probed, not an Ollama port nothing binds.
    assert "8000" in result.output


def test_doctor_checks_vllm_when_the_model_needs_it(isolated_home: Path) -> None:
    config.save(config.Config(mode="server", model=REPO))
    results = doctor.run_checks(
        LINUX_NVIDIA,
        which=lambda n: f"/usr/bin/{n}",
        api_up=lambda url, **k: pytest.fail("Ollama is not the engine"),
        vllm_up=lambda url, **k: False,
        webui_up=lambda port, **k: True,
        run=Runner({"docker info": '{"nvidia": {}}'}),
    )
    failed = {r.name: r for r in results if not r.ok}
    assert "Engine (vLLM) responding" in failed


def test_model_list_on_a_vllm_stack_says_there_is_no_ollama_list(isolated_home: Path) -> None:
    """Ollama is stopped by design, so "run `lepika up`" would loop forever."""
    config.save(config.Config(mode="server", model=REPO))
    result = runner.invoke(cli.app, ["model", "list"])
    assert result.exit_code != 0
    assert isinstance(result.exception, FriendlyError)
    assert "vLLM" in result.exception.problem
    assert "lepika model add" in result.exception.fix


def test_model_rm_on_a_vllm_stack_says_there_is_nothing_to_remove(isolated_home: Path) -> None:
    config.save(config.Config(mode="server", model=REPO))
    result = runner.invoke(cli.app, ["model", "rm", REPO])
    assert result.exit_code != 0
    assert isinstance(result.exception, FriendlyError)
    assert "nothing to remove" in result.exception.problem
    assert "lepika model add" in result.exception.fix


def test_expose_on_a_vllm_box_prints_an_openai_url_not_a_connect_line(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """`lepika connect` probes Ollama's /api/version; Caddy in front of vLLM never answers it."""
    config.save(config.Config(mode="server", model=REPO))
    monkeypatch.setattr(detect, "detect", lambda **k: LINUX_NVIDIA)
    monkeypatch.setattr(server, "start_stack", lambda info, cfg, **k: "u")
    monkeypatch.setattr(server, "lan_ip", lambda **k: "192.168.1.20")
    result = runner.invoke(cli.app, ["expose"])
    assert result.exit_code == 0, result.output
    key = server.read_env(paths.stack_dir() / server.ENV_FILE)["LEPIKA_API_KEY"]
    assert key in result.output
    assert "http://192.168.1.20:11435/v1" in result.output
    assert "lepika connect http" not in result.output


def test_model_add_starts_the_stack_for_the_model_it_is_about_to_pull(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """Switching off a vLLM model: the stack has to come up on the *new* model's profile.

    With the saved `org/repo` still in the config, `profiles()` says vllm, ollama stays
    stopped, and the pull fails with "Could not reach the engine".
    """
    config.save(config.Config(mode="server", model=REPO))
    started: list[str] = []
    monkeypatch.setattr(detect, "detect", lambda **k: LINUX_NVIDIA)
    monkeypatch.setattr(server, "start_stack", lambda info, cfg, **k: started.append(cfg.model))
    monkeypatch.setattr(engine, "pull_model", lambda *a, **k: None)
    result = runner.invoke(cli.app, ["model", "add", "qwen3:8b"])
    assert result.exit_code == 0, result.output
    assert started == ["qwen3:8b"]
    assert config.load().model == "qwen3:8b"


def test_model_add_keeps_the_old_model_when_the_pull_fails(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """The target model reaches the stack, but only a successful pull reaches the config."""
    config.save(config.Config(mode="server", model=REPO))
    monkeypatch.setattr(detect, "detect", lambda **k: LINUX_NVIDIA)
    monkeypatch.setattr(server, "start_stack", lambda info, cfg, **k: "u")

    def boom(*a: Any, **k: Any) -> None:
        raise FriendlyError("no", "fix")

    monkeypatch.setattr(engine, "pull_model", boom)
    result = runner.invoke(cli.app, ["model", "add", "qwen3:8b"])
    assert result.exit_code != 0
    assert config.load().model == REPO


def test_wizard_starts_the_stack_for_the_ollama_model_it_is_about_to_pull(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    config.save(config.Config(mode="server", model=REPO))
    started: list[str] = []
    monkeypatch.setattr(detect, "detect", lambda **k: LINUX_NVIDIA)
    monkeypatch.setattr(models, "load_curated", lambda **k: [])
    monkeypatch.setattr(wizard, "_ask", lambda *a, **k: "qwen3:8b")
    monkeypatch.setattr(engine, "pull_model", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_open_browser", lambda url: None)

    def fake_start(info: Any, cfg: Any, after_engine: Any = None, **k: Any) -> str:
        started.append(cfg.model)
        if after_engine is not None:
            after_engine()
        return "http://localhost:3000"

    monkeypatch.setattr(server, "start_stack", fake_start)
    wizard.run_wizard(mode="server")
    assert started == ["qwen3:8b"]
    assert config.load().model == "qwen3:8b"


def test_rotate_on_a_vllm_box_stays_off_the_ollama_connect_line(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """`lepika connect` rejects a vLLM engine, so the reconnect hint must not name it."""
    config.save(config.Config(mode="server", model=REPO))
    monkeypatch.setattr(detect, "detect", lambda **k: LINUX_NVIDIA)
    monkeypatch.setattr(server, "start_stack", lambda info, cfg, **k: "u")
    monkeypatch.setattr(server, "lan_ip", lambda **k: "192.168.1.20")
    runner.invoke(cli.app, ["expose"])
    result = runner.invoke(cli.app, ["expose", "--rotate"])
    assert result.exit_code == 0, result.output
    assert "lepika connect http" not in result.output
    assert "old key" in result.output


def test_engine_label_follows_the_model() -> None:
    assert server.engine_label(config.Config(mode="server", model=REPO)) == "vLLM"
    assert server.engine_label(config.Config(mode="server", model="qwen3:8b")) == "Ollama"
    assert server.engine_label(config.Config(mode="express", model=REPO)) == "Ollama"


def test_up_says_vllm_in_the_plan_sentence(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    config.save(config.Config(mode="server", model=REPO))
    monkeypatch.setattr(detect, "detect", lambda **k: LINUX_NVIDIA)
    monkeypatch.setattr(server, "gpu_note", lambda info, **k: None)
    monkeypatch.setattr(server, "start_stack", lambda info, cfg, **k: "http://localhost:3000")
    monkeypatch.setattr(cli, "_open_browser", lambda url: None)
    result = runner.invoke(cli.app, ["up"])
    assert result.exit_code == 0, result.output
    assert "OpenWebUI + vLLM" in result.output
