from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from lepika import acquire, cli, config, detect, engine, express, gguf, models, wizard
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


def _offline(request: Any, timeout: float = 0) -> Any:
    raise OSError("no network")


def test_choose_model_free_form_survives_an_offline_hub(capsys: pytest.CaptureFixture[str]) -> None:
    ref = wizard.choose_model(
        INFO,
        config.Config(),
        ask=lambda *a, **k: "hf.co/unsloth/gemma-3-4b-it-GGUF",
        curated=CURATED,
        urlopen=_offline,
    )
    assert ref == models.ModelRef(raw="hf.co/unsloth/gemma-3-4b-it-GGUF", kind="hf_gguf")
    # Collapsed: rich wraps the notice at 80 columns, between "Ollama" and "pick".
    assert "letting Ollama pick" in " ".join(capsys.readouterr().out.split())


CPU_ONLY = detect.SystemInfo("linux", "x86_64", "none", 16.0, False, True, True)


def test_choose_model_rejects_hf_repo_with_gguf_hint_where_nothing_can_serve_it() -> None:
    with pytest.raises(FriendlyError) as exc:
        wizard.choose_model(
            CPU_ONLY,
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


def test_dry_run_says_it_would_import_a_full_weight_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(wizard, "_ask", lambda *a, **k: "Qwen/Qwen3.5-2B")
    monkeypatch.setattr(models, "load_curated", lambda **k: CURATED)
    result = runner.invoke(cli.app, ["--dry-run"])
    assert result.exit_code == 0, result.output
    assert "would: import Qwen/Qwen3.5-2B into Ollama (nvfp4)" in result.output
    assert "would: pull" not in result.output


def test_wizard_acquires_the_model_through_acquire(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(wizard, "_ask", lambda *a, **k: "Qwen/Qwen3.5-2B")
    monkeypatch.setattr(models, "load_curated", lambda **k: CURATED)
    monkeypatch.setattr(express, "ensure_ollama", lambda info, **k: None)
    monkeypatch.setattr(express, "ensure_openwebui", lambda cfg, **k: None)
    monkeypatch.setattr(express, "webui_up", lambda port, **k: True)
    monkeypatch.setattr(cli, "_open_browser", lambda url: None)
    monkeypatch.setattr(acquire, "acquire", lambda info, cfg, ref: "Qwen/Qwen3.5-2B")
    result = runner.invoke(cli.app, [])
    assert result.exit_code == 0, result.output
    assert config.load().model == "Qwen/Qwen3.5-2B"


def test_wizard_declined_download_keeps_the_old_model(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """Saying no to the size leaves the stack up and the previous model still the default."""
    config.save(config.Config(model="previous:model"))
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(wizard, "_ask", lambda *a, **k: "Qwen/Qwen3.5-2B")
    monkeypatch.setattr(models, "load_curated", lambda **k: CURATED)
    monkeypatch.setattr(express, "ensure_ollama", lambda info, **k: None)
    monkeypatch.setattr(express, "ensure_openwebui", lambda cfg, **k: None)
    monkeypatch.setattr(express, "webui_up", lambda port, **k: True)
    monkeypatch.setattr(cli, "_open_browser", lambda url: None)
    monkeypatch.setattr(acquire, "acquire", lambda info, cfg, ref: None)
    result = runner.invoke(cli.app, [])
    assert result.exit_code == 0, result.output
    assert config.load().model == "previous:model"


# Already size-sorted, as `gguf.list_builds` returns them. On the 16 GB Mac (INFO) the
# Metal budget is 10.7 GB and 80 % of it 8.5 GB: IQ4_XS is the ★, Q4_K_M is "mixed"
# (under 80 % of RAM and under 1.5x the GPU), Q8_0 exceeds 80 % of RAM and is hidden.
BUILDS = [
    gguf.Build("UD-IQ2_M", int(6.0e9)),
    gguf.Build("UD-IQ4_XS", int(8.0e9)),
    gguf.Build("UD-Q4_K_M", int(12.0e9)),
    gguf.Build("Q8_0", int(28.7e9)),
]
GGUF_REF = models.ModelRef(raw="hf.co/unsloth/Qwen3.8-27B-GGUF", kind="hf_gguf")


def _listing(builds: list[gguf.Build]) -> Any:
    calls: list[str] = []

    def fake(repo: str, token: str = "", urlopen: Any = None) -> list[gguf.Build]:
        calls.append(repo)
        return builds

    fake.calls = calls  # type: ignore[attr-defined]
    return fake


def test_choose_quant_enter_takes_the_recommended_build(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(gguf, "list_builds", _listing(BUILDS))
    prompts: list[tuple[str, dict[str, Any]]] = []

    def ask(prompt: str, **kwargs: Any) -> str:
        prompts.append((prompt, kwargs))
        return ""

    ref = wizard.choose_quant(GGUF_REF, config.Config(), INFO, ask=ask)
    assert ref == models.ModelRef(raw="hf.co/unsloth/Qwen3.8-27B-GGUF:UD-IQ4_XS", kind="hf_gguf")
    assert prompts == [("Pick a number", {"default": "2"})]
    out = capsys.readouterr().out
    assert "UD-IQ4_XS ★" in out
    assert "fits your GPU" in out
    assert "1 larger build hidden" in out and "16 GB RAM" in out
    assert "GPU + some CPU" in out  # UD-Q4_K_M, row 3
    assert "Q8_0" not in out


def test_choose_quant_number_picks_another_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gguf, "list_builds", _listing(BUILDS))
    ref = wizard.choose_quant(GGUF_REF, config.Config(), INFO, ask=lambda *a, **k: "3")
    assert ref.raw == "hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"


def test_choose_quant_reprompts_once_then_takes_the_recommendation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(gguf, "list_builds", _listing(BUILDS))
    answers = iter(["9", "banana"])
    ref = wizard.choose_quant(GGUF_REF, config.Config(), INFO, ask=lambda *a, **k: next(answers))
    assert ref.raw == "hf.co/unsloth/Qwen3.8-27B-GGUF:UD-IQ4_XS"
    assert "Pick 1 to 3." in capsys.readouterr().out


def test_choose_quant_sends_the_saved_token(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def fake(repo: str, token: str = "", urlopen: Any = None) -> list[gguf.Build]:
        seen["token"] = token
        return BUILDS

    monkeypatch.setattr(gguf, "list_builds", fake)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    cfg = config.Config()
    cfg.hf_token = "hf_saved"
    wizard.choose_quant(GGUF_REF, cfg, INFO, ask=lambda *a, **k: "")
    assert seen["token"] == "hf_saved"


def test_choose_quant_leaves_an_explicit_tag_alone_without_asking_anyone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listing = _listing(BUILDS)
    monkeypatch.setattr(gguf, "list_builds", listing)
    tagged = models.ModelRef(raw="hf.co/unsloth/x-GGUF:Q4_K_M", kind="hf_gguf")

    def never(*a: Any, **k: Any) -> str:
        raise AssertionError("no prompt for a tagged ref")

    assert wizard.choose_quant(tagged, config.Config(), INFO, ask=never) == tagged
    assert listing.calls == []


def test_choose_quant_ignores_refs_that_are_not_hf_gguf(monkeypatch: pytest.MonkeyPatch) -> None:
    listing = _listing(BUILDS)
    monkeypatch.setattr(gguf, "list_builds", listing)
    for ref in (models.parse_model_ref("qwen3:8b"), models.parse_model_ref("Qwen/Qwen3.5-2B")):
        assert wizard.choose_quant(ref, config.Config(), INFO) == ref
    assert listing.calls == []


def test_choose_quant_falls_through_silently_on_a_repo_without_gguf(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(gguf, "list_builds", _listing([]))
    ref = models.ModelRef(raw="hf.co/Qwen/Qwen3.5-2B", kind="hf_gguf")
    assert wizard.choose_quant(ref, config.Config(), INFO) == ref
    assert capsys.readouterr().out == ""


def test_choose_quant_skips_a_malformed_repo_id_silently(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    listing = _listing(BUILDS)
    monkeypatch.setattr(gguf, "list_builds", listing)
    ref = models.ModelRef(raw="hf.co/../etc", kind="hf_gguf")
    assert wizard.choose_quant(ref, config.Config(), INFO) == ref
    assert listing.calls == []
    assert capsys.readouterr().out == ""


def test_choose_quant_says_so_when_nothing_fits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(gguf, "list_builds", _listing([gguf.Build("Q8_0", int(28.7e9))]))

    def never(*a: Any, **k: Any) -> str:
        raise AssertionError("nothing to pick from")

    assert wizard.choose_quant(GGUF_REF, config.Config(), INFO, ask=never) == GGUF_REF
    assert "None of the 1 builds fits in your 16 GB RAM" in capsys.readouterr().out


def test_choose_quant_names_the_ram_budget_without_a_gpu(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(gguf, "list_builds", _listing([gguf.Build("Q4_K_M", int(4e9))]))
    wizard.choose_quant(GGUF_REF, config.Config(), CPU_ONLY, ask=lambda *a, **k: "")
    out = capsys.readouterr().out
    assert "Fit (16 GB RAM)" in out and "CPU only — slow" in out


def test_choose_quant_probes_nvidia_memory_only_when_it_lists(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(gguf, "list_builds", _listing([gguf.Build("Q4_K_M", int(4e9))]))
    nvidia = detect.SystemInfo("linux", "x86_64", "nvidia", 32.0, False, True, True)

    def run(cmd: list[str], **kwargs: Any) -> Any:
        return SimpleNamespace(stdout="16303\n")

    wizard.choose_quant(GGUF_REF, config.Config(), nvidia, ask=lambda *a, **k: "", run=run)
    assert "Fit (17 GB GPU)" in capsys.readouterr().out
