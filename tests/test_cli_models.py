from __future__ import annotations

import dataclasses
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from lepika import acquire, cli, config, detect, engine, express, hf
from lepika.errors import FriendlyError
from lepika.models import ModelRef

runner = CliRunner()


def _plain(output: str) -> str:
    """CLI output with the colour stripped and the wrapping collapsed.

    Typer forces colour under CI (GITHUB_ACTIONS/FORCE_COLOR), and rich's highlighter
    then wraps fragments in escape codes that split the substring being looked for.
    """
    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", output).split())


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


CPU = detect.SystemInfo("linux", "x86_64", "none", 32.0, False, True, True)
MAC = detect.SystemInfo("macos", "arm64", "apple", 32.0, False, True, True)
SAFETENSORS = hf.Preflight(
    files=("config.json", "model.safetensors"),
    total_bytes=4_000_000_000,
    download_bytes=4_000_000_000,
)
GGUF_ONLY = hf.Preflight(files=("model-Q4_K_M.gguf",), total_bytes=4_000_000_000, download_bytes=0)


def test_model_add_rejects_full_weight_repo_where_nothing_can_serve_it(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    monkeypatch.setattr(detect, "detect", lambda **k: CPU)
    result = runner.invoke(cli.app, ["model", "add", "meta-llama/Llama-3.3-70B"])
    assert result.exit_code != 0
    # `runner.invoke` bypasses `cli.run`, which is where a FriendlyError is printed,
    # so the message is asserted on the exception itself (as the other CLI tests do).
    assert isinstance(result.exception, FriendlyError)
    assert "GGUF" in result.exception.fix


@pytest.fixture()
def fake_import(monkeypatch: pytest.MonkeyPatch, isolated_home: Path) -> dict[str, Any]:
    """Every effect of the import path, recorded: preflight, download, create, cleanup."""
    seen: dict[str, Any] = {
        "pulled": [],
        "downloaded": [],
        "imported": [],
        "confirms": [],
        "quants": [],
    }
    monkeypatch.setattr(detect, "detect", lambda **k: MAC)
    monkeypatch.setattr(express, "ensure_ollama", lambda info, **k: None)
    monkeypatch.setattr(
        express, "ensure_mlx", lambda info, **k: seen.setdefault("mlx", []).append(info.os)
    )
    monkeypatch.setattr(engine, "pull_model", lambda url, ref, **k: seen["pulled"].append(ref.raw))
    monkeypatch.setattr(engine, "list_models", lambda url, **k: [])
    monkeypatch.setattr(hf, "preflight", lambda repo, token="", **k: SAFETENSORS)

    def download(repo: str, dest: Path, token: str = "", **k: Any) -> None:
        seen["downloaded"].append((repo, dest))
        dest.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(hf, "download", download)

    def import_model(url: str, name: str, source: Path, **k: Any) -> None:
        seen["imported"].append((name, source))
        seen["quants"].append(k.get("quant"))

    monkeypatch.setattr(engine, "import_model", import_model)
    monkeypatch.setattr(acquire, "confirm", lambda q: seen["confirms"].append(q) or True)
    return seen


def test_model_add_imports_a_safetensors_repo_and_cleans_up(
    fake_import: dict[str, Any], isolated_home: Path
) -> None:
    result = runner.invoke(cli.app, ["model", "add", "Qwen/Qwen3.5-2B"])
    assert result.exit_code == 0, result.output
    dest = isolated_home / "hf" / "Qwen" / "Qwen3.5-2B"
    assert fake_import["downloaded"] == [("Qwen/Qwen3.5-2B", dest)]
    assert fake_import["imported"] == [("Qwen/Qwen3.5-2B", dest)]
    assert not dest.exists()  # Ollama holds its own copy; 4 GB is not kept twice
    assert not dest.parent.exists()  # and the org directory does not linger empty
    assert config.load().model == "Qwen/Qwen3.5-2B"
    # "Fetch", not "Download": a file already in the hub cache is copied, not fetched,
    # and it costs the disk the question is asking about all the same.
    assert fake_import["confirms"][0].startswith("Fetch 4.0 GB onto disk and import as nvfp4")
    assert fake_import["quants"] == ["nvfp4"]  # the default, with no --quant
    assert fake_import["pulled"] == []
    # The MLX engine is checked on the machine the import runs on, whatever it is.
    assert fake_import["mlx"] == ["macos"]


def test_model_add_keeps_an_org_directory_another_repo_still_uses(
    fake_import: dict[str, Any], isolated_home: Path
) -> None:
    """`rmdir` removes the org only when it is empty, so a sibling download survives."""
    sibling = isolated_home / "hf" / "Qwen" / "Qwen3.5-32B"
    sibling.mkdir(parents=True)
    result = runner.invoke(cli.app, ["model", "add", "Qwen/Qwen3.5-2B"])
    assert result.exit_code == 0, result.output
    assert sibling.is_dir()


def test_model_add_ensures_the_mlx_engine_before_downloading(
    fake_import: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    # Linux + NVIDIA: the one platform where the bundle is really installed, and the
    # engine has to be ready before a 55 GB download, never after it.
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(express, "ensure_mlx", lambda info, **k: order.append("mlx"))
    monkeypatch.setattr(
        hf,
        "download",
        lambda repo, dest, token="", **k: (
            order.append("download") or dest.mkdir(parents=True, exist_ok=True)
        ),
    )
    assert runner.invoke(cli.app, ["model", "add", "Qwen/Qwen3.5-2B"]).exit_code == 0
    assert order == ["mlx", "download"]


def test_model_add_quant_chooses_the_quantization(fake_import: dict[str, Any]) -> None:
    """--quant reaches `ollama create -q`, and the size question says which one it is."""
    result = runner.invoke(cli.app, ["model", "add", "Qwen/Qwen3.5-2B", "--quant", "int4"])
    assert result.exit_code == 0, result.output
    assert fake_import["quants"] == ["int4"]
    assert "int4" in fake_import["confirms"][0]


def test_model_add_rejects_a_quant_ollama_does_not_take(fake_import: dict[str, Any]) -> None:
    result = runner.invoke(cli.app, ["model", "add", "Qwen/Qwen3.5-2B", "--quant", "q8"])
    assert result.exit_code == 2  # Typer's usage error, before anything is downloaded
    assert "--quant must be one of nvfp4, int4." in _plain(result.output)
    assert fake_import["downloaded"] == []


def test_model_add_ignores_quant_on_a_pull(fake_engine: list[str]) -> None:
    """A tag is pulled, not built: the flag is only meaningful once an import happens."""
    result = runner.invoke(cli.app, ["model", "add", "qwen3:8b", "--quant", "int4"])
    assert result.exit_code == 0, result.output
    assert fake_engine == ["qwen3:8b"]
    assert config.load().model == "qwen3:8b"


def test_model_add_declined_import_adds_nothing(
    fake_import: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(acquire, "confirm", lambda q: False)
    result = runner.invoke(cli.app, ["model", "add", "Qwen/Qwen3.5-2B"])
    assert result.exit_code == 0
    assert fake_import["downloaded"] == []
    assert config.load().model == ""


def test_model_add_keeps_the_download_when_the_import_fails(
    fake_import: dict[str, Any], monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    def boom(url: str, name: str, source: Path, **k: Any) -> None:
        raise FriendlyError("Import of Qwen/Qwen3.5-2B failed.", "Run it again.")

    monkeypatch.setattr(engine, "import_model", boom)
    result = runner.invoke(cli.app, ["model", "add", "Qwen/Qwen3.5-2B"])
    assert result.exit_code != 0
    assert (isolated_home / "hf" / "Qwen" / "Qwen3.5-2B").is_dir()
    assert config.load().model == ""


def test_model_add_routes_a_gguf_only_repo_to_a_pull(
    fake_import: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hf, "preflight", lambda repo, token="", **k: GGUF_ONLY)
    result = runner.invoke(cli.app, ["model", "add", "unsloth/x-GGUF"])
    assert result.exit_code == 0, result.output
    assert fake_import["pulled"] == ["hf.co/unsloth/x-GGUF"]
    assert fake_import["downloaded"] == []
    assert config.load().model == "hf.co/unsloth/x-GGUF"


def test_model_add_falls_through_from_a_not_gguf_refusal_to_an_import(
    fake_import: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(url: str, ref: ModelRef, **k: Any) -> None:
        raise engine.NotGGUF("'hf.co/Qwen/Qwen3.5-2B' ships full weights.", "Use a GGUF build.")

    monkeypatch.setattr(engine, "pull_model", refuse)
    result = runner.invoke(cli.app, ["model", "add", "huggingface.co/Qwen/Qwen3.5-2B"])
    assert result.exit_code == 0, result.output
    assert fake_import["imported"][0][0] == "Qwen/Qwen3.5-2B"
    assert config.load().model == "Qwen/Qwen3.5-2B"


def test_model_add_not_gguf_refusal_stays_an_error_where_imports_cannot_run(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    monkeypatch.setattr(detect, "detect", lambda **k: CPU)
    monkeypatch.setattr(express, "ensure_ollama", lambda info, **k: None)

    def refuse(url: str, ref: ModelRef, **k: Any) -> None:
        raise engine.NotGGUF("'hf.co/Qwen/Qwen3.5-2B' ships full weights.", "Use a GGUF build.")

    monkeypatch.setattr(engine, "pull_model", refuse)
    result = runner.invoke(cli.app, ["model", "add", "hf.co/Qwen/Qwen3.5-2B"])
    assert result.exit_code != 0
    assert isinstance(result.exception, engine.NotGGUF)
    assert "full weights" in result.exception.problem


def test_model_add_gated_repo_asks_for_a_token_once_then_retries(
    fake_import: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def preflight(repo: str, token: str = "", **k: Any) -> hf.Preflight:
        calls.append(token)
        if not token:
            raise hf.GatedRepo("'x/y' is gated or private on Hugging Face.", "Accept its licence.")
        return SAFETENSORS

    monkeypatch.setattr(hf, "preflight", preflight)
    monkeypatch.setattr(hf, "ask_token", lambda cfg, **k: "hf_new")
    result = runner.invoke(cli.app, ["model", "add", "x/y"])
    assert result.exit_code == 0, result.output
    assert calls == ["", "hf_new"]


def test_model_add_gated_repo_without_a_token_is_the_licence_hint(
    fake_import: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def gated(repo: str, token: str = "", **k: Any) -> hf.Preflight:
        raise hf.GatedRepo("'x/y' is gated or private on Hugging Face.", "Accept its licence.")

    monkeypatch.setattr(hf, "preflight", gated)
    monkeypatch.setattr(hf, "ask_token", lambda cfg, **k: "")
    result = runner.invoke(cli.app, ["model", "add", "x/y"])
    assert result.exit_code != 0
    assert isinstance(result.exception, hf.GatedRepo)
    assert "gated" in result.exception.problem


def test_model_add_refuses_when_the_disk_is_too_small(
    fake_import: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    # Only `.free` is read; a namespace keeps the fake off CPython's private
    # `_ntuple_diskusage` name.
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=1_000_000))
    result = runner.invoke(cli.app, ["model", "add", "Qwen/Qwen3.5-2B"])
    assert result.exit_code != 0
    assert isinstance(result.exception, FriendlyError)
    assert "Not enough disk" in result.exception.problem
    assert fake_import["downloaded"] == []


def test_model_add_measures_ollamas_store_not_just_lepika_home(
    fake_import: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    isolated_home: Path,
    tmp_path: Path,
) -> None:
    """A stock Linux service keeps models on /usr/share — often a different, smaller disk."""
    import shutil

    store = tmp_path / "usr" / "share" / "ollama" / ".ollama"
    store.mkdir(parents=True)
    monkeypatch.setattr(express, "ollama_store", lambda **k: store)
    free = {isolated_home: 500 * 2**30, store: 1_000_000}
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=free[Path(path)]))
    result = runner.invoke(cli.app, ["model", "add", "Qwen/Qwen3.5-2B"])
    assert result.exit_code != 0
    assert isinstance(result.exception, FriendlyError)
    # ~/.lepika has room to spare; the engine's own disk is the one that decides.
    assert "Not enough disk" in result.exception.problem
    assert fake_import["downloaded"] == []


def test_model_add_repo_without_weights_is_refused(
    fake_import: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        hf,
        "preflight",
        lambda repo, token="", **k: hf.Preflight(
            files=("README.md",), total_bytes=1, download_bytes=1
        ),
    )
    result = runner.invoke(cli.app, ["model", "add", "x/y"])
    assert result.exit_code != 0
    assert isinstance(result.exception, FriendlyError)
    assert "no safetensors or GGUF" in result.exception.problem


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


def test_model_list_tells_the_engine_whether_it_is_ours(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """The unreachable hint depends on it: `lepika up` cannot start a remote engine."""
    seen: list[bool] = []
    config.save(config.Config(engine_managed=False, engine_url="http://gpu-box:11435"))
    monkeypatch.setattr(
        engine, "list_models", lambda url, managed=True, **k: bool(seen.append(managed)) or []
    )
    assert runner.invoke(cli.app, ["model", "list"]).exit_code == 0
    assert seen == [False]


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


def test_model_rm_of_the_default_under_its_latest_tag_clears_it(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """The config says `qwen3`; `model list` — and the user — say `qwen3:latest`."""
    config.save(config.Config(model="qwen3"))
    monkeypatch.setattr(engine, "delete_model", lambda url, name, **k: None)
    result = runner.invoke(cli.app, ["model", "rm", "qwen3:latest"])
    assert result.exit_code == 0, result.output
    assert config.load().model == ""


def test_model_list_marks_the_default_under_its_latest_tag(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    config.save(config.Config(model="qwen3"))
    monkeypatch.setattr(engine, "list_models", lambda url, **k: [("qwen3:latest", 1_000)])
    result = runner.invoke(cli.app, ["model", "list"])
    assert result.exit_code == 0, result.output
    assert "(default)" in result.output


def test_model_add_help_lists_all_three_model_ref_shapes() -> None:
    """Listing two of three reads as "a full-weight repo is not accepted here"."""
    result = runner.invoke(cli.app, ["model", "add", "--help"])
    assert result.exit_code == 0
    # The help is also wrapped into a box at 80 columns, which splits "(full weights)"
    # across two lines: drop the borders as well as the colour before looking.
    plain = " ".join(re.sub(r"[│╭╮╰╯─]", " ", _plain(result.output)).split())
    assert "qwen3:8b" in plain
    # The bare `<org>/<repo>` shape is the one that was missing; only it says what it is.
    assert "<org>/<repo> (full weights)" in plain


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


def test_model_add_skips_a_repo_that_is_already_imported(
    fake_import: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ollama already serves it: re-downloading 4 GB to rebuild the same model is waste."""
    monkeypatch.setattr(engine, "list_models", lambda url, **k: [("Qwen/Qwen3.5-2B:latest", 1)])
    monkeypatch.setattr(
        hf, "preflight", lambda repo, token="", **k: pytest.fail("no pre-flight when installed")
    )
    result = runner.invoke(cli.app, ["model", "add", "Qwen/Qwen3.5-2B"])
    assert result.exit_code == 0, result.output
    assert fake_import["downloaded"] == []
    assert fake_import["imported"] == []
    assert "already imported" in result.output
    assert config.load().model == "Qwen/Qwen3.5-2B"


def test_model_add_drops_an_ollama_tag_when_falling_through_to_an_import(
    fake_import: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`:latest` is Ollama's tag, not part of the Hugging Face repo id."""

    def refuse(url: str, ref: ModelRef, **k: Any) -> None:
        raise engine.NotGGUF("'hf.co/Qwen/Qwen3.5-2B:latest' ships full weights.", "Use GGUF.")

    monkeypatch.setattr(engine, "pull_model", refuse)
    result = runner.invoke(cli.app, ["model", "add", "hf.co/Qwen/Qwen3.5-2B:latest"])
    assert result.exit_code == 0, result.output
    assert fake_import["imported"][0][0] == "Qwen/Qwen3.5-2B"
    assert config.load().model == "Qwen/Qwen3.5-2B"


def test_model_add_import_without_a_terminal_is_friendly(
    fake_import: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Piped or in CI there is no one to answer the size question — say so, don't traceback."""

    def no_terminal(question: str, default: bool = False) -> bool:
        raise EOFError

    monkeypatch.setattr(acquire.Confirm, "ask", no_terminal)
    monkeypatch.setattr(acquire, "confirm", acquire.ask_confirm)  # the fixture's auto-yes, undone
    result = runner.invoke(cli.app, ["model", "add", "Qwen/Qwen3.5-2B"])
    assert result.exit_code != 0
    assert isinstance(result.exception, FriendlyError)
    assert "terminal" in result.exception.problem
    assert fake_import["downloaded"] == []


def test_model_add_warns_when_the_import_may_not_fit_in_ram_but_proceeds(
    fake_import: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The RAM figure is advice, not a refusal: the user asked for this model."""
    monkeypatch.setattr(detect, "detect", lambda **k: dataclasses.replace(MAC, ram_gb=1.0))
    result = runner.invoke(cli.app, ["model", "add", "Qwen/Qwen3.5-2B"])
    assert result.exit_code == 0, result.output
    assert "may not fit" in " ".join(result.output.split())
    assert fake_import["imported"][0][0] == "Qwen/Qwen3.5-2B"


@pytest.fixture()
def weights(tmp_path: Path) -> Path:
    """A folder shaped like what `hf download` leaves behind."""
    folder = tmp_path / "Qwen3.5-2B"
    folder.mkdir()
    (folder / "config.json").write_text("{}")
    (folder / "model.safetensors").write_bytes(b"\0" * 4096)
    return folder


@pytest.fixture()
def fake_local(monkeypatch: pytest.MonkeyPatch, isolated_home: Path) -> dict[str, Any]:
    """Every effect of `model import`, recorded — and no disk, engine or bundle touched."""
    seen: dict[str, Any] = {"imported": [], "ensured": []}
    monkeypatch.setattr(detect, "detect", lambda **k: MAC)
    monkeypatch.setattr(
        express, "ensure_ollama", lambda info, **k: seen["ensured"].append("ollama")
    )
    monkeypatch.setattr(express, "ensure_mlx", lambda info, **k: seen["ensured"].append("mlx"))
    # Not a directory, so the disk check falls back to ~/.lepika (the isolated one).
    monkeypatch.setattr(express, "ollama_store", lambda **k: isolated_home / "no-ollama-here")
    monkeypatch.setattr(engine, "list_models", lambda url, **k: [])
    monkeypatch.setattr(
        shutil, "disk_usage", lambda p: SimpleNamespace(total=0, used=0, free=10**12)
    )

    def import_model(url: str, name: str, source: Path, **k: Any) -> None:
        seen["imported"].append((name, source, k.get("owned"), k.get("quant")))

    monkeypatch.setattr(engine, "import_model", import_model)
    return seen


def test_model_import_imports_a_local_folder_and_makes_it_the_default(
    fake_local: dict[str, Any], weights: Path
) -> None:
    result = runner.invoke(cli.app, ["model", "import", str(weights)])
    assert result.exit_code == 0, result.output
    # owned=False: the folder is the user's, so Ollama reads it and nothing writes to it.
    assert fake_local["imported"] == [("Qwen3.5-2B", weights, False, "nvfp4")]
    assert config.load().model == "Qwen3.5-2B"
    assert "✓ Imported:" in result.output
    assert fake_local["ensured"] == ["ollama", "mlx"]
    assert not (weights / "Modelfile").exists()


def test_model_import_name_wins_over_the_folder_name(
    fake_local: dict[str, Any], weights: Path
) -> None:
    result = runner.invoke(
        cli.app, ["model", "import", str(weights), "--name", "Qwen/Qwen3.5-2B", "--quant", "int4"]
    )
    assert result.exit_code == 0, result.output
    assert fake_local["imported"] == [("Qwen/Qwen3.5-2B", weights, False, "int4")]
    assert config.load().model == "Qwen/Qwen3.5-2B"


def test_model_import_names_a_dot_path_after_the_real_folder(
    fake_local: dict[str, Any], weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`.` and a trailing slash have no `name` of their own — resolve to find one."""
    monkeypatch.chdir(weights)
    assert runner.invoke(cli.app, ["model", "import", "."]).exit_code == 0
    assert runner.invoke(cli.app, ["model", "import", f"{weights}/"]).exit_code == 0
    assert [name for name, *_rest in fake_local["imported"]] == ["Qwen3.5-2B", "Qwen3.5-2B"]


def test_model_import_never_names_a_model_dot_dot(
    fake_local: dict[str, Any], weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`Path("..").name` is `".."`, which Ollama would happily build under that name."""
    (weights / "checkpoint").mkdir()
    monkeypatch.chdir(weights / "checkpoint")
    result = runner.invoke(cli.app, ["model", "import", ".."])
    assert result.exit_code == 0, result.output
    assert fake_local["imported"][0][0] == "Qwen3.5-2B"
    assert config.load().model == "Qwen3.5-2B"


def test_model_import_refuses_a_folder_without_weights(
    fake_local: dict[str, Any], tmp_path: Path
) -> None:
    empty = tmp_path / "not-a-model"
    empty.mkdir()
    (empty / "model.safetensors").write_bytes(b"\0")  # weights, but no config.json
    result = runner.invoke(cli.app, ["model", "import", str(empty)])
    assert result.exit_code != 0
    assert isinstance(result.exception, FriendlyError)
    assert "does not look like a safetensors model folder" in result.exception.problem
    assert "lepika model add <org>/<repo>" in result.exception.fix
    assert fake_local["imported"] == []
    # A typo'd path must not be how a machine ends up with Ollama installed on it.
    assert fake_local["ensured"] == []


@pytest.mark.parametrize("bad", ["a/b/c", "..", ".", "./x"])
def test_model_import_refuses_a_name_ollama_cannot_use(
    fake_local: dict[str, Any], weights: Path, bad: str
) -> None:
    """Every part starts alphanumeric, so a path-shaped name is our refusal, not Ollama's."""
    result = runner.invoke(cli.app, ["model", "import", str(weights), "--name", bad])
    assert result.exit_code != 0
    assert isinstance(result.exception, FriendlyError)
    assert result.exception.problem == f"'{bad}' is not a valid model name."
    assert "at most one slash" in result.exception.fix
    assert fake_local["imported"] == []
    assert fake_local["ensured"] == []  # checked before the engine is installed


def test_model_import_needs_express_mode_with_a_local_engine(
    fake_local: dict[str, Any], weights: Path
) -> None:
    """Server mode runs no MLX engine of its own — refuse before touching the engine."""
    config.save(config.Config(mode="server"))
    result = runner.invoke(cli.app, ["model", "import", str(weights)])
    assert result.exit_code != 0
    assert isinstance(result.exception, FriendlyError)
    assert "Express mode" in result.exception.problem
    assert "GGUF" in result.exception.fix
    assert fake_local["ensured"] == []  # nothing installed, nothing started
    assert fake_local["imported"] == []


def test_model_import_skips_a_model_ollama_already_serves(
    fake_local: dict[str, Any], weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(engine, "list_models", lambda url, **k: [("Qwen3.5-2B:latest", 1)])
    result = runner.invoke(cli.app, ["model", "import", str(weights)])
    assert result.exit_code == 0, result.output
    assert "already imported" in result.output
    assert fake_local["imported"] == []
    assert config.load().model == "Qwen3.5-2B"


def test_model_import_refuses_when_the_store_disk_is_too_small(
    fake_local: dict[str, Any], weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "disk_usage", lambda p: SimpleNamespace(total=0, used=0, free=1))
    result = runner.invoke(cli.app, ["model", "import", str(weights)])
    assert result.exit_code != 0
    assert isinstance(result.exception, FriendlyError)
    assert "Not enough disk" in result.exception.problem
    assert "Free some space" in result.exception.fix
    assert fake_local["imported"] == []


def test_model_import_warns_when_it_may_not_fit_in_ram_but_proceeds(
    fake_local: dict[str, Any], weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same advisory `model add` prints, from the same helper."""
    monkeypatch.setattr(detect, "detect", lambda **k: dataclasses.replace(MAC, ram_gb=0.0))
    result = runner.invoke(cli.app, ["model", "import", str(weights)])
    assert result.exit_code == 0, result.output
    assert "may not fit" in " ".join(result.output.split())
    assert len(fake_local["imported"]) == 1
