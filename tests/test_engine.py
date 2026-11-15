from __future__ import annotations

import importlib.resources
import io
import itertools
import json
import sys
import urllib.error
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeResponse, FakeSystem, routes
from rich.progress import Progress

from lepika import engine, paths
from lepika.errors import FriendlyError
from lepika.models import ModelRef
from lepika.system import Response


def answering(
    table: dict[str, Response | BaseException],
    *,
    url: str = "http://x",
    stream: tuple[int, str] = (0, ""),
) -> FakeSystem:
    """An engine at `url` that answers each listed path, and nothing else."""
    return FakeSystem(urls=routes({url + path: v for path, v in table.items()}), stream=stream)


def serving(path: str, body: bytes) -> FakeSystem:
    """An engine at http://x that answers `path` with `body`."""
    return answering({path: FakeResponse(body)})


def test_list_models_parses_tags() -> None:
    body = json.dumps({"models": [{"name": "qwen3:8b", "size": 5_000_000_000}]}).encode()
    assert engine.list_models("http://x", sys=serving("/api/tags", body)) == [
        ("qwen3:8b", 5_000_000_000)
    ]


def test_list_models_sends_bearer_key() -> None:
    sys = serving("/api/tags", b'{"models": []}')
    engine.list_models("http://x", key="k", sys=sys)
    seen = sys.requested
    assert seen[0].full_url == "http://x/api/tags"
    assert seen[0].get_header("Authorization") == "Bearer k"


def test_list_models_unreachable_is_friendly() -> None:
    with pytest.raises(FriendlyError) as exc:
        engine.list_models("http://x", sys=answering({"/api/tags": OSError("refused")}))
    assert "lepika doctor" in exc.value.fix


def test_an_unreachable_remote_engine_is_not_told_to_run_lepika_up() -> None:
    """`lepika up` never starts someone else's engine (rule 9), so that hint is wrong."""
    down = answering({"/api/tags": urllib.error.URLError("refused")}, url="http://gpu-box:11435")
    with pytest.raises(FriendlyError) as exc:
        engine.list_models("http://gpu-box:11435", key="k", sys=down, managed=False)
    assert "gpu-box" in exc.value.problem
    assert "lepika up" not in exc.value.fix
    assert "lepika connect http://gpu-box:11435 --key" in exc.value.fix
    assert "lepika connect --local" in exc.value.fix


def test_an_unreachable_managed_engine_still_says_lepika_up() -> None:
    down = answering({"/api/tags": urllib.error.URLError("refused")}, url="http://127.0.0.1:11434")
    with pytest.raises(FriendlyError) as exc:
        engine.list_models("http://127.0.0.1:11434", sys=down)
    assert "lepika up" in exc.value.fix


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        ("qwen3", "qwen3:latest", True),
        ("qwen3:8b", "qwen3:8b", True),
        ("qwen3:8b", "qwen3:latest", False),
        ("hf.co/org/repo-GGUF", "hf.co/org/repo-GGUF:latest", True),
        ("hf.co/org/repo-GGUF:Q4_K_M", "hf.co/org/repo-GGUF", False),
        ("qwen3", "qwen3.5", False),
        ("Qwen/Qwen3.5-2B", "qwen/qwen3.5-2b:latest", True),
        ("Qwen/Qwen3.5-2B", "qwen/qwen3.5-4b:latest", False),
    ],
)
def test_same_model_treats_a_missing_tag_as_latest(a: str, b: str, same: bool) -> None:
    """Ollama stores an untagged ref as `name:latest` and lists it that way.

    It also lower-cases the name on create, so an imported `Qwen/Qwen3.5-2B` comes
    back from `/api/tags` as `qwen/qwen3.5-2b:latest`.
    """
    assert engine.same_model(a, b) is same


def test_delete_model_uses_delete_verb_and_json_body() -> None:
    sys = serving("/api/delete", b"")
    engine.delete_model("http://x", "qwen3:8b", sys=sys)
    req = sys.requested[0]
    assert req.get_method() == "DELETE"
    assert req.full_url == "http://x/api/delete"
    assert json.loads(req.data) == {"model": "qwen3:8b"}


def test_delete_missing_model_is_friendly() -> None:
    not_found = urllib.error.HTTPError("http://x/api/delete", 404, "Not Found", {}, None)  # type: ignore[arg-type]
    with pytest.raises(FriendlyError) as exc:
        engine.delete_model("http://x", "nope:1b", sys=answering({"/api/delete": not_found}))
    assert "not installed" in exc.value.problem


def test_pull_streams_progress_and_finishes_on_success() -> None:
    lines = [
        {"status": "pulling manifest"},
        {"status": "pulling abc", "digest": "abc", "total": 100, "completed": 40},
        {"status": "pulling abc", "digest": "abc", "total": 100, "completed": 100},
        {"status": "success"},
    ]
    body = "\n".join(json.dumps(line) for line in lines).encode()
    ticks: list[tuple[int, int]] = []
    engine.pull_model(
        "http://x",
        ModelRef(raw="qwen3:8b", kind="ollama"),
        sys=serving("/api/pull", body),
        progress=lambda done, total: ticks.append((done, total)),
    )
    assert ticks == [(40, 100), (100, 100)]


def test_pull_error_line_is_friendly() -> None:
    body = json.dumps({"error": "pull model manifest: file does not exist"}).encode()
    with pytest.raises(FriendlyError) as exc:
        engine.pull_model(
            "http://x", ModelRef(raw="nope:1b", kind="ollama"), sys=serving("/api/pull", body)
        )
    assert "nope:1b" in exc.value.problem
    assert "GGUF" in exc.value.fix


def test_pull_that_ends_without_success_is_friendly() -> None:
    body = json.dumps({"status": "pulling manifest"}).encode()
    with pytest.raises(FriendlyError):
        engine.pull_model(
            "http://x", ModelRef(raw="q:1b", kind="ollama"), sys=serving("/api/pull", body)
        )


def test_pull_sends_stream_true_and_timeout_is_generous() -> None:
    body = json.dumps({"status": "success"}).encode()
    sys = serving("/api/pull", body)
    engine.pull_model("http://x", ModelRef(raw="q:1b", kind="ollama"), sys=sys)
    seen = sys.requested
    assert json.loads(seen[0].data) == {"model": "q:1b", "stream": True}


def test_pull_that_dies_mid_stream_is_friendly() -> None:
    """The read timeout fires while iterating, long after the opener returned."""
    exploding = FakeResponse(fails=TimeoutError("timed out"))
    with pytest.raises(FriendlyError) as exc:
        engine.pull_model(
            "http://x",
            ModelRef(raw="qwen3:8b", kind="ollama"),
            sys=answering({"/api/pull": exploding}),
            progress=lambda done, total: None,
        )
    assert "qwen3:8b" in exc.value.problem


def test_pull_with_a_malformed_line_is_friendly() -> None:
    """A truncated stream ends mid-JSON; that must not surface as a JSONDecodeError."""
    with pytest.raises(FriendlyError) as exc:
        engine.pull_model(
            "http://x",
            ModelRef(raw="q:1b", kind="ollama"),
            sys=serving("/api/pull", b'{"status": "pulling ma'),
        )
    assert "q:1b" in exc.value.problem


@pytest.mark.parametrize("payload", [b"[]", b'{"models": [{"size": 1}]}', b"not json"])
def test_list_models_with_an_unreadable_reply_is_friendly(payload: bytes) -> None:
    """Something is answering, but it is not an Ollama /api/tags — never a traceback."""
    with pytest.raises(FriendlyError) as exc:
        engine.list_models("http://x", sys=serving("/api/tags", payload))
    assert "lepika doctor" in exc.value.fix


def rejecting(code: int) -> FakeSystem:
    """An engine that answers every request with `code`."""

    def refuse(req: Any) -> Any:
        raise urllib.error.HTTPError(req.full_url, code, "Unauthorized", {}, None)  # type: ignore[arg-type]

    return FakeSystem(urls=refuse)


CALLS: list[tuple[str, Any]] = [
    ("list_models", lambda sys: engine.list_models("http://gpu-box:11435", sys=sys)),
    (
        "delete_model",
        lambda sys: engine.delete_model("http://gpu-box:11435", "q:1b", sys=sys),
    ),
    (
        "pull_model",
        lambda sys: engine.pull_model(
            "http://gpu-box:11435", ModelRef(raw="q:1b", kind="ollama"), sys=sys
        ),
    ),
]


@pytest.mark.parametrize("name,call", CALLS, ids=[c[0] for c in CALLS])
@pytest.mark.parametrize("code", [401, 403])
def test_a_rejected_key_is_named_as_such(name: str, call: Any, code: int) -> None:
    """`lepika up` cannot fix someone else's engine saying no to our key."""
    with pytest.raises(FriendlyError) as exc:
        call(rejecting(code))
    assert exc.value.problem == "The engine at http://gpu-box:11435 rejected the API key."
    assert "lepika connect http://gpu-box:11435 --key <key>" in exc.value.fix


@pytest.mark.parametrize("name,call", CALLS, ids=[c[0] for c in CALLS])
def test_a_server_error_is_still_unreachable(name: str, call: Any) -> None:
    with pytest.raises(FriendlyError) as exc:
        call(rejecting(500))
    assert "Could not reach the engine" in exc.value.problem


def test_the_pull_bar_is_disabled_when_stdout_is_not_a_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-tty run got one frozen '0/549 bytes' line; the JSON log is the record."""
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    with engine._bar(ModelRef(raw="q:1b", kind="ollama"), None) as report:
        report(40, 100)
    assert out.getvalue() == ""


def test_speed_column_is_blank_until_a_speed_is_known() -> None:
    """A cached model completes in one event; rich would print '489/489 bytes ?'."""
    bar = Progress(disable=True)
    task_id = bar.add_task("q:1b", total=None)
    bar.update(task_id, completed=489, total=489)
    assert engine._SpeedColumn().render(bar.tasks[0]).plain == ""


def test_speed_column_shows_a_rate_once_bytes_flow() -> None:
    clock = itertools.count(0.0, 0.5)
    bar = Progress(disable=True, get_time=lambda: next(clock))
    task_id = bar.add_task("q:1b", total=1000)
    bar.update(task_id, completed=100)
    bar.update(task_id, completed=300)
    assert engine._SpeedColumn().render(bar.tasks[0]).plain.endswith("/s")


def test_human_size() -> None:
    assert engine.human_size(512) == "512 B"
    assert engine.human_size(4_700_000_000) == "4.7 GB"


def test_pull_of_full_weight_hf_repo_says_use_a_gguf_build() -> None:
    # Ollama's exact refusal for a safetensors repo (e.g. hf.co/Qwen/Qwen3.8-27B):
    # the URL is valid and the network is fine, so the generic hint would mislead.
    refusal = '{"error":"Repository is not GGUF or is not compatible with llama.cpp"}'
    body = json.dumps({"error": f"pull model manifest: 400: {refusal}"}).encode()
    with pytest.raises(engine.NotGGUF) as exc:
        engine.pull_model(
            "http://x",
            ModelRef(raw="hf.co/Qwen/Qwen3.8-27B", kind="hf_gguf"),
            sys=serving("/api/pull", body),
        )
    assert "full weights" in exc.value.problem
    assert "Qwen3.8-27B-GGUF" in exc.value.fix
    assert "internet" not in exc.value.fix
    assert isinstance(exc.value, FriendlyError)


def test_version_reads_api_version() -> None:
    body = json.dumps({"version": "0.33.0"}).encode()
    assert engine.version("http://x", sys=serving("/api/version", body)) == "0.33.0"


def _runner_failure(url: str) -> urllib.error.HTTPError:
    """Ollama's 500 when the MLX runner cannot load, with its JSON error body."""
    detail = "mlx runner failed: Error: MLX not available: failed to load MLX dynamic library"
    return urllib.error.HTTPError(
        url,
        500,
        "Internal Server Error",
        {},  # type: ignore[arg-type]
        io.BytesIO(json.dumps({"error": detail}).encode()),
    )


def test_load_model_posts_generate_and_maps_a_runner_failure() -> None:
    ok = serving("/api/generate", json.dumps({"done": True}).encode())
    engine.load_model("http://x", "Qwen/Qwen3.5-2B", sys=ok)
    seen = ok.requested
    assert seen[0].full_url == "http://x/api/generate"
    assert json.loads(seen[0].data)["model"] == "Qwen/Qwen3.5-2B"

    mlx_missing = answering({"/api/generate": _runner_failure("http://x/api/generate")})
    with pytest.raises(FriendlyError) as exc:
        engine.load_model("http://x", "Qwen/Qwen3.5-2B", sys=mlx_missing)
    assert "MLX engine" in exc.value.problem


def test_show_returns_capabilities_and_template() -> None:
    body = {"capabilities": ["completion", "thinking"], "template": "<|im_start|>{{ x }}"}
    sys = serving("/api/show", json.dumps(body).encode())
    assert engine.show("http://x", "o/r", sys=sys) == (
        ["completion", "thinking"],
        "<|im_start|>{{ x }}",
    )
    req = sys.requested[0]
    assert req.full_url.endswith("/api/show")
    assert json.loads(req.data)["model"] == "o/r"


def test_show_missing_fields_read_as_empty() -> None:
    assert engine.show("http://x", "o/r", sys=serving("/api/show", b"{}")) == ([], "")


def test_show_failure_reads_as_none() -> None:
    """Diagnostics must never break the pull that just succeeded."""
    down = answering({"/api/show": urllib.error.URLError("down")})
    assert engine.show("http://x", "o/r", sys=down) is None


def test_chatml_tools_template_ships_in_the_package() -> None:
    text = engine.chatml_tools_template()
    assert ".Tools" in text and ".ToolCalls" in text
    assert "<|im_start|>" in text


def test_chatml_tools_template_missing_from_the_install_is_friendly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken install must not surface as a raw traceback: `_ensure_tools` runs
    after a pull already succeeded and only knows how to swallow a FriendlyError."""

    class _Missing:
        def joinpath(self, _name: str) -> _Missing:
            return self

        def read_text(self, encoding: str = "utf-8") -> str:
            raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(importlib.resources, "files", lambda _pkg: _Missing())
    with pytest.raises(FriendlyError) as excinfo:
        engine.chatml_tools_template()
    assert excinfo.value.fix


def test_retemplate_rebuilds_the_model_with_the_given_template() -> None:
    seen: dict[str, Any] = {}

    def look(cmd: list[str], cwd: Path | None) -> None:
        # Read while the staging TemporaryDirectory is alive — gone once it returns.
        assert cwd is not None
        seen["modelfile"] = (cwd / "Modelfile").read_text()

    sys = FakeSystem(while_streaming=look)
    engine.retemplate("http://x", "hf.co/o/r-GGUF:Q4", "TPL {{ .Tools }}", sys=sys)
    seen["cmd"], kwargs = sys.streamed[0]
    seen["env"] = kwargs["env"]
    assert seen["cmd"] == ["ollama", "create", "hf.co/o/r-GGUF:Q4"]
    assert seen["env"]["OLLAMA_HOST"] == "http://x"
    assert seen["modelfile"].startswith("FROM hf.co/o/r-GGUF:Q4\n")
    assert 'TEMPLATE """TPL {{ .Tools }}"""' in seen["modelfile"]


def test_retemplate_failure_is_friendly() -> None:
    with pytest.raises(FriendlyError):
        engine.retemplate("http://x", "o/r", "TPL", sys=FakeSystem(stream=(1, "boom")))


def _versioned(text: str, stream: tuple[int, str] = (0, "")) -> FakeSystem:
    """An engine answering /api/version with `text` and /api/generate with success;
    `ollama create` exits with `stream`'s code and tail."""
    return answering(
        {
            "/api/version": FakeResponse(json.dumps({"version": text}).encode()),
            "/api/generate": FakeResponse(json.dumps({"done": True}).encode()),
        },
        stream=stream,
    )


def test_import_model_writes_a_modelfile_and_runs_ollama_create(tmp_path: Path) -> None:
    sys = _versioned("0.33.0")
    engine.import_model("http://x", "Qwen/Qwen3.5-2B", tmp_path, sys=sys)
    assert (tmp_path / "Modelfile").read_text() == "FROM .\n"
    cmd, kwargs = sys.streamed[0]
    assert cmd == ["ollama", "create", "Qwen/Qwen3.5-2B", "--experimental", "-q", "nvfp4"]
    assert kwargs["cwd"] == tmp_path
    assert kwargs["env"]["OLLAMA_HOST"] == "http://x"
    entry = json.loads((paths.logs_dir() / "lepika.log").read_text().splitlines()[-1])
    assert entry["event"] == "engine.import"
    assert entry["result"] == "success"


def test_import_model_logs_and_raises_when_the_imported_model_will_not_load(
    tmp_path: Path,
) -> None:
    """`ollama create` succeeded but the artifact will not run: still one import line."""
    sys = answering(
        {
            "/api/version": FakeResponse(json.dumps({"version": "0.33.0"}).encode()),
            "/api/generate": _runner_failure("http://x/api/generate"),
        }
    )
    with pytest.raises(FriendlyError) as exc:
        engine.import_model("http://x", "o/r", tmp_path, sys=sys)
    assert "MLX engine" in exc.value.problem
    assert len(sys.streamed) == 1
    entry = json.loads((paths.logs_dir() / "lepika.log").read_text().splitlines()[-1])
    assert entry["event"] == "engine.import"
    assert "MLX" in entry["result"]


def test_import_model_not_owned_never_writes_into_the_source(tmp_path: Path) -> None:
    """A folder the user owns is read, not staged: the Modelfile lives in a temp cwd."""
    seen: dict[str, Any] = {}

    def look(cmd: list[str], cwd: Path | None) -> None:
        assert cwd is not None
        # Read while the TemporaryDirectory is alive — it is gone once import_model returns.
        seen["cwd"] = cwd
        seen["modelfile"] = (cwd / "Modelfile").read_text()

    sys = _versioned("0.33.0")
    sys.while_streaming = look
    engine.import_model("http://x", "local-model", tmp_path, owned=False, sys=sys)
    assert not (tmp_path / "Modelfile").exists()
    assert seen["cwd"] != tmp_path
    assert seen["modelfile"] == f"FROM {tmp_path.resolve()}\n"


def test_import_model_logs_where_the_weights_came_from(tmp_path: Path) -> None:
    engine.import_model("http://x", "o/r", tmp_path, sys=_versioned("0.33.0"))
    entry = json.loads((paths.logs_dir() / "lepika.log").read_text().splitlines()[-1])
    assert entry["source"] == str(tmp_path)


def test_version_tuple_pads_a_short_version() -> None:
    """`0.32` means 0.32.0; unpadded it sorts below the floor and would be refused."""
    assert engine._version_tuple("0.32") == (0, 32, 0)


def test_import_model_accepts_a_two_component_version(tmp_path: Path) -> None:
    sys = _versioned("0.32")
    engine.import_model("http://x", "o/r", tmp_path, sys=sys)
    assert len(sys.streamed) == 1


def test_import_model_quantizes_with_the_quant_it_is_given(tmp_path: Path) -> None:
    sys = _versioned("0.33.0")
    engine.import_model("http://x", "o/r", tmp_path, quant="int4", sys=sys)
    assert sys.streamed[0][0][-2:] == ["-q", "int4"]


def test_import_model_without_a_quant_imports_the_source_as_is(tmp_path: Path) -> None:
    """A pre-quantized checkpoint: Ollama refuses `-q` on it but imports it bare."""
    sys = _versioned("0.33.0")
    engine.import_model("http://x", "o/r", tmp_path, quant=None, sys=sys)
    assert sys.streamed[0][0] == ["ollama", "create", "o/r", "--experimental"]


def test_import_model_refuses_an_old_ollama(tmp_path: Path) -> None:
    sys = _versioned("0.31.2")
    with pytest.raises(FriendlyError) as exc:
        engine.import_model("http://x", "o/r", tmp_path, sys=sys)
    assert "0.32" in exc.value.problem
    assert "lepika update" in exc.value.fix
    assert sys.streamed == []


@pytest.mark.parametrize(
    ("tail", "needle"),
    [
        ('Error: unsupported architecture "FooForCausalLM"', "architecture"),
        ("Error: /w is not a supported safetensors model directory", "architecture"),
        ("Error: something else", "Import of o/r failed"),
    ],
)
def test_import_model_maps_create_failures(tmp_path: Path, tail: str, needle: str) -> None:
    with pytest.raises(FriendlyError) as exc:
        engine.import_model("http://x", "o/r", tmp_path, sys=_versioned("0.33.0", (1, tail)))
    assert needle in exc.value.problem


@pytest.mark.parametrize(
    ("owned", "kept"),
    [(True, True), (False, False)],
)
def test_import_model_only_promises_a_kept_download_for_its_own_staging(
    tmp_path: Path, owned: bool, kept: bool
) -> None:
    """`~/.lepika/hf` is kept for a retry; the user's own folder was never downloaded."""
    failing = _versioned("0.33.0", (1, "Error: something else"))
    with pytest.raises(FriendlyError) as exc:
        engine.import_model("http://x", "o/r", tmp_path, owned=owned, sys=failing)
    assert ("the download is kept" in exc.value.fix) is kept
    assert "Run the same command again" in exc.value.fix
