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
from fakes import Streamer  # shared with tests/test_hf.py
from rich.progress import Progress

from lepika import engine, paths
from lepika.errors import FriendlyError
from lepika.models import ModelRef


class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, status: int = 200) -> None:
        super().__init__(body)
        self.status = status

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *a: Any) -> None:
        self.close()


class ExplodingResponse(io.BytesIO):
    """Opens fine, then dies mid-stream — a read timeout or a dropped connection."""

    def __init__(self, error: Exception) -> None:
        super().__init__(b"")
        self.error = error

    def __enter__(self) -> ExplodingResponse:
        return self

    def __exit__(self, *a: Any) -> None:
        self.close()

    def __iter__(self) -> Any:
        raise self.error


def opener_returning(body: bytes, seen: list[Any] | None = None) -> Any:
    def opener(req: Any, timeout: float = 0) -> FakeResponse:
        if seen is not None:
            seen.append(req)
        return FakeResponse(body)

    return opener


def test_list_models_parses_tags() -> None:
    body = json.dumps({"models": [{"name": "qwen3:8b", "size": 5_000_000_000}]}).encode()
    assert engine.list_models("http://x", urlopen=opener_returning(body)) == [
        ("qwen3:8b", 5_000_000_000)
    ]


def test_list_models_sends_bearer_key() -> None:
    seen: list[Any] = []
    engine.list_models("http://x", key="k", urlopen=opener_returning(b'{"models": []}', seen))
    assert seen[0].full_url == "http://x/api/tags"
    assert seen[0].get_header("Authorization") == "Bearer k"


def test_list_models_unreachable_is_friendly() -> None:
    def boom(req: Any, timeout: float = 0) -> Any:
        raise OSError("refused")

    with pytest.raises(FriendlyError) as exc:
        engine.list_models("http://x", urlopen=boom)
    assert "lepika doctor" in exc.value.fix


def test_an_unreachable_remote_engine_is_not_told_to_run_lepika_up() -> None:
    """`lepika up` never starts someone else's engine (rule 9), so that hint is wrong."""

    def down(request: Any, timeout: float = 0) -> Any:
        raise urllib.error.URLError("refused")

    with pytest.raises(FriendlyError) as exc:
        engine.list_models("http://gpu-box:11435", key="k", urlopen=down, managed=False)
    assert "gpu-box" in exc.value.problem
    assert "lepika up" not in exc.value.fix
    assert "lepika connect http://gpu-box:11435 --key" in exc.value.fix
    assert "lepika connect --local" in exc.value.fix


def test_an_unreachable_managed_engine_still_says_lepika_up() -> None:
    def down(request: Any, timeout: float = 0) -> Any:
        raise urllib.error.URLError("refused")

    with pytest.raises(FriendlyError) as exc:
        engine.list_models("http://127.0.0.1:11434", urlopen=down)
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
    seen: list[Any] = []
    engine.delete_model("http://x", "qwen3:8b", urlopen=opener_returning(b"", seen))
    req = seen[0]
    assert req.get_method() == "DELETE"
    assert req.full_url == "http://x/api/delete"
    assert json.loads(req.data) == {"model": "qwen3:8b"}


def test_delete_missing_model_is_friendly() -> None:
    import urllib.error

    def not_found(req: Any, timeout: float = 0) -> Any:
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)  # type: ignore[arg-type]

    with pytest.raises(FriendlyError) as exc:
        engine.delete_model("http://x", "nope:1b", urlopen=not_found)
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
        urlopen=opener_returning(body),
        progress=lambda done, total: ticks.append((done, total)),
    )
    assert ticks == [(40, 100), (100, 100)]


def test_pull_error_line_is_friendly() -> None:
    body = json.dumps({"error": "pull model manifest: file does not exist"}).encode()
    with pytest.raises(FriendlyError) as exc:
        engine.pull_model(
            "http://x", ModelRef(raw="nope:1b", kind="ollama"), urlopen=opener_returning(body)
        )
    assert "nope:1b" in exc.value.problem
    assert "GGUF" in exc.value.fix


def test_pull_that_ends_without_success_is_friendly() -> None:
    body = json.dumps({"status": "pulling manifest"}).encode()
    with pytest.raises(FriendlyError):
        engine.pull_model(
            "http://x", ModelRef(raw="q:1b", kind="ollama"), urlopen=opener_returning(body)
        )


def test_pull_sends_stream_true_and_timeout_is_generous() -> None:
    seen: list[Any] = []
    body = json.dumps({"status": "success"}).encode()
    engine.pull_model(
        "http://x", ModelRef(raw="q:1b", kind="ollama"), urlopen=opener_returning(body, seen)
    )
    assert json.loads(seen[0].data) == {"model": "q:1b", "stream": True}


def test_pull_that_dies_mid_stream_is_friendly() -> None:
    """The read timeout fires while iterating, long after the opener returned."""

    def opener(req: Any, timeout: float = 0) -> Any:
        return ExplodingResponse(TimeoutError("timed out"))

    with pytest.raises(FriendlyError) as exc:
        engine.pull_model(
            "http://x",
            ModelRef(raw="qwen3:8b", kind="ollama"),
            urlopen=opener,
            progress=lambda done, total: None,
        )
    assert "qwen3:8b" in exc.value.problem


def test_pull_with_a_malformed_line_is_friendly() -> None:
    """A truncated stream ends mid-JSON; that must not surface as a JSONDecodeError."""
    with pytest.raises(FriendlyError) as exc:
        engine.pull_model(
            "http://x",
            ModelRef(raw="q:1b", kind="ollama"),
            urlopen=opener_returning(b'{"status": "pulling ma'),
        )
    assert "q:1b" in exc.value.problem


@pytest.mark.parametrize("payload", [b"[]", b'{"models": [{"size": 1}]}', b"not json"])
def test_list_models_with_an_unreadable_reply_is_friendly(payload: bytes) -> None:
    """Something is answering, but it is not an Ollama /api/tags — never a traceback."""
    with pytest.raises(FriendlyError) as exc:
        engine.list_models("http://x", urlopen=opener_returning(payload))
    assert "lepika doctor" in exc.value.fix


def rejecting(code: int) -> Any:
    def opener(req: Any, timeout: float = 0) -> Any:
        raise urllib.error.HTTPError(req.full_url, code, "Unauthorized", {}, None)  # type: ignore[arg-type]

    return opener


CALLS: list[tuple[str, Any]] = [
    ("list_models", lambda opener: engine.list_models("http://gpu-box:11435", urlopen=opener)),
    (
        "delete_model",
        lambda opener: engine.delete_model("http://gpu-box:11435", "q:1b", urlopen=opener),
    ),
    (
        "pull_model",
        lambda opener: engine.pull_model(
            "http://gpu-box:11435", ModelRef(raw="q:1b", kind="ollama"), urlopen=opener
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
            urlopen=opener_returning(body),
        )
    assert "full weights" in exc.value.problem
    assert "Qwen3.8-27B-GGUF" in exc.value.fix
    assert "internet" not in exc.value.fix
    assert isinstance(exc.value, FriendlyError)


def test_version_reads_api_version() -> None:
    body = json.dumps({"version": "0.33.0"}).encode()
    assert engine.version("http://x", urlopen=opener_returning(body)) == "0.33.0"


def test_load_model_posts_generate_and_maps_a_runner_failure() -> None:
    seen: list[Any] = []

    def ok(req: Any, timeout: float = 0) -> Any:
        seen.append(req)
        return FakeResponse(json.dumps({"done": True}).encode())

    engine.load_model("http://x", "Qwen/Qwen3.5-2B", urlopen=ok)
    assert seen[0].full_url == "http://x/api/generate"
    assert json.loads(seen[0].data)["model"] == "Qwen/Qwen3.5-2B"

    def mlx_missing(req: Any, timeout: float = 0) -> Any:
        detail = "mlx runner failed: Error: MLX not available: failed to load MLX dynamic library"
        body = json.dumps({"error": detail}).encode()
        raise urllib.error.HTTPError(
            req.full_url,
            500,
            "Internal Server Error",
            {},  # type: ignore[arg-type]
            io.BytesIO(body),
        )

    with pytest.raises(FriendlyError) as exc:
        engine.load_model("http://x", "Qwen/Qwen3.5-2B", urlopen=mlx_missing)
    assert "MLX engine" in exc.value.problem


def test_show_returns_capabilities_and_template() -> None:
    def opener(req: Any, timeout: float = 0) -> Any:
        assert req.full_url.endswith("/api/show")
        assert json.loads(req.data)["model"] == "o/r"
        body = {"capabilities": ["completion", "thinking"], "template": "<|im_start|>{{ x }}"}
        return FakeResponse(json.dumps(body).encode())

    assert engine.show("http://x", "o/r", urlopen=opener) == (
        ["completion", "thinking"],
        "<|im_start|>{{ x }}",
    )


def test_show_missing_fields_read_as_empty() -> None:
    opener = lambda req, timeout=0: FakeResponse(b"{}")  # noqa: E731
    assert engine.show("http://x", "o/r", urlopen=opener) == ([], "")


def test_show_failure_reads_as_none() -> None:
    """Diagnostics must never break the pull that just succeeded."""

    def opener(req: Any, timeout: float = 0) -> Any:
        raise urllib.error.URLError("down")

    assert engine.show("http://x", "o/r", urlopen=opener) is None


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

    def stream(cmd: list[str], **kwargs: Any) -> tuple[int, str]:
        # Read while the staging TemporaryDirectory is alive — gone once it returns.
        seen["cmd"] = list(cmd)
        seen["env"] = kwargs["env"]
        seen["modelfile"] = (Path(kwargs["cwd"]) / "Modelfile").read_text()
        return 0, ""

    engine.retemplate(
        "http://x", "hf.co/o/r-GGUF:Q4", "TPL {{ .Tools }}", stream=stream, environ={}
    )
    assert seen["cmd"] == ["ollama", "create", "hf.co/o/r-GGUF:Q4"]
    assert seen["env"]["OLLAMA_HOST"] == "http://x"
    assert seen["modelfile"].startswith("FROM hf.co/o/r-GGUF:Q4\n")
    assert 'TEMPLATE """TPL {{ .Tools }}"""' in seen["modelfile"]


def test_retemplate_failure_is_friendly() -> None:
    with pytest.raises(FriendlyError):
        engine.retemplate(
            "http://x", "o/r", "TPL", stream=Streamer(code=1, tail="boom"), environ={}
        )


def _versioned(text: str) -> Any:
    """An opener answering /api/version with `text` and /api/generate with success."""

    def opener(req: Any, timeout: float = 0) -> Any:
        if req.full_url.endswith("/api/version"):
            return FakeResponse(json.dumps({"version": text}).encode())
        return FakeResponse(json.dumps({"done": True}).encode())

    return opener


def test_import_model_writes_a_modelfile_and_runs_ollama_create(tmp_path: Path) -> None:
    stream = Streamer()
    engine.import_model(
        "http://x",
        "Qwen/Qwen3.5-2B",
        tmp_path,
        stream=stream,
        urlopen=_versioned("0.33.0"),
        environ={},
    )
    assert (tmp_path / "Modelfile").read_text() == "FROM .\n"
    cmd, kwargs = stream.calls[0]
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

    def opener(req: Any, timeout: float = 0) -> Any:
        if req.full_url.endswith("/api/version"):
            return FakeResponse(json.dumps({"version": "0.33.0"}).encode())
        detail = "mlx runner failed: Error: MLX not available: failed to load MLX dynamic library"
        raise urllib.error.HTTPError(
            req.full_url,
            500,
            "Internal Server Error",
            {},  # type: ignore[arg-type]
            io.BytesIO(json.dumps({"error": detail}).encode()),
        )

    stream = Streamer()
    with pytest.raises(FriendlyError) as exc:
        engine.import_model(
            "http://x",
            "o/r",
            tmp_path,
            stream=stream,
            urlopen=opener,
            environ={},
        )
    assert "MLX engine" in exc.value.problem
    assert len(stream.calls) == 1
    entry = json.loads((paths.logs_dir() / "lepika.log").read_text().splitlines()[-1])
    assert entry["event"] == "engine.import"
    assert "MLX" in entry["result"]


def test_import_model_not_owned_never_writes_into_the_source(tmp_path: Path) -> None:
    """A folder the user owns is read, not staged: the Modelfile lives in a temp cwd."""
    seen: dict[str, Any] = {}

    def stream(cmd: list[str], **kwargs: Any) -> tuple[int, str]:
        cwd = Path(kwargs["cwd"])
        # Read while the TemporaryDirectory is alive — it is gone once import_model returns.
        seen["cwd"] = cwd
        seen["modelfile"] = (cwd / "Modelfile").read_text()
        return 0, ""

    engine.import_model(
        "http://x",
        "local-model",
        tmp_path,
        owned=False,
        stream=stream,
        urlopen=_versioned("0.33.0"),
        environ={},
    )
    assert not (tmp_path / "Modelfile").exists()
    assert seen["cwd"] != tmp_path
    assert seen["modelfile"] == f"FROM {tmp_path.resolve()}\n"


def test_import_model_logs_where_the_weights_came_from(tmp_path: Path) -> None:
    engine.import_model(
        "http://x",
        "o/r",
        tmp_path,
        stream=Streamer(),
        urlopen=_versioned("0.33.0"),
        environ={},
    )
    entry = json.loads((paths.logs_dir() / "lepika.log").read_text().splitlines()[-1])
    assert entry["source"] == str(tmp_path)


def test_version_tuple_pads_a_short_version() -> None:
    """`0.32` means 0.32.0; unpadded it sorts below the floor and would be refused."""
    assert engine._version_tuple("0.32") == (0, 32, 0)


def test_import_model_accepts_a_two_component_version(tmp_path: Path) -> None:
    stream = Streamer()
    engine.import_model(
        "http://x",
        "o/r",
        tmp_path,
        stream=stream,
        urlopen=_versioned("0.32"),
        environ={},
    )
    assert len(stream.calls) == 1


def test_import_model_quantizes_with_the_quant_it_is_given(tmp_path: Path) -> None:
    stream = Streamer()
    engine.import_model(
        "http://x",
        "o/r",
        tmp_path,
        quant="int4",
        stream=stream,
        urlopen=_versioned("0.33.0"),
        environ={},
    )
    assert stream.calls[0][0][-2:] == ["-q", "int4"]


def test_import_model_without_a_quant_imports_the_source_as_is(tmp_path: Path) -> None:
    """A pre-quantized checkpoint: Ollama refuses `-q` on it but imports it bare."""
    stream = Streamer()
    engine.import_model(
        "http://x",
        "o/r",
        tmp_path,
        quant=None,
        stream=stream,
        urlopen=_versioned("0.33.0"),
        environ={},
    )
    assert stream.calls[0][0] == ["ollama", "create", "o/r", "--experimental"]


def test_import_model_refuses_an_old_ollama(tmp_path: Path) -> None:
    stream = Streamer()
    with pytest.raises(FriendlyError) as exc:
        engine.import_model(
            "http://x",
            "o/r",
            tmp_path,
            stream=stream,
            urlopen=_versioned("0.31.2"),
            environ={},
        )
    assert "0.32" in exc.value.problem
    assert "lepika update" in exc.value.fix
    assert stream.calls == []


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
        engine.import_model(
            "http://x",
            "o/r",
            tmp_path,
            stream=Streamer(code=1, tail=tail),
            urlopen=_versioned("0.33.0"),
            environ={},
        )
    assert needle in exc.value.problem


@pytest.mark.parametrize(
    ("owned", "kept"),
    [(True, True), (False, False)],
)
def test_import_model_only_promises_a_kept_download_for_its_own_staging(
    tmp_path: Path, owned: bool, kept: bool
) -> None:
    """`~/.lepika/hf` is kept for a retry; the user's own folder was never downloaded."""
    with pytest.raises(FriendlyError) as exc:
        engine.import_model(
            "http://x",
            "o/r",
            tmp_path,
            owned=owned,
            stream=Streamer(code=1, tail="Error: something else"),
            urlopen=_versioned("0.33.0"),
            environ={},
        )
    assert ("the download is kept" in exc.value.fix) is kept
    assert "Run the same command again" in exc.value.fix
