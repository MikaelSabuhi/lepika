from __future__ import annotations

import io
import json
import sys
import urllib.error
from typing import Any

import pytest

from lepika import engine
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


def test_human_size() -> None:
    assert engine.human_size(512) == "512 B"
    assert engine.human_size(4_700_000_000) == "4.7 GB"
