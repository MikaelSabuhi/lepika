"""Talk to an Ollama engine over HTTP — local, containerised, or remote with a key."""

from __future__ import annotations

import contextlib
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from typing import Any

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    Task,
    TextColumn,
    TransferSpeedColumn,
)
from rich.text import Text

from lepika import log
from lepika.errors import FriendlyError
from lepika.models import ModelRef

UrlOpenFn = Callable[..., Any]
ProgressFn = Callable[[int, int], None]

_TAGS_TIMEOUT = 5.0
_HEALTH_TIMEOUT = 1.0
_DELETE_TIMEOUT = 30.0
# A pull streams for as long as the download takes; only silence is a failure.
_PULL_IDLE_TIMEOUT = 300.0


def _request(
    url: str,
    path: str,
    key: str,
    method: str = "GET",
    body: dict[str, Any] | None = None,
) -> urllib.request.Request:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(f"{url}{path}", data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    if key:
        request.add_header("Authorization", f"Bearer {key}")
    return request


def _opener(urlopen: UrlOpenFn | None) -> UrlOpenFn:
    return urlopen if urlopen is not None else urllib.request.urlopen


def _unreachable(url: str, managed: bool = True) -> FriendlyError:
    if managed:
        fix = "Run `lepika up` to start it, or `lepika doctor` to see what's wrong."
    else:
        # `lepika up` never starts someone else's engine (rule 9): the only moves
        # from here are checking that box, reconnecting, or going local again.
        fix = (
            "Check that machine is up and `lepika expose` is still on there, then "
            f"`lepika connect {url} --key <key>` again — or `lepika connect --local`."
        )
    return FriendlyError(f"Could not reach the engine at {url}.", fix)


def _rejected(url: str) -> FriendlyError:
    """A 401/403 is a key the engine refuses — `lepika up` cannot fix someone else's box."""
    return FriendlyError(
        f"The engine at {url} rejected the API key.",
        "Run `lepika expose --show` on that machine, then "
        f"`lepika connect {url} --key <key>` here.",
    )


def human_size(n_bytes: int) -> str:
    size = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1000 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1000
    return f"{size:.1f} GB"  # pragma: no cover - the loop always returns


def _canonical(name: str) -> str:
    """`qwen3` → `qwen3:latest`: the tag Ollama fills in when the ref has none.

    The tag is the `:` after the last `/`, so `hf.co/org/repo-GGUF` gets one and a
    port in a hostname never counts as one.
    """
    _, _, tail = name.rpartition("/")
    if ":" in tail:
        return name
    return f"{name}:latest"


def same_model(a: str, b: str) -> bool:
    """Do two refs name the same Ollama model, tag or no tag?"""
    return _canonical(a) == _canonical(b)


def vllm_up(url: str, urlopen: UrlOpenFn | None = None) -> bool:
    """Is the vLLM server answering? `GET /health` — a probe, so nothing is logged."""
    try:
        _opener(urlopen)(urllib.request.Request(f"{url}/health"), timeout=_HEALTH_TIMEOUT)
    except Exception:
        return False
    return True


def list_models(
    url: str, key: str = "", managed: bool = True, urlopen: UrlOpenFn | None = None
) -> list[tuple[str, int]]:
    """Every installed model as `(name, size_bytes)`, from `GET /api/tags`."""
    # Parsing sits inside the guard on purpose: something that answers on the port
    # without speaking Ollama is a failure to reach the engine, not a traceback.
    try:
        with _opener(urlopen)(_request(url, "/api/tags", key), timeout=_TAGS_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return [(str(m["name"]), int(m.get("size", 0))) for m in payload["models"]]
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise _rejected(url) from exc
        raise _unreachable(url, managed) from exc
    except Exception as exc:
        raise _unreachable(url, managed) from exc


def delete_model(
    url: str, name: str, key: str = "", managed: bool = True, urlopen: UrlOpenFn | None = None
) -> None:
    """Remove one model from the engine via `DELETE /api/delete`."""
    request = _request(url, "/api/delete", key, "DELETE", {"model": name})
    try:
        with _opener(urlopen)(request, timeout=_DELETE_TIMEOUT):
            pass
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise FriendlyError(
                f"Model '{name}' is not installed.", "See `lepika model list` for what is."
            ) from exc
        if exc.code in (401, 403):
            raise _rejected(url) from exc
        raise _unreachable(url, managed) from exc
    except Exception as exc:
        raise _unreachable(url, managed) from exc
    log.get_logger().info("engine.delete", model=name, url=url)


def _pull_failed(ref: ModelRef, detail: str) -> FriendlyError:
    if "not gguf" in detail.lower():
        # Ollama's refusal for a safetensors repo (`hf.co/Qwen/Qwen3.8-27B`): the
        # URL is right and the network is fine, so the generic hint would send the
        # user checking both. What they need is the GGUF build of the same model.
        repo = ref.raw.rpartition("/")[2].partition(":")[0]
        return FriendlyError(
            f"'{ref.raw}' ships full weights (safetensors), which Ollama cannot run.",
            f"Use a GGUF build of it instead — look for {repo}-GGUF on Hugging Face "
            f"and pass that, e.g. hf.co/<org>/{repo}-GGUF.",
        )
    return FriendlyError(
        f"Failed to pull model '{ref.raw}': {detail}",
        "Check the model name/URL — e.g. qwen3:8b or hf.co/<org>/<repo>-GGUF — "
        "and your internet connection.",
    )


def pull_model(
    url: str,
    ref: ModelRef,
    key: str = "",
    managed: bool = True,
    urlopen: UrlOpenFn | None = None,
    progress: ProgressFn | None = None,
) -> None:
    """Stream `POST /api/pull`, showing Ollama's per-layer progress as one bar."""
    logger = log.get_logger()
    logger.info("engine.pull", model=ref.raw, url=url)
    request = _request(url, "/api/pull", key, "POST", {"model": ref.raw, "stream": True})
    try:
        response = _opener(urlopen)(request, timeout=_PULL_IDLE_TIMEOUT)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise _rejected(url) from exc
        raise _unreachable(url, managed) from exc
    except Exception as exc:
        raise _unreachable(url, managed) from exc
    finished = False
    # The whole stream is guarded, not just the opener: the timeout above governs
    # every subsequent read, so a stall, a dropped connection or a truncated JSON
    # line all surface here — mid-download, long after the request succeeded.
    try:
        with response, _bar(ref, progress) as report:
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line:
                    continue
                event = json.loads(line)
                if "error" in event:
                    raise _pull_failed(ref, str(event["error"]))
                if "total" in event:
                    report(int(event.get("completed", 0)), int(event["total"]))
                if event.get("status") == "success":
                    finished = True
    except FriendlyError as exc:
        # The engine itself said no; its own message is already the detail.
        logger.warning("engine.pull", model=ref.raw, result=exc.problem)
        raise
    except Exception as exc:
        logger.warning("engine.pull", model=ref.raw, result=f"{type(exc).__name__}: {exc}")
        raise _pull_failed(ref, "the connection dropped mid-download") from exc
    if not finished:
        logger.warning("engine.pull", model=ref.raw, result="stream ended early")
        raise _pull_failed(ref, "the download ended before it completed")
    logger.info("engine.pull", model=ref.raw, result="success")


class _SpeedColumn(TransferSpeedColumn):
    """`TransferSpeedColumn` that stays blank while no speed has been measured.

    A model already on disk arrives as one `completed == total` event, and rich
    renders the unmeasured speed as a bare `?` after the byte count.
    """

    def render(self, task: Task) -> Text:
        if (task.finished_speed or task.speed) is None:
            return Text("")
        return super().render(task)


@contextlib.contextmanager
def _bar(ref: ModelRef, progress: ProgressFn | None) -> Iterator[ProgressFn]:
    """Yield a `(completed, total)` reporter: the injected one, or a rich bar.

    Ollama reports `completed`/`total` per layer digest, so the single bar tracks
    the layer being downloaded — the same thing `ollama pull` shows.
    """
    if progress is not None:
        yield progress
        return
    bar = Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        _SpeedColumn(),
        # Piped or redirected, a live bar renders as one frozen "0/549 bytes" line
        # that never updates. The JSON log already records the pull.
        disable=not sys.stdout.isatty(),
    )
    bar.start()
    task = bar.add_task(ref.raw, total=None)

    def report(completed: int, total: int) -> None:
        bar.update(task, completed=completed, total=total)

    try:
        yield report
    finally:
        bar.stop()
