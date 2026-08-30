"""Talk to an Ollama engine over HTTP — local, containerised, or remote with a key."""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
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

from lepika import log, proc
from lepika.errors import FriendlyError
from lepika.models import ModelRef

UrlOpenFn = Callable[..., Any]
ProgressFn = Callable[[int, int], None]
StreamFn = Callable[..., tuple[int, str]]

_TAGS_TIMEOUT = 5.0
_HEALTH_TIMEOUT = 1.0
_DELETE_TIMEOUT = 30.0
# A pull streams for as long as the download takes; only silence is a failure.
_PULL_IDLE_TIMEOUT = 300.0
_LOAD_TIMEOUT = 600  # a 27B read from disk into memory

IMPORT_QUANT = "nvfp4"  # what Ollama's own MLX library builds use
IMPORT_QUANTS: tuple[str, ...] = ("nvfp4", "int4")  # what its MLX runner accepts for `-q`
IMPORT_MIN_VERSION = (0, 32, 0)  # the x/create rewrite: the Qwen3.5 nvfp4 corruption fix
MLX_HINT = (
    "Apple Silicon has it built in; on Linux/Windows LePika installs Ollama's MLX bundle "
    "before an import — if it is already installed, restart Ollama (`lepika down` then "
    "`lepika up`, or `sudo systemctl restart ollama`) and check that `nvidia-smi` reports "
    "CUDA 13 or newer."
)


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
    """`Qwen3` → `qwen3:latest`: the name Ollama actually stores for a ref.

    The tag is the `:` after the last `/`, so `hf.co/org/repo-GGUF` gets one and a
    port in a hostname never counts as one. Case goes too: `create` and `pull`
    lower-case the name, so an imported `Qwen/Qwen3.5-2B` is listed as
    `qwen/qwen3.5-2b` while the config still holds the repo as it was typed.
    """
    name = name.lower()
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


class NotGGUF(FriendlyError):
    """Ollama refused an `hf.co/…` pull because the repo ships full weights.

    Its own class so the caller can fall through to an import instead of stopping:
    Ollama is the oracle for the format, not the repo's name.
    """


def _pull_failed(ref: ModelRef, detail: str) -> FriendlyError:
    if "not gguf" in detail.lower():
        # Ollama's refusal for a safetensors repo (`hf.co/Qwen/Qwen3.8-27B`): the
        # URL is right and the network is fine, so the generic hint would send the
        # user checking both. What they need is the GGUF build of the same model.
        repo = ref.raw.rpartition("/")[2].partition(":")[0]
        return NotGGUF(
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


def version(url: str, key: str = "", urlopen: UrlOpenFn | None = None) -> str:
    """`GET /api/version` — a probe, so nothing is logged."""
    try:
        with _opener(urlopen)(_request(url, "/api/version", key), timeout=_TAGS_TIMEOUT) as r:
            return str(json.loads(r.read().decode("utf-8"))["version"])
    except Exception as exc:
        raise _unreachable(url, True) from exc


def _version_tuple(text: str) -> tuple[int, ...]:
    """`0.32` → `(0, 32, 0)`. Padded, because a short tuple sorts below a long one:
    unpadded, `(0, 32) < (0, 32, 0)` is True and Ollama 0.32 would refuse itself."""
    parts = [int(part) for part in re.findall(r"\d+", text)[:3]]
    return tuple(parts + [0] * (3 - len(parts)))


def _runner_failed(name: str, detail: str) -> FriendlyError:
    if "mlx not available" in detail.lower():
        return FriendlyError(
            f"Ollama's MLX engine is missing, and '{name}' needs it to run.", MLX_HINT
        )
    return FriendlyError(
        f"Ollama could not load '{name}': {detail}",
        "Run `lepika logs` for the engine's reason, then `lepika doctor`.",
    )


def load_model(url: str, name: str, key: str = "", urlopen: UrlOpenFn | None = None) -> None:
    """Load a model once so an import is only reported as ✓ when it actually runs.

    An empty prompt makes Ollama load the weights and return; a missing MLX runner
    or a broken artifact surfaces here, not in the user's first chat.
    """
    request = _request(url, "/api/generate", key, "POST", {"model": name, "prompt": ""})
    try:
        with _opener(urlopen)(request, timeout=_LOAD_TIMEOUT):
            pass
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise _rejected(url) from exc
        try:
            detail = str(json.loads(exc.read().decode("utf-8")).get("error", exc.reason))
        except Exception:
            detail = str(exc.reason)
        raise _runner_failed(name, detail) from exc
    except Exception as exc:
        raise _unreachable(url, True) from exc


def _import_failed(name: str, tail: str, owned: bool) -> FriendlyError:
    low = tail.lower()
    repo = name.rpartition("/")[2]
    if "unsupported architecture" in low or "not a supported safetensors" in low:
        return FriendlyError(
            f"Ollama cannot import {name}'s architecture yet.",
            f"Use a GGUF build instead, e.g. hf.co/<org>/{repo}-GGUF — or `lepika update` "
            "for a newer Ollama.",
        )
    # Only LePika's own staging area is worth reassuring about: a retry there resumes a
    # download that is kept on purpose. A folder the user pointed at never went anywhere.
    kept = " (the download is kept)" if owned else ""
    return FriendlyError(
        f"Import of {name} failed.",
        f"Run the same command again{kept}; if it keeps failing, run `lepika doctor` and "
        "file an issue with the log.",
    )


@contextlib.contextmanager
def _staged(source: Path, owned: bool) -> Iterator[Path]:
    """The directory `ollama create` runs in — a Modelfile and nothing else is added.

    LePika's own download is staged in place (`FROM .`), which is the E2E-proven path.
    A folder the user pointed at is never written to, not even a file we would delete
    afterwards: the Modelfile goes in a temp directory and names the source absolutely.
    """
    if owned:
        # `FROM .` and nothing else: templates and parsers are Ollama's to detect.
        (source / "Modelfile").write_text("FROM .\n", encoding="utf-8")
        yield source
        return
    with tempfile.TemporaryDirectory(prefix="lepika-import-") as tmp:
        staging = Path(tmp)
        (staging / "Modelfile").write_text(f"FROM {source.resolve()}\n", encoding="utf-8")
        yield staging


def import_model(
    url: str,
    name: str,
    source: Path,
    key: str = "",
    quant: str = IMPORT_QUANT,
    owned: bool = True,
    stream: StreamFn = proc.stream,
    urlopen: UrlOpenFn | None = None,
    environ: Mapping[str, str] | None = None,
) -> None:
    """`ollama create <name> --experimental -q <quant>` from a safetensors directory.

    The name is the ref itself: Ollama accepts `Qwen/Qwen3.8-27B` verbatim, so the
    config, `model list` and `same_model` need no mapping table — it only lower-cases
    what it stores, which is why `same_model` compares case-insensitively. `owned` says
    whose directory `source` is — LePika's staged download, or a folder the user pointed at.
    """
    have = version(url, key, urlopen)
    if _version_tuple(have) < IMPORT_MIN_VERSION:
        raise FriendlyError(
            f"Importing full-weight repos needs Ollama 0.32 or newer (you have {have}).",
            "Run `lepika update`.",
        )
    env = dict(environ if environ is not None else os.environ)
    env["OLLAMA_HOST"] = url  # the CLI talks to the engine LePika manages, not a default
    logger = log.get_logger()
    argv = ["ollama", "create", name, "--experimental", "-q", quant]
    with _staged(source, owned) as cwd:
        code, tail = stream(argv, env=env, cwd=cwd)
    if code != 0:
        logger.warning("engine.import", model=name, quant=quant, source=str(source), result=tail)
        raise _import_failed(name, tail, owned)
    try:
        load_model(url, name, key, urlopen)
    except FriendlyError as exc:
        # The build worked and the weights are on disk, but nothing can run them —
        # an attempt that leaves no line would look like it never happened.
        logger.warning(
            "engine.import", model=name, quant=quant, source=str(source), result=exc.problem
        )
        raise
    logger.info("engine.import", model=name, quant=quant, source=str(source), result="success")
