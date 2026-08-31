"""Hugging Face repos as a model source: the `hf` CLI, driven — never reimplemented."""

from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.prompt import Prompt

from lepika import config, log, paths, proc
from lepika.config import Config
from lepika.errors import FriendlyError

RunFn = Callable[..., Any]
StreamFn = Callable[..., tuple[int, str]]
AskFn = Callable[..., str]

# Same shape as OpenWebUI's `uv tool run`: pinned interpreter, named package.
HF_PYTHON = "3.12"
HF_CMD: tuple[str, ...] = (
    "uv",
    "tool",
    "run",
    "--python",
    HF_PYTHON,
    "--from",
    "huggingface_hub",
    "hf",
)

# Only the safetensors and their configs come down: GGUF/PyTorch twins of the same
# weights would double a download Ollama never reads.
EXCLUDES: tuple[str, ...] = ("*.gguf", "*.bin", "*.pth", "*.pt", "original/*", "consolidated*")

_PREFLIGHT_TIMEOUT = 120

_UNITS = {"K": 10**3, "M": 10**6, "G": 10**9, "T": 10**12}

# A repo id is `<org>/<repo>` and nothing else — it becomes a directory name.
_REPO_ID = re.compile(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+")


class GatedRepo(FriendlyError):
    """The Hub answered 401/403: a licence to accept, or a token to supply."""


@dataclass(frozen=True)
class Preflight:
    files: tuple[str, ...]
    total_bytes: int
    # What the download leaves under `~/.lepika/hf`: `--dry-run` lists the whole repo
    # (so `has_gguf` can see a .gguf), but EXCLUDES keeps the twins off the disk, and
    # sizing by `total_bytes` would double-count a repo that ships both. Not the same
    # as bytes off the network — a file `hf` already holds is copied out of its hub
    # cache, which costs no download and the same disk.
    download_bytes: int

    @property
    def has_safetensors(self) -> bool:
        return any(f.lower().endswith(".safetensors") for f in self.files)

    @property
    def has_gguf(self) -> bool:
        return any(f.lower().endswith(".gguf") for f in self.files)


def check_repo(repo: str) -> None:
    """Refuse anything that is not a plain `<org>/<repo>`.

    The id becomes a directory under `~/.lepika/hf`, so an unchecked `../../x` would
    escape LePika's home. Guarded at both entry points rather than at the path join.
    """
    if not _REPO_ID.fullmatch(repo) or any(s in {".", ".."} for s in repo.split("/")):
        raise FriendlyError(
            f"'{repo}' is not a Hugging Face repo id.",
            "Use <org>/<repo>, e.g. Qwen/Qwen3.8-27B.",
        )


def _excluded(name: str) -> bool:
    """Would `download`'s `--exclude` patterns skip this file? (`hf` matches by glob.)"""
    return any(fnmatch.fnmatch(name, pattern) for pattern in EXCLUDES)


def _parse_size(text: str) -> int:
    """`hf` prints human sizes in powers of 1000: `390.0`, `2.9K`, `4.5G`."""
    value = text.strip().upper()
    if value == "-":
        # What `hf` prints for a file it already holds in ~/.cache/huggingface. The
        # caller stats the cached copy; 0 is only the answer when it cannot find it.
        return 0
    unit = value[-1] if value and value[-1] in _UNITS else ""
    number = value[:-1] if unit else value
    return int(float(number) * _UNITS.get(unit, 1))


def cache_dir(environ: Mapping[str, str] | None = None) -> Path:
    """Where `huggingface_hub` keeps its download cache, resolved as the library does."""
    env = environ if environ is not None else os.environ
    if hub := env.get("HF_HUB_CACHE", "").strip():
        return Path(hub)
    if home := env.get("HF_HOME", "").strip():
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _cached_size(repo: str, name: str, cache: Path) -> int:
    """Bytes of a file `hf` reported as `-`, read from the hub cache; 0 if it is not there.

    A download that costs no bytes still costs disk: `hf download --local-dir` copies
    the cached file into `~/.lepika/hf`, so it belongs in the estimate. The snapshot
    entry is a symlink into `blobs/`, which `stat()` follows. The first snapshot
    holding the file answers: this is an estimate, and two revisions of the same
    weights are the same size to within rounding.
    """
    if Path(name).is_absolute() or ".." in Path(name).parts:
        # Defence in depth: the name comes from `hf`'s own JSON, but a size estimate is
        # no reason to stat anything outside the cache directory we built the path from.
        return 0
    snapshots = cache / f"models--{repo.replace('/', '--')}" / "snapshots"
    try:
        revisions = sorted(snapshots.iterdir())
    except OSError:  # no cache, or nothing cached for this repo
        return 0
    for revision in revisions:
        try:
            return (revision / name).stat().st_size
        except OSError:  # not in this snapshot, or a broken symlink
            continue
    return 0


def _size_of(repo: str, name: str, text: str, cache: Path) -> int:
    """One listing entry's size on disk, cached or not."""
    return _cached_size(repo, name, cache) if text.strip() == "-" else _parse_size(text)


def _env(token: str, environ: Mapping[str, str] | None) -> dict[str, str]:
    env = dict(environ if environ is not None else os.environ)
    if token:
        # The child reads HF_TOKEN; the command line never carries it (rule 11).
        env["HF_TOKEN"] = token
    return env


def _parse_listing(stdout: str) -> list[dict[str, str]]:
    start = stdout.find("[")
    if start < 0:
        raise ValueError("no JSON array in hf output")
    listing: list[dict[str, str]] = json.loads(stdout[start:])
    return listing


def preflight(
    repo: str,
    # B107: an empty default means "no token", not a credential — a real one
    # arrives from the caller and travels only in the child's environment.
    token: str = "",  # nosec B107
    run: RunFn = proc.run_logged,
    environ: Mapping[str, str] | None = None,
) -> Preflight:
    """What the repo holds and how big it is, without downloading a byte.

    A pure read (`log=False`), and the one oracle for a bare `org/repo`: the file
    list says whether Ollama pulls it (GGUF) or LePika imports it (safetensors).
    """
    check_repo(repo)
    result: subprocess.CompletedProcess[str] = run(
        [*HF_CMD, "download", repo, "--dry-run", "--json"],
        check=False,
        log=False,
        env=_env(token, environ),
        timeout=_PREFLIGHT_TIMEOUT,
    )
    if result.returncode != 0:
        text = (result.stdout + result.stderr).lower()
        # Not-found is tested first: the Hub answers a mistyped repo with a 401 whose
        # body also mentions "gated", so a gated-first order would misread every typo.
        if "404" in text or "not found" in text:
            raise FriendlyError(
                f"'{repo}' is not on Hugging Face.",
                "Check the name — it is <org>/<repo>, e.g. Qwen/Qwen3.8-27B. "
                "A private repo needs HF_TOKEN.",
            )
        if "401" in text or "403" in text or "gated" in text:
            raise GatedRepo(
                f"'{repo}' is gated or private on Hugging Face.",
                "Accept its licence on huggingface.co, then set HF_TOKEN or answer the prompt.",
            )
        raise FriendlyError(
            f"Could not list the files of '{repo}'.",
            "Check your internet connection and that `uv` works, then try again.",
        )
    try:
        entries = _parse_listing(result.stdout)
        cache = cache_dir(environ)
        sized = [
            (str(e["file"]), _size_of(repo, str(e["file"]), str(e["size"]), cache)) for e in entries
        ]
        files = tuple(name for name, _size in sized)
        total = sum(size for _name, size in sized)
        wanted = sum(size for name, size in sized if not _excluded(name))
    except (ValueError, KeyError, TypeError) as exc:
        raise FriendlyError(
            f"Could not read the file list of '{repo}'.",
            "Run `uv tool upgrade huggingface_hub` and try again.",
        ) from exc
    return Preflight(files=files, total_bytes=total, download_bytes=wanted)


def download_dir(repo: str) -> Path:
    """`~/.lepika/hf/<org>/<repo>` — the staging area Ollama imports from."""
    check_repo(repo)
    return paths.hf_dir().joinpath(*repo.split("/"))


def load_config(source: Path) -> dict[str, Any]:
    """The repo's config.json as a dict; {} when absent or unreadable."""
    try:
        data = json.loads((source / "config.json").read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def quant_method(config_json: Mapping[str, Any]) -> str | None:
    """How already-quantized weights were quantized, or None for a raw checkpoint.

    Quantized safetensors declare themselves in config.json's `quantization_config`
    (e.g. `quant_method: "modelopt"` for NVFP4) — the one signal that decides whether
    `ollama create` gets a `-q` (it refuses to requantize) or imports the source bare.
    """
    qc = config_json.get("quantization_config")
    if not isinstance(qc, Mapping) or not qc:
        return None
    return str(qc.get("quant_method") or "unknown")


def fetch_config(
    repo: str,
    dest: Path,
    # B107: an empty default means "no token", not a credential — a real one
    # arrives from the caller and travels only in the child's environment.
    token: str = "",  # nosec B107
    run: RunFn = proc.run_logged,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """config.json alone, into the staging dir the full download will reuse.

    {} on any failure: the Hub not answering one small request must degrade to
    "unknown, treat as raw", never block an import that used to work.
    """
    check_repo(repo)
    dest.mkdir(parents=True, exist_ok=True)
    run(
        [*HF_CMD, "download", repo, "config.json", "--local-dir", str(dest)],
        check=False,
        log=False,
        env=_env(token, environ),
        timeout=_PREFLIGHT_TIMEOUT,
    )
    return load_config(dest)


def download(
    repo: str,
    dest: Path,
    # B107: an empty default means "no token", not a credential — a real one
    # arrives from the caller and travels only in the child's environment.
    token: str = "",  # nosec B107
    stream: StreamFn = proc.stream,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Fetch the weights, resumably: `hf` skips every file that is already complete."""
    dest.mkdir(parents=True, exist_ok=True)
    excludes = [arg for pattern in EXCLUDES for arg in ("--exclude", pattern)]
    code, _tail = stream(
        [*HF_CMD, "download", repo, "--local-dir", str(dest), *excludes],
        env=_env(token, environ),
    )
    logger = log.get_logger()
    if code != 0:
        logger.warning("hf.download", repo=repo, dest=str(dest), result="failed")
        raise FriendlyError(
            f"Download of '{repo}' failed.",
            "Check your connection and run the same command again — it resumes where it stopped.",
        )
    logger.info("hf.download", repo=repo, dest=str(dest), result="success")


def token_for(cfg: Config, environ: Mapping[str, str] | None = None) -> str:
    """HF_TOKEN from the environment wins; the one saved by `ask_token` is the fallback.

    Both are stripped: `export HF_TOKEN=$(cat token.txt)` carries a trailing newline,
    which the Hub rejects as a malformed bearer token.
    """
    env = environ if environ is not None else os.environ
    return env.get("HF_TOKEN", "").strip() or cfg.hf_token.strip()


def ask_token(cfg: Config, ask: AskFn | None = None) -> str:
    """Ask once — only after a gated pre-flight, so public repos never see a prompt.

    A non-empty answer is saved to config.toml (0600 from creation). Empty means the
    user has no token; the caller turns that into the licence hint. `config.save`
    writes every field, so pass the loaded Config, never a fresh one.
    """
    ask_fn: AskFn = ask if ask is not None else Prompt.ask
    token = ask_fn(
        "Hugging Face token (this repo is gated; Enter to cancel)", password=True, default=""
    ).strip()
    if token:
        cfg.hf_token = token
        config.save(cfg)
    return token
