"""Model references (3 shapes) and the curated model list."""

from __future__ import annotations

import dataclasses
import importlib.resources
import tomllib
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from lepika.errors import FriendlyError

ModelKind = Literal["ollama", "hf_gguf", "hf_repo"]

CURATED_MODELS_URL = "https://raw.githubusercontent.com/MikaelSabuhi/lepika/main/models.toml"


@dataclass(frozen=True)
class ModelRef:
    raw: str
    kind: ModelKind


@dataclass(frozen=True)
class CuratedModel:
    name: str
    ref: str
    min_ram_gb: float
    notes: str = ""


_CURATED_FIELDS = {f.name for f in dataclasses.fields(CuratedModel)}


def _usable_entry(entry: dict[str, Any]) -> bool:
    """Is a remote entry complete, with types that won't blow up downstream?

    A quoted number (`min_ram_gb = "8"`) is a plausible typo that constructs fine
    and then raises deep inside `fitting`, far from the fallback here — so the
    types are checked at the boundary, not just the field names.
    """
    ram = entry.get("min_ram_gb")
    return (
        isinstance(entry.get("name"), str)
        and isinstance(entry.get("ref"), str)
        # bool is an int subclass; `min_ram_gb = true` is not a memory size.
        and isinstance(ram, int | float)
        and not isinstance(ram, bool)
    )


def parse_model_ref(raw: str) -> ModelRef:
    ref = raw.strip()
    if not ref:
        raise FriendlyError(
            "Empty model reference.",
            "Examples: qwen3:8b · hf.co/unsloth/gemma-3-4b-it-GGUF · "
            "meta-llama/Llama-3.3-70B-Instruct",
        )
    for scheme in ("https://", "http://"):
        if ref.lower().startswith(scheme):
            ref = ref[len(scheme) :]
            break
    if ref.lower().startswith("huggingface.co/"):
        ref = "hf.co/" + ref[len("huggingface.co/") :]
    if ref.lower().startswith("hf.co/"):
        return ModelRef(raw=ref, kind="hf_gguf")
    if "/" in ref:
        return ModelRef(raw=ref, kind="hf_repo")
    return ModelRef(raw=ref, kind="ollama")


def _bundled_models_text() -> str:
    resource = importlib.resources.files("lepika").joinpath("models.toml")
    if resource.is_file():
        return resource.read_text(encoding="utf-8")
    # Editable/dev install: fall back to the repo-root copy.
    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / "models.toml").read_text(encoding="utf-8")


def _parse_curated(text: str, *, lenient: bool) -> list[CuratedModel]:
    """Parse a models.toml.

    `lenient` is for remote content, which is written by a NEWER LePika than the one
    reading it: unknown keys are dropped and unusable entries are skipped so that
    additive schema changes on main never break an older client. The bundled copy
    ships with the code, so it is parsed strictly and a defect there raises.
    """
    data = tomllib.loads(text)
    entries = data.get("models", [])
    if not isinstance(entries, list):
        raise TypeError("`models` must be an array of tables")
    curated: list[CuratedModel] = []
    for entry in entries:
        if not isinstance(entry, dict):
            if lenient:
                continue
            raise TypeError("each `models` entry must be a table")
        known = {k: v for k, v in entry.items() if k in _CURATED_FIELDS}
        if lenient and not _usable_entry(known):
            continue
        curated.append(CuratedModel(**known))
    return curated


def load_curated(
    fetch_remote: bool = True,
    urlopen: Callable[..., Any] | None = None,
) -> list[CuratedModel]:
    if fetch_remote:
        opener: Callable[..., Any] = urlopen if urlopen is not None else urllib.request.urlopen
        try:
            text = opener(CURATED_MODELS_URL, timeout=3).read().decode("utf-8")
            remote = _parse_curated(text, lenient=True)
        except Exception:
            # Unreachable, or served something unusable: the bundled list still works.
            remote = []
        if remote:
            return remote
    return _parse_curated(_bundled_models_text(), lenient=False)


def fitting(candidates: list[CuratedModel], ram_gb: float) -> list[CuratedModel]:
    return [m for m in candidates if m.min_ram_gb <= ram_gb]
