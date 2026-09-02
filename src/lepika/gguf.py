"""GGUF builds of a Hugging Face repo: which quantizations exist, and which one fits."""

from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# The quantization is the tail of the file stem, read the way the Hub reads it when it
# resolves `hf.co/<org>/<repo>:<TAG>` for Ollama — so the tag shown is the tag pulled.
# `UD-` is Unsloth's dynamic-quant prefix and part of the tag.
_QUANT = re.compile(
    r"(?:^|[-.])((?:UD-)?(?:IQ\d+_[A-Z0-9]+|Q\d+(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4(?:_[A-Z0-9]+)?))$",
    re.IGNORECASE,
)
_SHARD = re.compile(r"-\d{5}-of-\d{5}$")
# Not chat models: vision projectors, importance matrices, speculative-decoding drafts.
_SKIP_PREFIXES = ("mmproj", "imatrix", "mtp-")
_UNQUANTIZED = frozenset({"BF16", "F16", "F32"})

UrlOpenFn = Callable[..., Any]

HUB_API = "https://huggingface.co/api/models"
_TIMEOUT = 5


class Unavailable(Exception):
    """The Hub did not answer, or answered something that is not a file list."""


@dataclass(frozen=True)
class Build:
    quant: str
    size_bytes: int

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1e9

    @property
    def quantized(self) -> bool:
        return self.quant.upper() not in _UNQUANTIZED


def parse_builds(payload: Any) -> list[Build]:
    """Every GGUF build in a Hub model listing — shards summed, one row per quant.

    Shards may sit in a folder named after their quant (`BF16/…-00001-of-00002.gguf`);
    any other folder is skipped, because Ollama resolves a tag against the root.
    A root-level file beats a folder twin of the same quant.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("siblings"), list):
        raise Unavailable("not a model listing")
    root: dict[str, int] = {}
    nested: dict[str, int] = {}
    for entry in payload["siblings"]:
        if not isinstance(entry, dict):
            continue
        name, size = entry.get("rfilename"), entry.get("size")
        if not isinstance(name, str) or not isinstance(size, int) or isinstance(size, bool):
            continue
        folder, _, base = name.rpartition("/")
        lower = base.lower()
        if not lower.endswith(".gguf") or lower.startswith(_SKIP_PREFIXES):
            continue
        match = _QUANT.search(_SHARD.sub("", base[: -len(".gguf")]))
        if match is None:
            continue
        quant = match.group(1)
        if folder and folder.lower() != quant.lower():
            continue
        bucket = nested if folder else root
        bucket[quant] = bucket.get(quant, 0) + size
    merged = {**nested, **root}
    return sorted((Build(q, s) for q, s in merged.items()), key=lambda b: b.size_bytes)


def split_tag(raw: str) -> tuple[str, str]:
    """`hf.co/<org>/<repo>[:<tag>]` → (`<org>/<repo>`, tag or "")."""
    repo = raw[len("hf.co/") :] if raw.lower().startswith("hf.co/") else raw
    head, sep, tail = repo.rpartition("/")
    name, _colon, tag = tail.partition(":")
    return head + sep + name, tag


def list_builds(
    repo: str,
    # B107: an empty default means "no token", not a credential — a real one
    # arrives from the caller and travels only in the request header.
    token: str = "",  # nosec B107
    urlopen: UrlOpenFn | None = None,
) -> list[Build]:
    """The repo's GGUF builds from one Hub API call; `Unavailable` when it cannot say.

    `blobs=true` is what puts a byte size on every file. Everything that is not an
    answer — offline, 401/403/404/429, a rate-limit HTML page — is one exception the
    caller turns into "let Ollama pick": this is a convenience, never a gate.
    """
    opener: UrlOpenFn = urlopen if urlopen is not None else urllib.request.urlopen
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = urllib.request.Request(f"{HUB_API}/{repo}?blobs=true", headers=headers)
    try:
        payload = json.loads(opener(request, timeout=_TIMEOUT).read().decode("utf-8"))
    except Exception as exc:
        raise Unavailable(str(exc)) from exc
    return parse_builds(payload)
