"""config.toml under ~/.lepika — flat, versioned, migratable."""

from __future__ import annotations

import dataclasses
import difflib
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from lepika.errors import FriendlyError
from lepika.paths import lepika_home

SCHEMA_VERSION = 1

MODES: tuple[str, ...] = ("express", "server")

DEFAULT_ENGINE_URL = "http://127.0.0.1:11434"

# Ollama's own default is 4096 tokens: one pasted document, or a chat an hour old,
# fails with "exceeds the available context size" on any model. 16k fits long
# chats and still fits beside a Q4 27B model on a 16 GB card; the KV cache for a
# model that size costs roughly 0.25 MB per token, so 32k would spill to CPU there.
DEFAULT_CONTEXT_LENGTH = 16384


@dataclass
class Config:
    schema_version: int = SCHEMA_VERSION
    mode: str = "express"
    model: str = ""
    engine_url: str = DEFAULT_ENGINE_URL
    # True: LePika installs/starts the engine (native in Express, a container in
    # Server). False: `lepika connect` pointed us at an engine someone else runs.
    engine_managed: bool = True
    engine_key: str = ""
    # Hugging Face token for gated repos (Express imports). Server mode keeps its
    # own copy in stack/.env, the file compose reads. Never logged: `hf_token` is a
    # redacted key name (log.py).
    hf_token: str = ""
    webui_port: int = 3000
    api_port: int = 11435
    exposed: bool = False
    # Tokens of context the engine LePika starts gives every model (Express: the
    # `ollama serve` environment; Server: the container's). An engine LePika only
    # adopted keeps its own — `context_note` says so.
    context_length: int = DEFAULT_CONTEXT_LENGTH


def config_path() -> Path:
    return lepika_home() / "config.toml"


# TOML basic strings name these escapes; every other control character must be
# written as \uXXXX, or the file no longer parses.
_NAMED_ESCAPES = {
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
    '"': '\\"',
    "\\": "\\\\",
}


def _toml_string(text: str) -> str:
    out: list[str] = []
    for ch in text:
        if ch in _NAMED_ESCAPES:
            out.append(_NAMED_ESCAPES[ch])
        elif ch < " " or ch == "\x7f":
            out.append(f"\\u{ord(ch):04X}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _dump_toml(data: dict[str, object]) -> str:
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, bool):
            lines.append(f"{key} = {'true' if value else 'false'}")
        elif isinstance(value, int | float):
            lines.append(f"{key} = {value}")
        else:
            # A free-form model ref is user input. An unescaped quote, backslash or
            # control character writes a config.toml that `load` can't read back.
            lines.append(f"{key} = {_toml_string(str(value))}")
    return "\n".join(lines) + "\n"


def save(cfg: Config) -> None:
    """Write atomically so an interrupted save can't leave a half-written config."""
    path = config_path()
    tmp = path.with_suffix(".toml.tmp")
    # The file can hold an engine key, so it is private from the first byte: the
    # mode is set when the file is created, not chmod'ed after the key is already
    # on disk under the process umask. fchmod also tightens a stale tmp file left
    # world-readable by an interrupted run, which O_TRUNC alone would keep.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    if hasattr(os, "fchmod"):  # POSIX only; Windows ignores mode bits entirely
        os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(_dump_toml(dataclasses.asdict(cfg)))
    os.replace(tmp, path)


_SECRET_KEYS = frozenset({"engine_key", "hf_token"})
_PORT_KEYS = frozenset({"webui_port", "api_port"})


def _accepted(key: str, value: object, default: object) -> str | None:
    """What `key` accepts when `value` is not acceptable; None when it is fine."""
    # The declared default's type is the field's type. bool is checked first and
    # kept apart from int: TOML `true` is a Python int subclass, but not a port.
    if isinstance(default, bool):
        return None if isinstance(value, bool) else "true or false"
    if isinstance(default, int):
        if not isinstance(value, int) or isinstance(value, bool):
            return "a whole number"
        if key in _PORT_KEYS and not 1 <= value <= 65535:
            return "a port from 1 to 65535"
        if key in ("context_length", "schema_version") and value < 1:
            return "a whole number of 1 or more"
        return None
    if not isinstance(value, str):
        return "text"
    if key == "mode" and value not in MODES:
        return " or ".join(f'"{m}"' for m in MODES)
    if key == "engine_url" and not value.startswith(("http://", "https://")):
        return "a URL starting with http:// or https://"
    return None


def _validate(raw: dict[str, object], path: Path) -> Config:
    """Turn a parsed config.toml into a Config, or name the first thing wrong in it.

    The file is user-editable, so a typo must stop LePika with the setting's name
    rather than run with a quietly different meaning: `exposed = "no"` is a truthy
    string, and an unchecked `mode = "sever"` would run Express.
    """
    defaults = Config()
    known = [f.name for f in dataclasses.fields(Config)]
    for key, value in raw.items():
        if key not in known:
            match = difflib.get_close_matches(key, known, n=1)
            hint = f"Did you mean `{match[0]}`? " if match else ""
            raise FriendlyError(
                f"Unknown setting `{key}` in {path}.",
                f"{hint}Remove or rename it; the settings are: {', '.join(known)}.",
            )
        default = getattr(defaults, key)
        accepted = _accepted(key, value, default)
        if accepted is not None:
            # A secret is never echoed: a mistyped value may still be the real key.
            shown = "" if key in _SECRET_KEYS else f" = {value!r}"
            raise FriendlyError(
                f"Setting `{key}`{shown} must be {accepted}.",
                f"Fix it in {path}, or delete the line to use the default "
                f"({_dump_toml({key: default}).strip()}).",
            )
        if key == "schema_version" and isinstance(value, int) and value > SCHEMA_VERSION:
            raise FriendlyError(
                f"{path} was written by a newer LePika "
                f"(schema {value}; this version reads {SCHEMA_VERSION}).",
                "Upgrade LePika: run the install command from the README again.",
            )
    return Config(**raw)  # type: ignore[arg-type]


def load() -> Config:
    path = config_path()
    if not path.exists():
        return Config()
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise FriendlyError(
            f"Your config file is corrupted: {path}",
            "Delete it and run `lepika` again — it will be recreated.",
        ) from exc
    return _validate(raw, path)
