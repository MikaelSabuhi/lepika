"""config.toml under ~/.lepika — flat, versioned, migratable."""

from __future__ import annotations

import dataclasses
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from lepika.errors import FriendlyError
from lepika.paths import lepika_home

SCHEMA_VERSION = 1

DEFAULT_ENGINE_URL = "http://127.0.0.1:11434"


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


def config_path() -> Path:
    return lepika_home() / "config.toml"


def _dump_toml(data: dict[str, object]) -> str:
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, bool):
            lines.append(f"{key} = {'true' if value else 'false'}")
        elif isinstance(value, int | float):
            lines.append(f"{key} = {value}")
        else:
            # A free-form model ref is user input and may contain `"` or `\`.
            # Unescaped, either one writes a config.toml that no longer parses —
            # and `load` then refuses to read the file it just wrote.
            text = str(value).replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{key} = "{text}"')
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
    raw.setdefault("schema_version", SCHEMA_VERSION)
    known = {f.name for f in dataclasses.fields(Config)}
    return Config(**{k: v for k, v in raw.items() if k in known})
