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


@dataclass
class Config:
    schema_version: int = SCHEMA_VERSION
    mode: str = "express"
    model: str = ""
    engine_url: str = "http://127.0.0.1:11434"
    webui_port: int = 3000


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
    tmp.write_text(_dump_toml(dataclasses.asdict(cfg)), encoding="utf-8")
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
