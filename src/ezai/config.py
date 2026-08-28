"""config.toml under ~/.ezai — flat, versioned, migratable."""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass
from pathlib import Path

from ezai.paths import ezai_home

SCHEMA_VERSION = 1


@dataclass
class Config:
    schema_version: int = SCHEMA_VERSION
    mode: str = "express"
    model: str = ""
    engine_url: str = "http://127.0.0.1:11434"
    webui_port: int = 3000


def config_path() -> Path:
    return ezai_home() / "config.toml"


def _dump_toml(data: dict[str, object]) -> str:
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, bool):
            lines.append(f"{key} = {'true' if value else 'false'}")
        elif isinstance(value, int | float):
            lines.append(f"{key} = {value}")
        else:
            lines.append(f'{key} = "{value}"')
    return "\n".join(lines) + "\n"


def save(cfg: Config) -> None:
    config_path().write_text(_dump_toml(dataclasses.asdict(cfg)), encoding="utf-8")


def load() -> Config:
    path = config_path()
    if not path.exists():
        return Config()
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    raw.setdefault("schema_version", SCHEMA_VERSION)
    known = {f.name for f in dataclasses.fields(Config)}
    return Config(**{k: v for k, v in raw.items() if k in known})
