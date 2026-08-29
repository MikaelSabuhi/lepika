"""All LePika state lives under one directory, overridable via LEPIKA_HOME."""

from __future__ import annotations

import os
from pathlib import Path


def lepika_home() -> Path:
    home = Path(os.environ.get("LEPIKA_HOME", str(Path.home() / ".lepika")))
    home.mkdir(parents=True, exist_ok=True)
    return home


def logs_dir() -> Path:
    d = lepika_home() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def pid_file(name: str) -> Path:
    return lepika_home() / f"{name}.pid"


def stack_dir() -> Path:
    d = lepika_home() / "stack"
    d.mkdir(parents=True, exist_ok=True)
    return d


def openwebui_data_dir() -> Path:
    d = lepika_home() / "openwebui"
    d.mkdir(parents=True, exist_ok=True)
    return d
