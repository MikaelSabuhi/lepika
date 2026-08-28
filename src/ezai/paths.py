"""All ezai state lives under one directory, overridable via EZAI_HOME."""

from __future__ import annotations

import os
from pathlib import Path


def ezai_home() -> Path:
    home = Path(os.environ.get("EZAI_HOME", str(Path.home() / ".ezai")))
    home.mkdir(parents=True, exist_ok=True)
    return home


def logs_dir() -> Path:
    d = ezai_home() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def pid_file(name: str) -> Path:
    return ezai_home() / f"{name}.pid"
