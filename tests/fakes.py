"""Shared test doubles for the injected subprocess callables."""

from __future__ import annotations

import subprocess
from typing import Any


class Runner:
    """Stands in for proc.run_logged; canned stdout per leading argv words."""

    def __init__(self, stdout: dict[str, str] | None = None, code: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.stdout = stdout or {}
        self.code = code

    def __call__(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(cmd))
        out = next((v for k, v in self.stdout.items() if " ".join(cmd).startswith(k)), "")
        return subprocess.CompletedProcess(cmd, self.code, stdout=out, stderr="")


class Caller:
    """Stands in for subprocess.call: records the argv, returns a canned exit code."""

    def __init__(self, code: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.code = code

    def __call__(self, cmd: list[str]) -> int:
        self.calls.append(list(cmd))
        return self.code
