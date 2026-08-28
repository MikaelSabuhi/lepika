"""Single choke point for subprocess calls: captured, logged, friendly."""

from __future__ import annotations

import datetime
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from lepika.errors import FriendlyError
from lepika.paths import logs_dir


def _append_log(cmd: Sequence[str], outcome: str, output: str) -> Path:
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    log_path = logs_dir() / "lepika.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{stamp}] $ {' '.join(cmd)}\n({outcome})\n{output}\n")
    return log_path


def run_logged(
    cmd: Sequence[str],
    *,
    check: bool = True,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
            env=dict(env) if env is not None else None,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        _append_log(cmd, "not found", str(exc))
        raise FriendlyError(
            f"Command not found: {cmd[0]}",
            f"Install {cmd[0]} or run `lepika doctor` for setup help.",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        log_path = _append_log(cmd, f"timed out after {timeout}s", _decode(exc))
        raise FriendlyError(
            f"Command timed out after {timeout}s: {' '.join(cmd)}",
            f"Try again; details in {log_path}",
        ) from exc
    log_path = _append_log(cmd, f"exit {result.returncode}", f"{result.stdout}{result.stderr}")
    if check and result.returncode != 0:
        raise FriendlyError(
            f"Command failed: {' '.join(cmd)}",
            f"Details were logged to {log_path}",
        )
    return result


def _decode(exc: subprocess.TimeoutExpired) -> str:
    parts: list[str] = []
    for stream in (exc.stdout, exc.stderr):
        if stream is None:
            continue
        parts.append(stream.decode("utf-8", "replace") if isinstance(stream, bytes) else stream)
    return "".join(parts)
