"""Single choke point for subprocess calls: captured, logged, friendly."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from lepika.errors import FriendlyError
from lepika.log import LOG_FILE, get_logger
from lepika.paths import logs_dir


def _log_path() -> Path:
    return logs_dir() / LOG_FILE


def run_logged(
    cmd: Sequence[str],
    *,
    check: bool = True,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    log: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run `cmd` captured. Pass `log=False` for a pure read: only failures are recorded."""
    logger = get_logger()
    try:
        result = subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
            env=dict(env) if env is not None else None,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        logger.error("proc.run", cmd=list(cmd), outcome="not found", output=str(exc))
        raise FriendlyError(
            f"Command not found: {cmd[0]}",
            f"Install {cmd[0]} or run `lepika doctor` for setup help.",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        logger.error(
            "proc.run",
            cmd=list(cmd),
            outcome=f"timed out after {timeout}s",
            output=_tail(_decode(exc)),
        )
        raise FriendlyError(
            f"Command timed out after {timeout}s: {' '.join(cmd)}",
            f"Try again; details in {_log_path()}",
        ) from exc
    if result.returncode == 0:
        # Success is one line: what ran and that it worked. Output is noise here.
        # A read-only command changed nothing, so it earns no line at all.
        if log:
            logger.info("proc.run", cmd=list(cmd), exit=0)
    else:
        # Failures are always recorded, whether or not the caller asked for logging.
        logger.warning(
            "proc.run",
            cmd=list(cmd),
            exit=result.returncode,
            output=_tail(result.stdout + result.stderr),
        )
    if check and result.returncode != 0:
        raise FriendlyError(
            f"Command failed: {' '.join(cmd)}",
            f"Details were logged to {_log_path()}",
        )
    return result


def _decode(exc: subprocess.TimeoutExpired) -> str:
    parts: list[str] = []
    for stream in (exc.stdout, exc.stderr):
        if stream is None:
            continue
        parts.append(stream.decode("utf-8", "replace") if isinstance(stream, bytes) else stream)
    return "".join(parts)


def _tail(text: str, lines: int = 40) -> str:
    """The last few lines of a failed command — enough to diagnose, not a dump."""
    return "\n".join(text.splitlines()[-lines:])
