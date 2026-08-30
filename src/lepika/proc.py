"""Single choke point for subprocess calls: captured, logged, friendly."""

from __future__ import annotations

import re
import subprocess
import sys
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO

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


_SEGMENT = re.compile(rb"[\r\n]")


def stream(
    cmd: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    popen: Callable[..., Any] = subprocess.Popen,
    sink: BinaryIO | None = None,
) -> tuple[int, str]:
    """Run a command the user has to watch, keeping only its tail for the log.

    `hf download` and `ollama create` take minutes and draw progress bars; captured
    by `run_logged` they look like a hang. The merged output is copied to the
    terminal byte for byte (carriage returns included, so bars redraw in place),
    while the last 40 segments are kept: a failure still gets rule 12's tail
    without the log holding a download's worth of bars. A secret travels in
    `env`, never on the command line.
    """
    out: BinaryIO = sink if sink is not None else sys.stdout.buffer
    try:
        child = popen(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=dict(env) if env is not None else None,
            cwd=str(cwd) if cwd is not None else None,
        )
    except FileNotFoundError as exc:
        get_logger().error("proc.stream", cmd=list(cmd), outcome="not found", output=str(exc))
        raise FriendlyError(
            f"Command not found: {cmd[0]}",
            f"Install {cmd[0]} or run `lepika doctor` for setup help.",
        ) from exc
    tail: deque[str] = deque(maxlen=40)
    pending = b""
    with child:
        # `read1` returns whatever has arrived rather than blocking until EOF, which
        # is what keeps a progress bar live; it is a `BufferedReader` method the
        # `IO[bytes]` hint on `stdout` does not carry.
        pipe: Any = child.stdout
        while chunk := pipe.read1(65536):
            out.write(chunk)
            out.flush()
            *done, pending = _SEGMENT.split(pending + chunk)
            tail.extend(s.decode("utf-8", "replace") for s in done if s.strip())
    if pending.strip():
        tail.append(pending.decode("utf-8", "replace"))
    code = int(child.returncode)
    text = "\n".join(tail)
    if code != 0:
        get_logger().warning("proc.stream", cmd=list(cmd), exit=code, output=text)
    return code, text


def _decode(exc: subprocess.TimeoutExpired) -> str:
    parts: list[str] = []
    for part in (exc.stdout, exc.stderr):
        if part is None:
            continue
        parts.append(part.decode("utf-8", "replace") if isinstance(part, bytes) else part)
    return "".join(parts)


def _tail(text: str, lines: int = 40) -> str:
    """The last few lines of a failed command — enough to diagnose, not a dump."""
    return "\n".join(text.splitlines()[-lines:])
