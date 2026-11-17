"""The pids LePika wrote down, and asking a process to stop."""

from __future__ import annotations

import signal

from lepika import paths
from lepika.system import System


def record(name: str, pid: int) -> None:
    """Write `<name>.pid`."""
    paths.pid_file(name).write_text(str(pid))


def recorded(name: str) -> int | None:
    """The pid `<name>.pid` holds, or None when there is no file, it is not a number,
    or the number is not positive. Reads only: what a stale file means is the caller's.
    """
    pf = paths.pid_file(name)
    if not pf.exists():
        return None
    try:
        pid = int(pf.read_text().strip())
    except ValueError:
        # Truncated or garbage pid file: nothing to stop.
        return None
    if pid <= 0:
        # `os.kill(0, SIGTERM)` signals our whole process group, and negatives
        # signal a group too — neither is ever what a stale pid file meant.
        return None
    return pid


def forget(name: str) -> None:
    """Remove `<name>.pid` if it is there."""
    paths.pid_file(name).unlink(missing_ok=True)


def terminate(pid: int, os_name: str, *, sys: System) -> bool:
    """Ask `pid` to stop (taskkill /T /F on Windows, SIGTERM elsewhere).

    False when the signal found it gone or was refused; on Windows, only when
    launching `taskkill` was refused.
    """
    # One `try` around both, as the stops have always had it: `taskkill` itself
    # can be refused, and that is not ours to signal either.
    try:
        if os_name == "windows":
            sys.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False)
        else:
            sys.signal(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        # Gone, or recycled into a process we don't own (a truncated pid can
        # parse to a live foreign pid).
        return False
    return True
