"""The boundary between LePika and the machine it runs on.

Every external effect — a subprocess, a URL, a signal, a port probe, the clock —
is a method of `System`. `RealSystem` is the one implementation in `src`; the test
suite passes a fake with the same methods, so no test reaches the real machine.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, Self, runtime_checkable

from lepika import proc
from lepika.errors import CommandNotFound

# Windows: DETACHED_PROCESS (0x8) | CREATE_NEW_PROCESS_GROUP (0x200)
_WINDOWS_DETACH_FLAGS = 0x00000208

# Only defined on Windows, so it is looked up rather than referenced.
_WINDOWS_EXCLUSIVE_ADDR = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)


def _effective_uid() -> int:
    # Windows has no `os.geteuid`. Nothing asks there; the branch keeps the module
    # importable and type-checkable on every platform.
    if sys.platform == "win32":
        return -1
    return os.geteuid()


def _detach_kwargs(os_name: str) -> dict[str, Any]:
    if os_name == "windows":
        return {"creationflags": _WINDOWS_DETACH_FLAGS}
    return {"start_new_session": True}


def _bind_address(address: str, port: int) -> bool:
    """Try to bind one address, reporting whether the port was available there."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if os.name == "nt" and _WINDOWS_EXCLUSIVE_ADDR is not None:
            # Windows' SO_REUSEADDR lets two sockets own the same port outright,
            # which would hide every conflict. SO_EXCLUSIVEADDRUSE asks for the
            # exclusivity that POSIX gives by default.
            sock.setsockopt(socket.SOL_SOCKET, _WINDOWS_EXCLUSIVE_ADDR, 1)
        else:
            # SO_REUSEADDR matches what the OpenWebUI server itself sets, so a port
            # left in TIME_WAIT by the previous run reads as free here exactly as
            # it would there.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((address, port))
        except OSError:
            return False
    return True


def _spawn_with(
    popen: Callable[..., Any], cmd: Sequence[str], *, env: Mapping[str, str], log: Path
) -> int:
    """Start `cmd` detached through `popen`, its output appended to `log`; its pid."""
    handle = log.open("ab")
    try:
        child = popen(
            list(cmd),
            stdout=handle,
            stderr=handle,
            env=dict(env),
            **_detach_kwargs("windows" if os.name == "nt" else "posix"),
        )
    except FileNotFoundError as exc:
        raise CommandNotFound(cmd[0]) from exc
    finally:
        # The child holds its own copy of the descriptor; the parent's would
        # otherwise stay open for as long as LePika runs.
        handle.close()
    return int(child.pid)


def _global_memory_status() -> tuple[bool, int]:  # pragma: no cover - Windows only
    """Call GlobalMemoryStatusEx, returning (succeeded, total physical bytes)."""
    import ctypes

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MemoryStatusEx()
    stat.dwLength = ctypes.sizeof(MemoryStatusEx)
    ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
    return bool(ok), int(stat.ullTotalPhys)


class Response(Protocol):
    """What `open_url` hands back: the part of an HTTP response LePika reads."""

    status: int

    def read(self) -> bytes: ...
    def __iter__(self) -> Iterator[bytes]: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *exc: object) -> bool | None: ...


@runtime_checkable
class System(Protocol):
    @property
    def environ(self) -> Mapping[str, str]: ...

    def run(
        self,
        cmd: Sequence[str],
        *,
        check: bool = True,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        log: bool = True,
        ok: Collection[int] = (0,),
    ) -> subprocess.CompletedProcess[str]:
        """A captured command, logged and friendly (`proc.run_logged`)."""
        ...

    def call(self, cmd: Sequence[str]) -> int:
        """A command on the user's own terminal (`proc.call`)."""
        ...

    def stream(
        self,
        cmd: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
    ) -> tuple[int, str]:
        """A long command the user watches; returns its code and tail (`proc.stream`)."""
        ...

    def spawn(self, cmd: Sequence[str], *, env: Mapping[str, str], log: Path) -> int:
        """A detached child whose output appends to `log`; returns its pid."""
        ...

    def which(self, name: str) -> str | None: ...
    def open_url(self, request: urllib.request.Request, *, timeout: float) -> Response: ...
    def sleep(self, seconds: float) -> None: ...
    def signal(self, pid: int, sig: int) -> None: ...
    def can_bind(self, address: str, port: int) -> bool: ...
    def open_browser(self, url: str) -> None: ...

    def euid(self) -> int | None:
        """The effective uid, or None where the platform has none (Windows)."""
        ...

    def can_write(self, path: Path) -> bool:
        """May this process write into `path`? (`os.access` with `W_OK`.)"""
        ...

    def disk_free(self, path: Path) -> int: ...

    def ram_bytes(self) -> int | None:
        """Physical memory in bytes, or None when the machine will not say."""
        ...


class RealSystem:
    """The machine itself. Each method delegates to the module that owns the effect,
    looked up at call time, so the suite's guards (the `urlopen` one) still apply."""

    @property
    def environ(self) -> Mapping[str, str]:
        return os.environ

    def run(
        self,
        cmd: Sequence[str],
        *,
        check: bool = True,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        log: bool = True,
        ok: Collection[int] = (0,),
    ) -> subprocess.CompletedProcess[str]:
        return proc.run_logged(cmd, check=check, env=env, timeout=timeout, log=log, ok=ok)

    def call(self, cmd: Sequence[str]) -> int:
        return proc.call(cmd)

    def stream(
        self,
        cmd: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
    ) -> tuple[int, str]:
        return proc.stream(cmd, env=env, cwd=cwd)

    def spawn(self, cmd: Sequence[str], *, env: Mapping[str, str], log: Path) -> int:
        # Looked up at call time, so a test that patches `subprocess.Popen` reaches it.
        return _spawn_with(subprocess.Popen, cmd, env=env, log=log)

    def which(self, name: str) -> str | None:
        return shutil.which(name)

    def open_url(self, request: urllib.request.Request, *, timeout: float) -> Response:
        # B310 suppressed: every caller builds an http(s) URL; none comes from a file name.
        response: Response = urllib.request.urlopen(request, timeout=timeout)  # nosec B310
        return response

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def signal(self, pid: int, sig: int) -> None:
        os.kill(pid, sig)

    def can_bind(self, address: str, port: int) -> bool:
        return _bind_address(address, port)

    def open_browser(self, url: str) -> None:
        webbrowser.open(url)

    def euid(self) -> int | None:
        uid = _effective_uid()
        return None if uid < 0 else uid

    def can_write(self, path: Path) -> bool:
        return os.access(path, os.W_OK)

    def disk_free(self, path: Path) -> int:
        return shutil.disk_usage(path).free

    def ram_bytes(self) -> int | None:
        if sys.platform == "win32":
            # GlobalMemoryStatusEx signals failure through its BOOL return, leaving
            # the struct untouched: reading it anyway reported a 0 GB machine.
            ok, total = _global_memory_status()
            return total if ok else None
        # Asked of the C library, so no process is started (rule 6). An unknown
        # name is ValueError, and an answer of -1 is how sysconf says it does not know.
        try:
            total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError):
            return None
        return total if total > 0 else None


REAL: System = RealSystem()
