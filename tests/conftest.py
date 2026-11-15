from __future__ import annotations

import os
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from lepika import detect, system
from lepika.system import System

# Saved before any fixture patches it, so the guard's own test can tell the two apart.
REAL_URLOPEN = urllib.request.urlopen


class RealNetworkCall(BaseException):
    """A test reached the real `urllib.request.urlopen`.

    A BaseException on purpose: every opener in `src/` sits under an
    `except Exception` that turns failures into `False` or a fallback, which is
    exactly how an un-faked probe used to pass silently (docs/architecture.md rule 3).
    """


class RealProcessCall(BaseException):
    """A test started or signalled a real process.

    A BaseException for the same reason as `RealNetworkCall`: production code reads
    an `OSError` or any `Exception` from a launch or a signal as "not there" and
    carries on, which is how a real `ps` or a real SIGTERM went unnoticed.
    """


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point LEPIKA_HOME at a temp dir so no test touches the real ~/.lepika.

    HF_HUB_CACHE goes with it: `hf.preflight` stats the hub cache to size a file the
    listing reports as `-`, and the fallback is `~/.cache/huggingface/hub` — a real
    directory on a developer's machine, whose contents would decide the assertion.
    """
    home = tmp_path / "lepika-home"
    monkeypatch.setenv("LEPIKA_HOME", str(home))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub-cache"))
    return home


def guard_network(node: pytest.Item, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the real `urlopen` for every test except a `canary` one.

    The canary exists to talk to the real upstreams; it is opted out by marker only,
    and the default run deselects it (`addopts` in pyproject.toml).
    """
    if node.get_closest_marker("canary") is not None:
        return

    def refuse(*args: Any, **kwargs: Any) -> Any:
        target = args[0] if args else kwargs.get("url", "<unknown>")
        raise RealNetworkCall(f"test reached the network: {getattr(target, 'full_url', target)}")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)


@pytest.fixture(autouse=True)
def no_real_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that lets production code call the real `urlopen`."""
    guard_network(request.node, monkeypatch)


def guard_processes(node: pytest.Item, monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse a real child process and a real `os.kill`, except in a `canary` test,
    which is opted out by marker exactly as from the network guard.

    Launches are refused in `subprocess.Popen.__init__` on the class itself, which
    `run`, `call` and `check_output` all construct, so a reference to `Popen` captured
    in a parameter default (`proc.stream`) is refused too. Signals are refused by
    replacing the name `os.kill`. A builtin cannot be guarded any deeper, so a default
    that captured the real one before this fixture ran would be a gap; none does, and
    `test_no_function_captures_a_real_launcher_or_signal_as_a_default` fails on any
    new capture.

    A `real_subprocess` test may start a child, and may signal only a child it
    started: `subprocess.run` kills its own child on a timeout. Its registered
    description limits it to the test interpreter, but that is a convention: the
    guard does not check the program. A test that exercises `RealSystem` patches
    `subprocess.Popen` or `os.kill` with its own double; its `monkeypatch.setattr`
    runs after this autouse fixture and wins.
    """
    if node.get_closest_marker("canary") is not None:
        return
    children: set[int] = set()
    real_kill = os.kill
    real_init = vars(subprocess.Popen)["__init__"]
    marked = node.get_closest_marker("real_subprocess") is not None

    def guarded_init(self: subprocess.Popen[Any], *args: Any, **kwargs: Any) -> None:
        if not marked:
            argv = args[0] if args else kwargs.get("args", "<unknown>")
            raise RealProcessCall(f"test started a real process: {argv}")
        real_init(self, *args, **kwargs)
        children.add(self.pid)

    def guarded_kill(pid: int, sig: int) -> None:
        if pid not in children:
            raise RealProcessCall(f"test sent a real signal {sig} to pid {pid}")
        real_kill(pid, sig)

    monkeypatch.setattr(subprocess.Popen, "__init__", guarded_init)
    monkeypatch.setattr(os, "kill", guarded_kill)


@pytest.fixture(autouse=True)
def no_real_processes(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that lets production code start or signal a real process."""
    guard_processes(request.node, monkeypatch)


@pytest.fixture(autouse=True)
def no_real_docker_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep `docker info` a fake: a real one is slow and answers with the developer's own
    Docker Desktop memory. A test that hands it the real `System` gets "unknown" — what
    a stopped daemon answers — and sizes against its own SystemInfo, as before."""
    real = detect.docker_memory_gb

    def probe(*, sys: System) -> float:
        return 0.0 if sys is system.REAL else real(sys=sys)

    monkeypatch.setattr(detect, "docker_memory_gb", probe)
