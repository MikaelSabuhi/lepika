from __future__ import annotations

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


@pytest.fixture(autouse=True)
def no_real_docker_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep `docker info` a fake: a real one is slow and answers with the developer's own
    Docker Desktop memory. A test that hands it the real `System` gets "unknown" — what
    a stopped daemon answers — and sizes against its own SystemInfo, as before."""
    real = detect.docker_memory_gb

    def probe(*, sys: System) -> float:
        return 0.0 if sys is system.REAL else real(sys=sys)

    monkeypatch.setattr(detect, "docker_memory_gb", probe)
