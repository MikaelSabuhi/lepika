from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import pytest

from lepika import detect, proc


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


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that lets production code call the real `urlopen`."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        target = args[0] if args else kwargs.get("url", "<unknown>")
        raise RealNetworkCall(f"test reached the network: {getattr(target, 'full_url', target)}")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)


@pytest.fixture(autouse=True)
def no_real_docker_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep `docker info` a fake: a real one is slow and answers with the developer's own
    Docker Desktop memory. A test that does not fake `run` gets "unknown" — what a
    stopped daemon answers — and sizes against its own SystemInfo, as before."""
    real = detect.docker_memory_gb

    def probe(run: Any = proc.run_logged) -> float:
        return 0.0 if run is proc.run_logged else real(run)

    monkeypatch.setattr(detect, "docker_memory_gb", probe)
