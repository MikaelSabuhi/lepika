from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import pytest


class RealNetworkCall(BaseException):
    """A test reached the real `urllib.request.urlopen`.

    A BaseException on purpose: every opener in `src/` sits under an
    `except Exception` that turns failures into `False` or a fallback, which is
    exactly how an un-faked probe used to pass silently (docs/architecture.md rule 3).
    """


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point LEPIKA_HOME at a temp dir so no test touches the real ~/.lepika."""
    home = tmp_path / "lepika-home"
    monkeypatch.setenv("LEPIKA_HOME", str(home))
    return home


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that lets production code call the real `urlopen`."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        target = args[0] if args else kwargs.get("url", "<unknown>")
        raise RealNetworkCall(f"test reached the network: {getattr(target, 'full_url', target)}")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
