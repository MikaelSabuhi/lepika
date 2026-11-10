"""The suite's own safety rails (docs/architecture.md rule 3)."""

from __future__ import annotations

import re
import subprocess
import urllib.request
from typing import Any

import pytest
from conftest import RealNetworkCall

from lepika import detect, express


def test_real_urlopen_is_replaced_for_every_test() -> None:
    with pytest.raises(RealNetworkCall, match=re.escape("http://example.invalid/")):
        urllib.request.urlopen("http://example.invalid/", timeout=0.1)


def test_the_guard_also_catches_a_keyword_url() -> None:
    with pytest.raises(RealNetworkCall, match=re.escape("http://example.invalid/")):
        urllib.request.urlopen(url="http://example.invalid/", timeout=0.1)


def test_the_guard_is_not_swallowed_by_except_exception() -> None:
    # webui_up catches Exception and returns False; the guard must still fail the test.
    with pytest.raises(RealNetworkCall):
        express.webui_up(3000)


def test_dockers_memory_is_never_asked_of_the_real_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Server-mode wizard tests size models through `detect.docker_memory_gb` without
    faking `run`: the real `docker info` would make them slow and decide them by the
    developer's own Docker Desktop memory setting."""

    def spawned(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"a test started a real process: {args[0] if args else kwargs}")

    monkeypatch.setattr(subprocess, "run", spawned)
    monkeypatch.setattr(subprocess, "Popen", spawned)
    assert detect.docker_memory_gb() == 0.0
