"""The suite's own safety rails (docs/architecture.md rule 3)."""

from __future__ import annotations

import re
import subprocess
import tomllib
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from conftest import REAL_URLOPEN, RealNetworkCall, guard_network

from lepika import detect, health, system


def test_real_urlopen_is_replaced_for_every_test() -> None:
    with pytest.raises(RealNetworkCall, match=re.escape("http://example.invalid/")):
        urllib.request.urlopen("http://example.invalid/", timeout=0.1)


def test_the_guard_also_catches_a_keyword_url() -> None:
    with pytest.raises(RealNetworkCall, match=re.escape("http://example.invalid/")):
        urllib.request.urlopen(url="http://example.invalid/", timeout=0.1)


def test_the_guard_is_not_swallowed_by_except_exception() -> None:
    # webui_up catches Exception and returns False; the guard must still fail the test.
    with pytest.raises(RealNetworkCall):
        health.webui_up(3000, sys=system.REAL)


def test_dockers_memory_is_never_asked_of_the_real_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Server-mode wizard tests size models through `detect.docker_memory_gb` without
    faking `System`: the real `docker info` would make them slow and decide them by the
    developer's own Docker Desktop memory setting."""

    def spawned(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"a test started a real process: {args[0] if args else kwargs}")

    monkeypatch.setattr(subprocess, "run", spawned)
    monkeypatch.setattr(subprocess, "Popen", spawned)
    assert detect.docker_memory_gb(sys=system.REAL) == 0.0


def test_canary_tests_are_deselected_by_default(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The canary talks to real upstreams, so the default run must leave it out — and
    the network guard must step aside for it, and only for it."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    options = tomllib.loads(pyproject.read_text(encoding="utf-8"))["tool"]["pytest"]
    assert "not canary" in options["ini_options"]["addopts"]

    # This test is unmarked, so the guard is in place.
    assert urllib.request.urlopen is not REAL_URLOPEN

    # The same node, marked canary, is left with the real opener.
    monkeypatch.setattr(urllib.request, "urlopen", REAL_URLOPEN)
    request.node.add_marker(pytest.mark.canary)
    guard_network(request.node, monkeypatch)
    assert urllib.request.urlopen is REAL_URLOPEN
