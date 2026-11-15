"""The suite's own safety rails (docs/architecture.md rule 3)."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import tomllib
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from conftest import REAL_URLOPEN, RealNetworkCall, RealProcessCall, guard_network

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


@pytest.mark.parametrize("launch", ["run", "call", "Popen"])
def test_a_real_subprocess_fails_the_test(launch: str) -> None:
    # Looked up on the module at call time, as production code does.
    launcher = getattr(subprocess, launch)
    with pytest.raises(RealProcessCall, match=re.escape("lepika-guard-probe")):
        launcher(["lepika-guard-probe", "--help"])


def test_a_real_signal_fails_the_test() -> None:
    with pytest.raises(RealProcessCall, match="pid 4242"):
        os.kill(4242, 15)


@pytest.mark.real_subprocess
def test_a_marked_test_may_start_a_child_but_still_cannot_signal() -> None:
    assert subprocess.run([sys.executable, "-c", "pass"], check=False).returncode == 0
    with pytest.raises(RealProcessCall, match="pid 4242"):
        os.kill(4242, 15)
    # Its own child is the one exception: `subprocess.run` kills a timed-out child.
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    os.kill(child.pid, signal.SIGTERM)
    assert child.wait(timeout=10) != 0


def test_the_process_guard_is_not_swallowed_by_a_broad_except() -> None:
    # `stop_openwebui` catches OSError around its signal and `except Exception` is
    # everywhere; the guard must still fail the test through both.
    with pytest.raises(RealProcessCall):
        try:
            os.kill(4242, 15)
        except Exception:
            pytest.fail("the guard was an ordinary exception")
    with pytest.raises(RealProcessCall):
        try:
            subprocess.run(["ps", "-eo", "pid=,args="], check=False)
        except Exception:
            pytest.fail("the guard was an ordinary exception")


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
