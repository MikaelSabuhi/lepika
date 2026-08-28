from __future__ import annotations

import sys
from pathlib import Path

import pytest

from lepika import paths, proc
from lepika.errors import FriendlyError


def test_run_logged_captures_output_and_appends_log(isolated_home: Path) -> None:
    result = proc.run_logged([sys.executable, "-c", "print('hello')"])
    assert result.returncode == 0
    assert "hello" in result.stdout
    log_text = (paths.logs_dir() / "lepika.log").read_text()
    assert "hello" in log_text


def test_run_logged_raises_friendly_error_on_failure(isolated_home: Path) -> None:
    with pytest.raises(FriendlyError) as exc:
        proc.run_logged([sys.executable, "-c", "raise SystemExit(3)"])
    assert "lepika.log" in exc.value.fix


def test_run_logged_check_false_returns_result(isolated_home: Path) -> None:
    result = proc.run_logged([sys.executable, "-c", "raise SystemExit(3)"], check=False)
    assert result.returncode == 3


def test_run_logged_missing_executable_is_friendly(isolated_home: Path) -> None:
    with pytest.raises(FriendlyError) as exc:
        proc.run_logged(["lepika-no-such-binary", "--help"])
    assert "lepika-no-such-binary" in exc.value.problem
    assert "lepika doctor" in exc.value.fix


def test_run_logged_timeout_is_friendly(isolated_home: Path) -> None:
    with pytest.raises(FriendlyError) as exc:
        proc.run_logged([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.2)
    assert "timed out" in exc.value.problem
    assert "lepika.log" in exc.value.fix
