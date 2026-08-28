from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ezai import paths, proc
from ezai.errors import FriendlyError


def test_run_logged_captures_output_and_appends_log(isolated_home: Path) -> None:
    result = proc.run_logged([sys.executable, "-c", "print('hello')"])
    assert result.returncode == 0
    assert "hello" in result.stdout
    log_text = (paths.logs_dir() / "ezai.log").read_text()
    assert "hello" in log_text


def test_run_logged_raises_friendly_error_on_failure(isolated_home: Path) -> None:
    with pytest.raises(FriendlyError) as exc:
        proc.run_logged([sys.executable, "-c", "raise SystemExit(3)"])
    assert "ezai.log" in exc.value.fix


def test_run_logged_check_false_returns_result(isolated_home: Path) -> None:
    result = proc.run_logged([sys.executable, "-c", "raise SystemExit(3)"], check=False)
    assert result.returncode == 3
