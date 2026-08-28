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
    assert "proc.run" in log_text


def test_run_logged_records_a_structured_entry(isolated_home: Path) -> None:
    import json

    proc.run_logged([sys.executable, "-c", "print('hello')"])
    lines = (paths.logs_dir() / "lepika.log").read_text().splitlines()
    entry = json.loads(lines[-1])
    assert entry["event"] == "proc.run"
    assert entry["exit"] == 0
    # Success is one clean line; command output is only kept for failures.
    assert "output" not in entry


def test_run_logged_keeps_output_only_for_failures(isolated_home: Path) -> None:
    import json

    proc.run_logged([sys.executable, "-c", "print('boom'); raise SystemExit(2)"], check=False)
    entry = json.loads((paths.logs_dir() / "lepika.log").read_text().splitlines()[-1])
    assert entry["exit"] == 2
    assert "boom" in entry["output"]


def test_run_logged_with_log_false_records_nothing_for_a_read(isolated_home: Path) -> None:
    """Read-only commands (`ollama list`) change nothing, so they earn no log line."""
    proc.run_logged([sys.executable, "-c", "print('hello')"], log=False)
    log_path = paths.logs_dir() / "lepika.log"
    assert not log_path.exists() or log_path.read_text() == ""


def test_run_logged_with_log_false_still_records_failures(isolated_home: Path) -> None:
    """`log=False` silences the success line only — a failure is always worth a line."""
    import json

    proc.run_logged(
        [sys.executable, "-c", "print('boom'); raise SystemExit(2)"], check=False, log=False
    )
    entry = json.loads((paths.logs_dir() / "lepika.log").read_text().splitlines()[-1])
    assert entry["event"] == "proc.run"
    assert entry["exit"] == 2
    assert "boom" in entry["output"]


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
