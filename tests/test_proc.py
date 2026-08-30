from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

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
    proc.run_logged([sys.executable, "-c", "print('hello')"])
    lines = (paths.logs_dir() / "lepika.log").read_text().splitlines()
    entry = json.loads(lines[-1])
    assert entry["event"] == "proc.run"
    assert entry["exit"] == 0
    # Success is one clean line; command output is only kept for failures.
    assert "output" not in entry


def test_run_logged_keeps_output_only_for_failures(isolated_home: Path) -> None:
    proc.run_logged([sys.executable, "-c", "print('boom'); raise SystemExit(2)"], check=False)
    entry = json.loads((paths.logs_dir() / "lepika.log").read_text().splitlines()[-1])
    assert entry["exit"] == 2
    assert "boom" in entry["output"]


def test_run_logged_with_log_false_records_nothing_for_a_read(isolated_home: Path) -> None:
    """Read-only commands (`sysctl -n hw.memsize`) change nothing, so they earn no log line."""
    proc.run_logged([sys.executable, "-c", "print('hello')"], log=False)
    log_path = paths.logs_dir() / "lepika.log"
    assert not log_path.exists() or log_path.read_text() == ""


def test_run_logged_with_log_false_still_records_failures(isolated_home: Path) -> None:
    """`log=False` silences the success line only — a failure is always worth a line."""
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


class FakePopen:
    """A child whose merged output is a canned byte string; records how it was started."""

    def __init__(self, output: bytes, code: int = 0) -> None:
        self.output = output
        self.returncode = code
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, cmd: list[str], **kwargs: Any) -> FakePopen:
        self.calls.append((list(cmd), dict(kwargs)))
        self.stdout = io.BytesIO(self.output)
        return self

    def __enter__(self) -> FakePopen:
        return self

    def __exit__(self, *a: Any) -> None:
        pass


def test_stream_echoes_raw_output_and_returns_a_tail() -> None:
    child = FakePopen(b"downloading 10%\rdownloading 100%\ndone\n")
    sink = io.BytesIO()
    code, tail = proc.stream(["hf", "download"], popen=child, sink=sink)
    assert code == 0
    # The terminal sees the bytes untouched, carriage returns included: progress
    # bars redraw in place instead of arriving as one long line at the end.
    assert sink.getvalue() == b"downloading 10%\rdownloading 100%\ndone\n"
    assert tail == "downloading 10%\ndownloading 100%\ndone"


def test_stream_passes_env_and_cwd_and_merges_stderr(tmp_path: Path) -> None:
    child = FakePopen(b"ok\n")
    proc.stream(
        ["ollama", "create"],
        env={"OLLAMA_HOST": "http://x"},
        cwd=tmp_path,
        popen=child,
        sink=io.BytesIO(),
    )
    cmd, kwargs = child.calls[0]
    assert cmd == ["ollama", "create"]
    assert kwargs["env"] == {"OLLAMA_HOST": "http://x"}
    assert kwargs["cwd"] == str(tmp_path)
    assert kwargs["stderr"] is subprocess.STDOUT


def test_stream_keeps_only_forty_segments_and_logs_failures(isolated_home: Path) -> None:
    output = b"".join(f"line {i}\n".encode() for i in range(50))
    child = FakePopen(output, code=3)
    code, tail = proc.stream(["hf", "download"], popen=child, sink=io.BytesIO())
    assert code == 3
    assert tail.splitlines()[0] == "line 10"
    assert tail.splitlines()[-1] == "line 49"
    entry = json.loads((paths.logs_dir() / "lepika.log").read_text().splitlines()[-1])
    assert entry["event"] == "proc.stream"
    assert entry["exit"] == 3
    assert "line 49" in entry["output"]


def test_stream_success_logs_nothing(isolated_home: Path) -> None:
    proc.stream(["hf", "download"], popen=FakePopen(b"fine\n"), sink=io.BytesIO())
    assert not (paths.logs_dir() / "lepika.log").exists()


def test_stream_missing_binary_is_friendly() -> None:
    def gone(cmd: list[str], **kwargs: Any) -> Any:
        raise FileNotFoundError(cmd[0])

    with pytest.raises(FriendlyError) as exc:
        proc.stream(["hf", "download"], popen=gone, sink=io.BytesIO())
    assert "hf" in exc.value.problem
