from __future__ import annotations

import json
from pathlib import Path

import pytest

from lepika import log, paths


def _lines(directory: Path | None = None) -> list[dict[str, object]]:
    logs = directory if directory is not None else paths.logs_dir()
    text = (logs / log.LOG_FILE).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_get_logger_writes_json_lines_to_the_log_file(isolated_home: Path) -> None:
    log.get_logger().info("stack.start", mode="server", port=3000)
    entries = _lines()
    assert entries[-1]["event"] == "stack.start"
    assert entries[-1]["mode"] == "server"
    assert entries[-1]["port"] == 3000
    assert "timestamp" in entries[-1]


def test_secret_looking_keys_are_redacted(isolated_home: Path) -> None:
    """A key or token must never land in a log file a user is told to paste into an issue."""
    log.get_logger().info("connect", url="http://gpu-box:11435", key="s3cret", hf_token="hf_x")
    entry = _lines()[-1]
    assert entry["url"] == "http://gpu-box:11435"
    assert entry["key"] == "***"
    assert entry["hf_token"] == "***"
    assert "s3cret" not in (paths.logs_dir() / log.LOG_FILE).read_text()


def test_logger_follows_lepika_home_changes(
    isolated_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tests (and LEPIKA_HOME users) must not share one cached file handle across homes."""
    log.get_logger().info("first")
    first_home = paths.logs_dir()
    assert _lines(first_home)[-1]["event"] == "first"

    monkeypatch.setenv("LEPIKA_HOME", str(tmp_path / "other-home"))
    log.get_logger().info("second")
    second_home = paths.logs_dir()
    assert second_home != first_home

    assert [e["event"] for e in _lines(second_home)] == ["second"]
    assert [e["event"] for e in _lines(first_home)] == ["first"]
