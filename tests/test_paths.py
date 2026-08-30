from __future__ import annotations

from pathlib import Path

from lepika import paths


def test_lepika_home_honors_env_and_creates_dir(isolated_home: Path) -> None:
    home = paths.lepika_home()
    assert home == isolated_home
    assert home.is_dir()


def test_logs_dir_and_pid_file_live_under_home(isolated_home: Path) -> None:
    assert paths.logs_dir() == isolated_home / "logs"
    assert paths.logs_dir().is_dir()
    assert paths.pid_file("openwebui") == isolated_home / "openwebui.pid"


def test_stack_dir_lives_under_home(isolated_home: Path) -> None:
    assert paths.stack_dir() == isolated_home / "stack"
    assert paths.stack_dir().is_dir()


def test_openwebui_data_dir_lives_under_home(isolated_home: Path) -> None:
    assert paths.openwebui_data_dir() == isolated_home / "openwebui"
    assert paths.openwebui_data_dir().is_dir()


def test_hf_dir_is_under_home(isolated_home: Path) -> None:
    assert paths.hf_dir() == isolated_home / "hf"
    assert paths.hf_dir().is_dir()
