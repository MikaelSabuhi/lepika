from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point EZAI_HOME at a temp dir so no test touches the real ~/.ezai."""
    home = tmp_path / "ezai-home"
    monkeypatch.setenv("EZAI_HOME", str(home))
    return home
