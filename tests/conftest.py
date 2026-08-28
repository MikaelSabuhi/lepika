from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point LEPIKA_HOME at a temp dir so no test touches the real ~/.lepika."""
    home = tmp_path / "lepika-home"
    monkeypatch.setenv("LEPIKA_HOME", str(home))
    return home
