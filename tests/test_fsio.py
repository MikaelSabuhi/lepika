from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

import pytest

from lepika import fsio


@pytest.mark.skipif(not hasattr(os, "fchmod"), reason="POSIX permission bits")
def test_write_private_creates_the_file_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "secret"
    fsio.write_private(path, "s3cret\n")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_write_private_replaces_atomically_and_leaves_no_temp_file(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("old\n", encoding="utf-8")
    fsio.write_private(path, "new\n")
    assert path.read_text(encoding="utf-8") == "new\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["config.toml"]


def test_write_private_keeps_the_old_file_when_the_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / ".env"
    fsio.write_private(path, "KEEP=1\n")
    real_fdopen = os.fdopen

    def exploding_fdopen(fd: int, *args: Any, **kwargs: Any) -> Any:
        # Half-write the temp file, then die before it can be renamed into place.
        monkeypatch.setattr(os, "fdopen", real_fdopen)
        with real_fdopen(fd, *args, **kwargs) as handle:
            handle.write("half-written garbage")
        raise OSError("disk full")

    monkeypatch.setattr(os, "fdopen", exploding_fdopen)
    with pytest.raises(OSError, match="disk full"):
        fsio.write_private(path, "NEVER=1\n")
    assert path.read_text(encoding="utf-8") == "KEEP=1\n"


@pytest.mark.skipif(not hasattr(os, "fchmod"), reason="POSIX permission bits")
def test_write_private_tightens_a_stale_world_readable_temp_file(tmp_path: Path) -> None:
    """O_TRUNC keeps an existing file's bits: a temp file an interrupted run left
    world-readable would otherwise hand its secret to every user on the box."""
    path = tmp_path / "secret_key"
    tmp = tmp_path / "secret_key.tmp"
    tmp.write_text("leftover", encoding="utf-8")
    os.chmod(tmp, 0o644)
    fsio.write_private(path, "s3cret\n")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert path.read_text(encoding="utf-8") == "s3cret\n"
