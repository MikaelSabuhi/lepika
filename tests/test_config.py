from __future__ import annotations

import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from lepika import config
from lepika.errors import FriendlyError


def test_load_returns_defaults_when_no_file(isolated_home: Path) -> None:
    cfg = config.load()
    assert cfg.mode == "express"
    assert cfg.webui_port == 3000
    assert cfg.schema_version == config.SCHEMA_VERSION


def test_save_then_load_round_trips(isolated_home: Path) -> None:
    cfg = config.Config(model="qwen3:8b", webui_port=3210)
    config.save(cfg)
    loaded = config.load()
    assert loaded == cfg
    assert config.config_path().read_text().startswith("schema_version = 1")


def test_save_leaves_no_temp_file_behind(isolated_home: Path) -> None:
    config.save(config.Config(model="qwen3:8b"))
    assert config.config_path().exists()
    assert not config.config_path().with_suffix(".toml.tmp").exists()


def test_save_keeps_previous_config_when_write_fails(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A save that dies mid-write must not damage the config already on disk."""
    original = config.Config(model="keep-me", webui_port=1234)
    config.save(original)
    before = config.config_path().read_text()

    real_fdopen = os.fdopen

    def exploding_fdopen(fd: int, *args: Any, **kwargs: Any) -> Any:
        # Half-write the temp file, then die before `save` can rename it into place.
        monkeypatch.setattr(os, "fdopen", real_fdopen)
        with real_fdopen(fd, *args, **kwargs) as handle:
            handle.write("half-written garbage")
        raise OSError("disk full")

    # `save` writes through os.fdopen, so that is the seam to break; do NOT call
    # monkeypatch.undo() here — it would also revert the autouse LEPIKA_HOME fixture
    # and point these assertions at the real ~/.lepika.
    monkeypatch.setattr(os, "fdopen", exploding_fdopen)
    with pytest.raises(OSError, match="disk full"):
        config.save(config.Config(model="never-lands"))

    assert config.config_path().read_text() == before
    assert config.load() == original


def test_save_escapes_quotes_and_backslashes_in_strings(isolated_home: Path) -> None:
    """A free-form model ref may contain `"` or `\\`; an unescaped one corrupts the file."""
    cfg = config.Config(model='we"ird\\ref')
    config.save(cfg)
    assert config.load() == cfg


def test_load_drops_unknown_keys(isolated_home: Path) -> None:
    config.config_path().write_text('schema_version = 1\nmode = "express"\nfuture_option = "x"\n')
    cfg = config.load()
    assert cfg.mode == "express"


def test_load_raises_friendly_error_on_corrupt_file(isolated_home: Path) -> None:
    config.config_path().write_text("this is not = valid = toml [[[\n")
    with pytest.raises(FriendlyError) as excinfo:
        config.load()
    assert "corrupted" in excinfo.value.problem
    assert "lepika" in excinfo.value.fix


def test_defaults_describe_a_managed_local_engine(isolated_home: Path) -> None:
    cfg = config.load()
    assert cfg.engine_managed is True
    assert cfg.engine_key == ""
    assert cfg.engine_url == config.DEFAULT_ENGINE_URL
    assert cfg.api_port == 11435
    assert cfg.exposed is False


def test_remote_engine_round_trips(isolated_home: Path) -> None:
    cfg = config.Config(
        engine_managed=False, engine_url="http://gpu-box:11435", engine_key="abc", exposed=True
    )
    config.save(cfg)
    assert config.load() == cfg


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_config_file_is_private(isolated_home: Path) -> None:
    """config.toml can hold an engine key; other users on the box must not read it."""
    config.save(config.Config(engine_key="abc"))
    assert stat.S_IMODE(os.stat(config.config_path()).st_mode) == 0o600

    # A config written by an older version is world-readable; saving must tighten it.
    os.chmod(config.config_path(), 0o644)
    config.save(config.Config(engine_key="abc"))
    assert stat.S_IMODE(os.stat(config.config_path()).st_mode) == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_save_tightens_a_stale_world_readable_temp_file(isolated_home: Path) -> None:
    """O_TRUNC keeps an existing file's bits, so a tmp left behind by an older run
    would hand the engine key to every user on the box."""
    tmp = config.config_path().with_suffix(".toml.tmp")
    tmp.write_text("leftover")
    os.chmod(tmp, 0o644)
    config.save(config.Config(engine_key="abc"))
    assert stat.S_IMODE(os.stat(config.config_path()).st_mode) == 0o600


def test_hf_token_round_trips_and_defaults_empty(isolated_home: Path) -> None:
    assert config.load().hf_token == ""
    config.save(config.Config(hf_token="hf_abc"))
    assert config.load().hf_token == "hf_abc"
