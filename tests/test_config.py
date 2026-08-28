from __future__ import annotations

from pathlib import Path

import pytest

from ezai import config
from ezai.errors import FriendlyError


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

    real_write_text = Path.write_text

    def exploding_write_text(self: Path, data: str, **kwargs: object) -> int:
        real_write_text(self, "half-written garbage", **kwargs)  # type: ignore[arg-type]
        raise OSError("disk full")

    # Only write_text is patched, so the reads below still work; do NOT call
    # monkeypatch.undo() here — it would also revert the autouse EZAI_HOME fixture
    # and point these assertions at the real ~/.ezai.
    monkeypatch.setattr(Path, "write_text", exploding_write_text)
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
    assert "ezai" in excinfo.value.fix
