from __future__ import annotations

from pathlib import Path

from ezai import config


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


def test_load_drops_unknown_keys(isolated_home: Path) -> None:
    config.config_path().write_text('schema_version = 1\nmode = "express"\nfuture_option = "x"\n')
    cfg = config.load()
    assert cfg.mode == "express"
