from __future__ import annotations

import dataclasses
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


def test_a_config_is_never_edited_in_place() -> None:
    """A change is a new value; the one that was loaded keeps describing the file."""
    cfg = config.Config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.model = "qwen3:8b"  # type: ignore[misc]
    assert dataclasses.replace(cfg, model="qwen3:8b").model == "qwen3:8b"
    assert cfg.model == ""


def test_save_then_load_round_trips(isolated_home: Path) -> None:
    cfg = config.Config(model="qwen3:8b", webui_port=3210)
    config.save(cfg)
    loaded = config.load()
    assert loaded == cfg
    assert config.config_path().read_text().startswith("schema_version = 1")


def test_saving_always_writes_the_current_schema(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A loaded file is upgraded in memory: saving it back under its old number would
    pair a newer key with a schema an older LePika thinks it can read."""
    monkeypatch.setattr(config, "SCHEMA_VERSION", 2)
    config.config_path().write_text('schema_version = 1\nmodel = "qwen3:8b"\n')
    config.save(config.load())
    lines = config.config_path().read_text().splitlines()
    assert "schema_version = 2" in lines
    assert 'model = "qwen3:8b"' in lines


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


def _load_error(text: str) -> FriendlyError:
    config.config_path().write_text(text, encoding="utf-8")
    with pytest.raises(FriendlyError) as excinfo:
        config.load()
    return excinfo.value


def test_an_unknown_key_suggests_the_nearest_setting(isolated_home: Path) -> None:
    err = _load_error("schema_version = 1\nwebui_prot = 8080\n")
    assert "webui_prot" in err.problem
    assert str(config.config_path()) in err.problem + err.fix
    assert "webui_port" in err.fix


def test_a_mistyped_mode_is_named_not_silently_express(isolated_home: Path) -> None:
    err = _load_error('mode = "sever"\n')
    assert "mode" in err.problem
    assert "sever" in err.problem
    assert "express" in err.problem + err.fix
    assert "server" in err.problem + err.fix


def test_a_quoted_boolean_does_not_expose_the_ui(isolated_home: Path) -> None:
    err = _load_error('exposed = "no"\n')
    assert "exposed" in err.problem


def test_a_quoted_port_is_refused(isolated_home: Path) -> None:
    err = _load_error('webui_port = "abc"\n')
    assert "webui_port" in err.problem


@pytest.mark.parametrize("port", [0, 70000])
def test_a_port_out_of_range_is_refused(isolated_home: Path, port: int) -> None:
    err = _load_error(f"api_port = {port}\n")
    assert "api_port" in err.problem
    assert str(port) in err.problem


def test_a_boolean_is_not_a_port(isolated_home: Path) -> None:
    err = _load_error("webui_port = true\n")
    assert "webui_port" in err.problem


@pytest.mark.parametrize("length", [-5, 0])
def test_context_length_must_be_positive(isolated_home: Path, length: int) -> None:
    err = _load_error(f"context_length = {length}\n")
    assert "context_length" in err.problem


def test_engine_url_needs_a_scheme(isolated_home: Path) -> None:
    err = _load_error('engine_url = "127.0.0.1:11434"\n')
    assert "engine_url" in err.problem
    assert "http://" in err.problem + err.fix


def test_a_newer_schema_is_refused_with_an_upgrade_hint(isolated_home: Path) -> None:
    err = _load_error('schema_version = 99\nmode = "express"\n')
    assert "newer" in err.problem
    assert "uv tool upgrade lepika" in err.fix


def test_a_newer_schema_wins_over_an_unknown_key(isolated_home: Path) -> None:
    """A newer LePika's file carries keys this one doesn't know; upgrading is the fix."""
    err = _load_error("some_future_key = 1\nschema_version = 99\n")
    assert "newer" in err.problem
    assert "some_future_key" not in err.problem


def test_validation_error_never_echoes_a_secret(isolated_home: Path) -> None:
    for key in ("engine_key", "hf_token"):
        err = _load_error(f"{key} = 12345\n")
        assert key in err.problem
        assert "12345" not in err.problem
        assert "12345" not in err.fix


def test_the_first_problem_in_file_order_is_reported(isolated_home: Path) -> None:
    err = _load_error('exposed = "no"\nmode = "sever"\n')
    assert "exposed" in err.problem
    assert "sever" not in err.problem


def test_a_config_saved_by_this_version_loads_without_complaint(isolated_home: Path) -> None:
    cfg = config.Config(model="x", engine_key="k", exposed=True)
    config.save(cfg)
    assert config.load() == cfg


def test_any_string_round_trips_through_save_and_load(isolated_home: Path) -> None:
    """Every control character must be escaped, or the saved file no longer parses."""
    cfg = config.Config(model="".join(chr(c) for c in range(0x80)) + "é✓")
    config.save(cfg)
    assert config.load() == cfg


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


def test_context_length_defaults_and_round_trips(isolated_home: Path) -> None:
    """Ollama's own default is 4096, which a pasted document overruns in one turn."""
    assert config.Config().context_length == config.DEFAULT_CONTEXT_LENGTH == 16384
    config.save(config.Config(context_length=32768))
    assert config.load().context_length == 32768


def test_load_fills_context_length_for_a_config_written_before_it_existed(
    isolated_home: Path,
) -> None:
    config.config_path().write_text('schema_version = 1\nmodel = "qwen3:8b"\n')
    assert config.load().context_length == 16384


class _Refused(Exception):
    pass


def test_save_after_saves_only_once_the_action_has_returned(isolated_home: Path) -> None:
    config.save(config.Config(model="old:model"))
    new = config.Config(model="new:model")
    seen: list[str] = []
    assert config.save_after(new, lambda: seen.append(config.load().model) or "done") == "done"
    assert seen == ["old:model"]
    assert config.load() == new


def test_save_after_saves_nothing_when_the_action_raises(isolated_home: Path) -> None:
    config.save(config.Config(model="old:model"))

    def refuse() -> None:
        raise _Refused

    with pytest.raises(_Refused):
        config.save_after(config.Config(model="new:model"), refuse)
    assert config.load().model == "old:model"


def test_save_before_saves_first_and_keeps_it_when_the_action_raises(
    isolated_home: Path,
) -> None:
    config.save(config.Config(exposed=True))
    seen: list[bool] = []

    def refuse() -> None:
        seen.append(config.load().exposed)
        raise _Refused

    with pytest.raises(_Refused):
        config.save_before(config.Config(exposed=False), refuse)
    assert seen == [False]
    assert config.load().exposed is False
    assert config.save_before(config.Config(), lambda: 7) == 7
