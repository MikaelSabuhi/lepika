from __future__ import annotations

import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from lepika import config, detect, server

INFO = detect.SystemInfo("linux", "x86_64", "nvidia", 64.0, True, False, False)


def test_install_stack_copies_the_bundled_files_and_overwrites(isolated_home: Path) -> None:
    stack = server.install_stack()
    assert {p.name for p in stack.iterdir()} >= set(server.STACK_FILES)
    (stack / "compose.yml").write_text("tampered")
    server.install_stack()
    assert "tampered" not in (stack / "compose.yml").read_text()


def test_write_env_round_trips_and_quotes_values(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    server.write_env(env, {"A": "1", "JSON": '{"0": {"key": "x"}}', "EMPTY": ""})
    assert server.read_env(env) == {"A": "1", "JSON": '{"0": {"key": "x"}}', "EMPTY": ""}
    text = env.read_text()
    assert 'JSON=\'{"0": {"key": "x"}}\'' in text


def test_write_env_preserves_keys_it_does_not_manage(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("OLLAMA_IMAGE='ollama/ollama:0.11.4'\nMY_NOTE='keep me'\n")
    server.write_env(env, {"WEBUI_PORT": "3000"})
    loaded = server.read_env(env)
    assert loaded["OLLAMA_IMAGE"] == "ollama/ollama:0.11.4"
    assert loaded["MY_NOTE"] == "keep me"
    assert loaded["WEBUI_PORT"] == "3000"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_env_file_is_private(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    server.write_env(env, {"LEPIKA_API_KEY": "secret"})
    assert stat.S_IMODE(os.stat(env).st_mode) == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_write_env_tightens_a_world_readable_file(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("MY_NOTE='keep me'\n")
    os.chmod(env, 0o644)
    server.write_env(env, {"LEPIKA_API_KEY": "secret"})
    assert stat.S_IMODE(os.stat(env).st_mode) == 0o600
    assert server.read_env(env)["MY_NOTE"] == "keep me"


def test_env_values_for_a_managed_engine() -> None:
    values = server.env_values(config.Config(mode="server"), INFO, existing={})
    assert values["OLLAMA_BASE_URL"] == "http://ollama:11434"
    assert values["OLLAMA_API_CONFIGS"] == "{}"
    assert values["WEBUI_BIND"] == "127.0.0.1"
    assert values["WEBUI_PORT"] == "3000"
    assert values["API_PORT"] == "11435"
    assert values["ENABLE_OPENAI_API"] == "false"
    assert values["OPENWEBUI_IMAGE"] == "ghcr.io/open-webui/open-webui:main"
    assert values["OLLAMA_IMAGE"] == "ollama/ollama:latest"
    assert values["LEPIKA_UPSTREAM"] == "ollama:11434"
    assert values["VLLM_MODEL"] == ""


def test_env_values_keep_pinned_images_and_secrets() -> None:
    existing = {
        "OLLAMA_IMAGE": "ollama/ollama:0.11.4",
        "LEPIKA_API_KEY": "k",
        "HF_TOKEN": "hf",
    }
    values = server.env_values(config.Config(mode="server"), INFO, existing=existing)
    assert values["OLLAMA_IMAGE"] == "ollama/ollama:0.11.4"
    assert values["LEPIKA_API_KEY"] == "k"
    assert values["HF_TOKEN"] == "hf"


def test_env_values_for_a_remote_keyed_engine() -> None:
    cfg = config.Config(
        mode="server",
        engine_managed=False,
        engine_url="http://gpu-box:11435",
        engine_key="k",
    )
    values = server.env_values(cfg, INFO, existing={})
    assert values["OLLAMA_BASE_URL"] == "http://gpu-box:11435"
    assert values["OLLAMA_API_CONFIGS"] == '{"0": {"key": "k"}}'


def test_env_values_rewrites_loopback_for_the_container() -> None:
    cfg = config.Config(mode="server", engine_managed=False, engine_url="http://127.0.0.1:11434")
    values = server.env_values(cfg, INFO, existing={})
    assert values["OLLAMA_BASE_URL"] == "http://host.docker.internal:11434"
    assert server.container_engine_url("http://localhost:11434") == (
        "http://host.docker.internal:11434"
    )


def test_lepika_upstream_follows_a_remote_engine() -> None:
    # The `engine` profile is inactive with a remote engine, so proxying to
    # `ollama:11434` would be a 502 on every exposed request.
    cfg = config.Config(mode="server", engine_managed=False, engine_url="http://gpu-box:11435")
    assert server.env_values(cfg, INFO, existing={})["LEPIKA_UPSTREAM"] == "gpu-box:11435"
    local = config.Config(mode="server", engine_managed=False, engine_url="http://127.0.0.1:11434")
    upstream = server.env_values(local, INFO, existing={})["LEPIKA_UPSTREAM"]
    assert upstream == "host.docker.internal:11434"


def test_env_values_when_exposed_binds_the_ui_to_all_interfaces() -> None:
    values = server.env_values(config.Config(mode="server", exposed=True), INFO, existing={})
    assert values["WEBUI_BIND"] == "0.0.0.0"


def test_hf_token_from_the_shell_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "from-shell")
    values = server.env_values(config.Config(mode="server"), INFO, existing={"HF_TOKEN": "old"})
    assert values["HF_TOKEN"] == "from-shell"


def test_api_key_is_generated_once_and_reused(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    first = server.api_key(env)
    assert len(first) >= 32
    assert server.api_key(env) == first
    assert server.read_env(env)["LEPIKA_API_KEY"] == first


def test_api_key_rotate_replaces_it(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    first = server.api_key(env)
    assert server.api_key(env, rotate=True) != first


def test_api_key_keeps_the_rest_of_the_env(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    server.write_env(env, {"OLLAMA_IMAGE": "ollama/ollama:0.11.4"})
    server.api_key(env, rotate=True)
    assert server.read_env(env)["OLLAMA_IMAGE"] == "ollama/ollama:0.11.4"


def test_lan_ip_reports_the_address_the_socket_picked() -> None:
    # Binding stands in for the real UDP connect: either way the answer is the
    # local address the kernel chose, read back with getsockname.
    assert server.lan_ip(connect=lambda sock: sock.bind(("127.0.0.1", 0))) == "127.0.0.1"


def test_lan_ip_falls_back_to_a_placeholder() -> None:
    def boom(sock: Any) -> None:
        raise OSError("no route")

    assert server.lan_ip(connect=boom) == "<this machine's IP>"
