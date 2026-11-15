from __future__ import annotations

import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeSystem

from lepika import config, detect, server, serving, stack_env

INFO = detect.SystemInfo("linux", "x86_64", "nvidia", 64.0, True, False, False)


def test_install_stack_copies_the_bundled_files_and_overwrites(isolated_home: Path) -> None:
    stack = server.install_stack()
    assert {p.name for p in stack.iterdir()} >= set(server.STACK_FILES)
    (stack / "compose.yml").write_text("tampered")
    server.install_stack()
    assert "tampered" not in (stack / "compose.yml").read_text()


def test_write_env_round_trips_and_quotes_values(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    stack_env.write_env(env, {"A": "1", "JSON": '{"0": {"key": "x"}}', "EMPTY": ""})
    assert stack_env.read_env(env) == {"A": "1", "JSON": '{"0": {"key": "x"}}', "EMPTY": ""}
    text = env.read_text()
    assert 'JSON=\'{"0": {"key": "x"}}\'' in text


def test_write_env_preserves_keys_it_does_not_manage(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("OLLAMA_IMAGE='ollama/ollama:0.11.4'\nMY_NOTE='keep me'\n")
    stack_env.write_env(env, {"WEBUI_PORT": "3000"})
    loaded = stack_env.read_env(env)
    assert loaded["OLLAMA_IMAGE"] == "ollama/ollama:0.11.4"
    assert loaded["MY_NOTE"] == "keep me"
    assert loaded["WEBUI_PORT"] == "3000"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_env_file_is_private(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    stack_env.write_env(env, {"LEPIKA_API_KEY": "secret"})
    assert stat.S_IMODE(os.stat(env).st_mode) == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_write_env_tightens_a_world_readable_file(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("MY_NOTE='keep me'\n")
    os.chmod(env, 0o644)
    stack_env.write_env(env, {"LEPIKA_API_KEY": "secret"})
    assert stat.S_IMODE(os.stat(env).st_mode) == 0o600
    assert stack_env.read_env(env)["MY_NOTE"] == "keep me"


def test_env_values_for_a_managed_engine() -> None:
    values = server.env_values(config.Config(mode="server"), existing={}, environ={})
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
    values = server.env_values(config.Config(mode="server"), existing=existing, environ={})
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
    values = server.env_values(cfg, existing={}, environ={})
    assert values["OLLAMA_BASE_URL"] == "http://gpu-box:11435"
    assert values["OLLAMA_API_CONFIGS"] == '{"0": {"key": "k"}}'


def test_env_values_rewrites_loopback_for_the_container() -> None:
    cfg = config.Config(mode="server", engine_managed=False, engine_url="http://127.0.0.1:11434")
    values = server.env_values(cfg, existing={}, environ={})
    assert values["OLLAMA_BASE_URL"] == "http://host.docker.internal:11434"
    assert server.container_engine_url("http://localhost:11434") == (
        "http://host.docker.internal:11434"
    )


def test_lepika_upstream_follows_a_remote_engine() -> None:
    # The `engine` profile is inactive with a remote engine, so proxying to
    # `ollama:11434` would be a 502 on every exposed request.
    cfg = config.Config(mode="server", engine_managed=False, engine_url="http://gpu-box:11435")
    assert server.env_values(cfg, existing={}, environ={})["LEPIKA_UPSTREAM"] == "gpu-box:11435"
    local = config.Config(mode="server", engine_managed=False, engine_url="http://127.0.0.1:11434")
    upstream = server.env_values(local, existing={}, environ={})["LEPIKA_UPSTREAM"]
    assert upstream == "host.docker.internal:11434"


def test_env_values_when_exposed_binds_the_ui_to_all_interfaces() -> None:
    values = server.env_values(config.Config(mode="server", exposed=True), existing={}, environ={})
    assert values["WEBUI_BIND"] == "0.0.0.0"


def test_hf_token_from_the_shell_wins() -> None:
    values = server.env_values(
        config.Config(mode="server"),
        existing={"HF_TOKEN": "old"},
        environ={"HF_TOKEN": "from-shell"},
    )
    assert values["HF_TOKEN"] == "from-shell"


def test_token_prompt_without_a_terminal_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)

    def no_terminal(*a: Any, **k: Any) -> str:
        raise EOFError

    env = tmp_path / ".env"
    server.hf_token_prompt(env, ask=no_terminal)
    assert stack_env.read_env(env).get("HF_TOKEN", "") == ""


def test_api_key_is_generated_once_and_reused(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    first = stack_env.api_key(env)
    assert len(first) >= 32
    assert stack_env.api_key(env) == first
    assert stack_env.read_env(env)["LEPIKA_API_KEY"] == first


def test_api_key_rotate_replaces_it(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    first = stack_env.api_key(env)
    assert stack_env.api_key(env, rotate=True) != first


def test_api_key_keeps_the_rest_of_the_env(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    stack_env.write_env(env, {"OLLAMA_IMAGE": "ollama/ollama:0.11.4"})
    stack_env.api_key(env, rotate=True)
    assert stack_env.read_env(env)["OLLAMA_IMAGE"] == "ollama/ollama:0.11.4"


def test_lan_ip_reports_the_address_the_route_picked() -> None:
    assert server.lan_ip(sys=FakeSystem(route_address="192.168.1.20")) == "192.168.1.20"
    # The route pick is not filtered: it is the address the kernel chose.
    assert server.lan_ip(sys=FakeSystem(route_address="127.0.0.1")) == "127.0.0.1"


def test_lan_ip_falls_back_to_a_placeholder() -> None:
    assert server.lan_ip(sys=FakeSystem(route_address=None)) == "<this machine's IP>"


def test_write_env_round_trips_values_containing_quotes(tmp_path: Path) -> None:
    """`lepika connect --key "it's"` reaches .env: an unescaped `'` ends the value early."""
    env = tmp_path / ".env"
    values = {"SINGLE": "it's", "DOUBLE": 'say "hi"', "BOTH": 'it\'s "x"', "BACKSLASH": "a\\b"}
    stack_env.write_env(env, values)
    assert stack_env.read_env(env) == values
    text = env.read_text()
    # A value with a `'` is written double-quoted, which compose's parser unescapes.
    assert 'SINGLE="it\'s"' in text


def test_write_env_keeps_a_dollar_literal_in_a_double_quoted_value(tmp_path: Path) -> None:
    """Compose interpolates `$VAR` inside double quotes, so the escape has to survive."""
    env = tmp_path / ".env"
    stack_env.write_env(env, {"KEY": "it's $HOME", "PLAIN": "no $HOME here"})
    assert stack_env.read_env(env) == {"KEY": "it's $HOME", "PLAIN": "no $HOME here"}
    text = env.read_text()
    assert 'KEY="it\'s $$HOME"' in text
    # A value with no `'` stays single-quoted, where `$` is already literal.
    assert "PLAIN='no $HOME here'" in text


def test_engine_label_says_remote_when_the_engine_is_not_ours() -> None:
    """`lepika up` and the wizard both hand it a config: a remote engine is not "Ollama"."""
    remote = config.Config(mode="server", engine_managed=False, engine_url="http://gpu-box:11435")
    assert serving.engine_label(remote) == "a remote engine"
    assert serving.engine_label(config.Config(mode="server")) == "Ollama"
    assert serving.engine_label(config.Config(mode="express", engine_managed=False)) == (
        "a remote engine"
    )


def test_upstream_keeps_https_for_a_remote_behind_tls() -> None:
    """Caddy's reverse_proxy speaks plain HTTP to a bare host:port — TLS needs the scheme."""
    assert server._upstream("https://gpu-box:11435") == "https://gpu-box:11435"
    assert server._upstream("https://gpu-box") == "https://gpu-box"
    assert server._upstream("http://gpu-box:11435") == "gpu-box:11435"
    assert server._upstream("http://gpu-box") == "gpu-box"


def test_env_upstream_keeps_the_scheme_of_an_https_remote() -> None:
    cfg = config.Config(mode="server", engine_managed=False, engine_url="https://gpu-box:11435")
    assert (
        server.env_values(cfg, existing={}, environ={})["LEPIKA_UPSTREAM"]
        == "https://gpu-box:11435"
    )


def test_lan_ips_lists_every_address_route_pick_first() -> None:
    machine = FakeSystem(
        route_address="192.168.1.20",
        host_addresses=["127.0.0.1", "169.254.1.9", "192.168.1.20", "10.0.0.5"],
    )
    # The route pick leads; loopback and link-local are not addresses anyone dials,
    # and the pick must not repeat itself.
    assert server.lan_ips(sys=machine) == ["192.168.1.20", "10.0.0.5"]


def test_lan_ips_with_one_address_is_just_the_route_pick() -> None:
    machine = FakeSystem(route_address="192.168.1.20", host_addresses=["192.168.1.20"])
    assert server.lan_ips(sys=machine) == ["192.168.1.20"]


def test_lan_ips_passes_the_route_probe_through() -> None:
    """The route pick is the one `lan_ip` reads, kept even where the hostname finds nothing."""
    machine = FakeSystem(route_address="127.0.0.1", host_addresses=[])
    assert server.lan_ips(sys=machine) == ["127.0.0.1"]


def test_lan_ips_survives_a_hostname_that_does_not_resolve() -> None:
    """An unresolvable hostname is normal on a laptop; the route pick still stands."""
    # `RealSystem.host_addresses` turns the resolver's OSError into no addresses.
    machine = FakeSystem(route_address="192.168.1.20", host_addresses=[])
    assert server.lan_ips(sys=machine) == ["192.168.1.20"]


def test_lan_ips_drops_the_placeholder_when_there_is_no_route() -> None:
    """`lan_ip`'s "<this machine's IP>" is words, not an address: it is not a list entry."""
    machine = FakeSystem(route_address=None, host_addresses=["192.168.1.20"])
    assert server.lan_ips(sys=machine) == ["192.168.1.20"]


def test_lan_ips_keeps_the_order_and_drops_repeats_among_the_host_addresses() -> None:
    machine = FakeSystem(host_addresses=["10.0.0.5", "192.168.1.20", "10.0.0.5"])
    assert server.lan_ips(sys=machine) == ["10.0.0.5", "192.168.1.20"]


def test_env_values_carry_the_context_length_to_the_engine_container() -> None:
    cfg = config.Config(mode="server", context_length=8192)
    assert server.env_values(cfg, existing={}, environ={})["OLLAMA_CONTEXT_LENGTH"] == "8192"


def test_compose_hands_the_context_length_to_the_ollama_service() -> None:
    text = (Path(server.__file__).parent / "stack" / "compose.yml").read_text()
    ollama_block = text.split("\n  ollama:")[1].split("\n  vllm:")[0]
    assert "OLLAMA_CONTEXT_LENGTH: ${OLLAMA_CONTEXT_LENGTH}" in ollama_block
