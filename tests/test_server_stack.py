from __future__ import annotations

from pathlib import Path

import pytest
from fakes import Caller, Runner

from lepika import config, detect, server
from lepika.errors import FriendlyError

LINUX_NVIDIA = detect.SystemInfo("linux", "x86_64", "nvidia", 64.0, True, False, False)
MAC = detect.SystemInfo("macos", "arm64", "apple", 32.0, True, False, False)
NO_DOCKER = detect.SystemInfo("linux", "x86_64", "none", 16.0, False, False, False)


def test_profiles_follow_config() -> None:
    assert server.profiles(config.Config()) == ["engine"]
    assert server.profiles(config.Config(engine_managed=False)) == []
    assert server.profiles(config.Config(exposed=True)) == ["engine", "expose"]


def test_compose_cmd_lists_files_profiles_and_project_dir(tmp_path: Path) -> None:
    cmd = server.compose_cmd(tmp_path, ["engine", "expose"], gpu_overlay=True)
    assert cmd[:2] == ["docker", "compose"]
    assert cmd[cmd.index("--project-directory") + 1] == str(tmp_path)
    assert "-f" in cmd and str(tmp_path / "compose.yml") in cmd
    assert str(tmp_path / "compose.nvidia.yml") in cmd
    assert cmd.count("--profile") == 2


def test_compose_cmd_without_gpu_skips_the_overlay(tmp_path: Path) -> None:
    cmd = server.compose_cmd(tmp_path, ["engine"], gpu_overlay=False)
    assert str(tmp_path / "compose.nvidia.yml") not in cmd


def test_nvidia_in_docker_checks_the_runtime_on_linux() -> None:
    run = Runner({"docker info": '{"nvidia": {"path": "nvidia-container-runtime"}}\n'})
    assert server.nvidia_in_docker(LINUX_NVIDIA, run=run) is True
    assert server.nvidia_in_docker(LINUX_NVIDIA, run=Runner({"docker info": "{}\n"})) is False
    assert server.nvidia_in_docker(MAC, run=Runner()) is False


def test_ensure_docker_without_docker_never_asks_to_install_it_first() -> None:
    with pytest.raises(FriendlyError) as exc:
        server.ensure_docker(NO_DOCKER, run=Runner())
    assert "Express" in exc.value.fix
    assert "docs.docker.com" in exc.value.fix


def test_ensure_docker_daemon_down_is_friendly() -> None:
    with pytest.raises(FriendlyError) as exc:
        server.ensure_docker(LINUX_NVIDIA, run=Runner(code=1))
    assert "Docker" in exc.value.problem


def test_start_stack_writes_env_ups_and_waits(isolated_home: Path) -> None:
    cfg = config.Config(mode="server", model="qwen3:8b")
    run = Runner({"docker info": '{"nvidia": {}}\n'})
    call = Caller()
    events: list[str] = []
    url = server.start_stack(
        LINUX_NVIDIA,
        cfg,
        after_engine=lambda: events.append("after"),
        run=run,
        call=call,
        api_up=lambda url, **k: bool(events.append("engine-probe")) or True,
        up=lambda port, **k: bool(events.append("ui-probe")) or True,
        sleep=lambda s: None,
    )
    assert url == "http://localhost:3000"
    up_cmd = next(c for c in call.calls if "up" in c)
    assert up_cmd[-3:] == ["up", "-d", "--remove-orphans"]
    assert "--profile" in up_cmd and "engine" in up_cmd
    assert str(isolated_home / "stack" / "compose.nvidia.yml") in up_cmd
    env = server.read_env(isolated_home / "stack" / ".env")
    assert env["OLLAMA_BASE_URL"] == "http://ollama:11434"
    assert events.index("engine-probe") < events.index("after") < events.index("ui-probe")
    # Services from profiles that are not active are stopped explicitly.
    stops = [c for c in run.calls if "stop" in c]
    assert any("caddy" in c for c in stops)


def test_start_stack_refuses_when_native_ollama_holds_the_port(isolated_home: Path) -> None:
    info = detect.SystemInfo("macos", "arm64", "apple", 32.0, True, True, True)
    with pytest.raises(FriendlyError) as exc:
        server.start_stack(info, config.Config(mode="server"), run=Runner(), call=Caller())
    assert "11434" in exc.value.problem
    assert "lepika connect http://127.0.0.1:11434" in exc.value.fix


def test_start_stack_refuses_to_expose_without_an_api_key(isolated_home: Path) -> None:
    # An empty key makes Caddy's `Bearer {$LEPIKA_API_KEY}` matcher accept a bare
    # `Bearer ` header, so exposing without one is an open proxy.
    call = Caller()
    with pytest.raises(FriendlyError) as exc:
        server.start_stack(
            LINUX_NVIDIA,
            config.Config(mode="server", exposed=True),
            run=Runner({"docker info": "{}"}),
            call=call,
        )
    assert "lepika expose" in exc.value.fix
    assert call.calls == []


def test_start_stack_compose_failure_is_friendly(isolated_home: Path) -> None:
    with pytest.raises(FriendlyError) as exc:
        server.start_stack(
            LINUX_NVIDIA,
            config.Config(mode="server"),
            run=Runner({"docker info": "{}"}),
            call=Caller(1),
        )
    assert "lepika logs" in exc.value.fix


def test_start_stack_with_remote_engine_does_not_wait_for_a_container(
    isolated_home: Path,
) -> None:
    cfg = config.Config(
        mode="server",
        engine_managed=False,
        engine_url="http://gpu-box:11435",
        engine_key="k",
    )
    probed: list[tuple[str, str]] = []
    call = Caller()
    server.start_stack(
        LINUX_NVIDIA,
        cfg,
        run=Runner({"docker info": "{}"}),
        call=call,
        api_up=lambda url, key="", **k: bool(probed.append((url, key))) or True,
        up=lambda port, **k: True,
        sleep=lambda s: None,
    )
    assert probed == [("http://gpu-box:11435", "k")]
    up_cmd = next(c for c in call.calls if "up" in c)
    assert "engine" not in up_cmd


def test_stop_runs_compose_down(isolated_home: Path) -> None:
    run = Runner()
    assert server.stop(LINUX_NVIDIA, config.Config(mode="server"), run=run) is True
    assert any(c[-1] == "down" for c in run.calls)


def test_update_pulls_then_starts(isolated_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    call = Caller()
    started: list[str] = []
    monkeypatch.setattr(server, "start_stack", lambda info, cfg, **k: started.append("up") or "u")
    server.update(
        LINUX_NVIDIA,
        config.Config(mode="server"),
        run=Runner({"docker info": "{}"}),
        call=call,
    )
    assert any(c[-1] == "pull" for c in call.calls)
    assert started == ["up"]


def test_gpu_note_only_when_docker_cannot_see_the_gpu() -> None:
    note = server.gpu_note(LINUX_NVIDIA, run=Runner({"docker info": "{}"}))
    assert note is not None and "CPU" in note
    assert server.gpu_note(LINUX_NVIDIA, run=Runner({"docker info": '{"nvidia": {}}'})) is None
    assert server.gpu_note(MAC, run=Runner()) is None


def test_logs_returns_compose_and_lepika_logs(isolated_home: Path) -> None:
    from lepika import paths

    (paths.logs_dir() / "lepika.log").write_text('{"event": "x"}\n')
    run = Runner({"docker compose": "ollama-1 | ready\n"})
    sections = dict(server.logs(20, run=run))
    assert "ready" in sections["docker compose"]
    assert '"event": "x"' in sections["lepika.log"]
    tail_cmd = next(c for c in run.calls if "logs" in c)
    assert "--tail" in tail_cmd and "20" in tail_cmd
