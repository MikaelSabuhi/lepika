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
    # Services from profiles that are not active are stopped explicitly — in one
    # compose call, not one per profile.
    stops = [c for c in run.calls if "stop" in c]
    assert len(stops) == 1
    assert stops[0][stops[0].index("stop") + 1 :] == ["vllm", "caddy"]
    assert stops[0].count("--profile") == 2


# Port 11434 already answers: either our own ollama container (a second `lepika up`)
# or a native Ollama the container could never bind over.
ENGINE_ANSWERING = detect.SystemInfo("linux", "x86_64", "nvidia", 64.0, True, True, True)


def _start_with_ps(ps_stdout: str, call: Caller) -> tuple[str, Runner]:
    """Start the stack on a box where something already serves 11434.

    `docker compose ps -q ollama` is what tells the two apart: a container id means
    the engine on that port is ours.
    """
    run = Runner({"docker info": '{"nvidia": {}}', "docker compose": ps_stdout})
    url = server.start_stack(
        ENGINE_ANSWERING,
        config.Config(mode="server"),
        run=run,
        call=call,
        api_up=lambda url, **k: True,
        up=lambda port, **k: True,
        sleep=lambda s: None,
    )
    return url, run


def test_start_stack_proceeds_when_the_engine_on_11434_is_our_own_container(
    isolated_home: Path,
) -> None:
    """The stack publishes 127.0.0.1:11434 itself, so a second `lepika up` must not refuse."""
    call = Caller()
    url, run = _start_with_ps("9f8e7d6c5b4a\n", call)
    assert url == "http://localhost:3000"
    assert any(c[-3:] == ["up", "-d", "--remove-orphans"] for c in call.calls)
    ps_cmd = next(c for c in run.calls if "ps" in c)
    assert ps_cmd[-4:] == ["-q", "--status", "running", "ollama"]


def test_start_stack_refuses_when_native_ollama_holds_the_port(isolated_home: Path) -> None:
    call = Caller()
    with pytest.raises(FriendlyError) as exc:
        _start_with_ps("", call)
    assert "11434" in exc.value.problem
    # A loopback-bound native Ollama is not reachable from a container, so the only
    # real way forward is stopping it.
    assert "brew services stop ollama" in exc.value.fix
    assert "lepika connect" not in exc.value.fix
    assert call.calls == []


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


def test_stop_with_the_daemon_down_says_to_retry_down(isolated_home: Path) -> None:
    """`lepika down` cannot finish without the daemon: the containers come back with it."""
    with pytest.raises(FriendlyError) as exc:
        server.stop(LINUX_NVIDIA, config.Config(mode="server"), run=Runner(code=1))
    assert "not running" in exc.value.problem
    assert "lepika down" in exc.value.fix
    assert "lepika up" not in exc.value.fix


def test_update_with_the_daemon_down_says_to_retry_update(
    isolated_home: Path,
) -> None:
    with pytest.raises(FriendlyError) as exc:
        server.update(LINUX_NVIDIA, config.Config(mode="server"), run=Runner(code=1), call=Caller())
    assert "lepika update" in exc.value.fix
