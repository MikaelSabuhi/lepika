from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeResponse, FakeSystem, FakeUI, routes

from lepika import config, detect, log, paths, server, stack_env
from lepika.errors import FriendlyError
from lepika.updates import UpdateStep

LINUX_NVIDIA = detect.Machine("linux", "x86_64", "nvidia", 64.0)
MAC = detect.Machine("macos", "arm64", "apple", 32.0)
# Docker on the PATH: every Server-mode entry point asks before it runs `docker`.
DOCKER = {"docker": "/usr/bin/docker"}


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
    machine = FakeSystem(
        stdout={"docker info": '{"nvidia": {"path": "nvidia-container-runtime"}}\n'}
    )
    assert server.nvidia_in_docker(LINUX_NVIDIA, sys=machine) is True
    assert (
        server.nvidia_in_docker(LINUX_NVIDIA, sys=FakeSystem(stdout={"docker info": "{}\n"}))
        is False
    )
    assert server.nvidia_in_docker(MAC, sys=FakeSystem()) is False


def test_ensure_docker_without_docker_never_asks_to_install_it_first() -> None:
    machine = FakeSystem()
    with pytest.raises(FriendlyError) as exc:
        server.ensure_docker(sys=machine)
    assert "Express" in exc.value.fix
    assert "docs.docker.com" in exc.value.fix
    assert machine.looked_up == ["docker"]
    assert machine.ran == []


def test_ensure_docker_daemon_down_is_friendly() -> None:
    with pytest.raises(FriendlyError) as exc:
        server.ensure_docker(sys=FakeSystem(which=DOCKER, codes={"docker": 1}))
    assert "Docker" in exc.value.problem


_ENGINE = "http://127.0.0.1:11434/api/version"
_UI = "http://127.0.0.1:3000/health"
_UP = FakeResponse()


def _probed(machine: FakeSystem) -> list[str]:
    return [request.full_url for request in machine.requested]


def _box(
    stdout: dict[str, str | list[str]], *, port_taken: bool = False, **kwargs: Any
) -> FakeSystem:
    """A Docker host whose engine container and OpenWebUI both answer their probes.

    Port 11434 answers once `compose up` has run, or from the start when `port_taken`:
    an Ollama, native or the stack's own container, already holds it.
    """
    webui = routes({_UI: _UP})

    def answer(request: urllib.request.Request) -> Any:
        if request.full_url != _ENGINE:
            return webui(request)
        if port_taken or any("up" in c for c in machine.called):
            return _UP
        raise OSError("connection refused")

    machine = FakeSystem(stdout=stdout, which=DOCKER, urls=answer, **kwargs)
    return machine


def test_start_stack_writes_env_ups_and_waits(isolated_home: Path) -> None:
    # A key in the config does not reach the readiness probe of the engine LePika runs.
    cfg = config.Config(mode="server", model="qwen3:8b", engine_key="k")
    machine = _box({"docker info": '{"nvidia": {}}\n'})
    # How many probes had been made when the hook ran.
    hook_ran_after: list[int] = []
    url = server.start_stack(
        LINUX_NVIDIA,
        cfg,
        after_engine=lambda: hook_ran_after.append(len(machine.requested)),
        sys=machine,
        ui=FakeUI(),
    )
    assert url == "http://localhost:3000"
    up_cmd = next(c for c in machine.called if "up" in c)
    assert up_cmd[-3:] == ["up", "-d", "--remove-orphans"]
    assert "--profile" in up_cmd and "engine" in up_cmd
    assert str(isolated_home / "stack" / "compose.nvidia.yml") in up_cmd
    env = stack_env.read_env(isolated_home / "stack" / ".env")
    assert env["OLLAMA_BASE_URL"] == "http://ollama:11434"
    # Port 11434 is asked about before anything starts, then the engine is waited for
    # before the hook runs, and the UI after it.
    assert _probed(machine) == [_ENGINE, _ENGINE, _UI]
    assert hook_ran_after == [2]
    # Neither the port check nor the readiness wait sends the config's key.
    engine_keys = [
        r.get_header("Authorization") for r in machine.requested if r.full_url == _ENGINE
    ]
    assert engine_keys == [None, None]
    # Services from profiles that are not active are stopped explicitly — in one
    # compose call, not one per profile.
    stops = [c for c in machine.ran if "stop" in c]
    assert len(stops) == 1
    assert stops[0][stops[0].index("stop") + 1 :] == ["vllm", "caddy"]
    assert stops[0].count("--profile") == 2


def test_start_stack_takes_the_shell_hf_token_from_the_system(isolated_home: Path) -> None:
    """The shell's token wins over the file's, and the shell is the system's environment."""
    stack_env.write_env(paths.stack_dir() / ".env", {"HF_TOKEN": "old"})
    machine = _box({"docker info": '{"nvidia": {}}\n'}, environ={"HF_TOKEN": "from-shell"})
    server.start_stack(LINUX_NVIDIA, config.Config(mode="server"), sys=machine, ui=FakeUI())
    assert stack_env.read_env(isolated_home / "stack" / ".env")["HF_TOKEN"] == "from-shell"


def _with_ps(ps_stdout: str) -> FakeSystem:
    """A box where something already serves 11434: either our own ollama container (a
    second `lepika up`) or a native Ollama the container could never bind over.

    `docker compose ps -q ollama` is what tells the two apart: a container id means
    the engine on that port is ours.
    """
    return _box({"docker info": '{"nvidia": {}}', "docker compose": ps_stdout}, port_taken=True)


def test_start_stack_proceeds_when_the_engine_on_11434_is_our_own_container(
    isolated_home: Path,
) -> None:
    """The stack publishes 127.0.0.1:11434 itself, so a second `lepika up` must not refuse."""
    machine = _with_ps("9f8e7d6c5b4a\n")
    url = server.start_stack(LINUX_NVIDIA, config.Config(mode="server"), sys=machine, ui=FakeUI())
    assert url == "http://localhost:3000"
    assert any(c[-3:] == ["up", "-d", "--remove-orphans"] for c in machine.called)
    ps_cmd = next(c for c in machine.ran if "ps" in c)
    assert ps_cmd[-4:] == ["-q", "--status", "running", "ollama"]


def test_start_stack_refuses_when_native_ollama_holds_the_port(isolated_home: Path) -> None:
    machine = _with_ps("")
    with pytest.raises(FriendlyError) as exc:
        server.start_stack(LINUX_NVIDIA, config.Config(mode="server"), sys=machine, ui=FakeUI())
    assert "11434" in exc.value.problem
    # A loopback-bound native Ollama is not reachable from a container, so the only
    # real way forward is stopping it.
    assert "brew services stop ollama" in exc.value.fix
    assert "lepika connect" not in exc.value.fix
    assert machine.called == []


def test_the_port_check_asks_the_stacks_own_port_not_the_configured_engine(
    isolated_home: Path,
) -> None:
    """`engine_url` is the user's to edit; the port compose publishes is not."""
    machine = _with_ps("")
    cfg = config.Config(mode="server", engine_url="http://127.0.0.1:11500")
    with pytest.raises(FriendlyError) as exc:
        server.start_stack(LINUX_NVIDIA, cfg, sys=machine, ui=FakeUI())
    assert "11434" in exc.value.problem
    assert _probed(machine) == [server.HOST_ENGINE_URL + "/api/version"]


def test_the_port_check_runs_after_docker_is_known_to_be_running(isolated_home: Path) -> None:
    """Asked when the check runs: Docker is confirmed first, the port second, and only
    an answering port is worth a `compose ps`."""
    ran_when_asked: list[list[list[str]]] = []
    webui = routes({_UI: _UP})

    def answer(request: urllib.request.Request) -> Any:
        if request.full_url != _ENGINE:
            return webui(request)
        ran_when_asked.append([list(c) for c in machine.ran])
        if any("up" in c for c in machine.called):
            return _UP
        raise OSError("connection refused")

    machine = FakeSystem(stdout={"docker info": "{}"}, which=DOCKER, urls=answer)
    server.start_stack(LINUX_NVIDIA, config.Config(mode="server"), sys=machine, ui=FakeUI())
    # The port check: `docker info` had run, and nothing else had.
    assert ran_when_asked[0] == [["docker", "info"]]
    assert not any("ps" in c for c in machine.ran)


def test_start_stack_refuses_to_expose_without_an_api_key(isolated_home: Path) -> None:
    # An empty key makes Caddy's `Bearer {$LEPIKA_API_KEY}` matcher accept a bare
    # `Bearer ` header, so exposing without one is an open proxy.
    machine = _box({"docker info": "{}"})
    with pytest.raises(FriendlyError) as exc:
        server.start_stack(
            LINUX_NVIDIA, config.Config(mode="server", exposed=True), sys=machine, ui=FakeUI()
        )
    assert "lepika expose" in exc.value.fix
    assert machine.called == []


def test_start_stack_compose_failure_is_friendly(isolated_home: Path) -> None:
    machine = _box({"docker info": "{}"}, codes={"docker compose": 1})
    with pytest.raises(FriendlyError) as exc:
        server.start_stack(LINUX_NVIDIA, config.Config(mode="server"), sys=machine, ui=FakeUI())
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
    remote = "http://gpu-box:11435/api/version"
    machine = FakeSystem(
        stdout={"docker info": "{}"}, which=DOCKER, urls=routes({remote: _UP, _UI: _UP})
    )
    server.start_stack(LINUX_NVIDIA, cfg, sys=machine, ui=FakeUI())
    # Asked once, with its key; nothing waits for a local engine.
    engine_probes = [r for r in machine.requested if r.full_url != _UI]
    assert [(r.full_url, r.get_header("Authorization")) for r in engine_probes] == [
        (remote, "Bearer k")
    ]
    up_cmd = next(c for c in machine.called if "up" in c)
    assert "engine" not in up_cmd


def test_stop_runs_compose_down(isolated_home: Path) -> None:
    machine = FakeSystem(which=DOCKER)
    assert server.stop(LINUX_NVIDIA, config.Config(mode="server"), sys=machine) is True
    assert any(c[-1] == "down" for c in machine.ran)


def test_update_pulls_then_starts(isolated_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    machine = FakeSystem(stdout={"docker info": "{}"}, which=DOCKER)
    started: list[str] = []
    monkeypatch.setattr(server, "start_stack", lambda info, cfg, **k: started.append("up") or "u")
    server.update(LINUX_NVIDIA, config.Config(mode="server"), sys=machine, ui=FakeUI())
    assert any(c[-1] == "pull" for c in machine.called)
    assert started == ["up"]


def _image_ids(before: str, after: str) -> FakeSystem:
    """`docker image inspect` answers `before` until the pull has run, then `after`."""
    return FakeSystem(
        stdout={
            "docker info": "{}",
            "docker compose": "ollama/ollama:latest\n",
            "docker image inspect": [before, after],
        },
        which=DOCKER,
    )


def _server_update(machine: FakeSystem) -> list[UpdateStep]:
    steps = server.update(LINUX_NVIDIA, config.Config(mode="server"), sys=machine, ui=FakeUI())
    inspects = [
        kw
        for c, kw in zip(machine.ran, machine.ran_kwargs, strict=True)
        if c[:3] == ["docker", "image", "inspect"]
    ]
    # Once before the pull and once after; a quiet read, whatever it exits with.
    assert len(inspects) <= 2
    unchecked = {"check": False, "env": None, "timeout": None, "log": False, "ok": (0,)}
    assert all(kw == unchecked for kw in inspects)
    return steps


def test_server_update_reports_current_when_image_ids_are_unchanged(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "start_stack", lambda info, cfg, **k: "u")
    machine = _image_ids("sha256:aaa\n", "sha256:aaa\n")
    steps = _server_update(machine)
    assert steps == [UpdateStep("Container images", "current")]
    inspect = next(c for c in machine.ran if c[:3] == ["docker", "image", "inspect"])
    assert inspect[-1] == "ollama/ollama:latest"


def test_server_update_reports_upgraded_when_they_change(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "start_stack", lambda info, cfg, **k: "u")
    steps = _server_update(_image_ids("sha256:aaa\n", "sha256:bbb\n"))
    assert steps == [UpdateStep("Container images", "upgraded")]


def test_server_update_reports_upgraded_when_images_appear_for_the_first_time(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No id before the pull and one after: the pull certainly fetched something."""
    monkeypatch.setattr(server, "start_stack", lambda info, cfg, **k: "u")
    steps = _server_update(_image_ids("", "sha256:bbb\n"))
    assert steps == [UpdateStep("Container images", "upgraded")]


@pytest.mark.parametrize(
    "machine",
    [
        FakeSystem(stdout={"docker info": "{}", "docker compose": ""}, which=DOCKER),
        _image_ids("", ""),
        _image_ids("sha256:aaa\n", ""),
    ],
    ids=["no images listed", "no ids", "no ids after the pull"],
)
def test_server_update_does_not_claim_up_to_date_when_image_ids_are_unreadable(
    machine: FakeSystem, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "start_stack", lambda info, cfg, **k: "u")
    steps = _server_update(machine)
    assert steps == [
        UpdateStep(
            "Container images",
            "note",
            "pulled the latest images; could not tell whether anything changed",
        )
    ]


def test_gpu_note_only_when_docker_cannot_see_the_gpu() -> None:
    note = server.gpu_note(LINUX_NVIDIA, sys=FakeSystem(stdout={"docker info": "{}"}))
    assert note is not None and "CPU" in note
    assert (
        server.gpu_note(LINUX_NVIDIA, sys=FakeSystem(stdout={"docker info": '{"nvidia": {}}'}))
        is None
    )
    assert server.gpu_note(MAC, sys=FakeSystem()) is None


def test_logs_returns_compose_and_lepika_logs(isolated_home: Path) -> None:
    from lepika import paths

    (paths.logs_dir() / "lepika.log").write_text('{"event": "x"}\n')
    machine = FakeSystem(stdout={"docker compose": "ollama-1 | ready\n"})
    sections = dict(server.logs(20, sys=machine))
    assert "ready" in sections["docker compose"]
    assert '"event": "x"' in sections["lepika.log"]
    tail_cmd = next(c for c in machine.ran if "logs" in c)
    assert "--tail" in tail_cmd and "20" in tail_cmd


def test_stop_with_the_daemon_down_says_to_retry_down(isolated_home: Path) -> None:
    """`lepika down` cannot finish without the daemon: the containers come back with it."""
    with pytest.raises(FriendlyError) as exc:
        server.stop(
            LINUX_NVIDIA,
            config.Config(mode="server"),
            sys=FakeSystem(which=DOCKER, codes={"docker info": 1}),
        )
    assert "not running" in exc.value.problem
    assert "lepika down" in exc.value.fix
    assert "lepika up" not in exc.value.fix


def test_update_with_the_daemon_down_says_to_retry_update(
    isolated_home: Path,
) -> None:
    with pytest.raises(FriendlyError) as exc:
        server.update(
            LINUX_NVIDIA,
            config.Config(mode="server"),
            sys=FakeSystem(which=DOCKER, codes={"docker info": 1}),
            ui=FakeUI(),
        )
    assert "lepika update" in exc.value.fix


def test_start_stack_names_the_mode_in_the_stack_up_line(isolated_home: Path) -> None:
    """Both backends write `stack.up`; only the mode tells the two apart in the log."""
    server.start_stack(
        LINUX_NVIDIA,
        config.Config(mode="server", model="qwen3:8b"),
        sys=_box({"docker info": '{"nvidia": {}}\n'}),
        ui=FakeUI(),
    )
    text = (paths.logs_dir() / log.LOG_FILE).read_text(encoding="utf-8")
    line = next(json.loads(x) for x in text.splitlines() if json.loads(x)["event"] == "stack.up")
    assert line["mode"] == "server"
