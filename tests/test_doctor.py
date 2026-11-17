from __future__ import annotations

import urllib.error
from pathlib import Path
from typing import Any

import fakes
import pytest
from fakes import FakeResponse, FakeSystem, routes
from typer.testing import CliRunner

from lepika import cli, doctor
from lepika.detect import Machine

runner = CliRunner()

ENGINE = "http://127.0.0.1:11434"
DOWN = urllib.error.URLError("refused")


def machine(
    *,
    engine: bool = True,
    webui: bool = True,
    engine_url: str = ENGINE,
    which: dict[str, str] | None = None,
    **kwargs: Any,
) -> FakeSystem:
    """The engine at `engine_url` and OpenWebUI on 3000, each answering or refusing."""
    return FakeSystem(
        urls=routes(
            {
                f"{engine_url}/api/version": FakeResponse() if engine else DOWN,
                "http://127.0.0.1:3000/health": FakeResponse() if webui else DOWN,
            }
        ),
        which=which
        if which is not None
        else {
            "uv": "/usr/bin/uv",
            "docker": "/usr/bin/docker",
            "ollama": "/usr/local/bin/ollama",
        },
        **kwargs,
    )


def engine_probes(fake: FakeSystem) -> list[str]:
    return [r.full_url for r in fake.requested if r.full_url.endswith("/api/version")]


def info(ram: float = 32.0) -> Machine:
    return Machine(
        os="linux",
        arch="x86_64",
        gpu="nvidia",
        ram_gb=ram,
    )


def test_all_green_when_everything_up(isolated_home: Path) -> None:
    results = doctor.run_checks(info(), sys=machine())
    assert all(r.ok for r in results)


def test_missing_ollama_has_hint(isolated_home: Path) -> None:
    results = doctor.run_checks(
        info(),
        sys=machine(engine=False, webui=False, which={"uv": "/usr/bin/uv"}),
    )
    failed = {r.name: r for r in results if not r.ok}
    assert "Ollama installed" in failed
    assert failed["Ollama installed"].hint


def test_low_ram_is_flagged(isolated_home: Path) -> None:
    results = doctor.run_checks(info(ram=4.0), sys=machine())
    ram_check = next(r for r in results if r.name == doctor.RAM_CHECK)
    assert ram_check.ok is False


def test_checks_probe_the_configured_engine_url(isolated_home: Path) -> None:
    """A remote engine must be diagnosed where it actually lives, not on localhost."""
    from lepika import config

    config.save(config.Config(engine_url="http://gpu-box.local:11434"))
    fake = machine(engine_url="http://gpu-box.local:11434")
    doctor.run_checks(info(), sys=fake)
    assert engine_probes(fake) == ["http://gpu-box.local:11434/api/version"]


def test_remote_engine_skips_the_local_install_check_and_uses_the_key(isolated_home: Path) -> None:
    """A remote engine is someone else's to install — only whether it answers is ours."""
    from lepika import config

    config.save(
        config.Config(engine_managed=False, engine_url="http://gpu-box:11435", engine_key="k")
    )
    fake = machine(engine_url="http://gpu-box:11435")
    results = doctor.run_checks(info(), sys=fake)
    assert "Ollama installed" not in {r.name for r in results}
    seen = [
        r.get_header("Authorization") for r in fake.requested if r.full_url.endswith("/api/version")
    ]
    assert seen == ["Bearer k"]


def test_a_remote_engine_hint_offers_reconnecting_with_a_new_key(isolated_home: Path) -> None:
    """A rotated key looks exactly like a dead box; the hint has to cover both."""
    from lepika import config

    config.save(config.Config(engine_managed=False, engine_url="http://gpu-box:11435"))
    results = doctor.run_checks(
        info(), sys=machine(engine=False, engine_url="http://gpu-box:11435")
    )
    hint = next(r for r in results if r.name == "Engine responding").hint
    assert "lepika connect http://gpu-box:11435 --key <key>" in hint
    assert "lepika connect --local" in hint


def test_the_openwebui_hint_points_at_the_log_that_holds_the_cause(isolated_home: Path) -> None:
    results = doctor.run_checks(info(), sys=machine(webui=False))
    hint = next(r for r in results if r.name == "OpenWebUI responding").hint
    assert "lepika logs" in hint
    assert "openwebui.log" in hint


def test_doctor_command_exits_zero_when_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        doctor,
        "run_checks",
        lambda i, **k: [doctor.CheckResult("uv installed", True, "install uv")],
    )
    result = runner.invoke(cli.app, ["doctor"], obj=fakes.deps(machine=info()))
    assert result.exit_code == 0
    assert "uv installed" in result.output


def test_doctor_command_fails_on_core_check_and_prints_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        doctor,
        "run_checks",
        lambda i, **k: [doctor.CheckResult("Engine responding", False, "Run `lepika up`")],
    )
    result = runner.invoke(cli.app, ["doctor"], obj=fakes.deps(machine=info()))
    assert result.exit_code == 1
    assert "Run `lepika up`" in result.output


def test_doctor_command_ram_warning_does_not_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        doctor,
        "run_checks",
        lambda i, **k: [doctor.CheckResult(doctor.RAM_CHECK, False, "4 GB detected")],
    )
    result = runner.invoke(cli.app, ["doctor"], obj=fakes.deps(machine=info(ram=4.0)))
    assert result.exit_code == 0
    assert "4 GB detected" in result.output


def test_server_mode_checks_docker_instead_of_native_ollama(isolated_home: Path) -> None:
    """Server mode's prerequisites are Docker's, so its checks are Docker's too."""
    from lepika import config

    config.save(config.Config(mode="server"))
    results = doctor.run_checks(
        info(),
        sys=machine(
            which={"docker": "/usr/bin/docker"},
            stdout={"docker info --format": '{"nvidia": {}}'},
        ),
    )
    names = {r.name for r in results}
    assert {"Docker running", "docker compose available", "NVIDIA GPU visible to Docker"} <= names
    assert "Ollama installed" not in names
    assert "uv installed" not in names
    assert all(r.ok for r in results)


def test_server_mode_never_probes_docker_when_it_is_not_installed(isolated_home: Path) -> None:
    """`docker info` with no docker binary is a FriendlyError, not a red ✗ with a hint."""
    from lepika import config

    config.save(config.Config(mode="server"))
    fake = machine(which={})
    results = doctor.run_checks(info(), sys=fake)
    assert fake.ran == [], "must not probe a Docker that isn't there"
    failed = {r.name: r for r in results if not r.ok}
    assert "Docker running" in failed
    assert "lepika --mode express" in failed["Docker running"].hint


def test_a_docker_info_that_times_out_is_a_red_check_not_an_abort(isolated_home: Path) -> None:
    """Docker Desktop mid-start accepts the socket and never answers: still a ✗ with a hint."""
    from lepika import config
    from lepika.errors import FriendlyError

    config.save(config.Config(mode="server"))

    hang = FriendlyError("Command timed out after 20s: docker info", "Try again")
    results = doctor.run_checks(
        info(), sys=machine(which={"docker": "/usr/bin/docker"}, raises={"docker": hang})
    )
    failed = {r.name: r for r in results if not r.ok}
    assert "Docker running" in failed
    assert "docker compose available" in failed


def test_a_compose_probe_that_times_out_is_a_red_check_not_an_abort(isolated_home: Path) -> None:
    """`docker compose version` hangs the way `docker info` does: a finding, not an abort."""
    from lepika import config
    from lepika.errors import FriendlyError

    config.save(config.Config(mode="server"))
    hang = FriendlyError("Command timed out after 20s: docker compose version", "Try again")
    results = doctor.run_checks(
        info(),
        sys=machine(
            which={"docker": "/usr/bin/docker"},
            stdout={"docker info --format": '{"nvidia": {}}'},
            raises={"docker compose version": hang},
        ),
    )
    rows = {r.name: r for r in results}
    assert rows["Docker running"].ok is True
    assert rows["docker compose available"].ok is False
    assert rows["docker compose available"].hint


def test_the_compose_probe_is_bounded_by_a_timeout(isolated_home: Path) -> None:
    """Unbounded, `run_logged` waits forever on a daemon that accepts and never answers."""
    from lepika import config

    config.save(config.Config(mode="server"))
    fake = machine(
        which={"docker": "/usr/bin/docker"}, stdout={"docker info --format": '{"nvidia": {}}'}
    )
    doctor.run_checks(info(), sys=fake)
    seen = [{"cmd": cmd, **k} for cmd, k in zip(fake.ran, fake.ran_kwargs, strict=True)]
    compose = next(c for c in seen if c["cmd"][:3] == ["docker", "compose", "version"])
    assert compose["timeout"] == 20


def test_a_gpu_probe_that_times_out_is_a_red_check_not_an_abort(isolated_home: Path) -> None:
    """The GPU probe is a third `docker info`: a Docker that hangs must not abort doctor."""
    from lepika import config
    from lepika.errors import FriendlyError

    config.save(config.Config(mode="server"))
    hang = FriendlyError("Command timed out after 20s: docker info", "Try again")
    results = doctor.run_checks(
        info(),
        sys=machine(which={"docker": "/usr/bin/docker"}, raises={"docker info --format": hang}),
    )
    rows = {r.name: r for r in results}
    assert rows["Docker running"].ok is True
    assert rows["NVIDIA GPU visible to Docker"].ok is False
    assert rows["NVIDIA GPU visible to Docker"].hint


def test_the_gpu_probe_is_bounded_by_a_timeout(isolated_home: Path) -> None:
    from lepika import config

    config.save(config.Config(mode="server"))
    fake = machine(
        which={"docker": "/usr/bin/docker"}, stdout={"docker info --format": '{"nvidia": {}}'}
    )
    doctor.run_checks(info(), sys=fake)
    seen = [{"cmd": cmd, **k} for cmd, k in zip(fake.ran, fake.ran_kwargs, strict=True)]
    gpu = next(c for c in seen if "--format" in c["cmd"])
    assert gpu["timeout"] == 20
