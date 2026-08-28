from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ezai import express
from ezai.detect import SystemInfo
from ezai.errors import FriendlyError
from ezai.models import ModelRef
from ezai.paths import logs_dir


def info_for(os_name: str, has_ollama: bool = False) -> SystemInfo:
    return SystemInfo(
        os=os_name,  # type: ignore[arg-type]
        arch="x86_64",
        gpu="none",
        ram_gb=16.0,
        has_docker=False,
        has_ollama=has_ollama,
        ollama_running=False,
    )


class RunRecorder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str], **kwargs: Any) -> Any:
        self.calls.append(list(cmd))


class CallRecorder:
    """Stands in for subprocess.call: records argv, returns a canned exit code."""

    def __init__(self, code: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.code = code

    def __call__(self, cmd: list[str]) -> int:
        self.calls.append(list(cmd))
        return self.code


class PopenRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, cmd: list[str], **kwargs: Any) -> Any:
        self.calls.append((list(cmd), dict(kwargs)))
        return None


def test_install_ollama_macos_uses_brew() -> None:
    run = RunRecorder()
    express.install_ollama(
        info_for("macos"),
        run=run,
        which=lambda n: "/opt/homebrew/bin/brew" if n == "brew" else None,
    )
    assert ["brew", "install", "ollama"] in run.calls


def test_install_ollama_macos_without_brew_gives_download_link() -> None:
    with pytest.raises(FriendlyError) as exc:
        express.install_ollama(info_for("macos"), run=RunRecorder(), which=lambda n: None)
    assert "ollama.com" in exc.value.fix


def test_install_ollama_linux_streams_official_script() -> None:
    run = RunRecorder()
    call = CallRecorder()
    express.install_ollama(info_for("linux"), run=run, which=lambda n: None, call=call)
    assert any("ollama.com/install.sh" in " ".join(c) for c in call.calls)
    # Streamed, not captured: the script may prompt for sudo.
    assert run.calls == []


def test_install_ollama_linux_failure_is_friendly() -> None:
    with pytest.raises(FriendlyError) as exc:
        express.install_ollama(
            info_for("linux"), run=RunRecorder(), which=lambda n: None, call=CallRecorder(1)
        )
    assert "ollama.com/download/linux" in exc.value.fix


def test_install_ollama_windows_uses_winget() -> None:
    run = RunRecorder()
    express.install_ollama(info_for("windows"), run=run, which=lambda n: None)
    assert any(c[:2] == ["winget", "install"] for c in run.calls)


def test_start_ollama_windows_detaches_and_logs() -> None:
    popen = PopenRecorder()
    express.start_ollama("windows", popen=popen)
    cmd, kwargs = popen.calls[0]
    assert cmd == ["ollama", "serve"]
    assert kwargs["creationflags"] == 0x208
    assert "start_new_session" not in kwargs
    assert Path(kwargs["stdout"].name) == logs_dir() / "ollama.log"
    assert kwargs["stderr"] is kwargs["stdout"]
    kwargs["stdout"].close()


@pytest.mark.parametrize("os_name", ["linux", "macos"])
def test_start_ollama_posix_detaches_and_logs(os_name: str) -> None:
    popen = PopenRecorder()
    express.start_ollama(os_name, popen=popen)
    cmd, kwargs = popen.calls[0]
    assert cmd == ["ollama", "serve"]
    assert kwargs["start_new_session"] is True
    assert "creationflags" not in kwargs
    assert Path(kwargs["stdout"].name) == logs_dir() / "ollama.log"
    assert kwargs["stderr"] is kwargs["stdout"]
    kwargs["stdout"].close()


def test_start_ollama_missing_binary_is_friendly() -> None:
    def missing(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError(2, "No such file or directory: 'ollama'")

    with pytest.raises(FriendlyError) as exc:
        express.start_ollama("linux", popen=missing)
    assert "PATH" in exc.value.problem


def test_ensure_ollama_skips_install_and_start_when_running() -> None:
    run = RunRecorder()
    express.ensure_ollama(
        info_for("linux", has_ollama=True),
        run=run,
        which=lambda n: None,
        popen=lambda *a, **k: pytest.fail("should not start"),
        api_up=lambda *a, **k: True,
        sleep=lambda s: None,
    )
    assert run.calls == []


def test_ensure_ollama_probes_the_url_it_is_given() -> None:
    """A configured remote engine must be probed there, not on the local default."""
    probed: list[str] = []

    express.ensure_ollama(
        info_for("linux", has_ollama=True),
        run=RunRecorder(),
        which=lambda n: None,
        popen=lambda *a, **k: pytest.fail("should not start a local engine"),
        api_up=lambda url, **k: bool(probed.append(url)) or True,
        sleep=lambda s: None,
        url="http://gpu-box.local:11434",
    )
    assert probed == ["http://gpu-box.local:11434"]


def test_wait_for_raises_after_timeout() -> None:
    with pytest.raises(FriendlyError):
        express.wait_for(lambda: False, seconds=3, what="Ollama API", sleep=lambda s: None)


def test_pull_model_raises_friendly_on_nonzero_exit() -> None:
    with pytest.raises(FriendlyError):
        express.pull_model(ModelRef(raw="qwen3:8b", kind="ollama"), call=lambda cmd: 1)
