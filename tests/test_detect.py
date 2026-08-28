from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from lepika import detect
from lepika.errors import FriendlyError


def fake_run(stdout: str) -> Any:
    def _run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    return _run


def test_detect_os_maps_platform_names() -> None:
    assert detect.detect_os("Darwin") == "macos"
    assert detect.detect_os("Linux") == "linux"
    assert detect.detect_os("Windows") == "windows"
    with pytest.raises(FriendlyError):
        detect.detect_os("SunOS")


def test_detect_gpu_apple_silicon() -> None:
    assert detect.detect_gpu("macos", "arm64", which=lambda name: None) == "apple"


def test_detect_gpu_nvidia_via_nvidia_smi() -> None:
    def which(name: str) -> str | None:
        return "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None

    assert detect.detect_gpu("linux", "x86_64", which=which) == "nvidia"


def test_detect_gpu_none() -> None:
    assert detect.detect_gpu("linux", "x86_64", which=lambda name: None) == "none"


def test_ram_macos_uses_sysctl() -> None:
    ram = detect.detect_ram_gb("macos", run=fake_run(str(32 * 2**30)))
    assert ram == pytest.approx(32.0)


def test_ram_linux_reads_meminfo(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       16384000 kB\nMemFree: 1 kB\n")
    ram = detect.detect_ram_gb("linux", meminfo=meminfo)
    assert ram == pytest.approx(16384000 / 2**20)


def test_windows_ram_converts_bytes_to_gb() -> None:
    assert detect._windows_ram_gb(status=lambda: (True, 16 * 2**30)) == pytest.approx(16.0)


def test_windows_ram_raises_when_the_memory_api_fails() -> None:
    """GlobalMemoryStatusEx returns a BOOL; ignoring it reports a 0 GB machine."""
    with pytest.raises(FriendlyError) as exc:
        detect._windows_ram_gb(status=lambda: (False, 0))
    assert "memory" in exc.value.problem
    assert "lepika doctor" in exc.value.fix


def test_api_up_true_and_false() -> None:
    def ok(req: Any, timeout: float = 0) -> Any:
        class R:
            def read(self) -> bytes:
                return b"{}"

        return R()

    def boom(req: Any, timeout: float = 0) -> Any:
        raise OSError("refused")

    assert detect.api_up("http://x", urlopen=ok) is True
    assert detect.api_up("http://x", urlopen=boom) is False


def test_api_up_sends_bearer_key_when_given() -> None:
    seen: list[Any] = []

    def opener(req: Any, timeout: float = 0) -> Any:
        seen.append(req)
        return object()

    assert detect.api_up("http://x", urlopen=opener, key="s3cret") is True
    request = seen[0]
    assert request.full_url == "http://x/api/version"
    assert request.get_header("Authorization") == "Bearer s3cret"


def test_api_up_without_key_sends_no_authorization_header() -> None:
    seen: list[Any] = []

    def opener(req: Any, timeout: float = 0) -> Any:
        seen.append(req)
        return object()

    assert detect.api_up("http://x", urlopen=opener) is True
    assert seen[0].get_header("Authorization") is None


def test_plan_sentence_mentions_gpu_and_mode() -> None:
    info = detect.SystemInfo(
        os="macos",
        arch="arm64",
        gpu="apple",
        ram_gb=36.0,
        has_docker=False,
        has_ollama=True,
        ollama_running=False,
    )
    sentence = detect.plan_sentence(info)
    assert "Express" in sentence
    assert "Metal" in sentence
