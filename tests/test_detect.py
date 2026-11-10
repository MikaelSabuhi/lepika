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


def test_plan_sentence_describes_server_mode() -> None:
    info = detect.SystemInfo("linux", "x86_64", "nvidia", 64.0, True, False, False)
    assert "Server mode" in detect.plan_sentence(info, mode="server")
    assert "docker compose" in detect.plan_sentence(info, mode="server")


def test_plan_sentence_names_the_engine_it_is_given() -> None:
    info = detect.SystemInfo("linux", "x86_64", "nvidia", 64.0, True, False, False)
    assert "OpenWebUI + vLLM" in detect.plan_sentence(info, mode="server", engine="vLLM")
    assert "OpenWebUI + Ollama" in detect.plan_sentence(info, mode="server")


def test_plan_sentence_names_a_remote_engine() -> None:
    info = detect.SystemInfo("linux", "x86_64", "nvidia", 64.0, True, False, False)
    sentence = detect.plan_sentence(info, mode="server", engine="a remote engine")
    assert "OpenWebUI + a remote engine)." in sentence


def _info(gpu: detect.Gpu, ram_gb: float) -> detect.SystemInfo:
    return detect.SystemInfo(
        os="linux",
        arch="x86_64",
        gpu=gpu,
        ram_gb=ram_gb,
        has_docker=False,
        has_ollama=True,
        ollama_running=True,
    )


def test_gpu_memory_sums_every_nvidia_card_in_decimal_gb() -> None:
    calls: list[list[str]] = []

    def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        assert kwargs == {"check": False, "log": False, "timeout": 15}
        return subprocess.CompletedProcess(cmd, 0, stdout="16303\n24576\n", stderr="")

    total = detect.gpu_memory_gb(_info("nvidia", 64.0), run=run)
    assert total == pytest.approx((16303 + 24576) * 1024 * 1024 / 1e9)
    assert calls == [["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]]


def test_gpu_memory_is_zero_when_nvidia_smi_fails_or_babbles() -> None:
    def hang(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FriendlyError("nvidia-smi timed out.", "Check the driver.")

    assert detect.gpu_memory_gb(_info("nvidia", 64.0), run=hang) == 0.0
    assert detect.gpu_memory_gb(_info("nvidia", 64.0), run=fake_run("NVIDIA-SMI has failed")) == 0.0
    assert detect.gpu_memory_gb(_info("nvidia", 64.0), run=fake_run("")) == 0.0


def test_gpu_memory_on_apple_silicon_is_metals_share_of_ram() -> None:
    def never(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise AssertionError("no subprocess on Apple Silicon")

    assert detect.gpu_memory_gb(_info("apple", 16.0), run=never) == pytest.approx(16 * 2 / 3)
    assert detect.gpu_memory_gb(_info("apple", 36.0), run=never) == pytest.approx(27.0)
    assert detect.gpu_memory_gb(_info("apple", 128.0), run=never) == pytest.approx(96.0)


def test_gpu_memory_without_a_gpu_is_zero() -> None:
    def never(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise AssertionError("no subprocess without a GPU")

    assert detect.gpu_memory_gb(_info("none", 16.0), run=never) == 0.0
