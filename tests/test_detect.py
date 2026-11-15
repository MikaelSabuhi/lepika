from __future__ import annotations

import io
import subprocess
import urllib.request
from collections.abc import Collection, Mapping, Sequence
from typing import Any

import pytest
from fakes import FakeSystem

from lepika import config, detect
from lepika.errors import FriendlyError

_NVIDIA_SMI = ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]
_DOCKER_INFO = ["docker", "info", "--format", "{{.MemTotal}}"]
# What `gpu_memory_gb` and `docker_memory_gb` ask of `run`: a read with a timeout.
_PROBE_KWARGS = {"check": False, "env": None, "timeout": 15, "log": False, "ok": (0,)}


class HungSystem(FakeSystem):
    """A `run` that times out, the way `proc.run_logged` reports a wedged program."""

    def run(
        self,
        cmd: Sequence[str],
        *,
        check: bool = True,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        log: bool = True,
        ok: Collection[int] = (0,),
    ) -> subprocess.CompletedProcess[str]:
        self.ran.append(list(cmd))
        raise FriendlyError(f"{cmd[0]} timed out.", "Check that it still answers.")


def _engine_answers(request: urllib.request.Request) -> Any:
    return io.BytesIO(b'{"version": "0.33.2"}')


def test_detect_os_maps_platform_names() -> None:
    assert detect.detect_os("Darwin") == "macos"
    assert detect.detect_os("Linux") == "linux"
    assert detect.detect_os("Windows") == "windows"
    with pytest.raises(FriendlyError):
        detect.detect_os("SunOS")


def test_detect_gpu_apple_silicon() -> None:
    assert detect.detect_gpu("macos", "arm64", FakeSystem()) == "apple"


def test_detect_gpu_nvidia_via_nvidia_smi() -> None:
    sys = FakeSystem(which={"nvidia-smi": "/usr/bin/nvidia-smi"})
    assert detect.detect_gpu("linux", "x86_64", sys) == "nvidia"


def test_detect_gpu_none() -> None:
    assert detect.detect_gpu("linux", "x86_64", FakeSystem()) == "none"


def test_ram_comes_from_the_system_in_gib() -> None:
    sys = FakeSystem(ram_bytes=32 * 2**30)
    assert detect.detect_ram_gb(sys) == 32.0
    assert sys.ran == []


def test_unknown_memory_is_a_friendly_error() -> None:
    """Guessing 0 GB would silently filter the curated list down to nothing."""
    with pytest.raises(FriendlyError) as exc:
        detect.detect_ram_gb(FakeSystem(ram_bytes=None))
    assert exc.value.problem == "Could not detect system memory."
    # Not `lepika doctor`: it detects memory too, and would fail the same way.
    assert exc.value.fix == (
        "Re-run the command; if it keeps happening, report it at "
        "https://github.com/MikaelSabuhi/lepika/issues."
    )


def test_detect_makes_no_subprocess_call() -> None:
    """Rule 6: detecting the machine starts no process, whatever is installed."""
    sys = FakeSystem(
        which={"nvidia-smi": "/usr/bin/nvidia-smi", "docker": "/usr/bin/docker"},
        urls=_engine_answers,
    )
    info = detect.detect(sys=sys)
    assert sys.ran == []
    assert sys.called == []
    assert info.ram_gb == 16.0
    assert info.has_docker is True
    assert info.has_ollama is False
    assert info.ollama_running is True


def test_detect_probes_the_default_engine_url() -> None:
    sys = FakeSystem(urls=_engine_answers)
    detect.detect(sys=sys)
    assert [r.full_url for r in sys.requested] == [config.DEFAULT_ENGINE_URL + "/api/version"]


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


def test_gpu_memory_sums_every_nvidia_card_in_gib() -> None:
    """GiB, like RAM and the Metal share: a 16 GB card (16303 MiB) must read as 16,
    not the 17 decimal GB the picker once showed on a card sold as 16."""
    sys = FakeSystem(stdout={"nvidia-smi": "16303\n24576\n"})
    total = detect.gpu_memory_gb(_info("nvidia", 64.0), sys)
    assert total == pytest.approx((16303 + 24576) / 1024)
    assert sys.ran == [_NVIDIA_SMI]
    assert sys.ran_kwargs == [_PROBE_KWARGS]


def test_gpu_memory_is_zero_when_nvidia_smi_fails_or_babbles() -> None:
    nvidia = _info("nvidia", 64.0)
    hung = HungSystem()
    assert detect.gpu_memory_gb(nvidia, hung) == 0.0
    assert hung.ran == [_NVIDIA_SMI]
    babble = FakeSystem(stdout={"nvidia-smi": "NVIDIA-SMI has failed"})
    assert detect.gpu_memory_gb(nvidia, babble) == 0.0
    assert detect.gpu_memory_gb(nvidia, FakeSystem(stdout={"nvidia-smi": ""})) == 0.0


def test_gpu_memory_on_apple_silicon_is_metals_share_of_ram() -> None:
    sys = FakeSystem()
    assert detect.gpu_memory_gb(_info("apple", 16.0), sys) == pytest.approx(16 * 2 / 3)
    assert detect.gpu_memory_gb(_info("apple", 36.0), sys) == pytest.approx(27.0)
    assert detect.gpu_memory_gb(_info("apple", 128.0), sys) == pytest.approx(96.0)
    assert sys.ran == []  # no subprocess on Apple Silicon


def test_gpu_memory_without_a_gpu_is_zero() -> None:
    sys = FakeSystem()
    assert detect.gpu_memory_gb(_info("none", 16.0), sys) == 0.0
    assert sys.ran == []  # no subprocess without a GPU


def test_docker_memory_reads_what_the_engine_container_can_use_in_gib() -> None:
    """Docker Desktop runs containers in a VM with its own, smaller memory."""
    sys = FakeSystem(stdout={"docker info": "8320299008\n"})
    assert detect.docker_memory_gb(sys) == pytest.approx(7.749, abs=0.001)
    assert sys.ran == [_DOCKER_INFO]
    assert sys.ran_kwargs == [_PROBE_KWARGS]


def test_docker_memory_is_zero_when_docker_fails_or_babbles() -> None:
    """Zero means unknown: callers fall back to the machine's own RAM."""
    hung = HungSystem()
    assert detect.docker_memory_gb(hung) == 0.0
    assert hung.ran == [_DOCKER_INFO]
    babble = FakeSystem(stdout={"docker info": "Cannot connect to the Docker daemon"})
    assert detect.docker_memory_gb(babble) == 0.0
    assert detect.docker_memory_gb(FakeSystem(stdout={"docker info": ""})) == 0.0
