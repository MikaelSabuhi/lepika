from __future__ import annotations

import json
import shlex
import signal
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeResponse, FakeSystem, UnansweredURL, routes

from lepika import config, health, mlx, ollama, paths, serving
from lepika.detect import Machine
from lepika.errors import FriendlyError
from lepika.paths import logs_dir
from lepika.system import Response


def info_for(os_name: str) -> Machine:
    return Machine(
        os=os_name,  # type: ignore[arg-type]
        arch="x86_64",
        gpu="none",
        ram_gb=16.0,
    )


_URL = "http://127.0.0.1:11434"
INSTALLED = {"ollama": "/usr/local/bin/ollama"}


def answering(*up: bool, url: str = _URL, **kwargs: Any) -> FakeSystem:
    """A machine whose engine at `url` answers /api/version as `up` says, probe by probe.

    The last answer repeats. With no answers, a probe fails the test instead of
    reading as "down": nothing may be asked of the network at all.
    """
    if not up:
        return FakeSystem(**kwargs)
    said = [FakeResponse() if yes else OSError("connection refused") for yes in up]
    return FakeSystem(urls=routes({f"{url}/api/version": said}), **kwargs)


def test_install_ollama_macos_uses_brew() -> None:
    machine = FakeSystem(which={"brew": "/opt/homebrew/bin/brew"})
    ollama.install_ollama(info_for("macos"), sys=machine)
    assert ["brew", "install", "ollama"] in machine.ran


def test_install_ollama_macos_without_brew_gives_download_link() -> None:
    with pytest.raises(FriendlyError) as exc:
        ollama.install_ollama(info_for("macos"), sys=FakeSystem())
    assert "ollama.com" in exc.value.fix


def test_install_ollama_linux_streams_official_script() -> None:
    machine = FakeSystem()
    ollama.install_ollama(info_for("linux"), sys=machine)
    assert any("ollama.com/install.sh" in " ".join(c) for c in machine.called)
    # Streamed, not captured: the script may prompt for sudo.
    assert machine.ran == []


def _systemd(*more: str) -> dict[str, str]:
    """A Linux box with systemd on PATH; `more` names the other binaries there."""
    return {name: f"/usr/bin/{name}" for name in ("systemctl", *more)}


_DISABLE = ["sudo", "systemctl", "disable", "--now", "ollama.service"]


def test_install_ollama_linux_survives_a_failed_optional_step() -> None:
    """The script aborts on a wedged systemd long after the engine is in place."""
    machine = FakeSystem(which=_systemd("ollama"), codes={"sh -c": 1}, euid=1000)
    ollama.install_ollama(info_for("linux"), sys=machine)
    # And the service the aborted script left enabled still gets disabled.
    assert machine.called[1:] == [_DISABLE]


def test_install_ollama_linux_failure_is_friendly() -> None:
    machine = FakeSystem(codes={"sh -c": 1})
    with pytest.raises(FriendlyError) as exc:
        ollama.install_ollama(info_for("linux"), sys=machine)
    assert "ollama.com/download/linux" in exc.value.fix
    # A failed install has no service to disable.
    assert len(machine.called) == 1


def test_install_ollama_linux_disables_the_systemd_service_the_script_enables() -> None:
    machine = FakeSystem(which=_systemd(), euid=1000)
    ollama.install_ollama(info_for("linux"), sys=machine)
    assert machine.called[1:] == [_DISABLE]


def test_install_ollama_linux_without_systemctl_has_no_service_to_disable() -> None:
    machine = FakeSystem()
    ollama.install_ollama(info_for("linux"), sys=machine)
    assert len(machine.called) == 1


def test_install_ollama_as_root_disables_the_unit_without_sudo() -> None:
    machine = FakeSystem(which=_systemd(), euid=0)
    ollama.install_ollama(info_for("linux"), sys=machine)
    assert machine.called[1:] == [["systemctl", "disable", "--now", "ollama.service"]]


def test_install_ollama_without_a_uid_disables_the_unit_through_sudo() -> None:
    """Only a uid of exactly 0 is root: a system that reports none is not."""
    machine = FakeSystem(which=_systemd(), euid=None)
    ollama.install_ollama(info_for("linux"), sys=machine)
    assert machine.called[1:] == [_DISABLE]


@pytest.mark.parametrize(
    "disable",
    [
        1,
        FriendlyError("Command not found: sudo", "Install sudo or run `lepika doctor`."),
        PermissionError(13, "Permission denied", "sudo"),
    ],
    ids=["exits non-zero", "no sudo", "sudo not executable"],
)
def test_install_ollama_linux_survives_a_failed_service_disable(
    disable: int | Exception,
) -> None:
    if isinstance(disable, Exception):
        machine = FakeSystem(which=_systemd(), raises={"sudo": disable}, euid=1000)
    else:
        machine = FakeSystem(which=_systemd(), codes={"sudo": disable}, euid=1000)
    ollama.install_ollama(info_for("linux"), sys=machine)
    assert machine.called[1:] == [_DISABLE]


def test_install_ollama_windows_uses_winget() -> None:
    machine = FakeSystem()
    ollama.install_ollama(info_for("windows"), sys=machine)
    assert any(c[:2] == ["winget", "install"] for c in machine.ran)


def test_start_ollama_spawns_serve_with_its_log(isolated_home: Path) -> None:
    """Detached by `System.spawn`, whose flags per OS are pinned in test_system."""
    machine = FakeSystem(environ={"HOME": "/h"})
    ollama.start_ollama(sys=machine)
    assert machine.spawned == [
        (
            ["ollama", "serve"],
            {"HOME": "/h", "OLLAMA_CONTEXT_LENGTH": str(config.DEFAULT_CONTEXT_LENGTH)},
            logs_dir() / "ollama.log",
        )
    ]


def test_start_ollama_missing_binary_is_friendly() -> None:
    with pytest.raises(FriendlyError) as exc:
        ollama.start_ollama(sys=FakeSystem(missing={"ollama"}))
    assert "PATH" in exc.value.problem
    assert not paths.pid_file("ollama").exists()


def test_start_ollama_records_the_pid(isolated_home: Path) -> None:
    """Without the pid file a mode switch cannot tell our Ollama from anyone else's."""
    ollama.start_ollama(sys=FakeSystem(spawn_pid=5151))
    assert paths.pid_file("ollama").read_text() == "5151"


def test_stop_ollama_no_pidfile_returns_false(isolated_home: Path) -> None:
    """An Ollama LePika did not start (brew, systemd, tray app) is never ours to stop."""
    # No URL is answered: a probe would fail the test, and a signal is recorded.
    machine = answering()
    assert ollama.stop_ollama("linux", _URL, sys=machine) is False
    assert (machine.requested, machine.ran, machine.signals) == ([], [], [])


def test_stop_ollama_malformed_pidfile_is_cleaned_up(isolated_home: Path) -> None:
    pf = paths.pid_file("ollama")
    pf.write_text("not-a-pid\n")
    machine = answering(True)
    assert ollama.stop_ollama("linux", _URL, sys=machine) is False
    assert machine.signals == []
    assert not pf.exists()


@pytest.mark.parametrize("content", ["0", "-5\n"])
def test_stop_ollama_nonpositive_pid_is_never_signalled(isolated_home: Path, content: str) -> None:
    """`os.kill(0, ...)` signals the whole process group; 0 and negatives never reach kill."""
    pf = paths.pid_file("ollama")
    pf.write_text(content)
    machine = answering(True)
    assert ollama.stop_ollama("linux", _URL, sys=machine) is False
    assert machine.signals == []
    assert not pf.exists()


def test_stop_ollama_stale_pidfile_is_never_signalled(isolated_home: Path) -> None:
    """After a reboot the recorded pid can belong to a stranger — the API decides (rule 7)."""
    pf = paths.pid_file("ollama")
    pf.write_text("4242")
    machine = answering(False, stdout=OLLAMA_PS)
    stopped = ollama.stop_ollama("linux", _URL, sys=machine)
    assert stopped is False
    assert machine.signals == []
    # Nothing answering settles it: the process table is not even asked.
    assert machine.ran == []
    assert not pf.exists()


# A process listing that says pid 4242 (or 1) really is an Ollama.
OLLAMA_PS = {
    "ps -p 4242": "/usr/local/bin/ollama\n",
    "ps -p 1": "/usr/local/bin/ollama\n",
    "tasklist": '"ollama.exe","4242","Console","1","94,208 K"\n',
}


def _probed(machine: FakeSystem) -> list[str]:
    return [request.full_url for request in machine.requested]


def test_stop_ollama_signals_when_the_api_answers(isolated_home: Path) -> None:
    paths.pid_file("ollama").write_text("4242")
    url = "http://gpu-box.local:11434"
    machine = answering(True, False, url=url, stdout=OLLAMA_PS)
    stopped = ollama.stop_ollama("linux", url, sys=machine)
    assert stopped is True
    assert machine.signals == [(4242, signal.SIGTERM)]
    # Probed where the config says the engine is, not on the local default.
    assert _probed(machine) == [f"{url}/api/version"] * 2
    assert machine.ran == [["ps", "-p", "4242", "-o", "comm="]]
    assert not paths.pid_file("ollama").exists()


def test_stop_ollama_passes_the_engine_key_to_the_probe(isolated_home: Path) -> None:
    """A keyed engine answers 401 without the key, which would read as "already down"."""
    paths.pid_file("ollama").write_text("4242")
    machine = answering(True, False, stdout=OLLAMA_PS)
    ollama.stop_ollama("linux", _URL, key="s3cret", sys=machine)
    keys = [request.get_header("Authorization") for request in machine.requested]
    assert keys == ["Bearer s3cret", "Bearer s3cret"]


class Waiting(FakeSystem):
    """Records, at each wait between shutdown checks, whether the pid file is still there."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.kept: list[bool] = []

    def sleep(self, seconds: float) -> None:
        super().sleep(seconds)
        self.kept.append(paths.pid_file("ollama").exists())


def test_stop_ollama_waits_for_the_engine_to_actually_exit(isolated_home: Path) -> None:
    """Ollama unloads models on the way out; port 11434 is only free once it is gone."""
    pf = paths.pid_file("ollama")
    pf.write_text("4242")
    said = [FakeResponse(), FakeResponse(), FakeResponse(), OSError("connection refused")]
    machine = Waiting(urls=routes({f"{_URL}/api/version": said}), stdout=OLLAMA_PS)
    stopped = ollama.stop_ollama("linux", _URL, sys=machine)
    assert stopped is True
    # The pid file is what a retry would need: it stays until the engine is gone.
    assert machine.kept == [True, True]
    assert machine.slept == [1, 1]
    assert not pf.exists()


def test_stop_ollama_that_never_dies_is_friendly_and_keeps_the_pid_file(
    isolated_home: Path,
) -> None:
    """Returning True here would send the wizard straight into a busy port 11434."""
    pf = paths.pid_file("ollama")
    pf.write_text("4242")
    machine = answering(True, stdout=OLLAMA_PS)
    with pytest.raises(FriendlyError) as exc:
        ollama.stop_ollama("linux", _URL, attempts=2, sys=machine)
    assert "still answering" in exc.value.problem
    assert "ollama serve" in exc.value.fix
    # Left behind on purpose: the next attempt still needs to know which pid is ours.
    assert pf.read_text() == "4242"


def test_stop_ollama_never_signals_a_pid_that_is_not_an_ollama(isolated_home: Path) -> None:
    """`ollama.pid` outlives `lepika down`, so a reboot can recycle it onto a stranger."""
    pf = paths.pid_file("ollama")
    pf.write_text("4242")
    safari = {"ps -p 4242": "/Applications/Safari.app/Contents/MacOS/Safari\n"}
    machine = answering(True, stdout=safari)
    stopped = ollama.stop_ollama("linux", _URL, sys=machine)
    assert stopped is False
    assert machine.signals == []
    assert not pf.exists()


def test_stop_ollama_windows_checks_tasklist_then_taskkills(isolated_home: Path) -> None:
    paths.pid_file("ollama").write_text("4242")
    machine = answering(True, False, stdout=OLLAMA_PS)
    stopped = ollama.stop_ollama("windows", _URL, sys=machine)
    assert stopped is True
    # Windows has no SIGTERM.
    assert machine.signals == []
    assert machine.ran == [
        ["tasklist", "/FI", "PID eq 4242", "/FO", "CSV", "/NH"],
        ["taskkill", "/PID", "4242", "/T", "/F"],
    ]
    assert not paths.pid_file("ollama").exists()


def test_stop_ollama_windows_leaves_a_pid_that_is_not_an_ollama_alone(isolated_home: Path) -> None:
    pf = paths.pid_file("ollama")
    pf.write_text("4242")
    notepad = {"tasklist": '"notepad.exe","4242","Console","1","9,208 K"\n'}
    machine = answering(True, stdout=notepad)
    assert ollama.stop_ollama("windows", _URL, sys=machine) is False
    assert machine.ran == [["tasklist", "/FI", "PID eq 4242", "/FO", "CSV", "/NH"]]
    assert not pf.exists()


def test_stop_ollama_permission_error_is_not_fatal(isolated_home: Path) -> None:
    """A truncated pid can parse to a live foreign pid — refusing it is not a crash."""
    pf = paths.pid_file("ollama")
    pf.write_text("1")
    denied = {1: PermissionError(1, "Operation not permitted")}
    machine = answering(True, False, stdout=OLLAMA_PS, signal_errors=denied)
    assert ollama.stop_ollama("linux", _URL, sys=machine) is True
    assert machine.signals == [(1, signal.SIGTERM)]
    assert not pf.exists()


def test_stop_ollama_survives_a_process_that_is_already_gone(isolated_home: Path) -> None:
    """The engine can exit between the probe and the signal: the wait decides."""
    pf = paths.pid_file("ollama")
    pf.write_text("4242")
    gone = {4242: ProcessLookupError(3, "No such process")}
    machine = answering(True, False, stdout=OLLAMA_PS, signal_errors=gone)
    assert ollama.stop_ollama("linux", _URL, sys=machine) is True
    assert machine.signals == [(4242, signal.SIGTERM)]
    assert not pf.exists()


def test_ensure_ollama_skips_install_and_start_when_running() -> None:
    machine = answering(True, which=INSTALLED)
    ollama.ensure_ollama(info_for("linux"), context_length=16384, sys=machine)
    assert machine.ran == []
    assert machine.spawned == []
    assert machine.looked_up == ["ollama"]


def test_ensure_ollama_installs_when_ollama_is_not_on_the_path() -> None:
    """Asked when it is needed: nothing found means install, then use what answers."""
    machine = answering(True, which={"brew": "/opt/homebrew/bin/brew"})
    ollama.ensure_ollama(info_for("macos"), context_length=16384, sys=machine)
    assert machine.ran == [["brew", "install", "ollama"]]
    assert machine.looked_up[0] == "ollama"
    assert machine.spawned == []


def test_ensure_ollama_probes_the_url_it_is_given() -> None:
    """A configured remote engine must be probed there, not on the local default."""
    url = "http://gpu-box.local:11434"
    machine = answering(True, url=url, which=INSTALLED)
    ollama.ensure_ollama(info_for("linux"), context_length=16384, url=url, sys=machine)
    assert _probed(machine) == [f"{url}/api/version"]
    # The engine it manages is probed without a key.
    assert [r.get_header("Authorization") for r in machine.requested] == [None]
    assert machine.spawned == []


def test_wait_for_raises_after_timeout() -> None:
    with pytest.raises(FriendlyError):
        health.wait_for(lambda: False, seconds=3, what="Ollama API", sys=FakeSystem())


def _info(os_name: str, arch: str, gpu: str) -> Machine:
    return Machine(os_name, arch, gpu, 32.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("os_name", "arch", "gpu", "expected"),
    [
        ("macos", "arm64", "apple", True),
        # NVIDIA amd64: `ensure_mlx` installs Ollama's MLX bundle on the import path.
        ("linux", "x86_64", "nvidia", True),
        ("windows", "amd64", "nvidia", True),
        ("linux", "aarch64", "nvidia", False),  # no MLX-CUDA bundle for arm64 (Jetson)
        # An Intel Mac with nvidia-smi: no MLX bundle for macOS, only the arm64 build.
        ("macos", "x86_64", "nvidia", False),
        ("macos", "x86_64", "none", False),
        ("linux", "x86_64", "none", False),
        ("windows", "amd64", "none", False),
    ],
)
def test_import_allowed_needs_an_mlx_capable_machine(
    os_name: str, arch: str, gpu: str, expected: bool
) -> None:
    assert serving.import_allowed(config.Config(), _info(os_name, arch, gpu)) is expected


def test_import_allowed_is_express_only_and_needs_our_own_engine() -> None:
    mac = _info("macos", "arm64", "apple")
    assert serving.import_allowed(config.Config(mode="server"), mac) is False
    assert serving.import_allowed(config.Config(engine_managed=False), mac) is False


def test_ollama_install_dir_mirrors_install_sh(tmp_path: Path) -> None:
    exe = tmp_path / "usr" / "local" / "bin" / "ollama"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    linux = FakeSystem(which={"ollama": str(exe)})
    assert mlx.ollama_install_dir("linux", sys=linux) == tmp_path / "usr" / "local"
    win = tmp_path / "Programs" / "Ollama" / "ollama.exe"
    win.parent.mkdir(parents=True)
    win.write_text("")
    windows = FakeSystem(which={"ollama": str(win)})
    assert mlx.ollama_install_dir("windows", sys=windows) == win.parent
    assert mlx.ollama_install_dir("linux", sys=FakeSystem()) is None


def test_mlx_present_looks_for_the_cuda_backend_dir(tmp_path: Path) -> None:
    assert mlx.mlx_present(tmp_path) is False
    (tmp_path / "lib" / "ollama" / "mlx_cuda_v13").mkdir(parents=True)
    assert mlx.mlx_present(tmp_path) is True


def test_cuda_major_parses_nvidia_smi_and_defaults_to_zero() -> None:
    smi = "| NVIDIA-SMI 580.65   Driver Version: 580.65   CUDA Version: 13.0  |"
    assert mlx.cuda_major(sys=FakeSystem(stdout={"nvidia-smi": smi})) == 13
    assert mlx.cuda_major(sys=FakeSystem(stdout={"nvidia-smi": "no gpu"})) == 0
    gone = FriendlyError("Command not found: nvidia-smi", "Install it.")
    assert mlx.cuda_major(sys=FakeSystem(raises={"nvidia-smi": gone})) == 0


def test_cuda_major_bounds_a_wedged_driver() -> None:
    """A hung `nvidia-smi` must not stall the import: the probe is timed out and reads as 0."""
    timed_out = FriendlyError("nvidia-smi timed out after 15s.", "Try again.")
    machine = FakeSystem(raises={"nvidia-smi": timed_out})
    assert mlx.cuda_major(sys=machine) == 0
    seen = machine.ran_kwargs
    assert seen[0]["timeout"] == 15
    assert seen[0]["log"] is False and seen[0]["check"] is False


# What `nvidia-smi` prints on a driver new enough for the MLX bundle.
SMI13 = {"nvidia-smi": "CUDA Version: 13.0"}


class Installs(FakeSystem):
    """A machine whose streamed installer leaves the MLX runner where tar would."""

    def __init__(self, root: Path, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.root = root

    def call(self, cmd: Sequence[str]) -> int:
        (self.root / "lib" / "ollama" / "mlx_cuda_v13").mkdir()
        return super().call(cmd)


def _linux_install(tmp_path: Path, with_mlx: bool = False) -> tuple[Path, dict[str, str]]:
    """A standard Linux install under `tmp_path`, and a PATH with it and the pipeline."""
    root = tmp_path / "usr" / "local"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "ollama").write_text("")
    (root / "lib" / "ollama").mkdir(parents=True)
    if with_mlx:
        (root / "lib" / "ollama" / "mlx_cuda_v13").mkdir()
    return root, {"ollama": str(root / "bin" / "ollama"), **_PIPELINE}


# The rest of the Linux pipeline, on PATH.
_PIPELINE = {name: f"/usr/bin/{name}" for name in ("zstd", "curl", "tar")}

NVIDIA_LINUX = Machine("linux", "x86_64", "nvidia", 64.0)
NVIDIA_WINDOWS = Machine("windows", "amd64", "nvidia", 64.0)


class NeverProbes(FakeSystem):
    def which(self, name: str) -> str | None:
        pytest.fail("macOS must not probe for a binary")


def test_ensure_mlx_is_a_no_op_on_macos_and_when_the_bundle_is_present(tmp_path: Path) -> None:
    _root, which = _linux_install(tmp_path, with_mlx=True)
    # macOS returns before it looks at anything: every effect is faked, and none is used.
    mac = NeverProbes(stdout=SMI13)
    mlx.ensure_mlx(Machine("macos", "arm64", "apple", 32.0), sys=mac)
    assert (mac.ran, mac.called, mac.write_probes) == ([], [], [])
    linux = FakeSystem(which=which, stdout=SMI13)
    mlx.ensure_mlx(NVIDIA_LINUX, sys=linux)
    assert linux.called == []
    # The bundle is already there, so the driver is never asked about either.
    assert linux.ran == []


def test_ensure_mlx_installs_the_linux_bundle_with_sudo_when_needed(
    tmp_path: Path, isolated_home: Path
) -> None:
    root, which = _linux_install(tmp_path)
    machine = Installs(root, which=which, stdout=SMI13, writable=False)
    mlx.ensure_mlx(NVIDIA_LINUX, sys=machine)
    script = machine.called[0][2]
    assert machine.called[0][:2] == ["sh", "-c"]
    assert mlx.MLX_BUNDLE["linux"] in script
    assert "zstd -d" in script and "sudo tar" in script and str(root) in script
    entry = json.loads((logs_dir() / "lepika.log").read_text().splitlines()[-1])
    assert entry["event"] == "engine.mlx_install"
    assert entry["result"] == "success"


def test_ensure_mlx_skips_sudo_for_a_writable_install(tmp_path: Path) -> None:
    root, which = _linux_install(tmp_path)
    machine = Installs(root, which=which, stdout=SMI13, writable=True)
    mlx.ensure_mlx(NVIDIA_LINUX, sys=machine)
    # "sudo tar", not "sudo": the tmp_path this test extracts into is named after it.
    assert "sudo tar" not in machine.called[0][2] and "| tar -xf" in machine.called[0][2]
    # tar writes into lib/ollama, so that is the directory whose permissions decide.
    assert machine.write_probes == [root / "lib" / "ollama"]


def test_ensure_mlx_quotes_an_awkward_linux_install_dir(tmp_path: Path) -> None:
    root = tmp_path / "my dir$1" / "usr" / "local"
    (root / "bin").mkdir(parents=True)
    (root / "lib" / "ollama").mkdir(parents=True)
    which = {"ollama": str(root / "bin" / "ollama"), **_PIPELINE}
    machine = Installs(root, which=which, stdout=SMI13, writable=True)
    mlx.ensure_mlx(NVIDIA_LINUX, sys=machine)
    # A space would split the argument and `$1` would expand: sh sees neither.
    assert shlex.quote(str(root)) in machine.called[0][2]


def test_ensure_mlx_installs_the_windows_zip_with_powershell(tmp_path: Path) -> None:
    root = tmp_path / "Programs" / "Ollama"
    (root / "lib" / "ollama").mkdir(parents=True)
    (root / "ollama.exe").write_text("")
    machine = Installs(root, which={"ollama": str(root / "ollama.exe")}, stdout=SMI13)
    mlx.ensure_mlx(NVIDIA_WINDOWS, sys=machine)
    assert machine.called[0][:3] == ["powershell", "-NoProfile", "-Command"]
    assert mlx.MLX_BUNDLE["windows"] in machine.called[0][3]
    assert "Expand-Archive" in machine.called[0][3]


def test_ensure_mlx_quotes_an_apostrophe_in_the_windows_install_dir(tmp_path: Path) -> None:
    root = tmp_path / "O'Brien" / "Programs" / "Ollama"
    (root / "lib" / "ollama").mkdir(parents=True)
    (root / "ollama.exe").write_text("")
    machine = Installs(root, which={"ollama": str(root / "ollama.exe")}, stdout=SMI13)
    mlx.ensure_mlx(NVIDIA_WINDOWS, sys=machine)
    script = machine.called[0][3]
    # Doubled, not raw: a lone `'` would end the string PowerShell is parsing.
    assert f"-DestinationPath '{str(root).replace(chr(39), chr(39) * 2)}'" in script
    assert "O''Brien" in script
    # A 1 GB download must not go through the progress bar or the IE parser.
    assert "$ProgressPreference = 'SilentlyContinue'" in script
    assert "-UseBasicParsing" in script


def test_ensure_mlx_refuses_an_old_driver(tmp_path: Path) -> None:
    _root, which = _linux_install(tmp_path)
    machine = FakeSystem(which=which, stdout={"nvidia-smi": "CUDA Version: 12.8"})
    with pytest.raises(FriendlyError) as exc:
        mlx.ensure_mlx(NVIDIA_LINUX, sys=machine)
    assert "CUDA 13" in exc.value.problem


@pytest.mark.parametrize("missing", ["curl", "tar"])
def test_ensure_mlx_needs_curl_and_tar_on_linux(tmp_path: Path, missing: str) -> None:
    """The pipeline is `curl | zstd | tar`; a missing link fails halfway through a gigabyte."""
    _root, which = _linux_install(tmp_path)
    del which[missing]
    machine = FakeSystem(which=which, stdout=SMI13)
    with pytest.raises(FriendlyError) as exc:
        mlx.ensure_mlx(NVIDIA_LINUX, sys=machine)
    assert exc.value.problem == "Installing Ollama's MLX engine bundle needs curl and tar."
    assert "sudo apt install curl tar" in exc.value.fix
    assert machine.called == []  # nothing downloaded before the pipeline could run


def test_ensure_mlx_needs_zstd_on_linux(tmp_path: Path) -> None:
    root, _ = _linux_install(tmp_path)
    machine = FakeSystem(which={"ollama": str(root / "bin" / "ollama")}, stdout=SMI13)
    with pytest.raises(FriendlyError) as exc:
        mlx.ensure_mlx(NVIDIA_LINUX, sys=machine)
    assert "zstd" in exc.value.problem


def test_ensure_mlx_refuses_a_non_standard_install(tmp_path: Path) -> None:
    exe = tmp_path / "snap" / "bin" / "ollama"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    machine = FakeSystem(which={"ollama": str(exe)}, stdout=SMI13)
    with pytest.raises(FriendlyError) as exc:
        mlx.ensure_mlx(NVIDIA_LINUX, sys=machine)
    # The probed directory is named: "not a standard install" alone is not actionable.
    assert f"{tmp_path / 'snap'} is not a standard Ollama install" in exc.value.problem
    assert mlx.MLX_BUNDLE["linux"] in exc.value.fix


def test_ensure_mlx_reads_a_missing_binary_as_a_stale_path(tmp_path: Path) -> None:
    """A shell opened before OllamaSetup.exe ran has no `ollama` — that is not a broken install."""
    with pytest.raises(FriendlyError) as exc:
        mlx.ensure_mlx(NVIDIA_WINDOWS, sys=FakeSystem(stdout=SMI13))
    assert exc.value.problem == "Ollama is installed but not on your PATH yet."
    assert "open a new one" in exc.value.fix
    assert "standard Ollama install" not in exc.value.problem


def test_ensure_mlx_failed_install_is_friendly(tmp_path: Path, isolated_home: Path) -> None:
    _root, which = _linux_install(tmp_path)
    machine = FakeSystem(which=which, stdout=SMI13, codes={"sh -c": 1})
    with pytest.raises(FriendlyError) as exc:
        mlx.ensure_mlx(NVIDIA_LINUX, sys=machine)
    assert "MLX engine bundle failed" in exc.value.problem
    entry = json.loads((logs_dir() / "lepika.log").read_text().splitlines()[-1])
    assert entry["result"] == "failed"


def test_ensure_mlx_refuses_when_the_installer_reports_success_but_the_bundle_is_absent(
    tmp_path: Path, isolated_home: Path
) -> None:
    """Exit 0 is not proof: a truncated archive unpacks cleanly and leaves no runner."""
    _root, which = _linux_install(tmp_path)
    machine = FakeSystem(which=which, stdout=SMI13)
    with pytest.raises(FriendlyError) as exc:
        mlx.ensure_mlx(NVIDIA_LINUX, sys=machine)
    assert "MLX engine bundle failed" in exc.value.problem
    entry = json.loads((logs_dir() / "lepika.log").read_text().splitlines()[-1])
    assert entry["result"] == "failed"


def test_ollama_store_prefers_the_env_then_the_service_dir_then_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The disk an import fills: OLLAMA_MODELS, else the systemd store, else ~/.ollama."""
    # The path the official install script gives the systemd service.
    assert Path("/usr/share/ollama/.ollama") == mlx._SERVICE_STORE
    service = tmp_path / "usr" / "share" / "ollama" / ".ollama"
    monkeypatch.setattr(mlx, "_SERVICE_STORE", service)
    configured = FakeSystem(environ={"OLLAMA_MODELS": "/mnt/models"})
    assert mlx.ollama_store(sys=configured) == Path("/mnt/models")
    assert mlx.ollama_store(sys=FakeSystem()) == Path.home() / ".ollama"
    # An empty value is not a configured path.
    empty = FakeSystem(environ={"OLLAMA_MODELS": ""})
    assert mlx.ollama_store(sys=empty) == Path.home() / ".ollama"
    service.mkdir(parents=True)
    assert mlx.ollama_store(sys=FakeSystem()) == service
    assert mlx.ollama_store(sys=configured) == Path("/mnt/models")


def test_ollama_env_sets_the_context_length_from_config() -> None:
    env = ollama.ollama_env(16384, environ={"PATH": "/usr/bin"})
    assert env["OLLAMA_CONTEXT_LENGTH"] == "16384"
    assert env["PATH"] == "/usr/bin"


def test_ollama_env_lets_the_shell_win() -> None:
    """Like HF_TOKEN: a value the user just exported is fresher than the file."""
    env = ollama.ollama_env(16384, environ={"OLLAMA_CONTEXT_LENGTH": "65536"})
    assert env["OLLAMA_CONTEXT_LENGTH"] == "65536"


def test_start_ollama_passes_the_context_length_to_serve() -> None:
    machine = FakeSystem(environ={"HOME": "/h"})
    ollama.start_ollama(context_length=8192, sys=machine)
    cmd, env, _log = machine.spawned[0]
    assert cmd == ["ollama", "serve"]
    assert env["OLLAMA_CONTEXT_LENGTH"] == "8192"
    assert env["HOME"] == "/h"


def test_ensure_ollama_starts_the_engine_with_the_configured_context_length(
    isolated_home: Path,
) -> None:
    # Down when asked, up once started.
    machine = answering(False, True, which=INSTALLED)
    ollama.ensure_ollama(info_for("linux"), context_length=8192, sys=machine)
    _, env, _log = machine.spawned[0]
    assert env["OLLAMA_CONTEXT_LENGTH"] == "8192"


def test_context_note_when_someone_elses_engine_was_already_up(isolated_home: Path) -> None:
    """The tray app or brew started it: our env never reached it, so say how to set it."""
    machine = answering(True)
    note = ollama.context_note(
        info_for("windows"), config.Config(context_length=16384), sys=machine
    )
    assert note is not None
    assert "16384" in note
    assert "Settings" in note  # the Windows app has a Context length setting


def test_context_note_on_posix_says_to_restart(isolated_home: Path) -> None:
    note = ollama.context_note(info_for("linux"), config.Config(), sys=answering(True))
    assert note is not None
    assert "lepika up" in note


def test_context_note_is_silent_when_nothing_was_running(isolated_home: Path) -> None:
    machine = answering(False)
    assert ollama.context_note(info_for("linux"), config.Config(), sys=machine) is None
    assert _probed(machine) == [f"{_URL}/api/version"]
    assert machine.ran == []  # nothing answering: no pid to check


def test_context_note_asks_the_engine_the_config_names(isolated_home: Path) -> None:
    """The engine `lepika up` adopts is the one at `engine_url`, wherever that is."""
    url = "http://127.0.0.1:11500"
    machine = answering(True, url=url)
    cfg = config.Config(engine_url=url)
    assert ollama.context_note(info_for("linux"), cfg, sys=machine) is not None
    assert _probed(machine) == [f"{url}/api/version"]
    # A managed engine takes no key.
    assert [r.get_header("Authorization") for r in machine.requested] == [None]


def test_context_note_is_silent_for_our_own_engine(isolated_home: Path) -> None:
    """An Ollama LePika started got the env at launch; the note would be noise."""
    paths.pid_file("ollama").write_text("4242")
    machine = answering(True, stdout={"ps -p 4242": "/usr/local/bin/ollama\n"})
    assert ollama.context_note(info_for("linux"), config.Config(), sys=machine) is None
    assert machine.ran == [["ps", "-p", "4242", "-o", "comm="]]


def test_context_note_is_silent_for_a_remote_engine(isolated_home: Path) -> None:
    cfg = config.Config(engine_managed=False, engine_url="http://gpu-box:11435")
    # No answers: asking the remote engine would fail the test.
    assert ollama.context_note(info_for("linux"), cfg, sys=FakeSystem()) is None


def test_context_note_is_silent_in_server_mode_without_asking(isolated_home: Path) -> None:
    """The container gets the setting from `.env`; there is nothing to note or probe."""
    cfg = config.Config(mode="server")
    assert ollama.context_note(info_for("linux"), cfg, sys=FakeSystem()) is None


class OurEngine(FakeSystem):
    """An Ollama LePika started, as `ensure_ollama` sees it: answering until signalled,
    answering again once something starts a new one. `dies` False: a signal is ignored."""

    def __init__(self, environ: dict[str, str] | None = None) -> None:
        super().__init__(
            stdout={"ps -p 4242": "/usr/local/bin/ollama\n"},
            which=INSTALLED,
            environ=environ if environ is not None else {},
            urls=self._version,
        )
        self.up = True
        self.dies = True

    def _version(self, request: urllib.request.Request) -> Response:
        if request.full_url != f"{_URL}/api/version":
            raise UnansweredURL(request.full_url)
        if not self.up:
            raise OSError("connection refused")
        return FakeResponse()

    def signal(self, pid: int, sig: int) -> None:
        super().signal(pid, sig)
        if self.dies:
            self.up = False

    def spawn(self, cmd: Sequence[str], *, env: Mapping[str, str], log: Path) -> int:
        self.up = True
        return super().spawn(cmd, env=env, log=log)

    @property
    def killed(self) -> list[int]:
        return [pid for pid, _sig in self.signals]

    def ensure(self, context_length: int) -> None:
        ollama.ensure_ollama(info_for("linux"), context_length=context_length, sys=self)


def test_ensure_ollama_restarts_our_engine_when_the_context_length_changed(
    isolated_home: Path,
) -> None:
    """The README's recipe: edit context_length, run `lepika up`. `down` leaves our
    engine running, so without a restart the new value never reaches it."""
    engine = OurEngine()
    ollama.start_ollama(context_length=16384, sys=engine)
    engine.ensure(context_length=8192)
    assert engine.killed == [4242]
    assert engine.spawned[-1][1]["OLLAMA_CONTEXT_LENGTH"] == "8192"


def test_ensure_ollama_leaves_our_engine_alone_when_the_context_length_matches(
    isolated_home: Path,
) -> None:
    """A restart unloads every model: only a changed setting is worth that."""
    engine = OurEngine()
    ollama.start_ollama(context_length=16384, sys=engine)
    engine.ensure(context_length=16384)
    assert engine.killed == []
    assert len(engine.spawned) == 1


def test_ensure_ollama_restarts_our_engine_that_predates_the_context_setting(
    isolated_home: Path,
) -> None:
    """LePika 0.1.x started Ollama without OLLAMA_CONTEXT_LENGTH and recorded only the
    pid: that engine runs at Ollama's 4096, and an upgrade must not keep it there."""
    paths.pid_file("ollama").write_text("4242")
    engine = OurEngine()
    engine.ensure(context_length=16384)
    assert engine.killed == [4242]
    assert engine.spawned[-1][1]["OLLAMA_CONTEXT_LENGTH"] == "16384"


def test_ensure_ollama_never_restarts_an_engine_it_did_not_start(isolated_home: Path) -> None:
    """brew services, systemd, the tray app: `context_note` explains, nothing is signalled."""
    engine = OurEngine()
    engine.ensure(context_length=8192)
    assert engine.killed == []
    assert engine.spawned == []


def test_ensure_ollama_compares_against_the_shell_override(isolated_home: Path) -> None:
    """A shell OLLAMA_CONTEXT_LENGTH wins over config.toml at launch; the same shell on
    the next `up` asks for the same value, so restarting would change nothing."""
    engine = OurEngine(environ={"OLLAMA_CONTEXT_LENGTH": "65536"})
    ollama.start_ollama(context_length=16384, sys=engine)
    engine.ensure(context_length=16384)
    assert engine.killed == []


def test_ensure_ollama_restart_that_cannot_stop_the_engine_says_to_rerun_up(
    isolated_home: Path,
) -> None:
    engine = OurEngine()
    ollama.start_ollama(context_length=16384, sys=engine)
    engine.dies = False
    with pytest.raises(FriendlyError) as exc:
        engine.ensure(context_length=8192)
    assert "lepika up" in exc.value.fix
