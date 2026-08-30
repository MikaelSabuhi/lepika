from __future__ import annotations

import json
import shlex
import signal
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fakes import Runner

from lepika import config, express, paths
from lepika.detect import SystemInfo
from lepika.errors import FriendlyError
from lepika.paths import logs_dir


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


class FakeProc:
    pid = 4242


class PopenRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, cmd: list[str], **kwargs: Any) -> FakeProc:
        self.calls.append((list(cmd), dict(kwargs)))
        return FakeProc()


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
    assert not paths.pid_file("ollama").exists()


def test_start_ollama_records_the_pid(isolated_home: Path) -> None:
    """Without the pid file a mode switch cannot tell our Ollama from anyone else's."""
    popen = PopenRecorder()
    express.start_ollama("linux", popen=popen)
    assert paths.pid_file("ollama").read_text() == "4242"
    popen.calls[0][1]["stdout"].close()


def test_stop_ollama_no_pidfile_returns_false(isolated_home: Path) -> None:
    """An Ollama LePika did not start (brew, systemd, tray app) is never ours to stop."""

    def never(*a: Any, **k: Any) -> Any:
        raise AssertionError("must not probe or signal anything without a pid file")

    assert express.stop_ollama("linux", "http://127.0.0.1:11434", kill=never, api_up=never) is False


def test_stop_ollama_malformed_pidfile_is_cleaned_up(isolated_home: Path) -> None:
    pf = paths.pid_file("ollama")
    pf.write_text("not-a-pid\n")

    def never(pid: int, sig: int) -> None:
        raise AssertionError("should not signal anything")

    assert (
        express.stop_ollama(
            "linux", "http://127.0.0.1:11434", kill=never, api_up=lambda *a, **k: True
        )
        is False
    )
    assert not pf.exists()


@pytest.mark.parametrize("content", ["0", "-5\n"])
def test_stop_ollama_nonpositive_pid_is_never_signalled(isolated_home: Path, content: str) -> None:
    """`os.kill(0, ...)` signals the whole process group; 0 and negatives never reach kill."""
    pf = paths.pid_file("ollama")
    pf.write_text(content)

    def never(pid: int, sig: int) -> None:
        raise AssertionError("should not signal anything")

    assert (
        express.stop_ollama(
            "linux", "http://127.0.0.1:11434", kill=never, api_up=lambda *a, **k: True
        )
        is False
    )
    assert not pf.exists()


def test_stop_ollama_stale_pidfile_is_never_signalled(isolated_home: Path) -> None:
    """After a reboot the recorded pid can belong to a stranger — the API decides (rule 7)."""
    pf = paths.pid_file("ollama")
    pf.write_text("4242")

    def never(pid: int, sig: int) -> None:
        raise AssertionError("must not signal a process we cannot confirm is ours")

    stopped = express.stop_ollama(
        "linux", "http://127.0.0.1:11434", kill=never, api_up=lambda *a, **k: False
    )
    assert stopped is False
    assert not pf.exists()


def ollama_ps(pid: int = 4242) -> Runner:
    """A `run` whose process listing says the pid really is an Ollama."""
    return Runner(
        stdout={
            f"ps -p {pid}": "/usr/local/bin/ollama\n",
            "tasklist": f'"ollama.exe","{pid}","Console","1","94,208 K"\n',
        }
    )


def dying(*answers: bool) -> Callable[..., bool]:
    """An `api_up` that walks a canned sequence: the gate probe, then the wait."""
    values = iter(answers)
    return lambda url, **k: next(values)


def test_stop_ollama_signals_when_the_api_answers(isolated_home: Path) -> None:
    paths.pid_file("ollama").write_text("4242")
    killed: list[tuple[int, int]] = []
    probed: list[str] = []
    answers = iter([True, False])
    run = ollama_ps()
    stopped = express.stop_ollama(
        "linux",
        "http://gpu-box.local:11434",
        run=run,
        kill=lambda pid, sig: killed.append((pid, sig)),
        api_up=lambda url, **k: bool(probed.append(url)) or next(answers),
    )
    assert stopped is True
    assert killed == [(4242, signal.SIGTERM)]
    # Probed where the config says the engine is, not on the local default.
    assert probed == ["http://gpu-box.local:11434"] * 2
    assert run.calls == [["ps", "-p", "4242", "-o", "comm="]]
    assert not paths.pid_file("ollama").exists()


def test_stop_ollama_passes_the_engine_key_to_the_probe(isolated_home: Path) -> None:
    """A keyed engine answers 401 without the key, which would read as "already down"."""
    paths.pid_file("ollama").write_text("4242")
    keys: list[str] = []
    answers = iter([True, False])
    express.stop_ollama(
        "linux",
        "http://127.0.0.1:11434",
        run=ollama_ps(),
        kill=lambda pid, sig: None,
        api_up=lambda url, key="", **k: bool(keys.append(key)) or next(answers),
        key="s3cret",
    )
    assert keys == ["s3cret", "s3cret"]


def test_stop_ollama_waits_for_the_engine_to_actually_exit(isolated_home: Path) -> None:
    """Ollama unloads models on the way out; port 11434 is only free once it is gone."""
    pf = paths.pid_file("ollama")
    pf.write_text("4242")
    kept: list[bool] = []
    stopped = express.stop_ollama(
        "linux",
        "http://127.0.0.1:11434",
        run=ollama_ps(),
        kill=lambda pid, sig: None,
        api_up=dying(True, True, True, False),
        # The pid file is what a retry would need: it stays until the engine is gone.
        sleep=lambda s: kept.append(pf.exists()),
    )
    assert stopped is True
    assert kept == [True, True]
    assert not pf.exists()


def test_stop_ollama_that_never_dies_is_friendly_and_keeps_the_pid_file(
    isolated_home: Path,
) -> None:
    """Returning True here would send the wizard straight into a busy port 11434."""
    pf = paths.pid_file("ollama")
    pf.write_text("4242")
    with pytest.raises(FriendlyError) as exc:
        express.stop_ollama(
            "linux",
            "http://127.0.0.1:11434",
            run=ollama_ps(),
            kill=lambda pid, sig: None,
            api_up=lambda *a, **k: True,
            attempts=2,
            sleep=lambda s: None,
        )
    assert "still answering" in exc.value.problem
    assert "ollama serve" in exc.value.fix
    # Left behind on purpose: the next attempt still needs to know which pid is ours.
    assert pf.read_text() == "4242"


def test_stop_ollama_never_signals_a_pid_that_is_not_an_ollama(isolated_home: Path) -> None:
    """`ollama.pid` outlives `lepika down`, so a reboot can recycle it onto a stranger."""
    pf = paths.pid_file("ollama")
    pf.write_text("4242")
    run = Runner(stdout={"ps -p 4242": "/Applications/Safari.app/Contents/MacOS/Safari\n"})

    def never(pid: int, sig: int) -> None:
        raise AssertionError("must not signal a process that is not ours")

    stopped = express.stop_ollama(
        "linux", "http://127.0.0.1:11434", run=run, kill=never, api_up=lambda *a, **k: True
    )
    assert stopped is False
    assert not pf.exists()


def test_stop_ollama_windows_checks_tasklist_then_taskkills(isolated_home: Path) -> None:
    paths.pid_file("ollama").write_text("4242")
    run = ollama_ps()

    def never(pid: int, sig: int) -> None:
        raise AssertionError("Windows has no SIGTERM")

    stopped = express.stop_ollama(
        "windows",
        "http://127.0.0.1:11434",
        run=run,
        kill=never,
        api_up=dying(True, False),
    )
    assert stopped is True
    assert run.calls == [
        ["tasklist", "/FI", "PID eq 4242", "/FO", "CSV", "/NH"],
        ["taskkill", "/PID", "4242", "/T", "/F"],
    ]
    assert not paths.pid_file("ollama").exists()


def test_stop_ollama_windows_leaves_a_pid_that_is_not_an_ollama_alone(isolated_home: Path) -> None:
    pf = paths.pid_file("ollama")
    pf.write_text("4242")
    run = Runner(stdout={"tasklist": '"notepad.exe","4242","Console","1","9,208 K"\n'})
    assert (
        express.stop_ollama(
            "windows", "http://127.0.0.1:11434", run=run, api_up=lambda *a, **k: True
        )
        is False
    )
    assert run.calls == [["tasklist", "/FI", "PID eq 4242", "/FO", "CSV", "/NH"]]
    assert not pf.exists()


def test_stop_ollama_permission_error_is_not_fatal(isolated_home: Path) -> None:
    """A truncated pid can parse to a live foreign pid — refusing it is not a crash."""
    pf = paths.pid_file("ollama")
    pf.write_text("1")

    def denied(pid: int, sig: int) -> None:
        raise PermissionError(1, "Operation not permitted")

    assert (
        express.stop_ollama(
            "linux",
            "http://127.0.0.1:11434",
            run=ollama_ps(1),
            kill=denied,
            api_up=dying(True, False),
        )
        is True
    )
    assert not pf.exists()


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


def _info(os_name: str, arch: str, gpu: str) -> SystemInfo:
    return SystemInfo(os_name, arch, gpu, 32.0, False, True, True)  # type: ignore[arg-type]


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
    assert express.import_allowed(config.Config(), _info(os_name, arch, gpu)) is expected


def test_import_allowed_is_express_only_and_needs_our_own_engine() -> None:
    mac = _info("macos", "arm64", "apple")
    assert express.import_allowed(config.Config(mode="server"), mac) is False
    assert express.import_allowed(config.Config(engine_managed=False), mac) is False


def test_ollama_install_dir_mirrors_install_sh(tmp_path: Path) -> None:
    exe = tmp_path / "usr" / "local" / "bin" / "ollama"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    assert (
        express.ollama_install_dir("linux", which=lambda n: str(exe)) == tmp_path / "usr" / "local"
    )
    win = tmp_path / "Programs" / "Ollama" / "ollama.exe"
    win.parent.mkdir(parents=True)
    win.write_text("")
    assert express.ollama_install_dir("windows", which=lambda n: str(win)) == win.parent
    assert express.ollama_install_dir("linux", which=lambda n: None) is None


def test_mlx_present_looks_for_the_cuda_backend_dir(tmp_path: Path) -> None:
    assert express.mlx_present(tmp_path) is False
    (tmp_path / "lib" / "ollama" / "mlx_cuda_v13").mkdir(parents=True)
    assert express.mlx_present(tmp_path) is True


def test_cuda_major_parses_nvidia_smi_and_defaults_to_zero() -> None:
    smi = "| NVIDIA-SMI 580.65   Driver Version: 580.65   CUDA Version: 13.0  |"
    assert express.cuda_major(run=Runner(stdout={"nvidia-smi": smi})) == 13
    assert express.cuda_major(run=Runner(stdout={"nvidia-smi": "no gpu"})) == 0

    def gone(cmd: list[str], **k: Any) -> Any:
        raise FriendlyError("Command not found: nvidia-smi", "Install it.")

    assert express.cuda_major(run=gone) == 0


def test_cuda_major_bounds_a_wedged_driver() -> None:
    """A hung `nvidia-smi` must not stall the import: the probe is timed out and reads as 0."""
    seen: list[dict[str, Any]] = []

    def timed_out(cmd: list[str], **kwargs: Any) -> Any:
        seen.append(dict(kwargs))
        raise FriendlyError("nvidia-smi timed out after 15s.", "Try again.")

    assert express.cuda_major(run=timed_out) == 0
    assert seen[0]["timeout"] == 15
    assert seen[0]["log"] is False and seen[0]["check"] is False


def _linux_install(
    tmp_path: Path, with_mlx: bool = False
) -> tuple[Path, Callable[[str], str | None]]:
    root = tmp_path / "usr" / "local"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "ollama").write_text("")
    (root / "lib" / "ollama").mkdir(parents=True)
    if with_mlx:
        (root / "lib" / "ollama" / "mlx_cuda_v13").mkdir()
    return root, lambda name: str(root / "bin" / "ollama") if name == "ollama" else "/usr/bin/zstd"


NVIDIA_LINUX = SystemInfo("linux", "x86_64", "nvidia", 64.0, False, True, True)
NVIDIA_WINDOWS = SystemInfo("windows", "amd64", "nvidia", 64.0, False, True, True)
SMI13 = Runner(stdout={"nvidia-smi": "CUDA Version: 13.0"})


def test_ensure_mlx_is_a_no_op_on_macos_and_when_the_bundle_is_present(tmp_path: Path) -> None:
    _root, which = _linux_install(tmp_path, with_mlx=True)
    call = CallRecorder()
    smi = Runner(stdout={"nvidia-smi": "CUDA Version: 13.0"})
    # macOS returns before it looks at anything: every effect is faked, and none is used.
    express.ensure_mlx(
        SystemInfo("macos", "arm64", "apple", 32.0, False, True, True),
        which=lambda n: pytest.fail("macOS must not probe for a binary"),
        run=smi,
        call=call,
    )
    express.ensure_mlx(NVIDIA_LINUX, which=which, run=smi, call=call)
    assert call.calls == []
    # The bundle is already there, so the driver is never asked about either.
    assert smi.calls == []


def test_ensure_mlx_installs_the_linux_bundle_with_sudo_when_needed(
    tmp_path: Path, isolated_home: Path
) -> None:
    root, which = _linux_install(tmp_path)

    class Installer(CallRecorder):
        def __call__(self, cmd: list[str]) -> int:
            (root / "lib" / "ollama" / "mlx_cuda_v13").mkdir()
            return super().__call__(cmd)

    call = Installer()
    express.ensure_mlx(NVIDIA_LINUX, which=which, run=SMI13, call=call, writable=lambda p: False)
    script = call.calls[0][2]
    assert call.calls[0][:2] == ["sh", "-c"]
    assert express.MLX_BUNDLE["linux"] in script
    assert "zstd -d" in script and "sudo tar" in script and str(root) in script
    entry = json.loads((logs_dir() / "lepika.log").read_text().splitlines()[-1])
    assert entry["event"] == "engine.mlx_install"
    assert entry["result"] == "success"


def test_ensure_mlx_skips_sudo_for_a_writable_install(tmp_path: Path) -> None:
    root, which = _linux_install(tmp_path)

    class Installer(CallRecorder):
        def __call__(self, cmd: list[str]) -> int:
            (root / "lib" / "ollama" / "mlx_cuda_v13").mkdir()
            return super().__call__(cmd)

    call = Installer()
    probed: list[Path] = []
    express.ensure_mlx(
        NVIDIA_LINUX,
        which=which,
        run=SMI13,
        call=call,
        writable=lambda p: bool(probed.append(p)) or True,
    )
    # "sudo tar", not "sudo": the tmp_path this test extracts into is named after it.
    assert "sudo tar" not in call.calls[0][2] and "| tar -xf" in call.calls[0][2]
    # tar writes into lib/ollama, so that is the directory whose permissions decide.
    assert probed == [root / "lib" / "ollama"]


def test_ensure_mlx_quotes_an_awkward_linux_install_dir(tmp_path: Path) -> None:
    root = tmp_path / "my dir$1" / "usr" / "local"
    (root / "bin").mkdir(parents=True)
    (root / "lib" / "ollama").mkdir(parents=True)

    class Installer(CallRecorder):
        def __call__(self, cmd: list[str]) -> int:
            (root / "lib" / "ollama" / "mlx_cuda_v13").mkdir()
            return super().__call__(cmd)

    call = Installer()
    express.ensure_mlx(
        NVIDIA_LINUX,
        which=lambda n: str(root / "bin" / "ollama") if n == "ollama" else "/usr/bin/zstd",
        run=SMI13,
        call=call,
        writable=lambda p: True,
    )
    # A space would split the argument and `$1` would expand: sh sees neither.
    assert shlex.quote(str(root)) in call.calls[0][2]


def test_ensure_mlx_installs_the_windows_zip_with_powershell(tmp_path: Path) -> None:
    root = tmp_path / "Programs" / "Ollama"
    (root / "lib" / "ollama").mkdir(parents=True)
    (root / "ollama.exe").write_text("")

    class Installer(CallRecorder):
        def __call__(self, cmd: list[str]) -> int:
            (root / "lib" / "ollama" / "mlx_cuda_v13").mkdir()
            return super().__call__(cmd)

    call = Installer()
    express.ensure_mlx(
        NVIDIA_WINDOWS, which=lambda n: str(root / "ollama.exe"), run=SMI13, call=call
    )
    assert call.calls[0][:3] == ["powershell", "-NoProfile", "-Command"]
    assert express.MLX_BUNDLE["windows"] in call.calls[0][3]
    assert "Expand-Archive" in call.calls[0][3]


def test_ensure_mlx_quotes_an_apostrophe_in_the_windows_install_dir(tmp_path: Path) -> None:
    root = tmp_path / "O'Brien" / "Programs" / "Ollama"
    (root / "lib" / "ollama").mkdir(parents=True)
    (root / "ollama.exe").write_text("")

    class Installer(CallRecorder):
        def __call__(self, cmd: list[str]) -> int:
            (root / "lib" / "ollama" / "mlx_cuda_v13").mkdir()
            return super().__call__(cmd)

    call = Installer()
    express.ensure_mlx(
        NVIDIA_WINDOWS, which=lambda n: str(root / "ollama.exe"), run=SMI13, call=call
    )
    script = call.calls[0][3]
    # Doubled, not raw: a lone `'` would end the string PowerShell is parsing.
    assert f"-DestinationPath '{str(root).replace(chr(39), chr(39) * 2)}'" in script
    assert "O''Brien" in script
    # A 1 GB download must not go through the progress bar or the IE parser.
    assert "$ProgressPreference = 'SilentlyContinue'" in script
    assert "-UseBasicParsing" in script


def test_ensure_mlx_refuses_an_old_driver(tmp_path: Path) -> None:
    _root, which = _linux_install(tmp_path)
    with pytest.raises(FriendlyError) as exc:
        express.ensure_mlx(
            NVIDIA_LINUX,
            which=which,
            run=Runner(stdout={"nvidia-smi": "CUDA Version: 12.8"}),
            call=CallRecorder(),
        )
    assert "CUDA 13" in exc.value.problem


def test_ensure_mlx_needs_zstd_on_linux(tmp_path: Path) -> None:
    root, _ = _linux_install(tmp_path)

    def which(name: str) -> str | None:
        return str(root / "bin" / "ollama") if name == "ollama" else None

    with pytest.raises(FriendlyError) as exc:
        express.ensure_mlx(NVIDIA_LINUX, which=which, run=SMI13, call=CallRecorder())
    assert "zstd" in exc.value.problem


def test_ensure_mlx_refuses_a_non_standard_install(tmp_path: Path) -> None:
    exe = tmp_path / "snap" / "bin" / "ollama"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    with pytest.raises(FriendlyError) as exc:
        express.ensure_mlx(NVIDIA_LINUX, which=lambda n: str(exe), run=SMI13, call=CallRecorder())
    # The probed directory is named: "not a standard install" alone is not actionable.
    assert f"{tmp_path / 'snap'} is not a standard Ollama install" in exc.value.problem
    assert express.MLX_BUNDLE["linux"] in exc.value.fix


def test_ensure_mlx_reads_a_missing_binary_as_a_stale_path(tmp_path: Path) -> None:
    """A shell opened before OllamaSetup.exe ran has no `ollama` — that is not a broken install."""
    with pytest.raises(FriendlyError) as exc:
        express.ensure_mlx(NVIDIA_WINDOWS, which=lambda n: None, run=SMI13, call=CallRecorder())
    assert exc.value.problem == "Ollama is installed but not on your PATH yet."
    assert "open a new one" in exc.value.fix
    assert "standard Ollama install" not in exc.value.problem


def test_ensure_mlx_failed_install_is_friendly(tmp_path: Path, isolated_home: Path) -> None:
    _root, which = _linux_install(tmp_path)
    with pytest.raises(FriendlyError) as exc:
        express.ensure_mlx(NVIDIA_LINUX, which=which, run=SMI13, call=CallRecorder(code=1))
    assert "MLX engine bundle failed" in exc.value.problem
    entry = json.loads((logs_dir() / "lepika.log").read_text().splitlines()[-1])
    assert entry["result"] == "failed"


def test_ensure_mlx_refuses_when_the_installer_reports_success_but_the_bundle_is_absent(
    tmp_path: Path, isolated_home: Path
) -> None:
    """Exit 0 is not proof: a truncated archive unpacks cleanly and leaves no runner."""
    _root, which = _linux_install(tmp_path)
    with pytest.raises(FriendlyError) as exc:
        express.ensure_mlx(NVIDIA_LINUX, which=which, run=SMI13, call=CallRecorder())
    assert "MLX engine bundle failed" in exc.value.problem
    entry = json.loads((logs_dir() / "lepika.log").read_text().splitlines()[-1])
    assert entry["result"] == "failed"


def test_ollama_store_prefers_the_env_then_the_service_dir_then_home() -> None:
    """The disk an import fills: OLLAMA_MODELS, else the systemd store, else ~/.ollama."""
    assert express.ollama_store({"OLLAMA_MODELS": "/mnt/models"}) == Path("/mnt/models")
    assert express.ollama_store({}, exists=lambda p: True) == Path("/usr/share/ollama/.ollama")
    assert express.ollama_store({}, exists=lambda p: False) == Path.home() / ".ollama"
    # An empty value is not a configured path.
    assert express.ollama_store({"OLLAMA_MODELS": ""}, exists=lambda p: False) == (
        Path.home() / ".ollama"
    )
