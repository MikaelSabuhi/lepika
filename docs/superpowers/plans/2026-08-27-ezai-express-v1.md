# ezai Express v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `ezai` v0.1 — one command installs native Ollama + OpenWebUI (no Docker) and opens a local AI chat in the browser on macOS, Linux, and Windows.

**Architecture:** A thin Typer/Rich CLI that detects the platform/GPU/RAM, then orchestrates upstream tools (`ollama`, `uv tool run open-webui`) via subprocess. All state under `~/.ezai` (`EZAI_HOME` overridable). Every external effect is injected as a callable so unit tests run with fakes, no network, no Docker.

**Tech Stack:** Python ≥3.11, uv, Typer + Rich (only runtime deps), pytest, ruff (+format), mypy --strict, bandit, pip-audit, pre-commit, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-08-27-ezai-design.md` (this plan implements the Express-mode half plus scaffolding/CI; Server mode and release automation are follow-up plans).

## Global Constraints

- Python `>=3.11`. Runtime dependencies are ONLY `typer>=0.12` and `rich>=13.7`. Stdlib for HTTP (`urllib.request`) and TOML (`tomllib`).
- All subprocess calls go through `ezai.proc.run_logged` (captured + logged) except deliberately streaming ones (`ollama pull`) noted in-task.
- All filesystem state under `ezai.paths.ezai_home()` which honors env var `EZAI_HOME` — the autouse pytest fixture isolates every test.
- Unit tests never touch the network, Docker, brew, winget, or real binaries: inject fakes for `run`, `which`, `popen`, `urlopen`, `sleep`.
- Quality gates for EVERY task before commit: `uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -q && uv run bandit -r src -q`.
- mypy is strict; annotate everything, including tests' public functions.
- **Workflow:** each task = one branch in a git worktree = one PR to `main` titled with its conventional commit message. CI must pass; the user squash-merges. Never push to `main`.
- Conventional commits (`feat:`, `fix:`, `test:`, `ci:`, `docs:`, `chore:`).
- GitHub repo: `MikaelSabuhi/ezaiselfhost`.
- Any task that changes user-facing commands or install steps updates `README.md` in the same PR.

---

### Task 1: Project scaffolding, CI pipeline, `--version`

**Files:**
- Create: `pyproject.toml`, `src/ezai/__init__.py`, `src/ezai/cli.py`, `src/ezai/py.typed`, `tests/conftest.py`, `tests/test_cli.py`, `.github/workflows/ci.yml`, `.pre-commit-config.yaml`, `LICENSE`
- Modify: `README.md` (placeholder note that full README arrives in Task 13)

**Interfaces:**
- Produces: Typer app object `ezai.cli.app`; console-script entry `ezai = "ezai.cli:run"`; `ezai.cli.run() -> None` (wraps `app()` with friendly error handling added in Task 2); `ezai --version` prints `ezai 0.1.0`.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "ezai"
version = "0.1.0"
description = "One command → local AI chat in your browser. Self-host LLMs on your own GPU."
readme = "README.md"
license = "MIT"
requires-python = ">=3.11"
authors = [{ name = "Mikael Sabuhi" }]
keywords = ["llm", "ollama", "openwebui", "self-host", "local-ai"]
dependencies = ["typer>=0.12", "rich>=13.7"]

[project.urls]
Homepage = "https://github.com/MikaelSabuhi/ezaiselfhost"

[project.scripts]
ezai = "ezai.cli:run"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/ezai"]

[dependency-groups]
dev = [
  "pytest>=8",
  "ruff>=0.6",
  "mypy>=1.11",
  "bandit[toml]>=1.7",
  "pip-audit>=2.7",
  "pre-commit>=3.8",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "RUF"]

[tool.mypy]
strict = true
files = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.bandit]
# ezai's job is orchestrating trusted local CLIs (ollama, uv, brew, winget).
# All subprocess invocations are list-based and reviewed; no untrusted input
# reaches a shell.
skips = ["B404", "B603", "B607"]
```

- [ ] **Step 2: Create the package and CLI skeleton**

`src/ezai/__init__.py`:

```python
"""ezai — one command → local AI chat in your browser."""
```

`src/ezai/py.typed`: empty file (marks the package as typed).

`src/ezai/cli.py`:

```python
"""Typer entry point for ezai."""

from __future__ import annotations

import importlib.metadata

import typer

app = typer.Typer(
    help="One command → local AI chat in your browser.",
    add_completion=False,
)


def _version_string() -> str:
    return f"ezai {importlib.metadata.version('ezai')}"


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
) -> None:
    if version:
        typer.echo(_version_string())
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo("Setup wizard coming soon. Run `ezai --help` for available commands.")


def run() -> None:
    """Console-script entry point."""
    app()
```

- [ ] **Step 3: Write the failing test**

`tests/conftest.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point EZAI_HOME at a temp dir so no test touches the real ~/.ezai."""
    home = tmp_path / "ezai-home"
    monkeypatch.setenv("EZAI_HOME", str(home))
    return home
```

`tests/test_cli.py`:

```python
from __future__ import annotations

from typer.testing import CliRunner

from ezai.cli import app

runner = CliRunner()


def test_version_flag_prints_name_and_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == "ezai 0.1.0"
```

- [ ] **Step 4: Install and verify the test fails before implementation exists / passes after**

```bash
uv sync --dev
uv run pytest -q
```

Expected: PASS (skeleton was written in Step 2 — if anything fails, fix before proceeding).

- [ ] **Step 5: Add CI workflow**

`.github/workflows/ci.yml`:

```yaml
name: CI

on:
  pull_request:
  push:
    branches: [main]

jobs:
  checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
      - name: Sync dependencies
        run: uv sync --dev
      - name: Lint
        run: uv run ruff check .
      - name: Format check
        run: uv run ruff format --check .
      - name: Type check
        run: uv run mypy src
      - name: Tests
        run: uv run pytest -q
      - name: Security audit (code)
        run: uv run bandit -r src -q
      - name: Security audit (dependencies)
        run: uv run pip-audit
```

- [ ] **Step 6: Add pre-commit config**

`.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

- [ ] **Step 7: Add MIT `LICENSE`** (standard MIT text, copyright `2026 Mikael Sabuhi`) **and note in `README.md`** under the existing title: `> 🚧 Under construction — v0.1 (Express mode) is being built. Watch/star to follow along.`

- [ ] **Step 8: Run all quality gates**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -q && uv run bandit -r src -q
```

Expected: all pass.

- [ ] **Step 9: Commit and open PR**

```bash
git add -A
git commit -m "feat: scaffold ezai CLI with uv, CI pipeline, and pre-commit"
git push -u origin HEAD
gh pr create --fill --title "feat: scaffold ezai CLI with uv, CI pipeline, and pre-commit"
```

---

### Task 2: Foundations — paths, friendly errors, logged subprocess runner

**Files:**
- Create: `src/ezai/paths.py`, `src/ezai/errors.py`, `src/ezai/proc.py`
- Modify: `src/ezai/cli.py` (friendly error handling in `run()`)
- Test: `tests/test_paths.py`, `tests/test_proc.py`

**Interfaces:**
- Produces:
  - `paths.ezai_home() -> Path` (honors `EZAI_HOME`, mkdir -p), `paths.logs_dir() -> Path`, `paths.pid_file(name: str) -> Path`
  - `errors.FriendlyError(problem: str, fix: str)` with `.problem` / `.fix` attributes
  - `proc.run_logged(cmd: Sequence[str], *, check: bool = True, env: Mapping[str, str] | None = None, timeout: float | None = None) -> subprocess.CompletedProcess[str]` — captures text output, appends to `logs_dir()/ezai.log`, raises `FriendlyError` on nonzero exit when `check`.

- [ ] **Step 1: Write failing tests**

`tests/test_paths.py`:

```python
from __future__ import annotations

from pathlib import Path

from ezai import paths


def test_ezai_home_honors_env_and_creates_dir(isolated_home: Path) -> None:
    home = paths.ezai_home()
    assert home == isolated_home
    assert home.is_dir()


def test_logs_dir_and_pid_file_live_under_home(isolated_home: Path) -> None:
    assert paths.logs_dir() == isolated_home / "logs"
    assert paths.logs_dir().is_dir()
    assert paths.pid_file("openwebui") == isolated_home / "openwebui.pid"
```

`tests/test_proc.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ezai import paths, proc
from ezai.errors import FriendlyError


def test_run_logged_captures_output_and_appends_log(isolated_home: Path) -> None:
    result = proc.run_logged([sys.executable, "-c", "print('hello')"])
    assert result.returncode == 0
    assert "hello" in result.stdout
    log_text = (paths.logs_dir() / "ezai.log").read_text()
    assert "hello" in log_text


def test_run_logged_raises_friendly_error_on_failure(isolated_home: Path) -> None:
    with pytest.raises(FriendlyError) as exc:
        proc.run_logged([sys.executable, "-c", "raise SystemExit(3)"])
    assert "ezai.log" in exc.value.fix


def test_run_logged_check_false_returns_result(isolated_home: Path) -> None:
    result = proc.run_logged([sys.executable, "-c", "raise SystemExit(3)"], check=False)
    assert result.returncode == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_paths.py tests/test_proc.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ezai.paths'`.

- [ ] **Step 3: Implement**

`src/ezai/paths.py`:

```python
"""All ezai state lives under one directory, overridable via EZAI_HOME."""

from __future__ import annotations

import os
from pathlib import Path


def ezai_home() -> Path:
    home = Path(os.environ.get("EZAI_HOME", str(Path.home() / ".ezai")))
    home.mkdir(parents=True, exist_ok=True)
    return home


def logs_dir() -> Path:
    d = ezai_home() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def pid_file(name: str) -> Path:
    return ezai_home() / f"{name}.pid"
```

`src/ezai/errors.py`:

```python
"""User-facing errors: every problem ships with a one-line fix."""

from __future__ import annotations


class FriendlyError(Exception):
    def __init__(self, problem: str, fix: str) -> None:
        self.problem = problem
        self.fix = fix
        super().__init__(problem)
```

`src/ezai/proc.py`:

```python
"""Single choke point for subprocess calls: captured, logged, friendly."""

from __future__ import annotations

import datetime
import subprocess
from collections.abc import Mapping, Sequence

from ezai.errors import FriendlyError
from ezai.paths import logs_dir


def run_logged(
    cmd: Sequence[str],
    *,
    check: bool = True,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(cmd),
        capture_output=True,
        text=True,
        env=dict(env) if env is not None else None,
        timeout=timeout,
    )
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    log_path = logs_dir() / "ezai.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"[{stamp}] $ {' '.join(cmd)}\n"
            f"(exit {result.returncode})\n{result.stdout}{result.stderr}\n"
        )
    if check and result.returncode != 0:
        raise FriendlyError(
            f"Command failed: {' '.join(cmd)}",
            f"Details were logged to {log_path}",
        )
    return result
```

In `src/ezai/cli.py`, replace `run()` and add imports:

```python
from rich.console import Console

from ezai.errors import FriendlyError

err_console = Console(stderr=True)


def run() -> None:
    """Console-script entry point."""
    try:
        app()
    except FriendlyError as exc:
        err_console.print(f"[red]✗ {exc.problem}[/red]")
        err_console.print(f"[yellow]→ {exc.fix}[/yellow]")
        raise SystemExit(1) from exc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q` — Expected: PASS.

- [ ] **Step 5: Quality gates** (Global Constraints command). Expected: all pass.

- [ ] **Step 6: Commit and open PR**

```bash
git add -A
git commit -m "feat: add paths, FriendlyError, and logged subprocess runner"
git push -u origin HEAD
gh pr create --fill --title "feat: add paths, FriendlyError, and logged subprocess runner"
```

---

### Task 3: Config load/save with schema versioning

**Files:**
- Create: `src/ezai/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `paths.ezai_home()`
- Produces:
  - `config.SCHEMA_VERSION: int = 1`
  - `@dataclass config.Config(schema_version: int = 1, mode: str = "express", model: str = "", engine_url: str = "http://127.0.0.1:11434", webui_port: int = 3000)`
  - `config.config_path() -> Path`, `config.load() -> Config` (missing file → defaults; unknown keys dropped), `config.save(cfg: Config) -> None`

- [ ] **Step 1: Write failing tests**

`tests/test_config.py`:

```python
from __future__ import annotations

from pathlib import Path

from ezai import config


def test_load_returns_defaults_when_no_file(isolated_home: Path) -> None:
    cfg = config.load()
    assert cfg.mode == "express"
    assert cfg.webui_port == 3000
    assert cfg.schema_version == config.SCHEMA_VERSION


def test_save_then_load_round_trips(isolated_home: Path) -> None:
    cfg = config.Config(model="qwen3:8b", webui_port=3210)
    config.save(cfg)
    loaded = config.load()
    assert loaded == cfg
    assert config.config_path().read_text().startswith("schema_version = 1")


def test_load_drops_unknown_keys(isolated_home: Path) -> None:
    config.config_path().write_text(
        'schema_version = 1\nmode = "express"\nfuture_option = "x"\n'
    )
    cfg = config.load()
    assert cfg.mode == "express"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_config.py -q` — Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `src/ezai/config.py`**

```python
"""config.toml under ~/.ezai — flat, versioned, migratable."""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass
from pathlib import Path

from ezai.paths import ezai_home

SCHEMA_VERSION = 1


@dataclass
class Config:
    schema_version: int = SCHEMA_VERSION
    mode: str = "express"
    model: str = ""
    engine_url: str = "http://127.0.0.1:11434"
    webui_port: int = 3000


def config_path() -> Path:
    return ezai_home() / "config.toml"


def _dump_toml(data: dict[str, object]) -> str:
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, bool):
            lines.append(f"{key} = {'true' if value else 'false'}")
        elif isinstance(value, int | float):
            lines.append(f"{key} = {value}")
        else:
            lines.append(f'{key} = "{value}"')
    return "\n".join(lines) + "\n"


def save(cfg: Config) -> None:
    config_path().write_text(_dump_toml(dataclasses.asdict(cfg)), encoding="utf-8")


def load() -> Config:
    path = config_path()
    if not path.exists():
        return Config()
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    raw.setdefault("schema_version", SCHEMA_VERSION)
    known = {f.name for f in dataclasses.fields(Config)}
    return Config(**{k: v for k, v in raw.items() if k in known})
```

- [ ] **Step 4: Run tests** — `uv run pytest -q` — Expected: PASS.
- [ ] **Step 5: Quality gates.** Expected: all pass.
- [ ] **Step 6: Commit and open PR**

```bash
git add -A
git commit -m "feat: add versioned config load/save"
git push -u origin HEAD
gh pr create --fill --title "feat: add versioned config load/save"
```

---

### Task 4: Platform, GPU, RAM, and service detection

**Files:**
- Create: `src/ezai/detect.py`
- Test: `tests/test_detect.py`

**Interfaces:**
- Consumes: `proc.run_logged`, `errors.FriendlyError`
- Produces:
  - `detect.OsName = Literal["macos", "linux", "windows"]`, `detect.Gpu = Literal["apple", "nvidia", "none"]`
  - `@dataclass(frozen=True) detect.SystemInfo(os: OsName, arch: str, gpu: Gpu, ram_gb: float, has_docker: bool, has_ollama: bool, ollama_running: bool)`
  - `detect.detect_os(system: str | None = None) -> OsName`
  - `detect.detect_gpu(os_name: OsName, arch: str, which: Callable[[str], str | None] = shutil.which) -> Gpu`
  - `detect.detect_ram_gb(os_name: OsName, run: RunFn = proc.run_logged, meminfo: Path = Path("/proc/meminfo")) -> float`
  - `detect.api_up(url: str, timeout: float = 1.0, urlopen: UrlOpenFn | None = None) -> bool`
  - `detect.detect(which: Callable[[str], str | None] = shutil.which, run: RunFn = proc.run_logged, urlopen: UrlOpenFn | None = None) -> SystemInfo`
  - `detect.plan_sentence(info: SystemInfo) -> str` (one human sentence describing the plan)

- [ ] **Step 1: Write failing tests**

`tests/test_detect.py`:

```python
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from ezai import detect
from ezai.errors import FriendlyError


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


def test_api_up_true_and_false() -> None:
    def ok(url: str, timeout: float = 0) -> Any:
        class R:
            def read(self) -> bytes:
                return b"{}"

        return R()

    def boom(url: str, timeout: float = 0) -> Any:
        raise OSError("refused")

    assert detect.api_up("http://x", urlopen=ok) is True
    assert detect.api_up("http://x", urlopen=boom) is False


def test_plan_sentence_mentions_gpu_and_mode() -> None:
    info = detect.SystemInfo(
        os="macos", arch="arm64", gpu="apple", ram_gb=36.0,
        has_docker=False, has_ollama=True, ollama_running=False,
    )
    sentence = detect.plan_sentence(info)
    assert "Express" in sentence
    assert "Metal" in sentence
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_detect.py -q` — Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `src/ezai/detect.py`**

```python
"""Detect what machine we're on and what's already installed."""

from __future__ import annotations

import platform as _platform
import shutil
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ezai import proc
from ezai.errors import FriendlyError

OsName = Literal["macos", "linux", "windows"]
Gpu = Literal["apple", "nvidia", "none"]
RunFn = Callable[..., Any]
UrlOpenFn = Callable[..., Any]

OLLAMA_URL = "http://127.0.0.1:11434"


@dataclass(frozen=True)
class SystemInfo:
    os: OsName
    arch: str
    gpu: Gpu
    ram_gb: float
    has_docker: bool
    has_ollama: bool
    ollama_running: bool


def detect_os(system: str | None = None) -> OsName:
    s = (system or _platform.system()).lower()
    if s == "darwin":
        return "macos"
    if s == "linux":
        return "linux"
    if s == "windows":
        return "windows"
    raise FriendlyError(
        f"Unsupported operating system: {s}",
        "ezai supports macOS, Linux, and Windows.",
    )


def detect_gpu(
    os_name: OsName,
    arch: str,
    which: Callable[[str], str | None] = shutil.which,
) -> Gpu:
    if os_name == "macos" and arch == "arm64":
        return "apple"
    if which("nvidia-smi") is not None:
        return "nvidia"
    return "none"


def _windows_ram_gb() -> float:
    import ctypes

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MemoryStatusEx()
    stat.dwLength = ctypes.sizeof(MemoryStatusEx)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
    return float(stat.ullTotalPhys) / 2**30


def detect_ram_gb(
    os_name: OsName,
    run: RunFn = proc.run_logged,
    meminfo: Path = Path("/proc/meminfo"),
) -> float:
    if os_name == "macos":
        result = run(["sysctl", "-n", "hw.memsize"])
        return float(int(result.stdout.strip())) / 2**30
    if os_name == "linux":
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                return float(kb) / 2**20
        return 0.0
    return _windows_ram_gb()


def api_up(url: str, timeout: float = 1.0, urlopen: UrlOpenFn | None = None) -> bool:
    opener: UrlOpenFn = urlopen if urlopen is not None else urllib.request.urlopen
    try:
        opener(f"{url}/api/version", timeout=timeout)
    except Exception:
        return False
    return True


def detect(
    which: Callable[[str], str | None] = shutil.which,
    run: RunFn = proc.run_logged,
    urlopen: UrlOpenFn | None = None,
) -> SystemInfo:
    os_name = detect_os()
    arch = _platform.machine().lower()
    return SystemInfo(
        os=os_name,
        arch=arch,
        gpu=detect_gpu(os_name, arch, which),
        ram_gb=detect_ram_gb(os_name, run),
        has_docker=which("docker") is not None,
        has_ollama=which("ollama") is not None,
        ollama_running=api_up(OLLAMA_URL, urlopen=urlopen),
    )


_GPU_LABEL = {"apple": "Apple Metal GPU", "nvidia": "NVIDIA GPU", "none": "CPU only"}
_OS_LABEL = {"macos": "macOS", "linux": "Linux", "windows": "Windows"}


def plan_sentence(info: SystemInfo) -> str:
    speed_note = "" if info.gpu != "none" else " (no GPU found — models will run slowly)"
    return (
        f"Detected {_OS_LABEL[info.os]} on {info.arch} with {info.ram_gb:.0f} GB RAM "
        f"and {_GPU_LABEL[info.gpu]}{speed_note} — Express mode: native Ollama + "
        f"OpenWebUI, no Docker needed."
    )
```

Note: `api_up`'s `urlopen` call intentionally targets fixed localhost URLs built from constants — if bandit flags B310 here, add `# nosec B310` with a comment explaining the URL is constant.

- [ ] **Step 4: Run tests** — `uv run pytest -q` — Expected: PASS.
- [ ] **Step 5: Quality gates.** Expected: all pass (fix any mypy strictness fallout, e.g. `Callable` variance, without loosening `strict`).
- [ ] **Step 6: Commit and open PR**

```bash
git add -A
git commit -m "feat: add platform, GPU, RAM, and service detection"
git push -u origin HEAD
gh pr create --fill --title "feat: add platform, GPU, RAM, and service detection"
```

---

### Task 5: Model references and the curated model list

**Files:**
- Create: `src/ezai/models.py`, `models.toml` (repo root)
- Modify: `pyproject.toml` (bundle `models.toml` into the wheel)
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `errors.FriendlyError`
- Produces:
  - `models.ModelKind = Literal["ollama", "hf_gguf", "hf_repo"]`
  - `@dataclass(frozen=True) models.ModelRef(raw: str, kind: ModelKind)` — `raw` is normalized (scheme stripped, `huggingface.co` → `hf.co`)
  - `models.parse_model_ref(raw: str) -> ModelRef`
  - `@dataclass(frozen=True) models.CuratedModel(name: str, ref: str, min_ram_gb: float, notes: str = "")`
  - `models.load_curated(fetch_remote: bool = True, urlopen: Callable[..., Any] | None = None) -> list[CuratedModel]`
  - `models.fitting(candidates: list[CuratedModel], ram_gb: float) -> list[CuratedModel]`
  - `models.CURATED_MODELS_URL = "https://raw.githubusercontent.com/MikaelSabuhi/ezaiselfhost/main/models.toml"`

- [ ] **Step 1: Write failing tests**

`tests/test_models.py`:

```python
from __future__ import annotations

from typing import Any

import pytest

from ezai import models
from ezai.errors import FriendlyError


def test_parse_plain_tag_is_ollama() -> None:
    ref = models.parse_model_ref("qwen3:8b")
    assert ref == models.ModelRef(raw="qwen3:8b", kind="ollama")


def test_parse_hf_gguf_url_normalizes_scheme_and_domain() -> None:
    ref = models.parse_model_ref("https://huggingface.co/unsloth/gemma-3-4b-it-GGUF")
    assert ref.kind == "hf_gguf"
    assert ref.raw == "hf.co/unsloth/gemma-3-4b-it-GGUF"


def test_parse_bare_hf_co_prefix() -> None:
    ref = models.parse_model_ref("hf.co/unsloth/gemma-3-4b-it-GGUF:Q4_K_M")
    assert ref.kind == "hf_gguf"


def test_parse_org_slash_repo_is_hf_repo() -> None:
    ref = models.parse_model_ref("meta-llama/Llama-3.3-70B-Instruct")
    assert ref.kind == "hf_repo"


def test_parse_empty_raises() -> None:
    with pytest.raises(FriendlyError):
        models.parse_model_ref("   ")


def test_load_curated_falls_back_to_bundled_on_network_error(isolated_home: Any) -> None:
    def boom(url: str, timeout: float = 0) -> Any:
        raise OSError("no network")

    curated = models.load_curated(urlopen=boom)
    assert len(curated) >= 5
    assert all(m.min_ram_gb > 0 for m in curated)


def test_load_curated_skips_network_when_disabled() -> None:
    def fail_if_called(url: str, timeout: float = 0) -> Any:
        raise AssertionError("network should not be used")

    curated = models.load_curated(fetch_remote=False, urlopen=fail_if_called)
    assert curated


def test_fitting_filters_by_ram() -> None:
    small = models.CuratedModel(name="S", ref="s:1b", min_ram_gb=2)
    big = models.CuratedModel(name="B", ref="b:70b", min_ram_gb=48)
    assert models.fitting([small, big], ram_gb=16.0) == [small]
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_models.py -q` — Expected: FAIL.

- [ ] **Step 3: Create `models.toml` at repo root**

```toml
# Curated "known good" models shown by the ezai wizard.
# Keep this list short, current, and honest. min_ram_gb is the practical
# minimum system memory (or VRAM) to run the default quantization comfortably.
schema = 1

[[models]]
name = "Qwen 3 0.6B — tiny, instant"
ref = "qwen3:0.6b"
min_ram_gb = 2

[[models]]
name = "Llama 3.2 3B — light and quick"
ref = "llama3.2:3b"
min_ram_gb = 6

[[models]]
name = "Gemma 3 4B — great all-rounder"
ref = "gemma3:4b"
min_ram_gb = 6

[[models]]
name = "Qwen 3 8B — strong general model"
ref = "qwen3:8b"
min_ram_gb = 12

[[models]]
name = "DeepSeek-R1 8B — reasoning"
ref = "deepseek-r1:8b"
min_ram_gb = 12

[[models]]
name = "Gemma 3 12B — bigger, better answers"
ref = "gemma3:12b"
min_ram_gb = 16

[[models]]
name = "GPT-OSS 20B — OpenAI open-weight"
ref = "gpt-oss:20b"
min_ram_gb = 16

[[models]]
name = "Gemma 3 27B — high quality"
ref = "gemma3:27b"
min_ram_gb = 32

[[models]]
name = "DeepSeek-R1 32B — heavyweight reasoning"
ref = "deepseek-r1:32b"
min_ram_gb = 32

[[models]]
name = "Llama 3.3 70B — flagship class"
ref = "llama3.3:70b"
min_ram_gb = 48
```

- [ ] **Step 4: Bundle it — add to `pyproject.toml`**

```toml
[tool.hatch.build.targets.wheel.force-include]
"models.toml" = "ezai/models.toml"
```

- [ ] **Step 5: Implement `src/ezai/models.py`**

```python
"""Model references (3 shapes) and the curated model list."""

from __future__ import annotations

import importlib.resources
import tomllib
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ezai.errors import FriendlyError

ModelKind = Literal["ollama", "hf_gguf", "hf_repo"]

CURATED_MODELS_URL = (
    "https://raw.githubusercontent.com/MikaelSabuhi/ezaiselfhost/main/models.toml"
)


@dataclass(frozen=True)
class ModelRef:
    raw: str
    kind: ModelKind


@dataclass(frozen=True)
class CuratedModel:
    name: str
    ref: str
    min_ram_gb: float
    notes: str = ""


def parse_model_ref(raw: str) -> ModelRef:
    ref = raw.strip()
    if not ref:
        raise FriendlyError(
            "Empty model reference.",
            "Examples: qwen3:8b · hf.co/unsloth/gemma-3-4b-it-GGUF · meta-llama/Llama-3.3-70B-Instruct",
        )
    for scheme in ("https://", "http://"):
        if ref.lower().startswith(scheme):
            ref = ref[len(scheme):]
            break
    if ref.lower().startswith("huggingface.co/"):
        ref = "hf.co/" + ref[len("huggingface.co/"):]
    if ref.lower().startswith("hf.co/"):
        return ModelRef(raw=ref, kind="hf_gguf")
    if "/" in ref:
        return ModelRef(raw=ref, kind="hf_repo")
    return ModelRef(raw=ref, kind="ollama")


def _bundled_models_text() -> str:
    resource = importlib.resources.files("ezai").joinpath("models.toml")
    if resource.is_file():
        return resource.read_text(encoding="utf-8")
    # Editable/dev install: fall back to the repo-root copy.
    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / "models.toml").read_text(encoding="utf-8")


def load_curated(
    fetch_remote: bool = True,
    urlopen: Callable[..., Any] | None = None,
) -> list[CuratedModel]:
    text: str | None = None
    if fetch_remote:
        opener: Callable[..., Any] = (
            urlopen if urlopen is not None else urllib.request.urlopen
        )
        try:
            text = opener(CURATED_MODELS_URL, timeout=3).read().decode("utf-8")
        except Exception:
            text = None
    if text is None:
        text = _bundled_models_text()
    data = tomllib.loads(text)
    return [CuratedModel(**entry) for entry in data.get("models", [])]


def fitting(candidates: list[CuratedModel], ram_gb: float) -> list[CuratedModel]:
    return [m for m in candidates if m.min_ram_gb <= ram_gb]
```

If bandit flags the `urlopen` of `CURATED_MODELS_URL` (B310), add `# nosec B310` with a comment: constant HTTPS URL to this repo.

- [ ] **Step 6: Run tests** — `uv run pytest -q` — Expected: PASS.
- [ ] **Step 7: Quality gates.** Expected: all pass.
- [ ] **Step 8: Commit and open PR**

```bash
git add -A
git commit -m "feat: add model-ref parsing and curated model list"
git push -u origin HEAD
gh pr create --fill --title "feat: add model-ref parsing and curated model list"
```

---

### Task 6: Express engine — install, start, and pull with Ollama

**Files:**
- Create: `src/ezai/express.py`
- Test: `tests/test_express_ollama.py`

**Interfaces:**
- Consumes: `detect.SystemInfo`, `detect.api_up`, `detect.OLLAMA_URL`, `models.ModelRef`, `proc.run_logged`, `paths.logs_dir`, `errors.FriendlyError`
- Produces:
  - `express.install_ollama(info: SystemInfo, run: RunFn = proc.run_logged, which: WhichFn = shutil.which) -> None`
  - `express.start_ollama(os_name: str, popen: PopenFn = subprocess.Popen) -> None` (detached, logs to `logs_dir()/ollama.log`)
  - `express.wait_for(predicate: Callable[[], bool], seconds: int, what: str, sleep: SleepFn = time.sleep) -> None`
  - `express.ensure_ollama(info, run=..., which=..., popen=..., api_up=detect.api_up, sleep=...) -> None`
  - `express.pull_model(ref: ModelRef, call: CallFn = subprocess.call) -> None` — streams `ollama pull` output directly to the terminal (progress bar), deliberately NOT via `run_logged`

- [ ] **Step 1: Write failing tests**

`tests/test_express_ollama.py`:

```python
from __future__ import annotations

from typing import Any

import pytest

from ezai import express
from ezai.detect import SystemInfo
from ezai.errors import FriendlyError
from ezai.models import ModelRef


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


def test_install_ollama_macos_uses_brew() -> None:
    run = RunRecorder()
    express.install_ollama(
        info_for("macos"), run=run, which=lambda n: "/opt/homebrew/bin/brew" if n == "brew" else None
    )
    assert ["brew", "install", "ollama"] in run.calls


def test_install_ollama_macos_without_brew_gives_download_link() -> None:
    with pytest.raises(FriendlyError) as exc:
        express.install_ollama(info_for("macos"), run=RunRecorder(), which=lambda n: None)
    assert "ollama.com" in exc.value.fix


def test_install_ollama_linux_uses_official_script() -> None:
    run = RunRecorder()
    express.install_ollama(info_for("linux"), run=run, which=lambda n: None)
    assert any("ollama.com/install.sh" in " ".join(c) for c in run.calls)


def test_install_ollama_windows_uses_winget() -> None:
    run = RunRecorder()
    express.install_ollama(info_for("windows"), run=run, which=lambda n: None)
    assert any(c[:2] == ["winget", "install"] for c in run.calls)


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


def test_wait_for_raises_after_timeout() -> None:
    with pytest.raises(FriendlyError):
        express.wait_for(lambda: False, seconds=3, what="Ollama API", sleep=lambda s: None)


def test_pull_model_raises_friendly_on_nonzero_exit() -> None:
    with pytest.raises(FriendlyError):
        express.pull_model(ModelRef(raw="qwen3:8b", kind="ollama"), call=lambda cmd: 1)
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_express_ollama.py -q` — Expected: FAIL.

- [ ] **Step 3: Implement `src/ezai/express.py`**

```python
"""Express mode: native Ollama + OpenWebUI via uv. No Docker anywhere."""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Callable
from typing import Any

from ezai import detect, proc
from ezai.detect import SystemInfo
from ezai.errors import FriendlyError
from ezai.models import ModelRef
from ezai.paths import logs_dir

RunFn = Callable[..., Any]
WhichFn = Callable[[str], str | None]
PopenFn = Callable[..., Any]
SleepFn = Callable[[float], None]
CallFn = Callable[[list[str]], int]

# Windows: DETACHED_PROCESS (0x8) | CREATE_NEW_PROCESS_GROUP (0x200)
_WINDOWS_DETACH_FLAGS = 0x00000208


def _detach_kwargs(os_name: str) -> dict[str, Any]:
    if os_name == "windows":
        return {"creationflags": _WINDOWS_DETACH_FLAGS}
    return {"start_new_session": True}


def install_ollama(
    info: SystemInfo,
    run: RunFn = proc.run_logged,
    which: WhichFn = shutil.which,
) -> None:
    if info.os == "macos":
        if which("brew") is not None:
            run(["brew", "install", "ollama"])
        else:
            raise FriendlyError(
                "Ollama is not installed and Homebrew was not found.",
                "Install Ollama from https://ollama.com/download/mac then run `ezai` again.",
            )
    elif info.os == "linux":
        run(["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"])
    else:  # windows
        run(
            [
                "winget",
                "install",
                "--id",
                "Ollama.Ollama",
                "-e",
                "--accept-package-agreements",
                "--accept-source-agreements",
            ]
        )


def start_ollama(os_name: str, popen: PopenFn = subprocess.Popen) -> None:
    log = (logs_dir() / "ollama.log").open("ab")
    popen(["ollama", "serve"], stdout=log, stderr=log, **_detach_kwargs(os_name))


def wait_for(
    predicate: Callable[[], bool],
    seconds: int,
    what: str,
    sleep: SleepFn = time.sleep,
) -> None:
    for _ in range(seconds):
        if predicate():
            return
        sleep(1)
    raise FriendlyError(
        f"{what} did not become ready within {seconds}s.",
        f"Check the logs in {logs_dir()} and run `ezai doctor`.",
    )


def ensure_ollama(
    info: SystemInfo,
    run: RunFn = proc.run_logged,
    which: WhichFn = shutil.which,
    popen: PopenFn = subprocess.Popen,
    api_up: Callable[..., bool] = detect.api_up,
    sleep: SleepFn = time.sleep,
) -> None:
    if not info.has_ollama:
        install_ollama(info, run=run, which=which)
    if not api_up(detect.OLLAMA_URL):
        start_ollama(info.os, popen=popen)
        wait_for(lambda: api_up(detect.OLLAMA_URL), 30, "Ollama API", sleep=sleep)


def pull_model(ref: ModelRef, call: CallFn = subprocess.call) -> None:
    # Streams ollama's own progress bar to the terminal on purpose.
    code = call(["ollama", "pull", ref.raw])
    if code != 0:
        raise FriendlyError(
            f"Failed to pull model '{ref.raw}'.",
            "Check the model name/URL — e.g. qwen3:8b or hf.co/<org>/<repo>-GGUF — and your internet connection.",
        )
```

- [ ] **Step 4: Run tests** — `uv run pytest -q` — Expected: PASS.
- [ ] **Step 5: Quality gates.** Expected: all pass.
- [ ] **Step 6: Commit and open PR**

```bash
git add -A
git commit -m "feat: add Express-mode Ollama install, start, and model pull"
git push -u origin HEAD
gh pr create --fill --title "feat: add Express-mode Ollama install, start, and model pull"
```

---

### Task 7: Express UI — OpenWebUI lifecycle via uv

**Files:**
- Modify: `src/ezai/express.py`
- Test: `tests/test_express_webui.py`

**Interfaces:**
- Consumes: `config.Config`, `paths.pid_file`, `paths.logs_dir`, everything from Task 6
- Produces (appended to `express.py`):
  - `express.webui_url(port: int) -> str` → `http://localhost:{port}`
  - `express.webui_up(port: int, urlopen: Callable[..., Any] | None = None) -> bool` (GET `/health`, 1s timeout)
  - `express.install_openwebui(run: RunFn = proc.run_logged) -> None` — `uv tool install --python 3.11 open-webui` (idempotent)
  - `express.start_openwebui(port: int, engine_url: str, popen: PopenFn = subprocess.Popen, environ: Mapping[str, str] | None = None) -> int` — detached `uv tool run --from open-webui open-webui serve --port {port}` with `OLLAMA_BASE_URL={engine_url}` in env; writes `pid_file("openwebui")`; returns pid
  - `express.stop_openwebui(os_name: str, run: RunFn = proc.run_logged, kill: Callable[[int, int], None] = os.kill) -> bool` — True if a process was stopped
  - `express.ensure_openwebui(cfg: Config, run=..., popen=..., up=webui_up, sleep=...) -> None`

- [ ] **Step 1: Write failing tests**

`tests/test_express_webui.py`:

```python
from __future__ import annotations

import signal
from pathlib import Path
from typing import Any

from ezai import express, paths
from ezai.config import Config


class FakeProc:
    pid = 4242


class PopenRecorder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.envs: list[dict[str, str] | None] = []

    def __call__(self, cmd: list[str], **kwargs: Any) -> FakeProc:
        self.calls.append(list(cmd))
        self.envs.append(kwargs.get("env"))
        return FakeProc()


def test_start_openwebui_sets_engine_env_and_writes_pidfile(isolated_home: Path) -> None:
    popen = PopenRecorder()
    pid = express.start_openwebui(
        3000, "http://127.0.0.1:11434", popen=popen, environ={"PATH": "/usr/bin"}
    )
    assert pid == 4242
    assert paths.pid_file("openwebui").read_text() == "4242"
    assert popen.envs[0] is not None
    assert popen.envs[0]["OLLAMA_BASE_URL"] == "http://127.0.0.1:11434"
    assert "open-webui" in " ".join(popen.calls[0])


def test_stop_openwebui_kills_pid_and_removes_pidfile(isolated_home: Path) -> None:
    paths.pid_file("openwebui").write_text("4242")
    killed: list[tuple[int, int]] = []
    stopped = express.stop_openwebui(
        "linux", kill=lambda pid, sig: killed.append((pid, sig))
    )
    assert stopped is True
    assert killed == [(4242, signal.SIGTERM)]
    assert not paths.pid_file("openwebui").exists()


def test_stop_openwebui_no_pidfile_returns_false(isolated_home: Path) -> None:
    assert express.stop_openwebui("linux") is False


def test_ensure_openwebui_skips_start_when_healthy(isolated_home: Path) -> None:
    class NoStart:
        def __call__(self, *a: Any, **k: Any) -> FakeProc:
            raise AssertionError("should not start")

    run_calls: list[list[str]] = []
    express.ensure_openwebui(
        Config(),
        run=lambda cmd, **k: run_calls.append(list(cmd)),
        popen=NoStart(),
        up=lambda port, urlopen=None: True,
        sleep=lambda s: None,
    )
    # install is idempotent and still invoked; start is not
    assert any("open-webui" in " ".join(c) for c in run_calls)
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_express_webui.py -q` — Expected: FAIL (missing attributes).

- [ ] **Step 3: Append to `src/ezai/express.py`**

```python
import os
import urllib.request
from collections.abc import Mapping

from ezai.config import Config
from ezai.paths import pid_file


def webui_url(port: int) -> str:
    return f"http://localhost:{port}"


def webui_up(port: int, urlopen: Callable[..., Any] | None = None) -> bool:
    opener: Callable[..., Any] = urlopen if urlopen is not None else urllib.request.urlopen
    try:
        opener(f"http://127.0.0.1:{port}/health", timeout=1.0)
    except Exception:
        return False
    return True


def install_openwebui(run: RunFn = proc.run_logged) -> None:
    run(["uv", "tool", "install", "--python", "3.11", "open-webui"])


def start_openwebui(
    port: int,
    engine_url: str,
    popen: PopenFn = subprocess.Popen,
    environ: Mapping[str, str] | None = None,
) -> int:
    env = dict(environ if environ is not None else os.environ)
    env["OLLAMA_BASE_URL"] = engine_url
    log = (logs_dir() / "openwebui.log").open("ab")
    proc_handle = popen(
        [
            "uv", "tool", "run", "--from", "open-webui",
            "open-webui", "serve", "--port", str(port),
        ],
        env=env,
        stdout=log,
        stderr=log,
        **_detach_kwargs("windows" if os.name == "nt" else "posix"),
    )
    pid = int(proc_handle.pid)
    pid_file("openwebui").write_text(str(pid))
    return pid


def stop_openwebui(
    os_name: str,
    run: RunFn = proc.run_logged,
    kill: Callable[[int, int], None] = os.kill,
) -> bool:
    pf = pid_file("openwebui")
    if not pf.exists():
        return False
    pid = int(pf.read_text().strip())
    try:
        if os_name == "windows":
            run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False)
        else:
            import signal

            kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    pf.unlink(missing_ok=True)
    return True


def ensure_openwebui(
    cfg: Config,
    run: RunFn = proc.run_logged,
    popen: PopenFn = subprocess.Popen,
    up: Callable[..., bool] = webui_up,
    sleep: SleepFn = time.sleep,
) -> None:
    install_openwebui(run=run)
    if up(cfg.webui_port):
        return
    start_openwebui(cfg.webui_port, cfg.engine_url, popen=popen)
    wait_for(lambda: up(cfg.webui_port), 180, "OpenWebUI", sleep=sleep)
```

Refactor note: `_detach_kwargs` takes an os_name; passing `"posix"` is fine — it only special-cases `"windows"`. Move all new imports to the top of the file with the existing ones (ruff will enforce this).

- [ ] **Step 4: Run tests** — `uv run pytest -q` — Expected: PASS.
- [ ] **Step 5: Quality gates.** Expected: all pass.
- [ ] **Step 6: Commit and open PR**

```bash
git add -A
git commit -m "feat: add OpenWebUI lifecycle management via uv"
git push -u origin HEAD
gh pr create --fill --title "feat: add OpenWebUI lifecycle management via uv"
```

---

### Task 8: Lifecycle commands — `up`, `down`, `status`, `logs`

**Files:**
- Modify: `src/ezai/cli.py`
- Test: `tests/test_cli_lifecycle.py`

**Interfaces:**
- Consumes: `detect.detect`, `express.ensure_ollama`, `express.ensure_openwebui`, `express.stop_openwebui`, `express.webui_up`, `express.webui_url`, `detect.api_up`, `detect.OLLAMA_URL`, `config.load`, `paths.logs_dir`
- Produces: `ezai up` (start everything, print URL, open browser), `ezai down` (stop OpenWebUI; leave Ollama running — it's a shared service), `ezai status` (Rich table: Ollama API / OpenWebUI / configured model), `ezai logs` (tail last 50 lines of each file in `logs_dir()`). All commands accept the code paths being monkeypatched in tests (import modules, not names: call `detect.detect()`, `express.ensure_ollama(...)` so `monkeypatch.setattr` works).

- [ ] **Step 1: Write failing tests**

`tests/test_cli_lifecycle.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from ezai import cli, config, detect, express, paths

runner = CliRunner()

INFO = detect.SystemInfo(
    os="linux", arch="x86_64", gpu="nvidia", ram_gb=32.0,
    has_docker=False, has_ollama=True, ollama_running=True,
)


@pytest.fixture()
def quiet_stack(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    counts = {"ensure_ollama": 0, "ensure_openwebui": 0, "browser": 0, "stop": 0}
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(
        express, "ensure_ollama",
        lambda info, **k: counts.__setitem__("ensure_ollama", counts["ensure_ollama"] + 1),
    )
    monkeypatch.setattr(
        express, "ensure_openwebui",
        lambda cfg, **k: counts.__setitem__("ensure_openwebui", counts["ensure_openwebui"] + 1),
    )
    monkeypatch.setattr(
        express, "stop_openwebui",
        lambda os_name, **k: counts.__setitem__("stop", counts["stop"] + 1) or True,
    )
    monkeypatch.setattr(
        cli, "_open_browser",
        lambda url: counts.__setitem__("browser", counts["browser"] + 1),
    )
    return counts


def test_up_starts_stack_and_opens_browser(quiet_stack: dict[str, int]) -> None:
    result = runner.invoke(cli.app, ["up"])
    assert result.exit_code == 0
    assert quiet_stack["ensure_ollama"] == 1
    assert quiet_stack["ensure_openwebui"] == 1
    assert quiet_stack["browser"] == 1
    assert "http://localhost:3000" in result.output


def test_down_stops_webui(quiet_stack: dict[str, int]) -> None:
    result = runner.invoke(cli.app, ["down"])
    assert result.exit_code == 0
    assert quiet_stack["stop"] == 1


def test_status_reports_services(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    monkeypatch.setattr(detect, "api_up", lambda url, **k: True)
    monkeypatch.setattr(express, "webui_up", lambda port, **k: False)
    config.save(config.Config(model="qwen3:8b"))
    result = runner.invoke(cli.app, ["status"])
    assert result.exit_code == 0
    assert "qwen3:8b" in result.output


def test_logs_prints_tail(isolated_home: Path) -> None:
    (paths.logs_dir() / "ezai.log").write_text("line-one\nline-two\n")
    result = runner.invoke(cli.app, ["logs"])
    assert result.exit_code == 0
    assert "line-two" in result.output
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_cli_lifecycle.py -q` — Expected: FAIL (no such commands).

- [ ] **Step 3: Implement in `src/ezai/cli.py`** (add below the callback; keep imports at top: `webbrowser`, `from rich.console import Console`, `from rich.table import Table`, `from ezai import config, detect, express, paths`)

```python
console = Console()


def _open_browser(url: str) -> None:
    webbrowser.open(url)


@app.command()
def up() -> None:
    """Start the local AI stack and open the browser."""
    info = detect.detect()
    console.print(detect.plan_sentence(info))
    cfg = config.load()
    express.ensure_ollama(info)
    express.ensure_openwebui(cfg)
    url = express.webui_url(cfg.webui_port)
    console.print(f"[green]✓ Ready:[/green] {url}")
    _open_browser(url)


@app.command()
def down() -> None:
    """Stop OpenWebUI (Ollama keeps running as a shared service)."""
    info = detect.detect()
    if express.stop_openwebui(info.os):
        console.print("[green]✓ OpenWebUI stopped.[/green]")
    else:
        console.print("OpenWebUI was not running.")


@app.command()
def status() -> None:
    """Show what's running."""
    cfg = config.load()
    table = Table(title="ezai status")
    table.add_column("Service")
    table.add_column("State")
    ollama_ok = detect.api_up(detect.OLLAMA_URL)
    webui_ok = express.webui_up(cfg.webui_port)
    table.add_row("Ollama API", "[green]up[/green]" if ollama_ok else "[red]down[/red]")
    table.add_row("OpenWebUI", "[green]up[/green]" if webui_ok else "[red]down[/red]")
    table.add_row("Model", cfg.model or "[dim]not set[/dim]")
    console.print(table)


@app.command()
def logs(lines: int = typer.Option(50, help="Lines per log file.")) -> None:
    """Print the tail of ezai's log files."""
    for log_file in sorted(paths.logs_dir().glob("*.log")):
        console.rule(str(log_file.name))
        content = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in content[-lines:]:
            console.print(line, markup=False)
```

- [ ] **Step 4: Run tests** — `uv run pytest -q` — Expected: PASS.
- [ ] **Step 5: Quality gates.** Expected: all pass.
- [ ] **Step 6: Commit and open PR**

```bash
git add -A
git commit -m "feat: add up, down, status, and logs commands"
git push -u origin HEAD
gh pr create --fill --title "feat: add up, down, status, and logs commands"
```

---

### Task 9: The wizard — default `ezai` experience with `--dry-run`

**Files:**
- Create: `src/ezai/wizard.py`
- Modify: `src/ezai/cli.py` (callback runs the wizard; add `--dry-run` option)
- Test: `tests/test_wizard.py`

**Interfaces:**
- Consumes: `detect.detect`, `detect.plan_sentence`, `models.load_curated`, `models.fitting`, `models.parse_model_ref`, `config.Config/save`, `express.*`, `cli._open_browser`
- Produces:
  - `wizard.choose_model(info: SystemInfo, ask: AskFn = rich.prompt.Prompt.ask, curated: list[CuratedModel] | None = None) -> ModelRef` — prints a numbered Rich table of fitting curated models plus a free-form option; a digit picks from the table, anything else is parsed as a model ref
  - `wizard.run_wizard(dry_run: bool = False) -> None` — detect → announce plan → choose model → save config → (unless dry_run) ensure engine, pull model, ensure UI, open browser. With `dry_run`, prints the actions it *would* take, one per line, each starting with `would:`.
  - `ezai` with no subcommand runs `run_wizard()`; `ezai --dry-run` runs `run_wizard(dry_run=True)`.
  - `hf_repo`-kind refs raise `FriendlyError` (vLLM/Server mode not shipped yet) suggesting the GGUF shape.

- [ ] **Step 1: Write failing tests**

`tests/test_wizard.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from ezai import cli, config, detect, models, wizard
from ezai.errors import FriendlyError

runner = CliRunner()

INFO = detect.SystemInfo(
    os="macos", arch="arm64", gpu="apple", ram_gb=16.0,
    has_docker=False, has_ollama=True, ollama_running=True,
)

CURATED = [
    models.CuratedModel(name="Small", ref="llama3.2:3b", min_ram_gb=6),
    models.CuratedModel(name="Huge", ref="llama3.3:70b", min_ram_gb=48),
]


def test_choose_model_by_number_picks_fitting_curated() -> None:
    ref = wizard.choose_model(INFO, ask=lambda *a, **k: "1", curated=CURATED)
    assert ref.raw == "llama3.2:3b"
    assert ref.kind == "ollama"


def test_choose_model_free_form() -> None:
    ref = wizard.choose_model(
        INFO, ask=lambda *a, **k: "hf.co/unsloth/gemma-3-4b-it-GGUF", curated=CURATED
    )
    assert ref.kind == "hf_gguf"


def test_choose_model_rejects_hf_repo_with_gguf_hint() -> None:
    with pytest.raises(FriendlyError) as exc:
        wizard.choose_model(
            INFO, ask=lambda *a, **k: "meta-llama/Llama-3.3-70B-Instruct", curated=CURATED
        )
    assert "GGUF" in exc.value.fix


def test_dry_run_writes_config_and_executes_nothing(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(models, "load_curated", lambda **k: CURATED)
    monkeypatch.setattr(wizard, "_ask", lambda *a, **k: "1")
    result = runner.invoke(cli.app, ["--dry-run"])
    assert result.exit_code == 0
    assert "would:" in result.output
    assert config.load().model == "llama3.2:3b"
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_wizard.py -q` — Expected: FAIL.

- [ ] **Step 3: Implement `src/ezai/wizard.py`**

```python
"""The default `ezai` experience: detect, ask, install, open browser."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from ezai import config, detect, express, models
from ezai.detect import SystemInfo
from ezai.errors import FriendlyError
from ezai.models import CuratedModel, ModelRef

AskFn = Callable[..., str]
console = Console()

_ask: AskFn = Prompt.ask


def _validate(ref: ModelRef) -> ModelRef:
    if ref.kind == "hf_repo":
        raise FriendlyError(
            "Full-weight Hugging Face repos need vLLM (Server mode on Linux + NVIDIA), "
            "which isn't available yet.",
            "Use a GGUF build instead, e.g. hf.co/<org>/<model>-GGUF",
        )
    return ref


def choose_model(
    info: SystemInfo,
    ask: AskFn | None = None,
    curated: list[CuratedModel] | None = None,
) -> ModelRef:
    ask_fn: AskFn = ask if ask is not None else _ask
    candidates = curated if curated is not None else models.load_curated()
    fitting = models.fitting(candidates, info.ram_gb)
    table = Table(title=f"Models that fit your {info.ram_gb:.0f} GB")
    table.add_column("#")
    table.add_column("Model")
    table.add_column("Ref")
    for i, m in enumerate(fitting, start=1):
        table.add_row(str(i), m.name, m.ref)
    console.print(table)
    answer = ask_fn(
        "Pick a number, or type any model (qwen3:8b · hf.co/<org>/<repo>-GGUF)"
    ).strip()
    if answer.isdigit() and 1 <= int(answer) <= len(fitting):
        return _validate(models.parse_model_ref(fitting[int(answer) - 1].ref))
    return _validate(models.parse_model_ref(answer))


def run_wizard(dry_run: bool = False) -> None:
    info = detect.detect()
    console.print(detect.plan_sentence(info))
    ref = choose_model(info)
    cfg = config.load()
    cfg.model = ref.raw
    config.save(cfg)
    if dry_run:
        console.print("would: ensure Ollama is installed and running")
        console.print(f"would: pull model {ref.raw}")
        console.print(f"would: start OpenWebUI on port {cfg.webui_port}")
        console.print(f"would: open {express.webui_url(cfg.webui_port)}")
        return
    express.ensure_ollama(info)
    express.pull_model(ref)
    express.ensure_openwebui(cfg)
    url = express.webui_url(cfg.webui_port)
    console.print(f"[green]✓ Ready:[/green] {url}")
    from ezai.cli import _open_browser

    _open_browser(url)
```

In `src/ezai/cli.py`, update the callback:

```python
@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what the wizard would do without doing it."
    ),
) -> None:
    if version:
        typer.echo(_version_string())
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        from ezai import wizard

        wizard.run_wizard(dry_run=dry_run)
```

(The `wizard` import stays inside the function to avoid an import cycle with `wizard`'s import of `cli._open_browser`.)

- [ ] **Step 4: Run tests** — `uv run pytest -q` — Expected: PASS (including Task 1's version test, unchanged).
- [ ] **Step 5: Quality gates.** Expected: all pass.
- [ ] **Step 6: Commit and open PR**

```bash
git add -A
git commit -m "feat: add setup wizard as the default ezai command"
git push -u origin HEAD
gh pr create --fill --title "feat: add setup wizard as the default ezai command"
```

---

### Task 10: Model commands — `model add`, `model list`, `model rm`

**Files:**
- Modify: `src/ezai/cli.py`
- Test: `tests/test_cli_models.py`

**Interfaces:**
- Consumes: `wizard.choose_model`, `models.parse_model_ref`, `wizard._validate` behavior (re-implement via `choose_model`/`parse_model_ref` + the same `FriendlyError` for `hf_repo`), `express.ensure_ollama`, `express.pull_model`, `proc.run_logged`, `config.load/save`
- Produces: `model_app = typer.Typer()` registered as `app.add_typer(model_app, name="model")` with:
  - `ezai model add [REF]` — no arg → interactive `wizard.choose_model`; with arg → parse; `hf_repo` → `FriendlyError`; then `ensure_ollama` + `pull_model` + save `cfg.model`
  - `ezai model list` — `run_logged(["ollama", "list"], check=False)`, print stdout
  - `ezai model rm NAME` — `run_logged(["ollama", "rm", NAME])`

- [ ] **Step 1: Write failing tests**

`tests/test_cli_models.py`:

```python
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from ezai import cli, config, detect, express, proc

runner = CliRunner()

INFO = detect.SystemInfo(
    os="linux", arch="x86_64", gpu="nvidia", ram_gb=32.0,
    has_docker=False, has_ollama=True, ollama_running=True,
)


@pytest.fixture()
def fake_engine(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    pulled: list[str] = []
    monkeypatch.setattr(detect, "detect", lambda **k: INFO)
    monkeypatch.setattr(express, "ensure_ollama", lambda info, **k: None)
    monkeypatch.setattr(express, "pull_model", lambda ref, **k: pulled.append(ref.raw))
    return pulled


def test_model_add_with_ref_pulls_and_saves(
    fake_engine: list[str], isolated_home: Path
) -> None:
    result = runner.invoke(cli.app, ["model", "add", "qwen3:8b"])
    assert result.exit_code == 0
    assert fake_engine == ["qwen3:8b"]
    assert config.load().model == "qwen3:8b"


def test_model_add_rejects_full_weight_repo(
    fake_engine: list[str], isolated_home: Path
) -> None:
    result = runner.invoke(cli.app, ["model", "add", "meta-llama/Llama-3.3-70B"])
    assert result.exit_code != 0
    assert fake_engine == []


def test_model_list_shows_ollama_output(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert cmd == ["ollama", "list"]
        return subprocess.CompletedProcess(cmd, 0, stdout="NAME  SIZE\nqwen3:8b  5GB\n", stderr="")

    monkeypatch.setattr(proc, "run_logged", fake_run)
    result = runner.invoke(cli.app, ["model", "list"])
    assert result.exit_code == 0
    assert "qwen3:8b" in result.output


def test_model_rm_invokes_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(proc, "run_logged", fake_run)
    result = runner.invoke(cli.app, ["model", "rm", "qwen3:8b"])
    assert result.exit_code == 0
    assert ["ollama", "rm", "qwen3:8b"] in calls
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_cli_models.py -q` — Expected: FAIL.

- [ ] **Step 3: Implement in `src/ezai/cli.py`**

```python
model_app = typer.Typer(help="Add, list, or remove local models.")
app.add_typer(model_app, name="model")


@model_app.command("add")
def model_add(
    ref: str | None = typer.Argument(
        None, help="qwen3:8b · hf.co/<org>/<repo>-GGUF · leave empty to browse"
    ),
) -> None:
    """Download a model and make it the default."""
    from ezai import models as models_mod
    from ezai import wizard

    info = detect.detect()
    if ref is None:
        model_ref = wizard.choose_model(info)
    else:
        model_ref = models_mod.parse_model_ref(ref)
        if model_ref.kind == "hf_repo":
            raise FriendlyError(
                "Full-weight Hugging Face repos need vLLM (Server mode on Linux + NVIDIA), "
                "which isn't available yet.",
                "Use a GGUF build instead, e.g. hf.co/<org>/<model>-GGUF",
            )
    express.ensure_ollama(info)
    express.pull_model(model_ref)
    cfg = config.load()
    cfg.model = model_ref.raw
    config.save(cfg)
    console.print(f"[green]✓ Added:[/green] {model_ref.raw}")


@model_app.command("list")
def model_list() -> None:
    """List downloaded models."""
    result = proc.run_logged(["ollama", "list"], check=False)
    console.print(result.stdout or "No models yet — run `ezai model add`.", markup=False)


@model_app.command("rm")
def model_rm(name: str = typer.Argument(..., help="Model name as shown by `ezai model list`.")) -> None:
    """Remove a downloaded model."""
    proc.run_logged(["ollama", "rm", name])
    console.print(f"[green]✓ Removed:[/green] {name}")
```

Note: tests monkeypatch `proc.run_logged`, so `cli` must call it as `proc.run_logged(...)` (module attribute), and `import ... from ezai import proc` at top. `FriendlyError` raised inside a command surfaces via `run()`'s handler in real usage; in tests `CliRunner` reports nonzero exit via the raised exception — assert `result.exit_code != 0` only.

Typer note: commands raising `FriendlyError` under `CliRunner.invoke` mark `result.exit_code` as 1 only when the exception is handled; ensure `cli.run()` is the real entry and add `app` invocation robustness by catching `FriendlyError` in each command? NO — keep the single handler in `run()`. For tests, `CliRunner.invoke` sets `result.exception`; `result.exit_code` is 1 when any exception escapes. This is the documented Click behavior — asserting `result.exit_code != 0` is valid.

- [ ] **Step 4: Run tests** — `uv run pytest -q` — Expected: PASS.
- [ ] **Step 5: Quality gates.** Expected: all pass.
- [ ] **Step 6: Commit and open PR**

```bash
git add -A
git commit -m "feat: add model add/list/rm commands"
git push -u origin HEAD
gh pr create --fill --title "feat: add model add/list/rm commands"
```

---

### Task 11: `ezai doctor`

**Files:**
- Create: `src/ezai/doctor.py`
- Modify: `src/ezai/cli.py`
- Test: `tests/test_doctor.py`

**Interfaces:**
- Consumes: `detect.SystemInfo`, `detect.api_up`, `detect.OLLAMA_URL`, `express.webui_up`, `config.load`, `paths`
- Produces:
  - `@dataclass(frozen=True) doctor.CheckResult(name: str, ok: bool, hint: str = "")`
  - `doctor.run_checks(info: SystemInfo, which: Callable[[str], str | None] = shutil.which, api_up: Callable[..., bool] = detect.api_up, webui_up: Callable[..., bool] = express.webui_up) -> list[CheckResult]` — checks: `uv` on PATH; `ollama` installed; Ollama API responding; OpenWebUI responding on configured port; RAM ≥ 8 GB (warning-grade check); each failed check carries a `hint` with a one-line fix
  - `ezai doctor` command printing ✓/✗ per check + hints, exit code 1 if any core check failed (RAM check is advisory: prints a warning but does not fail the run)

- [ ] **Step 1: Write failing tests**

`tests/test_doctor.py`:

```python
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from ezai import cli, doctor
from ezai.detect import SystemInfo

runner = CliRunner()


def info(ram: float = 32.0, has_ollama: bool = True) -> SystemInfo:
    return SystemInfo(
        os="linux", arch="x86_64", gpu="nvidia", ram_gb=ram,
        has_docker=False, has_ollama=has_ollama, ollama_running=True,
    )


def test_all_green_when_everything_up(isolated_home: Path) -> None:
    results = doctor.run_checks(
        info(),
        which=lambda n: f"/usr/bin/{n}",
        api_up=lambda url, **k: True,
        webui_up=lambda port, **k: True,
    )
    assert all(r.ok for r in results)


def test_missing_ollama_has_hint(isolated_home: Path) -> None:
    results = doctor.run_checks(
        info(has_ollama=False),
        which=lambda n: "/usr/bin/uv" if n == "uv" else None,
        api_up=lambda url, **k: False,
        webui_up=lambda port, **k: False,
    )
    failed = {r.name: r for r in results if not r.ok}
    assert "Ollama installed" in failed
    assert failed["Ollama installed"].hint


def test_low_ram_is_flagged(isolated_home: Path) -> None:
    results = doctor.run_checks(
        info(ram=4.0),
        which=lambda n: f"/usr/bin/{n}",
        api_up=lambda url, **k: True,
        webui_up=lambda port, **k: True,
    )
    ram_check = next(r for r in results if r.name == "RAM")
    assert ram_check.ok is False
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_doctor.py -q` — Expected: FAIL.

- [ ] **Step 3: Implement `src/ezai/doctor.py`**

```python
"""ezai doctor: every red ✗ comes with a one-line fix."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass

from ezai import config, detect, express
from ezai.detect import SystemInfo


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    hint: str = ""


def run_checks(
    info: SystemInfo,
    which: Callable[[str], str | None] = shutil.which,
    api_up: Callable[..., bool] = detect.api_up,
    webui_up: Callable[..., bool] = express.webui_up,
) -> list[CheckResult]:
    cfg = config.load()
    results = [
        CheckResult(
            "uv installed",
            which("uv") is not None,
            "Install uv: https://docs.astral.sh/uv/getting-started/installation/",
        ),
        CheckResult(
            "Ollama installed",
            info.has_ollama,
            "Run `ezai` to install it, or see https://ollama.com/download",
        ),
        CheckResult(
            "Ollama API responding",
            api_up(detect.OLLAMA_URL),
            "Run `ezai up` to start it; logs: `ezai logs`",
        ),
        CheckResult(
            "OpenWebUI responding",
            webui_up(cfg.webui_port),
            "Run `ezai up`; if the port is busy, change webui_port in "
            f"{config.config_path()}",
        ),
        CheckResult(
            "RAM",
            info.ram_gb >= 8.0,
            f"{info.ram_gb:.0f} GB detected — 8 GB+ recommended; stick to small "
            "models like qwen3:0.6b or llama3.2:3b",
        ),
    ]
    return results
```

Add to `src/ezai/cli.py`:

```python
@app.command()
def doctor() -> None:
    """Diagnose the local setup."""
    from ezai import doctor as doctor_mod

    info = detect.detect()
    results = doctor_mod.run_checks(info)
    core_failed = False
    for r in results:
        mark = "[green]✓[/green]" if r.ok else "[red]✗[/red]"
        console.print(f"{mark} {r.name}")
        if not r.ok:
            console.print(f"  [yellow]→ {r.hint}[/yellow]")
            if r.name != "RAM":
                core_failed = True
    if core_failed:
        raise typer.Exit(code=1)
```

- [ ] **Step 4: Run tests** — `uv run pytest -q` — Expected: PASS.
- [ ] **Step 5: Quality gates.** Expected: all pass.
- [ ] **Step 6: Commit and open PR**

```bash
git add -A
git commit -m "feat: add ezai doctor diagnostics"
git push -u origin HEAD
gh pr create --fill --title "feat: add ezai doctor diagnostics"
```

---

### Task 12: `ezai update`

**Files:**
- Modify: `src/ezai/cli.py`
- Test: `tests/test_cli_update.py`

**Interfaces:**
- Consumes: `detect.detect`, `proc.run_logged`, `express.stop_openwebui`, `express.ensure_openwebui`, `config.load`
- Produces: `ezai update` — upgrades Ollama (macOS: `brew upgrade ollama` when brew exists, else skip with note; Linux: re-run official install script; Windows: `winget upgrade --id Ollama.Ollama -e`), upgrades OpenWebUI (`uv tool upgrade open-webui`), restarts OpenWebUI (stop + ensure). All upgrade calls use `check=False` — an up-to-date package returning nonzero must not abort the run.

- [ ] **Step 1: Write failing tests**

`tests/test_cli_update.py`:

```python
from __future__ import annotations

import subprocess
from typing import Any

import pytest
from typer.testing import CliRunner

from ezai import cli, detect, express, proc

runner = CliRunner()


def test_update_upgrades_engine_and_webui(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    info = detect.SystemInfo(
        os="linux", arch="x86_64", gpu="nvidia", ram_gb=32.0,
        has_docker=False, has_ollama=True, ollama_running=True,
    )
    monkeypatch.setattr(detect, "detect", lambda **k: info)
    monkeypatch.setattr(proc, "run_logged", fake_run)
    monkeypatch.setattr(express, "stop_openwebui", lambda os_name, **k: True)
    monkeypatch.setattr(express, "ensure_openwebui", lambda cfg, **k: None)

    result = runner.invoke(cli.app, ["update"])
    assert result.exit_code == 0
    assert any("ollama.com/install.sh" in " ".join(c) for c in calls)
    assert ["uv", "tool", "upgrade", "open-webui"] in calls
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_cli_update.py -q` — Expected: FAIL.

- [ ] **Step 3: Implement in `src/ezai/cli.py`** (needs `shutil` imported at top)

```python
@app.command()
def update() -> None:
    """Upgrade Ollama and OpenWebUI to their latest versions."""
    info = detect.detect()
    console.print("Upgrading Ollama…")
    if info.os == "macos":
        if shutil.which("brew") is not None:
            proc.run_logged(["brew", "upgrade", "ollama"], check=False)
        else:
            console.print("Ollama.app updates itself — skipping engine upgrade.")
    elif info.os == "linux":
        proc.run_logged(["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"], check=False)
    else:
        proc.run_logged(["winget", "upgrade", "--id", "Ollama.Ollama", "-e"], check=False)
    console.print("Upgrading OpenWebUI…")
    proc.run_logged(["uv", "tool", "upgrade", "open-webui"], check=False)
    express.stop_openwebui(info.os)
    express.ensure_openwebui(config.load())
    console.print("[green]✓ Everything is up to date and running.[/green]")
```

- [ ] **Step 4: Run tests** — `uv run pytest -q` — Expected: PASS.
- [ ] **Step 5: Quality gates.** Expected: all pass.
- [ ] **Step 6: Commit and open PR**

```bash
git add -A
git commit -m "feat: add ezai update command"
git push -u origin HEAD
gh pr create --fill --title "feat: add ezai update command"
```

---

### Task 13: Installers and the real README

**Files:**
- Create: `install.sh`, `install.ps1`
- Modify: `README.md` (full rewrite)

**Interfaces:**
- Consumes: the published CLI surface from Tasks 1-12 (commands and flags exactly as implemented — verify with `uv run ezai --help` before writing)
- Produces: `curl`-able POSIX installer, PowerShell installer, launch-ready README.

- [ ] **Step 1: Create `install.sh`**

```sh
#!/bin/sh
# ezai installer — installs uv (if needed) and ezai, then starts setup.
set -eu

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (Python package manager)…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # uv installs into ~/.local/bin
  PATH="$HOME/.local/bin:$PATH"
  export PATH
fi

echo "Installing ezai…"
uv tool install --force ezai

echo ""
echo "✓ ezai installed. Starting setup…"
exec uv tool run ezai
```

- [ ] **Step 2: Create `install.ps1`**

```powershell
# ezai installer — installs uv (if needed) and ezai, then starts setup.
$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (Python package manager)…"
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Write-Host "Installing ezai…"
uv tool install --force ezai

Write-Host ""
Write-Host "✓ ezai installed. Starting setup…"
uv tool run ezai
```

- [ ] **Step 3: Rewrite `README.md`**

Structure (write real content for each part, matching the implemented CLI exactly):

```markdown
# ezai

**One command → local AI chat in your browser.**

Self-host LLMs on your own GPU — Mac (Metal), Linux and Windows (NVIDIA) — with zero configuration. No Docker required.

<!-- demo GIF placeholder: added by the release plan via vhs -->

## Install

**Mac / Linux**
```sh
curl -fsSL https://raw.githubusercontent.com/MikaelSabuhi/ezaiselfhost/main/install.sh | sh
```

**Windows (PowerShell)**
```powershell
irm https://raw.githubusercontent.com/MikaelSabuhi/ezaiselfhost/main/install.ps1 | iex
```

**Already have [uv](https://docs.astral.sh/uv/)?**
```sh
uvx ezai
```

That's it. ezai detects your OS, GPU, and RAM, suggests models that actually fit your machine, installs [Ollama](https://ollama.com) + [OpenWebUI](https://openwebui.com), and opens your browser.

## What you get
(table: platform / GPU used / how)

## Everyday commands
(table of ezai up/down/status/logs/model add|list|rm/update/doctor with one-line descriptions — copy from `ezai --help`)

## Pick any model
(the three ref shapes with one example each)

## Requirements
(honest minimums: 8 GB RAM recommended, disk space for models, that's all)

## Why not just use Ollama directly?
(honesty table: what ezai adds — the wizard, RAM-fit model picks, OpenWebUI wiring, doctor, one-line updates; link out to the raw tools for people who prefer them)

## Roadmap
(Server mode with docker compose + vLLM + network exposure w/ API keys; PyPI note if not yet published)

## Contributing / License
(worktree + PR workflow, MIT)
```

- [ ] **Step 4: Verify docs match reality**

Run: `uv run ezai --help` and compare every documented command/flag against the README tables. Fix mismatches in the README.

- [ ] **Step 5: Quality gates** (still must pass — README/installer changes shouldn't break them, but run anyway).

- [ ] **Step 6: Commit and open PR**

```bash
git add -A
git commit -m "docs: add installers and launch-ready README"
git push -u origin HEAD
gh pr create --fill --title "docs: add installers and launch-ready README"
```

---

## Deferred to follow-up plans (do NOT implement here)

- **Plan 2 — Server mode:** `stack/` compose file (openwebui, ollama, vllm, caddy profiles), `.env` generation, `ezai expose` + API keys, `ezai connect <url>` remote mode, mode selection in the wizard, vLLM `hf_repo` support.
- **Plan 3 — Release & virality:** PyPI trusted publishing workflow on `v*` tags, weekly smoke-test workflow, vhs GIF, CONTRIBUTING.md, issue templates, Dependabot config, branch protection setup, launch checklist.
