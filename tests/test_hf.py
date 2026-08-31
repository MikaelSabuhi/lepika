"""hf.py: the Hugging Face CLI, driven for a pre-flight listing and a download."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from fakes import Runner, Streamer

from lepika import config, hf, paths
from lepika.errors import FriendlyError

LISTING = json.dumps(
    [
        {"file": "config.json", "size": "2.9K"},
        {"file": "model-00001-of-00002.safetensors", "size": "4.5G"},
        {"file": "model-00002-of-00002.safetensors", "size": "1.2G"},
        {"file": "README.md", "size": "390.0"},
    ]
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("390.0", 390),
        ("2.9K", 2_900),
        ("4.5G", 4_500_000_000),
        ("1.5T", 1_500_000_000_000),
        ("-", 0),  # already in the hub cache
    ],
)
def test_parse_size_reads_the_units_hf_prints(text: str, expected: int) -> None:
    assert hf._parse_size(text) == expected


def test_preflight_reads_a_listing_with_cached_files(tmp_path: Path) -> None:
    """`hf` prints `-` for anything already in ~/.cache/huggingface — not a parse error.

    The file still ships in the repo, so it must stay in `files`: a cached
    safetensors that vanished here would send a full-weight repo to a pull.
    """
    cached = json.dumps(
        [
            {"file": "config.json", "size": "-"},
            {"file": "model.safetensors", "size": "-"},
            {"file": "tokenizer.json", "size": "2.9K"},
        ]
    )
    pre = hf.preflight(
        "Qwen/Qwen3.5-2B",
        run=Runner(stdout={"uv tool run": cached}),
        # An empty cache: nothing to stat, so a `-` stays 0 rather than guessing.
        environ={"HF_HUB_CACHE": str(tmp_path / "empty-hub")},
    )
    assert pre.has_safetensors is True
    assert "model.safetensors" in pre.files
    assert pre.total_bytes == 2_900
    assert pre.download_bytes == 2_900


def test_cache_dir_follows_the_hub_environment_variables(tmp_path: Path) -> None:
    hub, home = str(tmp_path / "hub"), str(tmp_path / "hf")
    assert hf.cache_dir({"HF_HUB_CACHE": hub}) == tmp_path / "hub"
    assert hf.cache_dir({"HF_HOME": home}) == tmp_path / "hf" / "hub"
    # The library's own precedence: the specific variable beats the general one…
    assert hf.cache_dir({"HF_HUB_CACHE": hub, "HF_HOME": home}) == tmp_path / "hub"
    # …but an exported-and-empty one is not a choice, so it falls through.
    assert hf.cache_dir({"HF_HUB_CACHE": "  ", "HF_HOME": home}) == tmp_path / "hf" / "hub"
    assert hf.cache_dir({}) == Path.home() / ".cache" / "huggingface" / "hub"


def test_preflight_never_stats_outside_the_hub_cache(tmp_path: Path) -> None:
    """A `-` size is not a reason to stat a path that climbs out of the cache directory."""
    cache = tmp_path / "hub"
    (cache / "models--org--repo" / "snapshots" / "aaa").mkdir(parents=True)
    (tmp_path / "secret").write_bytes(b"\0" * 999)
    listing = json.dumps([{"file": "../../../secret", "size": "-"}])
    pre = hf.preflight(
        "org/repo", run=Runner(stdout={"uv": listing}), environ={"HF_HUB_CACHE": str(cache)}
    )
    assert pre.download_bytes == 0


def test_preflight_sizes_a_cached_file_from_the_hub_cache(tmp_path: Path) -> None:
    """A cached file costs no download, but `hf download` copies it onto our disk anyway.

    Sizing it 0 undercounts exactly the repos a user is most likely to import twice.
    """
    cache = tmp_path / "hub"
    snapshot = cache / "models--Qwen--Qwen3.5-2B" / "snapshots" / "a1b2c3"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"\0" * 4096)
    (snapshot / "model.gguf").write_bytes(b"\0" * 8192)
    listing = json.dumps(
        [
            {"file": "model.safetensors", "size": "-"},
            {"file": "model.gguf", "size": "-"},  # cached too, but EXCLUDES skips it
            {"file": "tokenizer.json", "size": "2.9K"},
        ]
    )
    pre = hf.preflight(
        "Qwen/Qwen3.5-2B",
        run=Runner(stdout={"uv": listing}),
        environ={"HF_HUB_CACHE": str(cache)},
    )
    assert pre.total_bytes == 4096 + 8192 + 2_900
    assert pre.download_bytes == 4096 + 2_900


def test_preflight_reads_the_cached_size_through_the_snapshot_symlink(tmp_path: Path) -> None:
    """The snapshot entry is a symlink into `blobs/` — `stat()` is what follows it."""
    cache = tmp_path / "hub"
    repo = cache / "models--org--repo"
    blob = repo / "blobs" / "deadbeef"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"\0" * 2048)
    # Two snapshots, and only the second holds the file: the first must not stop the search.
    (repo / "snapshots" / "aaa").mkdir(parents=True)
    snapshot = repo / "snapshots" / "bbb"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").symlink_to(blob)
    listing = json.dumps([{"file": "model.safetensors", "size": "-"}])
    pre = hf.preflight(
        "org/repo", run=Runner(stdout={"uv": listing}), environ={"HF_HUB_CACHE": str(cache)}
    )
    assert pre.download_bytes == 2048


def test_load_config_reads_the_repo_config(tmp_path: Path) -> None:
    assert hf.load_config(tmp_path) == {}  # absent
    (tmp_path / "config.json").write_text("not json")
    assert hf.load_config(tmp_path) == {}  # unreadable is not an error, just unknown
    (tmp_path / "config.json").write_text('{"architectures": ["X"]}')
    assert hf.load_config(tmp_path) == {"architectures": ["X"]}


def test_quant_method_reads_the_quantization_config() -> None:
    assert hf.quant_method({}) is None
    assert hf.quant_method({"quantization_config": {}}) is None
    method = hf.quant_method({"quantization_config": {"quant_method": "modelopt"}})
    assert method == "modelopt"
    # A quantization_config without a method name is still a quantized checkpoint.
    assert hf.quant_method({"quantization_config": {"bits": 4}}) == "unknown"


def test_fetch_config_downloads_only_the_config_into_the_staging_dir(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def run(cmd: list[str], **k: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        (tmp_path / "config.json").write_text('{"quantization_config": {"quant_method": "awq"}}')
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    cfg = hf.fetch_config("org/repo", tmp_path, run=run, environ={})
    assert hf.quant_method(cfg) == "awq"
    assert "config.json" in calls[0]
    assert "--dry-run" not in calls[0]  # a real (single-file) download, not a listing


def test_fetch_config_failure_reads_as_no_config(tmp_path: Path) -> None:
    """The Hub not answering must degrade to today's behavior, never block an import."""

    def run(cmd: list[str], **k: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    assert hf.fetch_config("org/repo", tmp_path, run=run, environ={}) == {}


def test_preflight_lists_files_and_sums_sizes() -> None:
    run = Runner(stdout={"uv tool run": "Warning: unauthenticated\n" + LISTING})
    pre = hf.preflight("Qwen/Qwen3.5-2B", run=run, environ={})
    assert pre.has_safetensors is True
    assert pre.has_gguf is False
    assert pre.total_bytes == 2_900 + 4_500_000_000 + 1_200_000_000 + 390
    cmd = run.calls[0]
    assert cmd[:6] == ["uv", "tool", "run", "--python", hf.HF_PYTHON, "--from"]
    assert cmd[6:] == [
        "huggingface_hub",
        "hf",
        "download",
        "Qwen/Qwen3.5-2B",
        "--dry-run",
        "--json",
    ]


def test_preflight_is_a_pure_read_with_the_token_in_the_environment_only() -> None:
    seen: list[dict[str, Any]] = []

    def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.append(kwargs)
        return subprocess.CompletedProcess(cmd, 0, stdout=LISTING, stderr="")

    hf.preflight("Qwen/Qwen3.5-2B", token="hf_secret", run=run, environ={"PATH": "/bin"})
    assert seen[0]["log"] is False
    assert seen[0]["check"] is False
    assert seen[0]["env"] == {"PATH": "/bin", "HF_TOKEN": "hf_secret"}
    assert seen[0]["timeout"] == 120


def test_preflight_sees_a_gguf_only_repo() -> None:
    listing = json.dumps([{"file": "model-Q4_K_M.gguf", "size": "4.9G"}])
    pre = hf.preflight("unsloth/x-GGUF", run=Runner(stdout={"uv": listing}), environ={})
    assert pre.has_gguf is True
    assert pre.has_safetensors is False


@pytest.mark.parametrize(
    ("output", "exc_type", "needle"),
    [
        ("401 Client Error: Unauthorized", hf.GatedRepo, "gated"),
        ("403 Forbidden: gated repo", hf.GatedRepo, "gated"),
        ("404 Client Error: Repository Not Found", FriendlyError, "not on Hugging Face"),
        ("ConnectionError: boom", FriendlyError, "Could not list"),
    ],
)
def test_preflight_failures_are_friendly(output: str, exc_type: type, needle: str) -> None:
    run = Runner(stdout={"uv": output}, code=1)
    with pytest.raises(exc_type) as exc:
        hf.preflight("org/repo", run=run, environ={})
    assert needle in exc.value.problem


def test_preflight_reads_the_hubs_real_mistyped_repo_message_as_not_found() -> None:
    """The Hub answers a typo with a 401 that also says "gated": it is still a typo.

    `RepositoryNotFoundError` carries 401, "gated" and "not found" at once, so a
    gated-first order would tell everyone who mistyped a name to accept a licence.
    """
    output = (
        "401 Client Error. Repository Not Found for url: "
        "https://huggingface.co/api/models/x/y. If you are trying to access a "
        "private or gated repo, make sure you are authenticated."
    )
    with pytest.raises(FriendlyError) as exc:
        hf.preflight("x/y", run=Runner(stdout={"uv": output}, code=1), environ={})
    assert not isinstance(exc.value, hf.GatedRepo)
    assert "not on Hugging Face" in exc.value.problem
    assert "HF_TOKEN" in exc.value.fix


MIXED_LISTING = json.dumps(
    [
        {"file": "config.json", "size": "1.0K"},
        {"file": "model.safetensors", "size": "2.0G"},
        {"file": "model-Q4_K_M.gguf", "size": "4.0G"},
        {"file": "original/consolidated.00.pth", "size": "3.0G"},
    ]
)


def test_preflight_sizes_only_the_files_download_actually_fetches() -> None:
    """`--dry-run` lists everything (has_gguf needs to see the .gguf), but `download`
    excludes the GGUF/PyTorch twins — so the byte count PR B shows the user and checks
    against free disk must be the post-exclude one, or a repo shipping both is doubled.
    """
    pre = hf.preflight("org/repo", run=Runner(stdout={"uv": MIXED_LISTING}), environ={})
    assert pre.total_bytes == 1_000 + 2_000_000_000 + 4_000_000_000 + 3_000_000_000
    # The .gguf and the original/*.pth are excluded; the safetensors and config are not.
    assert pre.download_bytes == 1_000 + 2_000_000_000


@pytest.mark.parametrize("repo", ["Qwen/Qwen3.8-27B", "mlx-community/x_y.z"])
def test_check_repo_accepts_real_repo_ids(repo: str) -> None:
    assert hf.check_repo(repo) is None


@pytest.mark.parametrize("repo", ["../../x", "a/..", "noslash", "a/b/c", "a b/c", "/x"])
def test_download_dir_refuses_a_bad_repo_id_without_touching_disk(
    repo: str, isolated_home: Path
) -> None:
    with pytest.raises(FriendlyError) as exc:
        hf.download_dir(repo)
    assert "not a Hugging Face repo id" in exc.value.problem
    # Refused before `paths.hf_dir()` could create anything: no traversal, no mkdir.
    assert not (isolated_home / "hf").exists()


@pytest.mark.parametrize("repo", ["../../x", "a/..", "noslash", "a/b/c", "a b/c", "/x"])
def test_preflight_refuses_a_bad_repo_id_before_running_anything(repo: str) -> None:
    run = Runner(stdout={"uv": LISTING})
    with pytest.raises(FriendlyError):
        hf.preflight(repo, run=run, environ={})
    assert run.calls == []


def test_preflight_garbled_listing_is_friendly() -> None:
    with pytest.raises(FriendlyError):
        hf.preflight("org/repo", run=Runner(stdout={"uv": "not json"}), environ={})


def test_download_dir_nests_org_and_repo_under_lepika_home(isolated_home: Path) -> None:
    assert hf.download_dir("Qwen/Qwen3.5-2B") == isolated_home / "hf" / "Qwen" / "Qwen3.5-2B"


def test_download_streams_hf_with_excludes_and_the_token_in_env(isolated_home: Path) -> None:
    dest = isolated_home / "hf" / "org" / "repo"
    stream = Streamer()
    hf.download("org/repo", dest, token="hf_secret", stream=stream, environ={})
    cmd, kwargs = stream.calls[0]
    assert cmd[: len(hf.HF_CMD)] == list(hf.HF_CMD)
    assert cmd[len(hf.HF_CMD) :][:4] == ["download", "org/repo", "--local-dir", str(dest)]
    assert cmd.count("--exclude") == len(hf.EXCLUDES)
    assert "hf_secret" not in " ".join(cmd)
    assert kwargs["env"]["HF_TOKEN"] == "hf_secret"
    assert dest.is_dir()
    entry = json.loads((paths.logs_dir() / "lepika.log").read_text().splitlines()[-1])
    assert entry["event"] == "hf.download"
    assert entry["result"] == "success"


def test_download_failure_hints_that_a_retry_resumes(isolated_home: Path) -> None:
    with pytest.raises(FriendlyError) as exc:
        hf.download(
            "org/repo", isolated_home / "x", stream=Streamer(code=1, tail="boom"), environ={}
        )
    assert "resumes" in exc.value.fix
    entry = json.loads((paths.logs_dir() / "lepika.log").read_text().splitlines()[-1])
    assert entry["event"] == "hf.download"
    assert entry["result"] == "failed"


def test_token_for_prefers_the_environment_then_the_config() -> None:
    cfg = config.Config(hf_token="from-config")
    assert hf.token_for(cfg, environ={"HF_TOKEN": "from-env"}) == "from-env"
    assert hf.token_for(cfg, environ={}) == "from-config"
    assert hf.token_for(config.Config(), environ={}) == ""


def test_ask_token_saves_a_non_empty_answer_privately(isolated_home: Path) -> None:
    prompts: list[dict[str, Any]] = []

    def ask(prompt: str, **kwargs: Any) -> str:
        prompts.append(kwargs)
        return " hf_new "

    cfg = config.Config()
    assert hf.ask_token(cfg, ask=ask) == "hf_new"
    assert prompts[0]["password"] is True
    assert config.load().hf_token == "hf_new"


def test_ask_token_empty_answer_saves_nothing(isolated_home: Path) -> None:
    cfg = config.Config()
    assert hf.ask_token(cfg, ask=lambda *a, **k: "") == ""
    assert not config.config_path().exists()
