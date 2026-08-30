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
    [("390.0", 390), ("2.9K", 2_900), ("4.5G", 4_500_000_000), ("1.5T", 1_500_000_000_000)],
)
def test_parse_size_reads_the_units_hf_prints(text: str, expected: int) -> None:
    assert hf._parse_size(text) == expected


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
