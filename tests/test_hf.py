"""hf.py: the Hugging Face CLI, driven for a pre-flight listing and a download."""

from __future__ import annotations

import http.client
import json
import urllib.error
from email.message import Message
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeResponse, FakeSystem, FakeUI, routes

from lepika import acquire, config, hf, paths
from lepika.errors import FriendlyError
from lepika.system import Response

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
        sys=FakeSystem(
            stdout={"uv tool run": cached},
            # An empty cache: nothing to stat, so a `-` stays 0 rather than guessing.
            environ={"HF_HUB_CACHE": str(tmp_path / "empty-hub")},
        ),
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
        "org/repo", sys=FakeSystem(stdout={"uv": listing}, environ={"HF_HUB_CACHE": str(cache)})
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
        sys=FakeSystem(stdout={"uv": listing}, environ={"HF_HUB_CACHE": str(cache)}),
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
        "org/repo", sys=FakeSystem(stdout={"uv": listing}, environ={"HF_HUB_CACHE": str(cache)})
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
    def download(cmd: list[str]) -> None:
        # Written while `hf download` runs, so a read before it would find nothing.
        (tmp_path / "config.json").write_text('{"quantization_config": {"quant_method": "awq"}}')

    sys = FakeSystem(while_running=download)
    cfg = hf.fetch_config("org/repo", tmp_path, sys=sys)
    assert hf.quant_method(cfg) == "awq"
    assert "config.json" in sys.ran[0]
    assert "--dry-run" not in sys.ran[0]  # a real (single-file) download, not a listing


def test_fetch_config_failure_reads_as_no_config(tmp_path: Path) -> None:
    """The Hub not answering must degrade to today's behavior, never block an import."""
    assert hf.fetch_config("org/repo", tmp_path, sys=FakeSystem(codes={"uv": 1})) == {}


def test_preflight_lists_files_and_sums_sizes() -> None:
    sys = FakeSystem(stdout={"uv tool run": "Warning: unauthenticated\n" + LISTING})
    pre = hf.preflight("Qwen/Qwen3.5-2B", sys=sys)
    assert pre.has_safetensors is True
    assert pre.has_gguf is False
    assert pre.total_bytes == 2_900 + 4_500_000_000 + 1_200_000_000 + 390
    cmd = sys.ran[0]
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
    sys = FakeSystem(stdout={"uv": LISTING}, environ={"PATH": "/bin"})
    hf.preflight("Qwen/Qwen3.5-2B", token="hf_secret", sys=sys)
    seen = sys.ran_kwargs
    assert seen[0]["log"] is False
    assert seen[0]["check"] is False
    assert seen[0]["env"] == {"PATH": "/bin", "HF_TOKEN": "hf_secret"}
    assert seen[0]["timeout"] == 120


def test_preflight_sees_a_gguf_only_repo() -> None:
    listing = json.dumps([{"file": "model-Q4_K_M.gguf", "size": "4.9G"}])
    pre = hf.preflight("unsloth/x-GGUF", sys=FakeSystem(stdout={"uv": listing}))
    assert pre.has_gguf is True
    assert pre.has_safetensors is False


# What `huggingface_hub` 2.4.0 prints for a gated repo: no status code, no "gated".
GATED_CLI_OUTPUT = "Error: Access denied. Model 'meta-llama/Llama-3.2-1B' requires approval."


_HUB_MODEL = "https://huggingface.co/api/models/meta-llama/Llama-3.2-1B"


def _hub(
    answer: Any = None,
    *,
    raises: BaseException | None = None,
    reply: Response | None = None,
    output: str = GATED_CLI_OUTPUT,
) -> FakeSystem:
    """`hf` fails to list the repo, printing `output`; the Hub's model API answers
    `answer` as JSON — or `reply` as it is, or raises `raises`."""
    found: Response | BaseException
    if raises is not None:
        found = raises
    elif reply is not None:
        found = reply
    else:
        found = FakeResponse(json.dumps(answer).encode())
    return FakeSystem(stdout={"uv": output}, codes={"uv": 1}, urls=routes({_HUB_MODEL: found}))


def _http_error(code: int) -> urllib.error.HTTPError:
    url = "https://huggingface.co/api/models/x/y"
    return urllib.error.HTTPError(url, code, "error", Message(), None)


def _failing_preflight(sys: FakeSystem, token: str = "") -> None:
    hf.preflight("meta-llama/Llama-3.2-1B", token=token, sys=sys)


def test_a_gated_repo_is_recognised_whatever_the_cli_says() -> None:
    hub = _hub({"id": "meta-llama/Llama-3.2-1B", "gated": "manual"})
    with pytest.raises(hf.GatedRepo) as exc:
        _failing_preflight(hub)
    assert "gated" in exc.value.problem
    request, timeout = hub.requested[0], hub.request_timeouts[0]
    assert request.full_url == "https://huggingface.co/api/models/meta-llama/Llama-3.2-1B"
    assert timeout == 5


def test_a_gated_repo_is_recognised_for_auto_approval() -> None:
    with pytest.raises(hf.GatedRepo):
        _failing_preflight(_hub({"gated": "auto"}))


@pytest.mark.parametrize("code", [401, 404])
def test_a_missing_repo_is_not_found_on_401_and_404(code: int) -> None:
    """The Hub answers a repo that does not exist with a 401 — still a typo, not a licence."""
    with pytest.raises(FriendlyError) as exc:
        _failing_preflight(_hub(raises=_http_error(code)))
    assert not isinstance(exc.value, hf.GatedRepo)
    assert "not on Hugging Face" in exc.value.problem
    assert "HF_TOKEN" in exc.value.fix


def test_a_forbidden_repo_is_gated() -> None:
    with pytest.raises(hf.GatedRepo):
        _failing_preflight(_hub(raises=_http_error(403)))


def test_a_public_repo_that_failed_to_list_is_a_generic_failure() -> None:
    with pytest.raises(FriendlyError) as exc:
        _failing_preflight(_hub({"gated": False}, output="ConnectionError: boom"))
    assert not isinstance(exc.value, hf.GatedRepo)
    assert "Could not list" in exc.value.problem


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.URLError("no network"),
        TimeoutError("timed out"),
        _http_error(500),
        http.client.IncompleteRead(b""),  # a truncated body: not an OSError
    ],
)
def test_an_unreachable_hub_is_a_generic_failure(failure: BaseException) -> None:
    with pytest.raises(FriendlyError) as exc:
        _failing_preflight(_hub(raises=failure))
    assert not isinstance(exc.value, hf.GatedRepo)
    assert "Could not list" in exc.value.problem


def test_an_unreadable_hub_answer_is_a_generic_failure() -> None:
    garbled = _hub(reply=FakeResponse(b"<html>rate limited</html>"))
    with pytest.raises(FriendlyError) as exc:
        _failing_preflight(garbled)
    assert "Could not list" in exc.value.problem


def test_cli_wording_alone_never_classifies() -> None:
    with pytest.raises(FriendlyError) as exc:
        _failing_preflight(_hub({"gated": False}, output="404 not found gated 401"))
    assert not isinstance(exc.value, hf.GatedRepo)
    assert "Could not list" in exc.value.problem


def test_the_token_travels_as_a_header_only() -> None:
    hub = _hub({"gated": "manual"})
    with pytest.raises(hf.GatedRepo):
        _failing_preflight(hub, token="hf_secret")
    request = hub.requested[0]
    assert request.get_header("Authorization") == "Bearer hf_secret"
    assert "hf_secret" not in request.full_url
    # Unredirected: urllib copies `headers` onto a 30x hop, wherever it points.
    assert "Authorization" not in request.headers
    assert "Authorization" in request.unredirected_hdrs


def test_no_token_sends_no_authorization_header() -> None:
    hub = _hub({"gated": "manual"})
    with pytest.raises(hf.GatedRepo):
        _failing_preflight(hub, token="")
    request = hub.requested[0]
    assert request.get_header("Authorization") is None


def test_a_successful_listing_never_asks_the_hub() -> None:
    # No URL is answered: asking the Hub would raise past every except.
    hub = FakeSystem(stdout={"uv": LISTING})
    hf.preflight("Qwen/Qwen3.5-2B", sys=hub)
    assert hub.requested == []


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
    pre = hf.preflight("org/repo", sys=FakeSystem(stdout={"uv": MIXED_LISTING}))
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
    sys = FakeSystem(stdout={"uv": LISTING})
    with pytest.raises(FriendlyError):
        hf.preflight(repo, sys=sys)
    assert sys.ran == []


def test_preflight_garbled_listing_is_friendly() -> None:
    with pytest.raises(FriendlyError):
        hf.preflight("org/repo", sys=FakeSystem(stdout={"uv": "not json"}))


def test_download_dir_nests_org_and_repo_under_lepika_home(isolated_home: Path) -> None:
    assert hf.download_dir("Qwen/Qwen3.5-2B") == isolated_home / "hf" / "Qwen" / "Qwen3.5-2B"


def test_download_streams_hf_with_excludes_and_the_token_in_env(isolated_home: Path) -> None:
    dest = isolated_home / "hf" / "org" / "repo"
    sys = FakeSystem()
    hf.download("org/repo", dest, token="hf_secret", sys=sys)
    cmd, kwargs = sys.streamed[0]
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
        hf.download("org/repo", isolated_home / "x", sys=FakeSystem(stream=(1, "boom")))
    assert "resumes" in exc.value.fix
    entry = json.loads((paths.logs_dir() / "lepika.log").read_text().splitlines()[-1])
    assert entry["event"] == "hf.download"
    assert entry["result"] == "failed"


def test_token_for_prefers_the_environment_then_the_config() -> None:
    cfg = config.Config(hf_token="from-config")
    assert hf.token_for(cfg, sys=FakeSystem(environ={"HF_TOKEN": "from-env"})) == "from-env"
    assert hf.token_for(cfg, sys=FakeSystem(environ={})) == "from-config"
    assert hf.token_for(config.Config(), sys=FakeSystem()) == ""


def test_ask_token_returns_the_stripped_answer_and_saves_nothing(isolated_home: Path) -> None:
    prompts: list[dict[str, Any]] = []

    def ask(prompt: str, **kwargs: Any) -> str:
        prompts.append(kwargs)
        return " hf_new "

    assert hf.ask_token(ui=FakeUI(reader=ask)) == "hf_new"
    assert prompts[0]["password"] is True
    assert not config.config_path().exists()


def _gated_until_a_token(
    monkeypatch: pytest.MonkeyPatch, calls: list[str] | None = None
) -> hf.Preflight:
    listed = hf.Preflight(files=("model.safetensors",), total_bytes=1, download_bytes=1)

    def preflight(repo: str, token: str = "", **k: Any) -> hf.Preflight:
        if calls is not None:
            calls.append(token)
        if not token:
            raise hf.GatedRepo("'x/y' is gated or private on Hugging Face.", "Accept its licence.")
        return listed

    monkeypatch.setattr(hf, "preflight", preflight)
    return listed


def test_preflight_with_token_saves_a_typed_token_at_once(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    """A credential, saved the moment it is typed — and handed back with the rest kept."""
    calls: list[str] = []
    listed = _gated_until_a_token(monkeypatch, calls)
    cfg = config.Config(model="kept:model")
    pre, token, now = acquire.preflight_with_token(
        "x/y", cfg, sys=FakeSystem(), ui=FakeUI(reader=lambda *a, **k: " hf_new ")
    )
    assert (pre, token, calls) == (listed, "hf_new", ["", "hf_new"])
    assert now == config.Config(model="kept:model", hf_token="hf_new")
    assert config.load() == now


def test_ask_token_empty_answer_saves_nothing(
    monkeypatch: pytest.MonkeyPatch, isolated_home: Path
) -> None:
    assert hf.ask_token(ui=FakeUI(reader=lambda *a, **k: "")) == ""
    _gated_until_a_token(monkeypatch)
    with pytest.raises(hf.GatedRepo):
        acquire.preflight_with_token(
            "x/y", config.Config(), sys=FakeSystem(), ui=FakeUI(reader=lambda *a, **k: "")
        )
    assert not config.config_path().exists()


def test_token_prompt_without_a_terminal_skips(isolated_home: Path) -> None:
    def no_terminal(*a: Any, **k: Any) -> str:
        raise EOFError

    assert hf.ask_token(ui=FakeUI(reader=no_terminal)) == ""
    assert not config.config_path().exists()
