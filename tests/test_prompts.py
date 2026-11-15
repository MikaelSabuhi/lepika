from __future__ import annotations

from typing import Any

import pytest

from lepika import prompts
from lepika.errors import FriendlyError


def _no_terminal(*a: Any, **k: Any) -> Any:
    raise EOFError("stdin is not a terminal")


def test_ask_returns_the_default_when_there_is_no_terminal(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert prompts.ask("Pick 1 or 2", default="1", reader=_no_terminal) == "1"
    assert "No terminal to ask — using 1." in capsys.readouterr().out


def test_ask_without_a_default_raises_no_terminal() -> None:
    with pytest.raises(prompts.NoTerminal) as exc:
        prompts.ask("Pick a model", reader=_no_terminal)
    assert isinstance(exc.value, FriendlyError)
    assert "“Pick a model” needs an answer" in exc.value.problem
    assert "interactive terminal" in exc.value.fix


def test_ask_never_prints_a_secret_default(capsys: pytest.CaptureFixture[str]) -> None:
    answer = prompts.ask("Token", default="hunter2", password=True, reader=_no_terminal)
    assert answer == "hunter2"
    out = capsys.readouterr().out
    assert "hunter2" not in out
    assert "No terminal to ask — skipping." in out


def test_ask_passes_rich_keywords_through() -> None:
    seen: list[dict[str, Any]] = []

    def reader(prompt: str, **kwargs: Any) -> str:
        seen.append(kwargs)
        return "answer"

    assert prompts.ask("Q", default="d", password=True, reader=reader) == "answer"
    assert prompts.ask("Q", reader=reader) == "answer"
    assert seen[0] == {"default": "d", "password": True}
    assert seen[1] == {}


def test_confirm_default_and_no_default(capsys: pytest.CaptureFixture[str]) -> None:
    assert prompts.confirm("Go?", default=True, reader=_no_terminal) is True
    assert prompts.confirm("Go?", default=False, reader=_no_terminal) is False
    assert prompts.confirm("Go?", reader=lambda q, **k: True) is True
    with pytest.raises(prompts.NoTerminal):
        prompts.confirm("Go?", reader=_no_terminal)


def test_require_terminal_raises_even_with_a_default(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(prompts.NoTerminal):
        prompts.ask("Pick", default="1", require_terminal=True, reader=_no_terminal)
    with pytest.raises(prompts.NoTerminal):
        prompts.confirm("Go?", default=False, require_terminal=True, reader=_no_terminal)
    assert "No terminal to ask" not in capsys.readouterr().out


def test_require_terminal_still_hands_the_default_to_the_reader() -> None:
    seen: list[dict[str, Any]] = []

    def reader(prompt: str, **kwargs: Any) -> Any:
        seen.append(kwargs)
        return kwargs["default"]  # what rich returns for a bare Enter

    assert prompts.ask("Pick", default="2", require_terminal=True, reader=reader) == "2"
    assert prompts.confirm("Go?", default=False, require_terminal=True, reader=reader) is False
    assert seen == [{"default": "2"}, {"default": False}]
