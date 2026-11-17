from __future__ import annotations

from typing import Any

import pytest
from fakes import FakeUI, UnansweredQuestion

from lepika.errors import FriendlyError
from lepika.ui import UI, NoTerminal


def _no_terminal(*a: Any, **k: Any) -> Any:
    raise EOFError("stdin is not a terminal")


def test_ask_returns_the_default_when_there_is_no_terminal(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert UI(reader=_no_terminal).ask("Pick 1 or 2", default="1") == "1"
    assert "No terminal to ask — using 1." in capsys.readouterr().out


def test_ask_without_a_default_raises_no_terminal() -> None:
    with pytest.raises(NoTerminal) as exc:
        UI(reader=_no_terminal).ask("Pick a model")
    assert isinstance(exc.value, FriendlyError)
    assert "“Pick a model” needs an answer" in exc.value.problem
    assert "interactive terminal" in exc.value.fix


def test_ask_never_prints_a_secret_default(capsys: pytest.CaptureFixture[str]) -> None:
    answer = UI(reader=_no_terminal).ask("Token", default="hunter2", password=True)
    assert answer == "hunter2"
    out = capsys.readouterr().out
    assert "hunter2" not in out
    assert "No terminal to ask — skipping." in out


def test_ask_passes_rich_keywords_through() -> None:
    seen: list[dict[str, Any]] = []

    def reader(prompt: str, **kwargs: Any) -> str:
        seen.append(kwargs)
        return "answer"

    assert UI(reader=reader).ask("Q", default="d", password=True) == "answer"
    assert UI(reader=reader).ask("Q") == "answer"
    assert seen[0] == {"default": "d", "password": True}
    assert seen[1] == {}


def test_confirm_default_and_no_default(capsys: pytest.CaptureFixture[str]) -> None:
    assert UI(confirm_reader=_no_terminal).confirm("Go?", default=True) is True
    assert UI(confirm_reader=_no_terminal).confirm("Go?", default=False) is False
    assert UI(confirm_reader=lambda q, **k: True).confirm("Go?") is True
    with pytest.raises(NoTerminal):
        UI(confirm_reader=_no_terminal).confirm("Go?")


def test_require_terminal_raises_even_with_a_default(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(NoTerminal):
        UI(reader=_no_terminal).ask("Pick", default="1", require_terminal=True)
    with pytest.raises(NoTerminal):
        UI(confirm_reader=_no_terminal).confirm("Go?", default=False, require_terminal=True)
    assert "No terminal to ask" not in capsys.readouterr().out


def test_require_terminal_still_hands_the_default_to_the_reader() -> None:
    seen: list[dict[str, Any]] = []

    def reader(prompt: str, **kwargs: Any) -> Any:
        seen.append(kwargs)
        return kwargs["default"]  # what rich returns for a bare Enter

    ui = UI(reader=reader, confirm_reader=reader)
    assert ui.ask("Pick", default="2", require_terminal=True) == "2"
    assert ui.confirm("Go?", default=False, require_terminal=True) is False
    assert seen == [{"default": "2"}, {"default": False}]


def test_without_a_reader_rich_is_looked_up_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    """A UI built before a test patches rich still reaches the patch."""
    ui = UI()
    monkeypatch.setattr("rich.prompt.Prompt.ask", lambda prompt, **k: "typed")
    monkeypatch.setattr("rich.prompt.Confirm.ask", lambda prompt, **k: True)
    assert ui.ask("Q") == "typed"
    assert ui.confirm("Go?") is True


def test_say_reaches_stdout_and_say_error_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    UI().say("[green]shown[/green]")
    UI().say_error("[red]went wrong[/red]")
    captured = capsys.readouterr()
    assert captured.out == "shown\n"
    assert captured.err == "went wrong\n"


def test_say_forwards_markup_and_soft_wrap(capsys: pytest.CaptureFixture[str]) -> None:
    long = " ".join(["word"] * 60)  # wider than any console rich assumes off a terminal
    UI().say("[bold]literal[/bold]", markup=False)
    UI().say(long, soft_wrap=True)
    out = capsys.readouterr().out
    assert out == f"[bold]literal[/bold]\n{long}\n"


def test_fake_ui_raises_unanswered_question_when_nothing_was_scripted() -> None:
    with pytest.raises(UnansweredQuestion):
        FakeUI().ask("Pick a model")
    with pytest.raises(UnansweredQuestion):
        FakeUI().confirm("Go?", default=False)


def test_unanswered_question_is_never_read_as_no_terminal() -> None:
    """An EOF would be taken as "no terminal, use the default" and pass the test."""
    assert not issubclass(UnansweredQuestion, Exception)
    with pytest.raises(UnansweredQuestion):
        FakeUI().ask("Pick 1 or 2", default="1")


def test_fake_ui_answers_in_order_and_records_what_was_asked() -> None:
    fake = FakeUI(answers=["2", " qwen3:8b "], confirms=[True])
    assert fake.ask("Pick 1 or 2", default="1", require_terminal=True) == "2"
    assert fake.ask("Pick a model") == " qwen3:8b "
    assert fake.confirm("Go?", default=False) is True
    assert fake.asked == [("Pick 1 or 2", {"default": "1"}), ("Pick a model", {})]
    assert fake.confirmed == [("Go?", {"default": False})]
    with pytest.raises(UnansweredQuestion):
        fake.ask("One more?")


def test_fake_ui_without_a_terminal_follows_the_real_policy(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeUI(no_terminal=True)
    assert fake.ask("Pick 1 or 2", default="1") == "1"
    assert "No terminal to ask — using 1." in capsys.readouterr().out
    with pytest.raises(NoTerminal):
        fake.confirm("Go?", default=False, require_terminal=True)


def test_fake_ui_hands_questions_to_a_callers_reader() -> None:
    fake = FakeUI(reader=lambda prompt, **k: prompt.upper(), confirm_reader=lambda q, **k: False)
    assert fake.ask("abc") == "ABC"
    assert fake.confirm("Go?") is False
    assert fake.asked == [("abc", {})]
    assert fake.confirmed == [("Go?", {})]
