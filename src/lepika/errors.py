"""User-facing errors: every problem ships with a one-line fix."""

from __future__ import annotations


class FriendlyError(Exception):
    def __init__(self, problem: str, fix: str) -> None:
        self.problem = problem
        self.fix = fix
        super().__init__(problem)
