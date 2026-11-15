"""User-facing errors: every problem ships with a one-line fix."""

from __future__ import annotations

# Where a user reports what no fix line covers.
ISSUES_URL = "https://github.com/MikaelSabuhi/lepika/issues"


class FriendlyError(Exception):
    def __init__(self, problem: str, fix: str) -> None:
        self.problem = problem
        self.fix = fix
        super().__init__(problem)
