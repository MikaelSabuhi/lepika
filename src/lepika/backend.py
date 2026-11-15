"""What the two backends report back to the CLI: one step per upgraded component."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# "stale": the newer build is installed but the running process still serves the old
# one, and it is not LePika's to restart. "note": something the user should read that
# is not a version mismatch.
Outcome = Literal["upgraded", "current", "stale", "note"]


@dataclass(frozen=True)
class UpdateStep:
    component: str
    outcome: Outcome
    detail: str = ""
