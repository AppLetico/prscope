"""
Shared context for a single adversarial planning round.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...core import PlanningCore
from ..state import PlanningState


@dataclass
class PlanningRoundContext:
    core: PlanningCore
    session_id: str
    round_number: int
    requirements: str
    state: PlanningState
    issue_tracker: Any
    selected_author_model: str
    selected_critic_model: str
    event_callback: Any | None
