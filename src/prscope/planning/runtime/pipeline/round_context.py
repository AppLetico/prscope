"""
Shared context for a single adversarial planning round.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...core import PlanningCore
from ..model_policy import ResolvedModelPolicy
from ..state import PlanningState


@dataclass
class PlanningRoundContext:
    core: PlanningCore
    session_id: str
    round_number: int
    requirements: str
    state: PlanningState
    issue_tracker: Any
    selected_author_model: str | None = None
    selected_critic_model: str | None = None
    model_policy: ResolvedModelPolicy | None = None
    event_callback: Any | None = None
    refinement_evidence: dict[str, Any] | None = None
    # True when re-running design review for the same session round (pending critique replaced).
    same_round_repeat: bool = False
    # When True, skip critic validation_review LLM after repair/revise (e.g. PlanPanel issue follow-up).
    skip_validation_review: bool = False
