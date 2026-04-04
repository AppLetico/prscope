"""
Planning session state container.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ...memory import ParsedConstraint

if TYPE_CHECKING:
    from .author import ArchitectureDesign, DesignRecord, RepoUnderstanding
    from .critic import ImplementabilityResult, ReviewResult


@dataclass
class PlanningState:
    """
    Single source of truth for evolving planning-session knowledge.
    """

    session_id: str
    requirements: str

    # Repository knowledge
    repo_memory: dict[str, str] = field(default_factory=dict)
    manifesto: str = ""
    constraints: list[ParsedConstraint] = field(default_factory=list)

    # Context enrichment
    skills_context: str = ""
    recall_context: str = ""
    no_recall: bool = False
    # Last seen disk fingerprint for manifesto + skills (see PlanningConfig.instruction_context_refresh)
    instruction_context_fingerprint: str = ""

    # Pipeline artifacts
    repo_understanding: RepoUnderstanding | None = None
    design_record: DesignRecord | None = None
    architecture: ArchitectureDesign | None = None
    plan_markdown: str | None = None
    # SHA-256 hex digest of plan markdown last reviewed in design_review (critic turn).
    last_critic_turn_plan_fingerprint: str = ""
    review: ReviewResult | None = None
    constraint_eval: ReviewResult | ImplementabilityResult | None = None

    # Runtime bookkeeping
    revision_round: int = 0
    accessed_paths: set[str] = field(default_factory=set)
    issue_tracker: Any | None = None
    architecture_change_count: int = 0
    architecture_change_rounds: list[bool] = field(default_factory=list)
    # True only for the apply_critique round: skip critic validation_review; user runs Review when ready.
    defer_validation_after_apply: bool = False

    # Clarification and summaries
    clarification_logs: list[dict[str, Any]] = field(default_factory=list)
    suppressed_critic_clarifications: int = 0
    working_summary: str = ""
    review_score_history: list[float] = field(default_factory=list)
    open_issue_history: list[int] = field(default_factory=list)

    # Telemetry
    session_cost_usd: float = 0.0
    max_prompt_tokens: int = 0
    round_cost_usd: float = 0.0
    author_prompt_tokens: int = 0
    author_completion_tokens: int = 0
    critic_prompt_tokens: int = 0
    critic_completion_tokens: int = 0
