from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SignalEvidence:
    source: str
    detail: str
    path: str | None = None
    line: int | None = None


@dataclass
class ReasoningContext:
    signals: Any
    plan_state: Any | None = None
    decision_graph: Any | None = None
    issue_graph: Any | None = None
    session_metadata: dict[str, Any] = field(default_factory=dict)
    revision_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningDecision:
    confidence: float
    evidence: list[str] = field(default_factory=list)
    decision_source: str = ""
    reasoner_version: str = "v1"


@dataclass
class FrameworkSignals:
    candidates: dict[str, int] = field(default_factory=dict)
    inferred_framework: str | None = None
    evidence: list[str] = field(default_factory=list)


@dataclass
class ExistingFeatureSignals:
    feature_label: str
    evidence_count: int
    runtime_path_count: int
    top_route_score: int
    strong_existing_feature: bool
    matched_paths: list[str] = field(default_factory=list)
    evidence_lines: list[str] = field(default_factory=list)
    inferred_framework: str | None = None
    architecture: str | None = None
    signal_scores: dict[str, int] = field(default_factory=dict)
    existing_feature: bool = False


@dataclass
class DiscoveryFollowupSignals:
    heuristic_choice: str | None = None
    model_choice: str | None = None
    model_confidence: str = "low"
    rephrased_request: str | None = None
    concrete_enhancement_request: bool = False
    awaiting_proposal_review: bool = False
    awaiting_revision_input: bool = False
    enhance_existing: bool = False
    proposal_summary: str | None = None


@dataclass
class DiscoveryChoiceSignals:
    question_text: str
    options: dict[str, str]
    latest_user_message: str
    feature_label: str
    signal_summary: dict[str, Any] = field(default_factory=dict)
    evidence_lines: list[str] = field(default_factory=list)
    extra_context: str = ""


@dataclass
class DiscoveryDecision(ReasoningDecision):
    mode: str = "continue_discovery"
    routing_source: str = ""
    unresolved_decisions: list[str] = field(default_factory=list)
    question_suppression: list[str] = field(default_factory=list)
    rephrased_request: str | None = None
    complete: bool = False


@dataclass
class RefinementMessageSignals:
    user_message: str
    intent: str
    starts_like_question: bool
    has_question_mark: bool
    small_refinement: bool
    ambiguous: bool
    open_question_answer: bool
    open_question_reopen: bool
    issue_reference_candidates: list[str] = field(default_factory=list)
    heuristic_route: str | None = None
    heuristic_confidence: str = "low"
    model_route: str | None = None
    model_confidence: str = "low"


@dataclass
class OpenQuestionResolutionSignals:
    user_message: str
    current_items: list[str]
    proposed_items: list[str]


@dataclass
class IssueReferenceSignals:
    user_message: str
    issues: list[dict[str, str]] = field(default_factory=list)


@dataclass
class OpenQuestionResolutionDecision(ReasoningDecision):
    resolved_action: str = "keep"
    resulting_open_questions: str | None = None


@dataclass
class RefinementDecision(ReasoningDecision):
    route: str = "full_refine"
    question_resolution: OpenQuestionResolutionDecision | None = None
    issue_resolution: list[str] = field(default_factory=list)
    target_sections: list[str] = field(default_factory=list)


@dataclass
class ReviewSignals:
    issue_text: str
    decision_graph_json: str | None = None
    plan_content: str | None = None
    candidate_violations: list[str] = field(default_factory=list)
    confirmed_violations: list[str] = field(default_factory=list)


@dataclass
class ReviewDecision(ReasoningDecision):
    issue_links: list[str] = field(default_factory=list)
    decision_relation: str = "related"
    validated_constraint_violations: list[str] = field(default_factory=list)


@dataclass
class ConvergenceSignals:
    round_number: int
    design_quality_score: float
    review_complete: bool
    blocking_issue_count: int
    architectural_concern_count: int
    has_primary_issue: bool
    constraint_violation_count: int
    root_open_issue_count: int
    unresolved_dependency_chains: int
    architecture_change_rounds: list[bool] = field(default_factory=list)
    review_score_history: list[float] = field(default_factory=list)
    open_issue_history: list[int] = field(default_factory=list)
    implementable: bool = True


@dataclass
class ConvergenceDecision(ReasoningDecision):
    should_continue: bool = True
    converged: bool = False
    rationale: str = ""
    stability_signals: list[str] = field(default_factory=list)
