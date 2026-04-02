from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AuthorResult:
    plan: str
    unverified_references: set[str]
    accessed_paths: set[str]
    design_record: dict[str, Any] | None = None
    grounding_ratio: float | None = None
    rejection_counts: dict[str, int] = field(default_factory=dict)
    rejection_reasons: list[dict[str, str]] = field(default_factory=list)
    average_read_depth: float | None = None
    average_time_between_tool_calls: float | None = None
    draft_diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationResult:
    failure_messages: tuple[str, ...]
    reason_codes: tuple[str, ...]
    retryable: bool
    failure_count: int

    @property
    def ok(self) -> bool:
        return self.failure_count == 0

    @property
    def normalized_signature(self) -> frozenset[str]:
        return frozenset(self.reason_codes)

    @classmethod
    def success(cls) -> ValidationResult:
        return cls(failure_messages=(), reason_codes=(), retryable=False, failure_count=0)


@dataclass(frozen=True)
class EvidenceBundle:
    relevant_files: tuple[str, ...] = ()
    existing_components: tuple[str, ...] = ()
    test_targets: tuple[str, ...] = ()
    related_modules: tuple[str, ...] = ()
    existing_routes_or_helpers: tuple[str, ...] = ()
    evidence_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AttemptContext:
    attempt_number: int
    previous_failures: tuple[str, ...] = ()
    revision_hints: tuple[str, ...] = ()
    elapsed_ms: int = 0


@dataclass
class PlanDocument:
    title: str
    summary: str
    goals: str
    non_goals: str
    files_changed: str
    architecture: str
    implementation_steps: str
    test_strategy: str
    rollback_plan: str
    # Decision-graph / follow-up field; rendered only when non-trivial (see render_markdown).
    open_questions: str


@dataclass
class RepairPlan:
    problem_understanding: str
    accepted_issues: list[str]
    rejected_issues: list[str]
    root_causes: list[str]
    repair_strategy: str
    target_sections: list[str]
    revision_plan: str


@dataclass
class RevisionResult:
    problem_understanding: str
    updates: dict[str, str]
    justification: dict[str, str]
    what_changed: dict[str, str]  # section_id -> one-sentence concrete description of what was added/changed
    review_prediction: str


@dataclass
class RepoCandidates:
    entrypoints: list[str]
    source_modules: list[str]
    tests_and_config: list[str]
    all_paths: list[str]


@dataclass
class RepoUnderstanding:
    entrypoints: list[str]
    core_modules: list[str]
    relevant_modules: list[str]
    relevant_tests: list[str]
    architecture_summary: str
    risks: list[str]
    file_contents: dict[str, str]
    from_mental_model: bool = False


@dataclass
class ArchitectureDesign:
    problem_summary: str
    proposed_components: list[str]
    responsibilities: dict[str, str]
    data_flow: str
    integration_points: list[str]
    alternatives_considered: list[str]
    chosen_design: str
    simplification_opportunities: list[str]


@dataclass
class DesignRecord:
    problem_summary: str
    constraints: list[str]
    architecture: str
    alternatives_considered: list[str]
    tradeoffs: list[str]
    chosen_design: str
    assumptions: list[str]
    potential_failure_modes: list[str]


PLAN_SECTION_ORDER: tuple[str, ...] = (
    "summary",
    "goals",
    "non_goals",
    "files_changed",
    "architecture",
    "implementation_steps",
    "test_strategy",
    "rollback_plan",
)

PLAN_SECTION_TITLES: dict[str, str] = {
    "summary": "Summary",
    "goals": "Goals",
    "non_goals": "Non-Goals",
    "files_changed": "Files Changed",
    "architecture": "Architecture",
    "implementation_steps": "Implementation Steps",
    "test_strategy": "Test Strategy",
    "rollback_plan": "Rollback Plan",
    "open_questions": "Open Questions",
}


def open_questions_visible_in_markdown(open_questions: str) -> bool:
    """Whether open_questions should appear as a ## Open Questions section in the plan panel."""
    s = str(open_questions or "").strip()
    if not s:
        return False
    if re.match(r"^-\s*None\.?\s*$", s, flags=re.IGNORECASE):
        return False
    if s.casefold() in {"none", "n/a", "—", "-"}:
        return False
    return True


def render_markdown(plan: PlanDocument) -> str:
    lines: list[str] = [f"# {plan.title.strip() or 'Plan'}", ""]
    for section_id in PLAN_SECTION_ORDER:
        content = str(getattr(plan, section_id, "") or "").strip()
        if not content:
            continue
        lines.append(f"## {PLAN_SECTION_TITLES[section_id]}")
        lines.append(content)
        lines.append("")
    oq = str(getattr(plan, "open_questions", "") or "").strip()
    if open_questions_visible_in_markdown(oq):
        lines.append(f"## {PLAN_SECTION_TITLES['open_questions']}")
        lines.append(oq)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def apply_section_updates(plan: PlanDocument, updates: dict[str, str]) -> PlanDocument:
    payload = {
        "title": plan.title,
        "summary": plan.summary,
        "goals": plan.goals,
        "non_goals": plan.non_goals,
        "files_changed": plan.files_changed,
        "architecture": plan.architecture,
        "implementation_steps": plan.implementation_steps,
        "test_strategy": plan.test_strategy,
        "rollback_plan": plan.rollback_plan,
        "open_questions": plan.open_questions,
    }
    for key, value in updates.items():
        if key in payload:
            payload[key] = str(value)
    return PlanDocument(**payload)
