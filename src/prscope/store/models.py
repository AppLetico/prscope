from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class RepoProfile:
    """Stored repo profile."""

    id: int | None
    repo_root: str
    profile_sha: str
    profile_json: str
    created_at: str

    @property
    def profile_data(self) -> dict[str, Any]:
        return json.loads(self.profile_json)


@dataclass
class UpstreamRepo:
    """Stored upstream repo."""

    id: int | None
    full_name: str
    last_synced_at: str | None
    last_seen_updated_at: str | None


@dataclass
class PullRequest:
    """Stored pull request."""

    id: int | None
    repo_id: int
    number: int
    state: str
    title: str
    body: str | None
    author: str | None
    labels_json: str | None
    updated_at: str | None
    merged_at: str | None
    head_sha: str | None
    html_url: str | None

    @property
    def labels(self) -> list[str]:
        if self.labels_json:
            return json.loads(self.labels_json)
        return []


@dataclass
class PRFile:
    """Stored PR file change."""

    id: int | None
    pr_id: int
    path: str
    additions: int
    deletions: int


@dataclass
class Evaluation:
    """Stored evaluation result."""

    id: int | None
    pr_id: int
    local_profile_sha: str
    pr_head_sha: str
    rule_score: float | None
    final_score: float | None
    matched_features_json: str | None
    signals_json: str | None
    llm_json: str | None
    decision: str | None
    created_at: str

    @property
    def matched_features(self) -> list[str]:
        if self.matched_features_json:
            return json.loads(self.matched_features_json)
        return []

    @property
    def signals(self) -> dict[str, Any]:
        if self.signals_json:
            return json.loads(self.signals_json)
        return {}


@dataclass
class Artifact:
    """Stored artifact (plan/spec file, etc.)."""

    id: int | None
    evaluation_id: int
    type: str
    path: str
    created_at: str


@dataclass
class PlanningSession:
    """Stored planning session."""

    id: str
    repo_name: str
    title: str
    requirements: str
    author_model: str | None
    critic_model: str | None
    status: str
    seed_type: str
    seed_ref: str | None
    current_round: int
    no_recall: int
    created_at: str
    updated_at: str
    pending_questions_json: str | None = None
    phase_message: str | None = None
    is_processing: int = 0
    processing_started_at: str | None = None
    last_commands_json: str | None = None
    active_tool_calls_json: str | None = None
    completed_tool_call_groups_json: str | None = None
    current_command_id: str | None = None
    session_total_cost_usd: float | None = None
    max_prompt_tokens: int | None = None
    confidence_trend: float | None = None
    converged_early: int | None = None
    clarifications_log_json: str | None = None
    diagnostics_json: str | None = None
    event_seq: int = 0

    @property
    def clarifications_log(self) -> list[dict[str, Any]]:
        if not self.clarifications_log_json:
            return []
        try:
            parsed = json.loads(self.clarifications_log_json)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        return []


@dataclass
class PlanningCommand:
    """Durable command-log metadata for executor execution."""

    id: str
    session_id: str
    command: str | None
    command_id: str
    status: str
    payload_json: str
    result_snapshot_json: str | None
    started_at: str | None
    completed_at: str | None
    last_error: str | None
    attempt_count: int
    worker_id: str | None
    lease_expires_at: str | None
    created_at: str
    updated_at: str

    @property
    def payload(self) -> dict[str, Any]:
        try:
            parsed = json.loads(self.payload_json)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}


@dataclass
class PlanningTurn:
    """Stored planning conversation turn."""

    id: int | None
    session_id: str
    role: str
    content: str
    round: int
    major_issues_remaining: int | None = None
    minor_issues_remaining: int | None = None
    hard_constraint_violations: list[str] | None = None
    parse_error: str | None = None
    created_at: str = ""
    sequence: int | None = None


@dataclass
class PlanVersion:
    """Stored plan snapshot."""

    id: int | None
    session_id: str
    round: int
    plan_content: str
    plan_json: str | None
    decision_graph_json: str | None
    followups_json: str | None
    plan_sha: str
    created_at: str
    changed_sections: str | None = None
    diff_from_previous: str | None = None
    convergence_score: float | None = None


@dataclass
class PlanningRoundMetrics:
    id: int | None
    session_id: str
    round: int
    timestamp: str
    author_prompt_tokens: int | None = None
    author_completion_tokens: int | None = None
    critic_prompt_tokens: int | None = None
    critic_completion_tokens: int | None = None
    max_prompt_tokens: int | None = None
    major_issues: int | None = None
    minor_issues: int | None = None
    critic_confidence: float | None = None
    vagueness_score: float | None = None
    citation_count: int | None = None
    constraint_violations_json: str | None = None
    resolved_since_last_round_json: str | None = None
    clarifications_this_round: int | None = None
    call_cost_usd: float | None = None
    issues_resolved: int | None = None
    issues_introduced: int | None = None
    net_improvement: int | None = None
    time_to_first_tool_call: int | None = None
    grounding_ratio: float | None = None
    static_injection_tokens_pct: float | None = None
    rejected_for_no_discovery: int | None = None
    rejected_for_grounding: int | None = None
    rejected_for_budget: int | None = None
    average_read_depth_per_round: float | None = None
    time_between_tool_calls: float | None = None
    rejection_reasons_json: str | None = None
    plan_quality_score: float | None = None
    unsupported_claims_count: int | None = None
    missing_evidence_count: int | None = None

    @property
    def constraint_violations(self) -> list[str]:
        if not self.constraint_violations_json:
            return []
        try:
            parsed = json.loads(self.constraint_violations_json)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return []
