from __future__ import annotations

from .base import Reasoner
from .models import ConvergenceDecision, ConvergenceSignals, ReasoningContext


class ConvergenceReasoner(Reasoner[ConvergenceDecision]):
    async def decide(self, context: ReasoningContext) -> ConvergenceDecision:
        signals = context.signals
        if not isinstance(signals, ConvergenceSignals):
            return ConvergenceDecision(
                should_continue=True,
                converged=False,
                rationale="missing_signals",
                confidence=0.25,
                evidence=[],
                decision_source="convergence_reasoner",
            )
        score_history = [*signals.review_score_history, float(signals.design_quality_score)]
        architecture_recent = signals.architecture_change_rounds[-2:]
        architecture_stable = bool(architecture_recent) and not any(architecture_recent)
        score_window = score_history[-3:] if len(score_history) >= 3 else score_history[-2:]
        score_stable = len(score_window) >= 2 and (max(score_window) - min(score_window)) <= 0.75
        no_major_open_findings = (
            signals.blocking_issue_count == 0
            and signals.architectural_concern_count == 0
            and not signals.has_primary_issue
            and signals.constraint_violation_count == 0
        )
        heuristic_ready = no_major_open_findings and signals.design_quality_score >= 7.8 and signals.round_number >= 1
        base_ready = (
            (signals.review_complete or heuristic_ready)
            and signals.root_open_issue_count == 0
            and signals.unresolved_dependency_chains == 0
            and signals.design_quality_score >= 7.8
        )
        stability_ready = architecture_stable and score_stable
        open_issue_history = [*signals.open_issue_history, signals.root_open_issue_count]
        recent_issue_history = open_issue_history[-3:]
        issue_trend_ready = len(recent_issue_history) >= 3 and all(
            recent_issue_history[idx] <= recent_issue_history[idx - 1] for idx in range(1, len(recent_issue_history))
        )
        legacy_converged = base_ready and stability_ready and issue_trend_ready and signals.implementable
        harness_ok = (
            signals.rubric_floor_ok
            and signals.blockers_ok
            and not signals.rubric_incomplete
            and signals.acceptance_structurally_valid
            and signals.acceptance_satisfied
        )
        converged = legacy_converged and harness_ok
        stability_signals = [
            f"architecture_stable:{architecture_stable}",
            f"score_stable:{score_stable}",
            f"issue_trend_ready:{issue_trend_ready}",
            f"harness_ok:{harness_ok}",
        ]
        rationale = "review_complete" if converged else "review_open_issues"
        if legacy_converged and not harness_ok:
            if not signals.rubric_floor_ok:
                rationale = "rubric_below_floor"
            elif signals.rubric_incomplete:
                rationale = "rubric_incomplete"
            elif not signals.blockers_ok:
                rationale = "blocking_categories"
            elif not signals.acceptance_structurally_valid:
                rationale = "acceptance_criteria_invalid"
            elif not signals.acceptance_satisfied:
                rationale = "acceptance_not_evidence"
            else:
                rationale = "harness_gate_failed"
        elif not stability_ready:
            rationale = "stability_not_met"
        return ConvergenceDecision(
            should_continue=not converged,
            converged=converged,
            rationale=rationale,
            stability_signals=stability_signals,
            confidence=0.85 if converged else 0.65,
            evidence=stability_signals,
            decision_source="convergence_reasoner",
        )
