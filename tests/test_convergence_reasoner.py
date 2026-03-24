from __future__ import annotations

import pytest

from prscope.planning.runtime.reasoning import (
    ConvergenceReasoner,
    ConvergenceSignals,
    ReasoningContext,
)


@pytest.mark.asyncio
async def test_convergence_reasoner_no_stalled_escape_hatch_with_open_roots() -> None:
    """Stalled-refinement shortcut removed: open root issues must block convergence."""
    reasoner = ConvergenceReasoner()

    decision = await reasoner.decide(
        ReasoningContext(
            signals=ConvergenceSignals(
                round_number=3,
                design_quality_score=7.2,
                review_complete=False,
                blocking_issue_count=0,
                architectural_concern_count=0,
                has_primary_issue=False,
                constraint_violation_count=0,
                root_open_issue_count=1,
                unresolved_dependency_chains=0,
                architecture_change_rounds=[False, False, False],
                review_score_history=[7.1, 7.2],
                open_issue_history=[1, 1],
                implementable=True,
            )
        )
    )

    assert decision.converged is False
    assert decision.rationale == "review_open_issues"
