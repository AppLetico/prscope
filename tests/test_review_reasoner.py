from __future__ import annotations

import pytest

from prscope.planning.runtime.followups import decision_graph_to_json
from prscope.planning.runtime.followups.decision_graph import DecisionGraph, DecisionNode
from prscope.planning.runtime.reasoning import ReasoningContext, ReviewReasoner, ReviewSignals


@pytest.mark.asyncio
async def test_review_reasoner_links_issue_to_related_decision() -> None:
    graph = DecisionGraph(
        nodes={
            "architecture.database": DecisionNode(
                id="architecture.database",
                description="Which database should the feature use?",
                options=["postgres", "sqlite"],
                value=None,
                section="architecture",
                concept="database",
            )
        }
    )
    reasoner = ReviewReasoner()

    decision = await reasoner.decide(
        ReasoningContext(
            signals=ReviewSignals(
                issue_text="The database choice is still missing from the plan.",
                decision_graph_json=decision_graph_to_json(graph),
                confirmed_violations=["HARD_CONSTRAINT_001"],
            )
        )
    )

    assert decision.issue_links == ["architecture.database"]
    assert decision.decision_relation == "missing"
    assert decision.validated_constraint_violations == ["HARD_CONSTRAINT_001"]
