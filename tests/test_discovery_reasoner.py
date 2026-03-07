from __future__ import annotations

import pytest

from prscope.planning.runtime.discovery_support.existing_feature import build_existing_feature_signals
from prscope.planning.runtime.discovery_support.signals import build_framework_signals, build_signal_index
from prscope.planning.runtime.reasoning import (
    DiscoveryChoiceSignals,
    DiscoveryFollowupSignals,
    DiscoveryReasoner,
    ReasoningContext,
)


def test_build_framework_signals_infers_fastapi() -> None:
    index = build_signal_index(
        [
            {"path": "src/api.py", "line": 10, "text": "from fastapi import FastAPI"},
            {"path": "src/api.py", "line": 22, "text": "@app.get('/health')"},
        ]
    )
    signals = build_framework_signals(index)

    assert signals.inferred_framework == "fastapi"
    assert signals.candidates["fastapi"] > 0


def test_build_existing_feature_signals_marks_strong_runtime_evidence() -> None:
    signals = build_existing_feature_signals(
        {
            "feature_label": "health endpoint",
            "existing_feature": True,
            "matched_paths": ["src/api.py"],
            "inferred_framework": "fastapi",
        },
        lambda path: 6 if "api.py" in path else 0,
        ["`src/api.py:22` @app.get('/health')"],
    )

    assert signals.strong_existing_feature is True
    assert signals.runtime_path_count == 1
    assert signals.inferred_framework == "fastapi"


@pytest.mark.asyncio
async def test_discovery_reasoner_prefers_existing_feature_override() -> None:
    reasoner = DiscoveryReasoner()
    signals = build_existing_feature_signals(
        {
            "feature_label": "health endpoint",
            "existing_feature": True,
            "matched_paths": ["src/api.py"],
        },
        lambda path: 6 if "api.py" in path else 0,
        ["`src/api.py:22` @app.get('/health')"],
    )

    decision = await reasoner.decide(
        ReasoningContext(
            signals=signals,
            session_metadata={"scenario": "first_turn_existing_feature"},
        )
    )

    assert decision.mode == "existing_feature_first_turn"
    assert decision.confidence >= 0.8


@pytest.mark.asyncio
async def test_discovery_reasoner_routes_followup_revision_input() -> None:
    reasoner = DiscoveryReasoner()

    decision = await reasoner.decide(
        ReasoningContext(
            signals=DiscoveryFollowupSignals(
                heuristic_choice=None,
                model_choice="unknown",
                model_confidence="low",
                rephrased_request="add auth and rate limiting",
                concrete_enhancement_request=True,
                awaiting_proposal_review=True,
                proposal_summary="Current proposal",
            ),
            session_metadata={"scenario": "existing_feature_followup"},
        )
    )

    assert decision.mode == "proposal_review_revision_input"
    assert decision.rephrased_request == "add auth and rate limiting"


def test_discovery_reasoner_choice_prompt_and_payload_round_trip() -> None:
    prompt = DiscoveryReasoner.build_choice_prompt(
        DiscoveryChoiceSignals(
            question_text="How should we proceed?",
            options={"A": "Proceed", "B": "Revise", "C": "Cancel"},
            latest_user_message="revise it and add security",
            feature_label="health endpoint",
            signal_summary={"strong_existing_feature": True},
            evidence_lines=["`src/api.py:22` @app.get('/health')"],
            extra_context="Current proposal summary",
        )
    )

    parsed = DiscoveryReasoner.parse_choice_payload(
        {
            "choice": "b",
            "confidence": "medium",
            "rephrased_request": "add security",
            "reasoning": "user requested a revision",
        }
    )

    assert "Current proposal summary" in prompt
    assert parsed == {
        "choice": "B",
        "confidence": "medium",
        "rephrased_request": "add security",
        "reasoning": "user requested a revision",
    }
