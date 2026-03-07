from __future__ import annotations

import json
from typing import Any

from .base import Reasoner
from .models import (
    DiscoveryChoiceSignals,
    DiscoveryDecision,
    DiscoveryFollowupSignals,
    ExistingFeatureSignals,
    FrameworkSignals,
    ReasoningContext,
)


class DiscoveryReasoner(Reasoner[DiscoveryDecision]):
    async def decide(self, context: ReasoningContext) -> DiscoveryDecision:
        scenario = str(context.session_metadata.get("scenario", "")).strip()
        if scenario == "question_filter":
            return self._question_filter_decision(context)
        if scenario == "first_turn_existing_feature":
            return self._first_turn_decision(context)
        if scenario == "existing_feature_followup":
            return self._existing_feature_followup(context)
        return DiscoveryDecision(
            mode="continue_discovery",
            routing_source="default",
            confidence=0.25,
            evidence=[],
            decision_source="discovery_reasoner",
        )

    @staticmethod
    def _first_turn_decision(context: ReasoningContext) -> DiscoveryDecision:
        signals = context.signals
        if not isinstance(signals, ExistingFeatureSignals):
            return DiscoveryDecision(
                mode="continue_discovery",
                routing_source="default",
                confidence=0.25,
                evidence=[],
                decision_source="discovery_reasoner",
            )
        if signals.existing_feature and signals.strong_existing_feature:
            evidence = list(signals.evidence_lines[:3])
            if signals.inferred_framework:
                evidence.append(f"inferred_framework:{signals.inferred_framework}")
            return DiscoveryDecision(
                mode="existing_feature_first_turn",
                routing_source="existing_feature_signals",
                confidence=0.9,
                evidence=evidence,
                decision_source="discovery_reasoner",
            )
        return DiscoveryDecision(
            mode="continue_discovery",
            routing_source="existing_feature_signals",
            confidence=0.4,
            evidence=list(signals.evidence_lines[:2]),
            decision_source="discovery_reasoner",
        )

    @staticmethod
    def _question_filter_decision(context: ReasoningContext) -> DiscoveryDecision:
        signals = context.signals
        if not isinstance(signals, FrameworkSignals) or not signals.inferred_framework:
            return DiscoveryDecision(
                mode="continue_discovery",
                routing_source="framework_signals",
                confidence=0.35,
                evidence=[],
                decision_source="discovery_reasoner",
            )
        return DiscoveryDecision(
            mode="continue_discovery",
            routing_source="framework_signals",
            confidence=0.85,
            evidence=list(signals.evidence[:3]),
            question_suppression=["framework_identification"],
            decision_source="discovery_reasoner",
        )

    @staticmethod
    def _existing_feature_followup(context: ReasoningContext) -> DiscoveryDecision:
        signals = context.signals
        if not isinstance(signals, DiscoveryFollowupSignals):
            return DiscoveryDecision(
                mode="continue_discovery",
                routing_source="default",
                confidence=0.25,
                evidence=[],
                decision_source="discovery_reasoner",
            )
        source = "model" if signals.model_choice and signals.model_confidence != "low" else "heuristic"
        choice = signals.model_choice if source == "model" else signals.heuristic_choice
        confidence_map = {"low": 0.35, "medium": 0.65, "high": 0.9}
        confidence = confidence_map.get(signals.model_confidence if source == "model" else "high", 0.5)
        evidence = []
        if choice:
            evidence.append(f"choice:{choice}")
        if signals.rephrased_request:
            evidence.append(f"rephrased_request:{signals.rephrased_request}")
        if signals.proposal_summary:
            evidence.append("proposal_summary_present")

        if signals.awaiting_proposal_review:
            mode_map = {
                "A": "proposal_review_proceed",
                "B": "proposal_review_revise",
                "C": "proposal_review_cancel",
            }
            if choice in mode_map:
                return DiscoveryDecision(
                    mode=mode_map[choice],
                    routing_source=source,
                    confidence=confidence,
                    evidence=evidence,
                    decision_source="discovery_reasoner",
                    complete=choice == "A",
                    rephrased_request=signals.rephrased_request,
                )
            if signals.concrete_enhancement_request:
                return DiscoveryDecision(
                    mode="proposal_review_revision_input",
                    routing_source=source,
                    confidence=max(confidence, 0.6),
                    evidence=evidence,
                    decision_source="discovery_reasoner",
                    rephrased_request=signals.rephrased_request,
                )
            if signals.proposal_summary:
                return DiscoveryDecision(
                    mode="proposal_review_reprompt",
                    routing_source="fallback",
                    confidence=0.3,
                    evidence=evidence,
                    decision_source="discovery_reasoner",
                )
        if signals.awaiting_revision_input:
            if choice == "A":
                return DiscoveryDecision(
                    mode="revision_input_proceed",
                    routing_source=source,
                    confidence=confidence,
                    evidence=evidence,
                    decision_source="discovery_reasoner",
                    complete=True,
                    rephrased_request=signals.rephrased_request,
                )
            if choice == "C":
                return DiscoveryDecision(
                    mode="revision_input_cancel",
                    routing_source=source,
                    confidence=confidence,
                    evidence=evidence,
                    decision_source="discovery_reasoner",
                )
            if signals.concrete_enhancement_request:
                return DiscoveryDecision(
                    mode="revision_input_update_proposal",
                    routing_source=source,
                    confidence=max(confidence, 0.6),
                    evidence=evidence,
                    decision_source="discovery_reasoner",
                    rephrased_request=signals.rephrased_request,
                )
            return DiscoveryDecision(
                mode="revision_input_reprompt",
                routing_source="fallback",
                confidence=0.3,
                evidence=evidence,
                decision_source="discovery_reasoner",
            )
        if signals.enhance_existing and signals.concrete_enhancement_request:
            return DiscoveryDecision(
                mode="enhance_existing_complete",
                routing_source="heuristic",
                confidence=0.85,
                evidence=evidence,
                decision_source="discovery_reasoner",
                complete=True,
                rephrased_request=signals.rephrased_request,
            )
        mode_map = {
            "A": "existing_feature_review_only",
            "B": "existing_feature_enhancement_proposal",
            "C": "existing_feature_cancel",
        }
        if choice in mode_map:
            return DiscoveryDecision(
                mode=mode_map[choice],
                routing_source=source,
                confidence=confidence,
                evidence=evidence,
                decision_source="discovery_reasoner",
            )
        return DiscoveryDecision(
            mode="continue_discovery",
            routing_source="fallback",
            confidence=0.25,
            evidence=evidence,
            decision_source="discovery_reasoner",
        )

    @staticmethod
    def build_choice_prompt(signals: DiscoveryChoiceSignals) -> str:
        option_lines = "\n".join(f"{letter}: {text}" for letter, text in signals.options.items())
        evidence_block = (
            "\n".join(f"- {line}" for line in signals.evidence_lines[:4]) or "- No concrete evidence lines available"
        )
        context_block = (
            f"\n\nAdditional context:\n{signals.extra_context}" if str(signals.extra_context).strip() else ""
        )
        return (
            "Interpret the user's latest reply in context.\n"
            "You are classifying which option the user most likely intends.\n"
            "Treat repository signals as hints, not mandatory routing rules.\n"
            "Return JSON only with fields:\n"
            '- choice: "A" | "B" | "C" | "D" | "unknown"\n'
            '- confidence: "low" | "medium" | "high"\n'
            "- rephrased_request: string (empty if none)\n"
            "- reasoning: string\n\n"
            f"Feature label: {signals.feature_label}\n"
            f"Question: {signals.question_text}\n"
            f"Options:\n{option_lines}\n\n"
            f"Signals: {json.dumps(signals.signal_summary, separators=(',', ':'))}\n"
            f"Evidence:\n{evidence_block}{context_block}\n\n"
            f"Latest user reply:\n{signals.latest_user_message}"
        )

    @staticmethod
    def parse_choice_payload(payload: dict[str, Any] | None) -> dict[str, str] | None:
        if payload is None:
            return None
        choice = str(payload.get("choice", "unknown")).strip().upper()
        confidence = str(payload.get("confidence", "low")).strip().lower()
        if choice not in {"A", "B", "C", "D", "UNKNOWN"}:
            choice = "UNKNOWN"
        if confidence not in {"low", "medium", "high"}:
            confidence = "low"
        return {
            "choice": choice if choice != "UNKNOWN" else "unknown",
            "confidence": confidence,
            "rephrased_request": str(payload.get("rephrased_request", "")).strip(),
            "reasoning": str(payload.get("reasoning", "")).strip(),
        }
