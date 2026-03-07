from __future__ import annotations

import re

from ..followups import decision_graph_from_json
from .base import Reasoner
from .models import ReasoningContext, ReviewDecision, ReviewSignals


class ReviewReasoner(Reasoner[ReviewDecision]):
    async def decide(self, context: ReasoningContext) -> ReviewDecision:
        signals = context.signals
        if not isinstance(signals, ReviewSignals):
            return ReviewDecision(
                confidence=0.25,
                evidence=[],
                decision_source="review_reasoner",
            )
        decision_links = self._related_decision_ids(signals.issue_text, signals.decision_graph_json)
        relation = self._decision_relation(signals.issue_text)
        evidence = [f"decision_links:{','.join(decision_links)}"] if decision_links else []
        return ReviewDecision(
            issue_links=decision_links,
            decision_relation=relation,
            validated_constraint_violations=list(signals.confirmed_violations),
            confidence=0.8 if decision_links else 0.4,
            evidence=evidence,
            decision_source="review_reasoner",
        )

    @staticmethod
    def _normalized_tokens(text: str) -> set[str]:
        return {token for token in re.split(r"[^a-z0-9]+", text.lower()) if len(token) >= 3}

    def _related_decision_ids(self, issue_text: str, decision_graph_json: str | None) -> list[str]:
        graph = decision_graph_from_json(decision_graph_json)
        if not graph.nodes:
            return []
        issue_tokens = self._normalized_tokens(issue_text)
        issue_lower = issue_text.lower()
        if not issue_tokens:
            return []
        decision_markers = ("choice", "decision", "strategy", "schema", "protocol", "scope")
        matches: list[str] = []
        for node in graph.nodes.values():
            node_tokens = self._normalized_tokens(
                " ".join(
                    [
                        node.id.replace(".", " ").replace("_", " "),
                        node.description,
                        node.section,
                        str(node.concept or "").replace("_", " "),
                        " ".join(node.options or []),
                        str(node.value or ""),
                    ]
                )
            )
            overlap = len(issue_tokens & node_tokens)
            if overlap >= 2 or (overlap >= 1 and any(marker in issue_lower for marker in decision_markers)):
                matches.append(node.id)
        return sorted(set(matches))

    @staticmethod
    def _decision_relation(issue_text: str) -> str:
        lowered = issue_text.lower()
        missing_markers = (
            "missing",
            "underspecified",
            "unspecified",
            "unclear",
            "ambiguous",
            "undecided",
            "unresolved",
            "open question",
            "not specified",
            "needs clarification",
        )
        conflict_markers = (
            "conflict",
            "conflicting",
            "inconsistent",
            "contradict",
            "mismatch",
            "incompatible",
        )
        if any(marker in lowered for marker in missing_markers):
            return "missing"
        if any(marker in lowered for marker in conflict_markers):
            return "conflict"
        return "related"

    async def link_issue(self, *, issue_text: str, decision_graph_json: str | None) -> ReviewDecision:
        return await self.decide(
            ReasoningContext(
                signals=ReviewSignals(
                    issue_text=issue_text,
                    decision_graph_json=decision_graph_json,
                )
            )
        )
