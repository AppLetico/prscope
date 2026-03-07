from __future__ import annotations

import re
from typing import Any

from .base import Reasoner
from .models import (
    IssueReferenceSignals,
    OpenQuestionResolutionDecision,
    OpenQuestionResolutionSignals,
    ReasoningContext,
    RefinementDecision,
    RefinementMessageSignals,
)


class RefinementReasoner(Reasoner[RefinementDecision]):
    async def decide(self, context: ReasoningContext) -> RefinementDecision:
        scenario = str(context.session_metadata.get("scenario", "")).strip()
        if scenario == "open_question_resolution":
            resolution = self.resolve_open_questions(context)
            return RefinementDecision(
                route="question_resolution",
                confidence=resolution.confidence,
                evidence=list(resolution.evidence),
                decision_source="refinement_reasoner",
                question_resolution=resolution,
            )
        if scenario == "issue_resolution":
            return self.resolve_issue_references(context)
        return self.route_message(context)

    @staticmethod
    def classify_message_intent(user_message: str) -> str:
        text = user_message.strip()
        if not text:
            return "chat"
        normalized = " ".join(text.lower().split())
        starts_like_question = bool(
            re.match(r"^(what|why|how|when|where|who|which|is|are|can|could|would|should|do|does|did)\b", normalized)
        )
        has_question_mark = "?" in normalized
        if starts_like_question and has_question_mark:
            return "chat"
        explicit_refine_patterns = [
            r"\b(update|change|revise|modify|rewrite|adjust|fix|improve)\b.{0,40}\b(plan|draft|section|content)\b",
            r"\b(add|remove|replace|include|exclude)\b.{0,60}\b(plan|draft|section|open questions?|architecture|requirements)\b",
            r"\b(we|you)\s+should\b",
            r"\bshould be\b",
            r"\bmake sure\b",
            r"\bdo not\b",
            r"\bdon't\b",
            r"\bplease\b.{0,40}\b(add|change|update|remove|replace|revise|modify|fix)\b",
            r"\blet'?s\b.{0,40}\b(add|change|update|remove|replace|revise|modify|fix)\b",
            r"^(yes|no)\b",
        ]
        if any(re.search(pattern, normalized) for pattern in explicit_refine_patterns):
            return "refine"
        if re.search(r"^(add|remove|replace|update|change|revise|modify|rewrite|fix|include|exclude)\b", normalized):
            return "refine"
        return "chat"

    @staticmethod
    def is_small_request(user_message: str) -> bool:
        normalized = " ".join(user_message.lower().split())
        if not normalized or len(normalized) > 280:
            return False
        broad_change_signals = [
            "architecture",
            "system design",
            "refactor",
            "migration",
            "database",
            "auth",
            "authentication",
            "authorization",
            "security",
            "new service",
            "new module",
            "across all endpoints",
            "across the codebase",
            "multiple files",
            "end-to-end",
            "rewrite the plan",
            "overhaul",
        ]
        if any(signal in normalized for signal in broad_change_signals):
            return False
        small_change_signals = [
            "you should",
            "we should",
            "should be",
            "log",
            "verify",
            "validation",
            "test strategy",
            "test coverage",
            "monitor",
            "monitoring",
            "observability",
            "rollback",
            "roll back",
            "error handling",
            "guardrail",
            "guardrails",
            "wording",
            "clarify",
            "open question",
            "summary",
            "non-goal",
            "non goal",
            "files changed",
            "section",
            "rename",
            "typo",
            "small change",
        ]
        return any(signal in normalized for signal in small_change_signals)

    @classmethod
    def heuristic_route(cls, user_message: str) -> tuple[str | None, str, bool]:
        normalized = " ".join(user_message.lower().split())
        if not normalized:
            return ("author_chat", "high", False)
        starts_like_question = bool(
            re.match(r"^(what|why|how|when|where|who|which|is|are|can|could|would|should|do|does|did)\b", normalized)
        )
        has_question_mark = "?" in normalized
        if starts_like_question and has_question_mark:
            return ("author_chat", "high", False)
        if cls.classify_message_intent(user_message) == "refine":
            is_small = cls.is_small_request(user_message)
            return ("lightweight_refine" if is_small else "full_refine", "high", is_small)
        ambiguous_refine_signals = [
            r"^(actually|instead|except|also)\b",
            r"\b(need|needs|must|should|shouldn't|should not)\b",
            r"\b(only for|not for|for admin|for admins|for internal)\b",
            r"\b(make this|change this|this should|that should)\b",
            r"\b(async|idempotent|retry|queue|permission|access|security|schema)\b",
        ]
        if any(re.search(pattern, normalized) for pattern in ambiguous_refine_signals):
            return (None, "low", False)
        if has_question_mark:
            return ("author_chat", "medium", False)
        return (None, "low", False)

    @staticmethod
    def looks_like_open_question_answer(user_message: str) -> bool:
        normalized = " ".join(user_message.lower().split())
        if not normalized or "?" in normalized:
            return False
        answer_like_starts = ("yes", "no", "it should", "we should", "should be", "prefer", "i prefer")
        if normalized.startswith(answer_like_starts):
            return True
        return any(token in normalized for token in ["should", "must", "prefer", "we'll", "we will"])

    @staticmethod
    def looks_like_open_question_reopen(user_message: str) -> bool:
        normalized = " ".join(user_message.lower().split())
        if not normalized:
            return False
        reopen_markers = (
            "reopen",
            "open question",
            "leave this open",
            "leave it open",
            "still undecided",
            "not decided",
            "not sure yet",
            "defer this",
            "defer decision",
            "supersede",
            "changed my mind",
        )
        return any(marker in normalized for marker in reopen_markers)

    @classmethod
    def extract_message_signals(
        cls,
        user_message: str,
        *,
        model_route: dict[str, str] | None = None,
    ) -> RefinementMessageSignals:
        normalized = " ".join(user_message.lower().split())
        starts_like_question = bool(
            re.match(r"^(what|why|how|when|where|who|which|is|are|can|could|would|should|do|does|did)\b", normalized)
        )
        has_question_mark = "?" in normalized
        heuristic_route, heuristic_confidence, _ = cls.heuristic_route(user_message)
        return RefinementMessageSignals(
            user_message=user_message,
            intent=cls.classify_message_intent(user_message),
            starts_like_question=starts_like_question,
            has_question_mark=has_question_mark,
            small_refinement=cls.is_small_request(user_message),
            ambiguous=heuristic_route is None,
            open_question_answer=cls.looks_like_open_question_answer(user_message),
            open_question_reopen=cls.looks_like_open_question_reopen(user_message),
            heuristic_route=heuristic_route,
            heuristic_confidence=heuristic_confidence,
            model_route=str((model_route or {}).get("route", "")).strip() or None,
            model_confidence=str((model_route or {}).get("confidence", "low")),
        )

    @staticmethod
    def build_routing_prompt(current_plan_content: str, recent_turns: list[Any], user_message: str) -> str:
        history_lines = "\n".join(
            f"{turn.role}: {turn.content.strip()}" for turn in recent_turns if str(turn.content).strip()
        )
        return (
            "Classify the user's latest message in a planning refinement session.\n"
            "Return JSON only with fields:\n"
            '- route: "author_chat" | "lightweight_refine" | "full_refine"\n'
            '- confidence: "low" | "medium" | "high"\n'
            "- reasoning: string\n\n"
            "Choose `lightweight_refine` only when the request is clearly a small localized plan edit.\n"
            "Choose `full_refine` for ambiguity, scope changes, architecture changes, "
            "or anything that may require deeper reasoning.\n\n"
            f"Current plan excerpt:\n{current_plan_content[:2500]}\n\n"
            f"Recent conversation:\n{history_lines}\n\n"
            f"Latest user message:\n{user_message}"
        )

    @staticmethod
    def parse_routing_payload(payload: dict[str, Any]) -> dict[str, str] | None:
        route = str(payload.get("route", "")).strip()
        confidence = str(payload.get("confidence", "low")).strip().lower()
        if route not in {"author_chat", "lightweight_refine", "full_refine"}:
            return None
        if confidence not in {"low", "medium", "high"}:
            confidence = "low"
        return {
            "route": route,
            "confidence": confidence,
            "reasoning": str(payload.get("reasoning", "")).strip(),
        }

    @staticmethod
    def issue_match_tokens(text: str) -> set[str]:
        stopwords = {
            "please",
            "update",
            "plan",
            "address",
            "these",
            "review",
            "notes",
            "note",
            "start",
            "here",
            "with",
            "without",
            "should",
            "would",
            "could",
            "need",
            "needed",
            "include",
            "adjust",
            "approach",
            "task",
            "tasks",
            "success",
            "where",
            "when",
            "into",
            "from",
        }
        tokens: set[str] = set()
        for raw in re.findall(r"[a-z0-9_]+", text.lower()):
            token = raw
            if token.startswith("issue_"):
                tokens.add(token)
                continue
            if token.endswith("ies") and len(token) > 4:
                token = f"{token[:-3]}y"
            elif token.endswith("es") and len(token) > 4:
                token = token[:-2]
            elif token.endswith("s") and len(token) > 4:
                token = token[:-1]
            if len(token) < 4 or token in stopwords:
                continue
            tokens.add(token)
        return tokens

    @staticmethod
    def route_message(context: ReasoningContext) -> RefinementDecision:
        signals = context.signals
        if not isinstance(signals, RefinementMessageSignals):
            return RefinementDecision(
                route="full_refine",
                confidence=0.25,
                evidence=[],
                decision_source="refinement_reasoner",
            )
        if signals.model_route and signals.model_confidence != "low":
            confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.4}
            return RefinementDecision(
                route=signals.model_route,
                confidence=confidence_map.get(signals.model_confidence, 0.4),
                evidence=[f"model_route:{signals.model_route}", *signals.issue_reference_candidates[:2]],
                decision_source="refinement_reasoner",
            )
        if signals.heuristic_route:
            confidence_map = {"high": 0.85, "medium": 0.6, "low": 0.35}
            return RefinementDecision(
                route=signals.heuristic_route,
                confidence=confidence_map.get(signals.heuristic_confidence, 0.35),
                evidence=[
                    f"intent:{signals.intent}",
                    f"small_refinement:{signals.small_refinement}",
                    f"ambiguous:{signals.ambiguous}",
                ],
                decision_source="refinement_reasoner",
            )
        return RefinementDecision(
            route="full_refine",
            confidence=0.3,
            evidence=[f"ambiguous:{signals.ambiguous}", f"intent:{signals.intent}"],
            decision_source="refinement_reasoner",
        )

    @staticmethod
    def resolve_open_questions(context: ReasoningContext) -> OpenQuestionResolutionDecision:
        signals = context.signals
        if not isinstance(signals, OpenQuestionResolutionSignals):
            return OpenQuestionResolutionDecision(
                resolved_action="keep",
                resulting_open_questions=None,
                confidence=0.25,
                evidence=[],
                decision_source="refinement_reasoner",
            )
        current_items = signals.current_items
        proposed_items = signals.proposed_items
        proposed_is_none = (
            not proposed_items
            or (len(proposed_items) == 1 and proposed_items[0].strip().lower() in {"none", "- none.", "none."})
        )
        if not current_items:
            return OpenQuestionResolutionDecision(
                resolved_action="keep",
                resulting_open_questions=None,
                confidence=0.6,
                evidence=["no_current_or_proposed_items"],
                decision_source="refinement_reasoner",
            )
        removed_count = len(current_items) if proposed_is_none else max(0, len(current_items) - len(proposed_items))
        if removed_count <= 1:
            return OpenQuestionResolutionDecision(
                resolved_action="keep",
                resulting_open_questions=None,
                confidence=0.7,
                evidence=[f"removed_count:{removed_count}"],
                decision_source="refinement_reasoner",
            )
        normalized = " ".join(signals.user_message.lower().split())
        answer_markers = ("yes", "use ", "keep ", "choose ", "should ", "make ")
        if any(marker in normalized for marker in answer_markers):
            remaining = current_items[1:]
            resulting = "- None." if not remaining else "\n".join(remaining)
            return OpenQuestionResolutionDecision(
                resolved_action="preserve_unanswered",
                resulting_open_questions=resulting,
                confidence=0.75,
                evidence=[f"removed_count:{removed_count}", "answer_like_message"],
                decision_source="refinement_reasoner",
            )
        return OpenQuestionResolutionDecision(
            resolved_action="keep",
            resulting_open_questions=None,
            confidence=0.45,
            evidence=[f"removed_count:{removed_count}", "non_answer_like_message"],
            decision_source="refinement_reasoner",
        )

    @classmethod
    def resolve_issue_references(cls, context: ReasoningContext) -> RefinementDecision:
        signals = context.signals
        if not isinstance(signals, IssueReferenceSignals):
            return RefinementDecision(
                route="issue_resolution",
                confidence=0.25,
                evidence=[],
                decision_source="refinement_reasoner",
            )
        issues = signals.issues
        if not issues:
            return RefinementDecision(
                route="issue_resolution",
                confidence=0.35,
                evidence=["no_open_issues"],
                decision_source="refinement_reasoner",
            )
        normalized_message = " ".join(signals.user_message.lower().split())
        explicit_matches = [issue["id"] for issue in issues if issue.get("id") and issue["id"].lower() in normalized_message]
        if len(explicit_matches) == 1:
            return RefinementDecision(
                route="issue_resolution",
                confidence=0.9,
                evidence=[f"explicit_issue_id:{explicit_matches[0]}"],
                decision_source="refinement_reasoner",
                issue_resolution=explicit_matches,
            )
        exact_description_matches = [
            issue["id"]
            for issue in issues
            if issue.get("description") and " ".join(issue["description"].lower().split()) in normalized_message
        ]
        if len(exact_description_matches) == 1:
            return RefinementDecision(
                route="issue_resolution",
                confidence=0.85,
                evidence=[f"exact_issue_description:{exact_description_matches[0]}"],
                decision_source="refinement_reasoner",
                issue_resolution=exact_description_matches,
            )
        message_tokens = cls.issue_match_tokens(signals.user_message)
        if not message_tokens:
            return RefinementDecision(
                route="issue_resolution",
                confidence=0.35,
                evidence=["no_issue_match_tokens"],
                decision_source="refinement_reasoner",
            )
        scored: list[tuple[float, str]] = []
        for issue in issues:
            issue_id = str(issue.get("id", "")).strip()
            issue_tokens = cls.issue_match_tokens(str(issue.get("description", "")))
            if issue_id and issue_tokens:
                overlap = len(issue_tokens & message_tokens) / len(issue_tokens)
                if overlap > 0:
                    scored.append((overlap, issue_id))
        if not scored:
            return RefinementDecision(
                route="issue_resolution",
                confidence=0.35,
                evidence=["no_issue_overlap"],
                decision_source="refinement_reasoner",
            )
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_issue_id = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        if best_score >= 0.45 and (len(scored) == 1 or (best_score - second_score) >= 0.15):
            return RefinementDecision(
                route="issue_resolution",
                confidence=0.75,
                evidence=[f"best_issue_overlap:{best_score:.2f}", f"issue_id:{best_issue_id}"],
                decision_source="refinement_reasoner",
                issue_resolution=[best_issue_id],
            )
        return RefinementDecision(
            route="issue_resolution",
            confidence=0.4,
            evidence=["issue_overlap_ambiguous"],
            decision_source="refinement_reasoner",
        )
