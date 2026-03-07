from __future__ import annotations

import re

from .decision_graph import (
    DecisionGraph,
    FollowupSuggestionArtifact,
    PlanFollowupsArtifact,
    graph_to_followup_questions,
)

SECTION_RE = re.compile(r"^##\s+([^\n]+)\n(.*?)(?=^##\s+[^\n]+\n|\Z)", re.MULTILINE | re.DOTALL)


def _normalize_heading(heading: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", heading.strip().lower()).strip("_")


def _extract_sections(plan_content: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    for heading, body in SECTION_RE.findall(plan_content):
        sections[_normalize_heading(heading)] = body.strip()
    return sections


def _mentions_any(text: str, patterns: tuple[str, ...]) -> bool:
    normalized = text.lower()
    return any(pattern in normalized for pattern in patterns)


class FollowupEngine:
    def generate(
        self,
        *,
        current_graph: DecisionGraph,
        plan_content: str,
        plan_version_id: int | None,
    ) -> PlanFollowupsArtifact:
        questions = graph_to_followup_questions(current_graph)
        suggestions: list[FollowupSuggestionArtifact] = []
        if not questions:
            lowered = plan_content.lower()
            sections = _extract_sections(plan_content)
            test_strategy = sections.get("test_strategy", "")
            implementation_steps = sections.get("implementation_steps", "")
            rollback_plan = sections.get("rollback_plan", "")
            architecture = sections.get("architecture", "")

            has_verification_coverage = _mentions_any(
                "\n".join([test_strategy, implementation_steps]),
                ("verify", "verification", "test", "benchmark", "assert", "coverage", "exercise"),
            )
            has_rollout_coverage = _mentions_any(
                "\n".join([rollback_plan, implementation_steps]),
                ("rollback", "roll back", "rollout", "deploy", "release", "revert"),
            )
            has_monitoring_coverage = _mentions_any(
                "\n".join([architecture, implementation_steps, rollback_plan]),
                ("monitor", "monitoring", "observability", "metric", "metrics", "alert", "logging"),
            )

            if ("api" in lowered or "endpoint" in lowered) and not has_verification_coverage:
                suggestions.append(
                    FollowupSuggestionArtifact(
                        id="add_api_contract_tests",
                        suggestion="Add how we'll verify the new behavior works.",
                    )
                )
            if ("migration" in lowered or "schema" in lowered) and not has_rollout_coverage:
                suggestions.append(
                    FollowupSuggestionArtifact(
                        id="generate_migration_steps",
                        suggestion="Add rollout steps and how to roll back if needed.",
                    )
                )
            if re.search(r"\b(log|metric|observab|monitor)\b", lowered) and not has_monitoring_coverage:
                suggestions.append(
                    FollowupSuggestionArtifact(
                        id="expand_observability",
                        suggestion="Add how we'll monitor this in production.",
                    )
                )
        return PlanFollowupsArtifact(
            plan_version_id=plan_version_id,
            questions=questions,
            suggestions=suggestions[:3],
        )
