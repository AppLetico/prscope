"""
Stage implementations for adversarial planning rounds.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from typing import Any, Literal

from ...core import ConvergenceResult
from ..author import PlanDocument, RepairPlan, RepoUnderstanding, RevisionResult, apply_section_updates, render_markdown
from ..authoring.discovery import is_localized_frontend_request
from ..critic import ImplementabilityResult, ReviewResult
from ..followups import decision_graph_from_json, decision_graph_from_plan, merge_decision_graphs
from ..reasoning import (
    ConvergenceReasoner,
    ConvergenceSignals,
    ReasoningContext,
    ReviewReasoner,
)
from ..review import ManifestoCheckResult, build_impact_view, infer_issue_type
from ..tools import extract_file_references
from .round_context import PlanningRoundContext


def review_issue_severity(issue_kind: str) -> str:
    normalized = str(issue_kind).strip().lower()
    if normalized in {"architectural_concern", "validation_architectural_concern", "implementability_detail"}:
        return "minor"
    if normalized in {"recommended_change", "reviewer_question"}:
        return "info"
    return "major"


def review_issue_type(issue_text: str, *, issue_kind: str | None = None, decision_relation: str | None = None) -> str:
    return infer_issue_type(
        issue_text,
        source_kind=issue_kind,
        decision_relation=decision_relation,
    )


class PlanningStages:
    def __init__(
        self,
        *,
        emit_event: Callable[[Any | None, dict[str, Any], str], Any],
        attach_plan_artifacts: Callable[..., Any],
        repo_memory: Callable[[Any], dict[str, str]],
        critic: Any,
        author: Any,
        design_record_payload: Callable[[Any], dict[str, Any] | None],
        design_record_from_payload: Callable[[dict[str, Any] | None], Any],
        review_chat_summary: Callable[[ReviewResult], str],
        manifesto_checker: Any,
        causality_extractor: Any | None = None,
    ) -> None:
        self._emit_event = emit_event
        self._attach_plan_artifacts = attach_plan_artifacts
        self._repo_memory = repo_memory
        self._critic = critic
        self._author = author
        self._design_record_payload = design_record_payload
        self._design_record_from_payload = design_record_from_payload
        self._review_chat_summary = review_chat_summary
        self._manifesto_checker = manifesto_checker
        self._causality_extractor = causality_extractor
        self._review_reasoner = ReviewReasoner()
        self._convergence_reasoner = ConvergenceReasoner()

    @staticmethod
    def _critic_fallback_model(ctx: PlanningRoundContext) -> str | None:
        if ctx.model_policy is None:
            return None
        return ctx.model_policy.critic_review.first_fallback_model

    @staticmethod
    def _author_fallback_model(ctx: PlanningRoundContext) -> str | None:
        if ctx.model_policy is None:
            return None
        return ctx.model_policy.author_refine.first_fallback_model

    @staticmethod
    def _load_graph_payload(raw: str | None) -> dict[str, Any] | None:
        if not raw:
            return None
        try:
            payload = json.loads(str(raw))
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _explicit_payload_change_requested(requirements: str) -> bool:
        lowered = str(requirements or "").lower()
        mentions_payload_terms = any(
            token in lowered for token in ("payload", "response", "serialization", "serializer", "shape", "contract")
        )
        if not mentions_payload_terms:
            return False
        conditional_only_phrases = (
            "only if the response shape must change",
            "if the response shape must change",
            "only if response shape must change",
            "only if the payload must change",
            "if the payload must change",
            "only if payload must change",
        )
        return not any(phrase in lowered for phrase in conditional_only_phrases)

    @staticmethod
    def _source_of_truth_hint(requirements: str) -> str | None:
        match = re.search(
            r"(?:keep|preserve)\s+(.+?)\s+as the source of truth",
            str(requirements),
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return None
        subject = " ".join(match.group(1).split()).strip(" .,:;")
        if not subject:
            return None
        return f"State explicitly that {subject} remains the source of truth; do not weaken or omit that decision."

    @classmethod
    def _pressure_revision_hints(
        cls,
        *,
        requirements: str,
        reconsideration_candidates: list[dict[str, Any]],
    ) -> list[str]:
        hints: list[str] = []
        top_candidate = reconsideration_candidates[0] if reconsideration_candidates else {}
        if isinstance(top_candidate, dict) and str(top_candidate.get("decision_id", "")).strip():
            cluster = top_candidate.get("dominant_cluster")
            cluster_payload = cluster if isinstance(cluster, dict) else {}
            decision_id = str(top_candidate.get("decision_id", "")).strip()
            suggested_action = str(top_candidate.get("suggested_action", "")).strip() or "clarify pressured decision"
            root_issue = str(cluster_payload.get("root_issue", "")).strip() or "unspecified root issue"
            hints.append(
                f"For pressured decision `{decision_id}`, make a visible Architecture change that preserves, narrows, or replaces it with rationale. Root issue: {root_issue}. Suggested action: {suggested_action}."
            )
        source_of_truth_hint = cls._source_of_truth_hint(requirements)
        if source_of_truth_hint:
            hints.append(source_of_truth_hint)
        normalized = " ".join(str(requirements).lower().split())
        if "cache invalidation" in normalized or "invalidation" in normalized:
            hints.append(
                "Name only the concrete invalidation triggers already requested; do not add broad catch-all triggers like manual admin actions, expiration, or 'all related data changes' unless the requirements explicitly ask for them."
            )
        if "avoid broad architecture churn" in normalized or "avoid adding new subsystems" in normalized:
            hints.append(
                "Keep the design localized; do not introduce wrapper layers, new subsystems, broad cache-management abstractions, or speculative concurrency/synchronization language unless explicitly required."
            )
        return hints[:4]

    def _top_reconsideration_candidates(
        self,
        *,
        ctx: PlanningRoundContext,
        decision_graph_json: str | None,
    ) -> list[dict[str, Any]]:
        decision_graph = self._load_graph_payload(decision_graph_json)
        if decision_graph is None or not hasattr(ctx.issue_tracker, "graph_snapshot"):
            return []
        try:
            issue_graph = ctx.issue_tracker.graph_snapshot()
        except Exception:  # noqa: BLE001
            return []
        if not isinstance(issue_graph, dict):
            return []
        try:
            versions = ctx.core.store.get_plan_versions(ctx.session_id, limit=2)
        except Exception:  # noqa: BLE001
            versions = []
        previous_graph = None
        if len(versions) > 1:
            previous_graph = self._load_graph_payload(getattr(versions[1], "decision_graph_json", None))
        impact_view = build_impact_view(
            decision_graph=decision_graph,
            issue_graph=issue_graph,
            previous_decision_graph=previous_graph,
        )
        raw_candidates = impact_view.get("reconsideration_candidates", []) if isinstance(impact_view, dict) else []
        raw_decisions = impact_view.get("decisions", []) if isinstance(impact_view, dict) else []
        candidates: list[dict[str, Any]] = []
        for candidate in raw_candidates:
            if not isinstance(candidate, dict):
                continue
            if candidate.get("eligible") is False:
                continue
            dominant_cluster = candidate.get("dominant_cluster")
            cluster_payload = dominant_cluster if isinstance(dominant_cluster, dict) else {}
            candidates.append(
                {
                    "decision_id": str(candidate.get("decision_id", "")).strip(),
                    "reason": str(candidate.get("reason", "")).strip(),
                    "decision_pressure": int(candidate.get("decision_pressure", 0) or 0),
                    "suggested_action": str(candidate.get("suggested_action", "")).strip(),
                    "recently_changed": bool(candidate.get("recently_changed", False)),
                    "dominant_cluster": {
                        "root_issue_id": str(cluster_payload.get("root_issue_id", "")).strip(),
                        "root_issue": str(cluster_payload.get("root_issue", "")).strip(),
                        "severity": str(cluster_payload.get("severity", "")).strip(),
                        "affected_plan_sections": list(cluster_payload.get("affected_plan_sections", []) or []),
                        "suggested_action": str(cluster_payload.get("suggested_action", "")).strip(),
                    },
                }
            )
        filtered_candidates = [candidate for candidate in candidates if candidate.get("decision_id")][:2]
        if filtered_candidates:
            return filtered_candidates

        fallback_candidates: list[dict[str, Any]] = []
        for decision in raw_decisions:
            if not isinstance(decision, dict):
                continue
            dominant_cluster = decision.get("dominant_cluster")
            cluster_payload = dominant_cluster if isinstance(dominant_cluster, dict) else {}
            decision_id = str(decision.get("decision_id", "")).strip()
            decision_pressure = int(decision.get("decision_pressure", 0) or 0)
            risk_level = str(decision.get("risk_level", "")).strip().lower()
            highest_severity = str(decision.get("highest_severity", "")).strip().lower()
            if not decision_id:
                continue
            if decision_pressure < 4 and risk_level not in {"medium", "high"} and highest_severity != "major":
                continue
            fallback_candidates.append(
                {
                    "decision_id": decision_id,
                    "reason": "pressure_guidance",
                    "decision_pressure": decision_pressure,
                    "suggested_action": str(cluster_payload.get("suggested_action", "")).strip()
                    or "clarify pressured decision",
                    "recently_changed": bool(decision.get("recently_changed", False)),
                    "dominant_cluster": {
                        "root_issue_id": str(cluster_payload.get("root_issue_id", "")).strip(),
                        "root_issue": str(cluster_payload.get("root_issue", "")).strip(),
                        "severity": str(cluster_payload.get("severity", "")).strip(),
                        "affected_plan_sections": list(cluster_payload.get("affected_plan_sections", []) or []),
                        "suggested_action": str(cluster_payload.get("suggested_action", "")).strip(),
                    },
                }
            )
        fallback_candidates.sort(
            key=lambda candidate: (
                -int(candidate.get("decision_pressure", 0) or 0),
                candidate.get("decision_id", ""),
            )
        )
        return fallback_candidates[:2]

    async def _link_issue_to_decisions(
        self,
        *,
        ctx: PlanningRoundContext,
        issue_id: str,
        issue_text: str,
        decision_graph_json: str | None,
        issue_kind: str | None = None,
    ) -> None:
        review_decision = await self._review_reasoner.link_issue(
            issue_text=issue_text,
            decision_graph_json=decision_graph_json,
        )
        ctx.issue_tracker.set_issue_type(
            issue_id,
            review_issue_type(
                issue_text,
                issue_kind=issue_kind,
                decision_relation=review_decision.decision_relation,
            ),
        )
        if not review_decision.issue_links:
            return
        ctx.issue_tracker.link_issue_to_decisions(
            issue_id,
            review_decision.issue_links,
            relation=review_decision.decision_relation,
        )

    @staticmethod
    def _open_questions_from_graph(graph: Any) -> str:
        unresolved = [
            node.description.strip() for node in graph.unresolved_nodes() if str(node.description or "").strip()
        ]
        if not unresolved:
            return "- None."
        return "\n".join(f"- {item}" for item in unresolved)

    @staticmethod
    def _revision_repo_understanding(
        *,
        verified_paths: set[str],
        previous_plan_content: str,
    ) -> RepoUnderstanding:
        referenced_paths = verified_paths | set(extract_file_references(previous_plan_content))
        relevant_tests = sorted(
            path for path in referenced_paths if path.startswith("tests/") or ".test." in path or path.endswith("_test.py")
        )
        relevant_modules = sorted(path for path in referenced_paths if path not in relevant_tests)
        entrypoints = sorted(
            path
            for path in relevant_modules
            if any(
                token in path.lower()
                for token in (
                    "/web/api.py",
                    "planningview",
                    "planpanel",
                    "actionbar",
                    "/lib/api.ts",
                )
            )
        )
        return RepoUnderstanding(
            entrypoints=entrypoints or relevant_modules[:4],
            core_modules=relevant_modules[:8],
            relevant_modules=relevant_modules[:12],
            relevant_tests=relevant_tests[:6],
            architecture_summary="",
            risks=[],
            file_contents={},
            from_mental_model=False,
        )

    @staticmethod
    def _append_files_changed_entry(files_changed: str, path: str, rationale: str) -> str:
        normalized = str(files_changed or "").strip()
        if not path:
            return normalized
        if path in normalized:
            return normalized
        entry = f"- `{path}`: {rationale}"
        if not normalized:
            return entry
        return normalized.rstrip() + "\n" + entry

    @staticmethod
    def _prioritized_frontend_test_targets(paths: list[str]) -> list[str]:
        def _score(path: str) -> tuple[int, str]:
            normalized = str(path).strip()
            if "/pages/" in normalized:
                return (0, normalized)
            if "/components/" in normalized:
                return (1, normalized)
            if "/lib/" in normalized:
                return (3, normalized)
            return (2, normalized)

        return [path for _, path in sorted((_score(path) for path in paths), key=lambda item: item[0])]

    @staticmethod
    def _parse_files_changed_entries(files_changed: str) -> list[tuple[str, str]]:
        entries: list[tuple[str, str]] = []
        seen: set[str] = set()
        for raw_line in str(files_changed or "").splitlines():
            line = str(raw_line or "").strip()
            if not line:
                continue
            match = re.search(r"`([^`]+)`", line)
            if not match:
                continue
            path = match.group(1).strip()
            if not path or ("/" not in path and "." not in path) or path in seen:
                continue
            seen.add(path)
            remainder = line[match.end() :].strip()
            remainder = re.sub(r"^[\s:.-]+", "", remainder).strip()
            remainder = re.sub(r"^\*+\s*", "", remainder).strip()
            remainder = re.sub(r"\s*\*+$", "", remainder).strip()
            entries.append((path, remainder))
        return entries

    @classmethod
    def _normalize_files_changed_entries(
        cls,
        *,
        files_changed: str,
        preferred_owner_paths: tuple[str, ...] = (),
        preferred_test_targets: tuple[str, ...] = (),
        rationale_overrides: dict[str, str] | None = None,
    ) -> str:
        parsed = cls._parse_files_changed_entries(files_changed)
        if not parsed:
            return str(files_changed or "").strip()

        rationale_by_path: dict[str, str] = {}
        original_order: list[str] = []
        for path, rationale in parsed:
            if path not in rationale_by_path:
                original_order.append(path)
            if rationale and not rationale_by_path.get(path):
                rationale_by_path[path] = rationale
            else:
                rationale_by_path.setdefault(path, rationale)

        ordered_paths: list[str] = []

        def _append(path: str) -> None:
            if path in rationale_by_path and path not in ordered_paths:
                ordered_paths.append(path)

        for path in preferred_owner_paths:
            _append(path)
        for path in original_order:
            if re.search(r"\.test\.(?:[jt]sx?)$", path, flags=re.IGNORECASE):
                continue
            _append(path)
        for path in preferred_test_targets:
            _append(path)
        for path in original_order:
            _append(path)

        lines = []
        for path in ordered_paths:
            rationale = str(rationale_by_path.get(path, "") or "").strip()
            if not rationale:
                rationale = str((rationale_overrides or {}).get(path, "") or "").strip()
            lines.append(f"- `{path}`: {rationale}" if rationale else f"- `{path}`")
        return "\n".join(lines).strip()

    @staticmethod
    def _localized_frontend_owner_paths_from_plan(plan: PlanDocument, requirements: str) -> tuple[str, ...]:
        if not is_localized_frontend_request(requirements):
            return ()
        files_changed_refs = [str(path).strip() for path in extract_file_references(str(plan.files_changed or "")) if str(path).strip()]
        preferred: list[str] = []

        def _append_if_present(path: str) -> None:
            if path in files_changed_refs and path not in preferred:
                preferred.append(path)

        lower_requirements = str(requirements or "").lower()
        if "planning" in lower_requirements:
            _append_if_present("src/prscope/web/frontend/src/pages/PlanningView.tsx")
        if "planpanel" in lower_requirements:
            _append_if_present("src/prscope/web/frontend/src/components/PlanPanel.tsx")
        if any(token in lower_requirements for token in ("export", "download", "snapshot")):
            _append_if_present("src/prscope/web/frontend/src/lib/api.ts")

        for path in files_changed_refs:
            lowered = path.lower()
            if "/frontend/" not in lowered or re.search(r"\.test\.(?:[jt]sx?)$", lowered):
                continue
            if path not in preferred:
                preferred.append(path)
            if len(preferred) >= 3:
                break
        return tuple(preferred[:3])

    def _preserve_localized_refinement_detail(
        self,
        *,
        plan: PlanDocument,
        current_plan: PlanDocument,
        requirements: str,
    ) -> PlanDocument:
        if not is_localized_frontend_request(requirements):
            return plan

        updates: dict[str, str] = {}
        files_changed = str(plan.files_changed or "")
        localized_owner_rationales = {
            "src/prscope/web/frontend/src/pages/PlanningView.tsx": (
                "Keep the localized export UI wiring in the existing planning page instead of introducing a new owner."
            ),
            "src/prscope/web/frontend/src/components/PlanPanel.tsx": (
                "Preserve existing PlanPanel export and health behavior while wiring the localized UI change."
            ),
            "src/prscope/web/frontend/src/lib/api.ts": (
                "Preserve the existing frontend API helper wiring for the reused export/snapshot helpers."
            ),
        }
        for path in self._localized_frontend_owner_paths_from_plan(current_plan, requirements):
            files_changed = self._append_files_changed_entry(
                files_changed,
                path,
                localized_owner_rationales.get(
                    path,
                    "Keep the localized UI change anchored to the verified existing frontend owner.",
                ),
            )

        previous_test_targets = [
            path
            for path in {
                *extract_file_references(str(current_plan.files_changed or "")),
                *extract_file_references(str(current_plan.test_strategy or "")),
            }
            if re.search(r"\.test\.(?:[jt]sx?)$", str(path), flags=re.IGNORECASE)
        ]
        for path in self._prioritized_frontend_test_targets(previous_test_targets)[:1]:
            files_changed = self._append_files_changed_entry(
                files_changed,
                path,
                "Keep the localized UI behavior covered by the focused frontend regression test.",
            )
        files_changed = self._normalize_files_changed_entries(
            files_changed=files_changed,
            preferred_owner_paths=self._localized_frontend_owner_paths_from_plan(current_plan, requirements),
            preferred_test_targets=tuple(self._prioritized_frontend_test_targets(previous_test_targets)[:1]),
            rationale_overrides={
                **localized_owner_rationales,
                **{
                    path: "Keep the localized UI behavior covered by the focused frontend regression test."
                    for path in self._prioritized_frontend_test_targets(previous_test_targets)[:1]
                },
            },
        )
        if files_changed != str(plan.files_changed or ""):
            updates["files_changed"] = files_changed

        updated_impl = str(plan.implementation_steps or "").strip()
        if not extract_file_references(updated_impl):
            implementation_steps: list[str] = []
            owner_paths = self._localized_frontend_owner_paths_from_plan(current_plan, requirements)
            if "src/prscope/web/frontend/src/pages/PlanningView.tsx" in owner_paths:
                implementation_steps.append(
                    "1. Update `src/prscope/web/frontend/src/pages/PlanningView.tsx` to keep the localized export UI flow wired through the existing planning page."
                )
            if "src/prscope/web/frontend/src/components/PlanPanel.tsx" in owner_paths:
                implementation_steps.append(
                    "2. Update `src/prscope/web/frontend/src/components/PlanPanel.tsx` to render the requested export status while preserving existing PlanPanel behavior."
                )
            if "src/prscope/web/frontend/src/lib/api.ts" in owner_paths or any(
                token in str(requirements or "").lower() for token in ("export", "download", "snapshot")
            ):
                implementation_steps.append(
                    "3. Reuse `exportSession` and `downloadFile` from `src/prscope/web/frontend/src/lib/api.ts` exactly as already wired."
                )
            if previous_test_targets:
                implementation_steps.append(
                    f"4. Cover the localized export flow in `{previous_test_targets[0]}`."
                )
            if implementation_steps:
                updates["implementation_steps"] = "\n".join(implementation_steps)

        updated_tests = str(plan.test_strategy or "").strip()
        if not extract_file_references(updated_tests) and previous_test_targets:
            updates["test_strategy"] = (
                f"- Assert the export action is disabled while work is in progress in `{previous_test_targets[0]}`.\n"
                f"- Assert the last export result is rendered after success or failure in `{previous_test_targets[0]}`."
            )

        if updates:
            return apply_section_updates(plan, updates)
        return plan

    @staticmethod
    def _mentioned_validation_targets(failures: list[str], prefix: str) -> list[str]:
        targets: list[str] = []
        for failure in failures:
            text = str(failure or "").strip()
            if not text.startswith(prefix):
                continue
            match = re.search(r"mention `([^`]+)`", text)
            if match:
                target = match.group(1).strip()
                if target and target not in targets:
                    targets.append(target)
        return targets

    def _supplement_refinement_plan(
        self,
        *,
        plan: PlanDocument,
        failures: list[str],
        requirements: str,
    ) -> PlanDocument:
        updates: dict[str, str] = {}
        failure_text = "\n".join(str(item or "").strip() for item in failures)
        lower_requirements = str(requirements or "").lower()
        explicit_payload_change_requested = self._explicit_payload_change_requested(requirements)
        helper_phrase = "Reuse the verified helper and API wiring already present in the codebase"
        if "export" in lower_requirements and "download" in lower_requirements:
            helper_phrase = "Reuse `exportSession` and `downloadFile` exactly as already wired"
        elif "snapshot" in lower_requirements:
            helper_phrase = "Reuse the existing snapshot helper wiring exactly as already implemented"
        if "required section is empty: Implementation Steps" in failure_text:
            implementation_steps = [
                "1. Update the existing localized UI/component flow to implement the requested behavior.",
                f"2. {helper_phrase} instead of introducing new abstractions.",
            ]
            if explicit_payload_change_requested:
                implementation_steps.append(
                    "3. Keep any backend contract adjustment scoped to the existing API path only if the response shape must change."
                )
            updates["implementation_steps"] = "\n".join(implementation_steps)
        if "required section is empty: Architecture" in failure_text:
            architecture = "Keep the change localized to the existing component and verified helper wiring."
            if explicit_payload_change_requested:
                architecture += " Only touch the existing API path if the response shape must change."
            updates["architecture"] = architecture
        if "required section is empty: Test Strategy" in failure_text:
            backend_note = (
                "\n- If the response shape changes, assert the localized backend contract through the existing API model regression test."
                if explicit_payload_change_requested
                else ""
            )
            updates["test_strategy"] = (
                "- Assert the requested user-visible behavior in the focused frontend regression test.\n"
                "- Assert the export action is disabled while work is in progress and the last export result is rendered after success."
                + backend_note
            )
        if "required section is empty: Rollback Plan" in failure_text:
            updates["rollback_plan"] = (
                "- If the localized export UI behavior regresses, revert the touched UI/API wiring and restore the previous export flow."
            )
        files_changed = str(plan.files_changed or "")
        for path in self._mentioned_validation_targets(
            failures,
            "localized backend payload/response change must reference the existing API path;",
        ):
            files_changed = self._append_files_changed_entry(
                files_changed,
                path,
                "Keep the small backend payload/response adjustment localized to the existing web API path.",
            )
        for path in self._mentioned_validation_targets(
            failures,
            "localized backend payload/response change must reference the API model regression target;",
        ):
            files_changed = self._append_files_changed_entry(
                files_changed,
                path,
                "Cover the localized backend payload/response adjustment with the existing API model regression test.",
            )
        for path in self._mentioned_validation_targets(
            failures,
            "localized frontend UI change should reference a frontend regression target;",
        ):
            files_changed = self._append_files_changed_entry(
                files_changed,
                path,
                "Keep the localized UI behavior covered by the focused frontend regression test.",
            )
        if files_changed != str(plan.files_changed or ""):
            updates["files_changed"] = files_changed
        supplemented = apply_section_updates(plan, updates) if updates else plan
        return self._preserve_localized_refinement_detail(
            plan=supplemented,
            current_plan=plan,
            requirements=requirements,
        )

    def _confirmed_constraint_violations(
        self,
        *,
        plan_content: str,
        constraints: list[Any],
        candidate_violations: list[str],
    ) -> list[str]:
        if not candidate_violations:
            return []
        manifesto_result: ManifestoCheckResult = self._manifesto_checker.validate(
            plan_content=plan_content,
            constraints=constraints,
        )
        confirmed = set(manifesto_result.violations)
        return [constraint_id for constraint_id in candidate_violations if constraint_id in confirmed]

    async def design_review(
        self,
        *,
        ctx: PlanningRoundContext,
        current_plan_content: str,
        emit_tool: Callable[..., Any],
    ) -> ReviewResult:
        snapshot = ctx.core.transition_and_snapshot(
            "refining",
            phase_message="Running design review",
            current_round=ctx.round_number,
        )
        await self._emit_event(ctx.event_callback, snapshot, ctx.session_id)
        review_mode: Literal["initial", "validation", "stabilization", "implementability"] = "initial"
        architecture_change_count = int(ctx.state.architecture_change_count)
        if ctx.round_number > 1:
            review_mode = "stabilization"
        if ctx.round_number > 0 and architecture_change_count / float(ctx.round_number) > 0.7:
            review_mode = "stabilization"
        await emit_tool("design_review", "running", stage="reviewer")
        review_started = time.perf_counter()
        blocks = self._repo_memory(ctx.state)
        review_payload = await self._critic.run_design_review(
            requirements=ctx.requirements,
            plan_content=current_plan_content,
            manifesto=ctx.state.manifesto,
            architecture=blocks.get("architecture", ""),
            design_record=json.dumps(self._design_record_payload(ctx.state.design_record) or {}, indent=2),
            modules=blocks.get("modules", ""),
            patterns=blocks.get("patterns", ""),
            constraints=ctx.state.constraints,
            prior_critique=ctx.issue_tracker.distilled_context(),
            model_override=ctx.selected_critic_model,
            fallback_model_override=self._critic_fallback_model(ctx),
            session_id=ctx.session_id,
            round_number=ctx.round_number,
            mode=review_mode,
        )
        if not isinstance(review_payload, ReviewResult):
            raise RuntimeError("Expected ReviewResult from design review phase")
        review_result = review_payload
        review_result.constraint_violations = self._confirmed_constraint_violations(
            plan_content=current_plan_content,
            constraints=ctx.state.constraints,
            candidate_violations=review_result.constraint_violations,
        )
        current_plan = ctx.core.get_current_plan()
        current_decision_graph_json = getattr(current_plan, "decision_graph_json", None)
        ctx.state.review = review_result
        await emit_tool(
            "design_review",
            "done",
            stage="reviewer",
            duration_ms=round((time.perf_counter() - review_started) * 1000),
        )
        ctx.core.add_turn(
            "critic",
            self._review_chat_summary(review_result),
            round_number=ctx.round_number,
            parse_error=review_result.parse_error,
        )
        for item in [*review_result.blocking_issues, *review_result.architectural_concerns]:
            if item.strip():
                kind = "blocking_issue" if item in review_result.blocking_issues else "architectural_concern"
                added_issue = ctx.issue_tracker.add_issue(
                    item,
                    ctx.round_number,
                    severity=review_issue_severity(kind),  # type: ignore[arg-type]
                    source="critic",
                    issue_type=review_issue_type(item, issue_kind=kind),  # type: ignore[arg-type]
                )
                if added_issue.id:
                    await self._link_issue_to_decisions(
                        ctx=ctx,
                        issue_id=added_issue.id,
                        issue_text=item,
                        decision_graph_json=current_decision_graph_json,
                        issue_kind=kind,
                    )
        primary_issue_id = ""
        if review_result.primary_issue and review_result.primary_issue.strip():
            primary_issue_id = ctx.issue_tracker.add_issue(
                review_result.primary_issue,
                ctx.round_number,
                source="critic",
                issue_type=review_issue_type(review_result.primary_issue, issue_kind="primary_issue"),
            ).id
            if primary_issue_id:
                await self._link_issue_to_decisions(
                    ctx=ctx,
                    issue_id=primary_issue_id,
                    issue_text=review_result.primary_issue,
                    decision_graph_json=current_decision_graph_json,
                    issue_kind="primary_issue",
                )
            for derived_item in [*review_result.blocking_issues, *review_result.architectural_concerns]:
                if not derived_item.strip():
                    continue
                kind = "blocking_issue" if derived_item in review_result.blocking_issues else "architectural_concern"
                derived = ctx.issue_tracker.add_issue(
                    derived_item,
                    ctx.round_number,
                    severity=review_issue_severity(kind),  # type: ignore[arg-type]
                    source="critic",
                    issue_type=review_issue_type(derived_item, issue_kind=kind),  # type: ignore[arg-type]
                )
                if primary_issue_id and derived.id:
                    ctx.issue_tracker.add_edge(primary_issue_id, derived.id, "causes")
                if derived.id:
                    await self._link_issue_to_decisions(
                        ctx=ctx,
                        issue_id=derived.id,
                        issue_text=derived_item,
                        decision_graph_json=current_decision_graph_json,
                        issue_kind=kind,
                    )
        for constraint_id in review_result.constraint_violations:
            constraint_issue = ctx.issue_tracker.add_issue(
                f"Constraint violation: {constraint_id}",
                ctx.round_number,
                severity=review_issue_severity("constraint_violation"),  # type: ignore[arg-type]
                source="critic",
                issue_type=review_issue_type(
                    f"Constraint violation: {constraint_id}",
                    issue_kind="constraint_violation",
                ),
            )
            if primary_issue_id and constraint_issue.id:
                ctx.issue_tracker.add_edge(primary_issue_id, constraint_issue.id, "depends_on")
        for issue_id in review_result.resolved_issues:
            ctx.issue_tracker.resolve_issue(issue_id, ctx.round_number)
        if self._causality_extractor is not None:
            self._causality_extractor.extract_edges(
                graph=ctx.issue_tracker,
                review=review_result,
                round_number=ctx.round_number,
            )
        return review_result

    def _emit_plan_artifacts(
        self,
        *,
        ctx: PlanningRoundContext,
        version_id: int | None,
        plan_document: PlanDocument,
        plan_content: str,
        previous_graph_json: str | None,
    ) -> None:
        self._attach_plan_artifacts(
            version_id=version_id,
            plan_document=plan_document,
            plan_content=plan_content,
            previous_graph_json=previous_graph_json,
        )

    async def repair_plan(
        self,
        *,
        ctx: PlanningRoundContext,
        current_plan_doc: PlanDocument,
        review_result: ReviewResult,
        emit_tool: Callable[..., Any],
    ) -> RepairPlan:
        snapshot = ctx.core.transition_and_snapshot(
            "refining", phase_message="Planning repair", allow_round_stability=True
        )
        await self._emit_event(ctx.event_callback, snapshot, ctx.session_id)
        await emit_tool("repair_planning", "running", stage="author")
        repair_started = time.perf_counter()
        current_plan = ctx.core.get_current_plan()
        reconsideration_candidates = self._top_reconsideration_candidates(
            ctx=ctx,
            decision_graph_json=getattr(current_plan, "decision_graph_json", None),
        )
        design_record_payload = self._design_record_payload(ctx.state.design_record)
        if design_record_payload:
            try:
                updated_record = await self._author.update_design_record(
                    design_record=design_record_payload,
                    review=review_result,
                    requirements=ctx.requirements,
                    model_override=ctx.selected_author_model,
                    fallback_model_override=self._author_fallback_model(ctx),
                )
                ctx.state.design_record = self._design_record_from_payload(updated_record)
            except Exception as exc:  # noqa: BLE001
                await self._emit_event(
                    ctx.event_callback,
                    {
                        "type": "warning",
                        "message": f"Design record update failed; continuing with existing record: {exc}",
                    },
                    ctx.session_id,
                )
        repair_plan = await self._author.plan_repair(
            review=review_result,
            plan=current_plan_doc,
            requirements=ctx.requirements,
            design_record=self._design_record_payload(ctx.state.design_record),
            reconsideration_candidates=reconsideration_candidates,
            model_override=ctx.selected_author_model,
            fallback_model_override=self._author_fallback_model(ctx),
        )
        must_fix_count = 2
        must_fix_issues: list[str] = []
        if review_result.issue_priority:
            must_fix_issues = [item for item in review_result.issue_priority[:must_fix_count] if item.strip()]
        elif review_result.primary_issue:
            must_fix_issues = [review_result.primary_issue]
        accepted = {item.strip() for item in repair_plan.accepted_issues}
        rejected = {item.strip() for item in repair_plan.rejected_issues}
        missing_must_fix = [item for item in must_fix_issues if item not in accepted and item not in rejected]
        if missing_must_fix:
            repair_plan = await self._author.plan_repair(
                review=review_result,
                plan=current_plan_doc,
                requirements=(
                    f"{ctx.requirements}\n\n"
                    "You MUST address or explicitly reject each must-fix issue with rationale:\n"
                    + "\n".join(f"- {item}" for item in missing_must_fix)
                ),
                design_record=self._design_record_payload(ctx.state.design_record),
                reconsideration_candidates=reconsideration_candidates,
                model_override=ctx.selected_author_model,
                fallback_model_override=self._author_fallback_model(ctx),
            )
            accepted = {item.strip() for item in repair_plan.accepted_issues}
            rejected = {item.strip() for item in repair_plan.rejected_issues}
            unresolved_must_fix = [item for item in must_fix_issues if item not in accepted and item not in rejected]
            for unresolved in unresolved_must_fix:
                ctx.issue_tracker.add_issue(
                    f"Must-fix unresolved without rationale: {unresolved}",
                    ctx.round_number,
                    severity=review_issue_severity("must_fix"),  # type: ignore[arg-type]
                )
                await self._emit_event(
                    ctx.event_callback,
                    {
                        "type": "warning",
                        "message": (
                            f"Must-fix issue remained unresolved without explicit rejection rationale: {unresolved}"
                        ),
                    },
                    ctx.session_id,
                )
        await emit_tool(
            "repair_planning",
            "done",
            stage="author",
            duration_ms=round((time.perf_counter() - repair_started) * 1000),
        )
        ctx.core.add_turn(
            "author",
            (
                "Repair planning complete.\n\n"
                f"Problem understanding: {repair_plan.problem_understanding}\n"
                "Target sections: "
                f"{', '.join(repair_plan.target_sections) if repair_plan.target_sections else '(none)'}\n"
                f"Repair strategy: {repair_plan.repair_strategy}"
            ),
            round_number=ctx.round_number,
        )
        return repair_plan

    async def revise_plan(
        self,
        *,
        ctx: PlanningRoundContext,
        current_plan_doc: PlanDocument,
        repair_plan: RepairPlan,
        review_result: ReviewResult,
        emit_tool: Callable[..., Any],
    ) -> tuple[str, list[str], RevisionResult]:
        snapshot = ctx.core.transition_and_snapshot(
            "refining", phase_message="Revising design", allow_round_stability=True
        )
        await self._emit_event(ctx.event_callback, snapshot, ctx.session_id)
        await emit_tool("apply_critique", "running", stage="author")
        open_issues = ctx.issue_tracker.open_issues()
        revise_started = time.perf_counter()
        previous_plan = ctx.core.get_current_plan()
        previous_plan_content = getattr(previous_plan, "plan_content", "") if previous_plan is not None else ""
        reconsideration_candidates = self._top_reconsideration_candidates(
            ctx=ctx,
            decision_graph_json=getattr(previous_plan, "decision_graph_json", None),
        )
        base_revision_hints = self._pressure_revision_hints(
            requirements=ctx.requirements,
            reconsideration_candidates=reconsideration_candidates,
        )
        revision_hints: list[str] | None = list(base_revision_hints) if base_revision_hints else None
        verified_paths = set(ctx.state.accessed_paths) | set(extract_file_references(previous_plan_content))
        revision_repo_understanding = self._revision_repo_understanding(
            verified_paths=verified_paths,
            previous_plan_content=previous_plan_content,
        )
        current_refinement_validation = self._author.validate_refinement_result(
            plan_content=render_markdown(current_plan_doc),
            repo_understanding=revision_repo_understanding,
            verified_paths_extra=verified_paths,
            requirements_text=ctx.requirements,
        )
        missing_section_failures = [
            failure
            for failure in current_refinement_validation.failure_messages
            if str(failure).startswith("required section is empty:")
        ]
        base_budget = 1 if review_result.primary_issue else min(3, max(1, len(open_issues)))
        revision_budget = max(base_budget, min(5, len(missing_section_failures) or 1))
        if missing_section_failures:
            revision_hints = [*base_revision_hints, *missing_section_failures]
        for attempt in range(2):
            revision_result = await self._author.revise_plan(
                repair_plan=repair_plan,
                current_plan=current_plan_doc,
                requirements=ctx.requirements,
                design_record=self._design_record_payload(ctx.state.design_record),
                revision_budget=revision_budget,
                model_override=ctx.selected_author_model,
                fallback_model_override=self._author_fallback_model(ctx),
                simplest_possible_design=review_result.simplest_possible_design,
                revision_hints=revision_hints,
                reconsideration_candidates=reconsideration_candidates,
            )
            updated_plan = apply_section_updates(current_plan_doc, revision_result.updates)
            updated_plan = self._preserve_localized_refinement_detail(
                plan=updated_plan,
                current_plan=current_plan_doc,
                requirements=ctx.requirements,
            )
            preliminary_markdown = render_markdown(updated_plan)
            grounding_failures = self._author.incremental_grounding_failures(
                previous_plan_content=previous_plan_content,
                updated_plan_content=preliminary_markdown,
                verified_paths=verified_paths,
            )
            revision_validation = self._author.validate_refinement_result(
                plan_content=preliminary_markdown,
                repo_understanding=revision_repo_understanding,
                verified_paths_extra=verified_paths,
                requirements_text=ctx.requirements,
            )
            retryable_failures = [*grounding_failures, *list(revision_validation.failure_messages)]
            if retryable_failures:
                original_updated_plan = updated_plan
                supplemented_plan = self._supplement_refinement_plan(
                    plan=updated_plan,
                    failures=retryable_failures,
                    requirements=ctx.requirements,
                )
                supplemented_markdown = render_markdown(supplemented_plan)
                supplemented_validation = self._author.validate_refinement_result(
                    plan_content=supplemented_markdown,
                    repo_understanding=revision_repo_understanding,
                    verified_paths_extra=verified_paths,
                    requirements_text=ctx.requirements,
                )
                if supplemented_validation.failure_messages:
                    supplemented_plan = self._supplement_refinement_plan(
                        plan=supplemented_plan,
                        failures=[*retryable_failures, *list(supplemented_validation.failure_messages)],
                        requirements=ctx.requirements,
                    )
                    supplemented_markdown = render_markdown(supplemented_plan)
                    supplemented_validation = self._author.validate_refinement_result(
                        plan_content=supplemented_markdown,
                        repo_understanding=revision_repo_understanding,
                        verified_paths_extra=verified_paths,
                        requirements_text=ctx.requirements,
                    )
                if not supplemented_validation.failure_messages:
                    revision_result = RevisionResult(
                        problem_understanding=revision_result.problem_understanding,
                        updates=revision_result.updates
                        | {
                            section: str(getattr(supplemented_plan, section, ""))
                            for section in (
                                "architecture",
                                "files_changed",
                                "implementation_steps",
                                "test_strategy",
                                "rollback_plan",
                            )
                            if str(getattr(original_updated_plan, section, ""))
                            != str(getattr(supplemented_plan, section, ""))
                        },
                        justification=dict(revision_result.justification),
                        review_prediction=revision_result.review_prediction,
                    )
                    updated_plan = supplemented_plan
                    preliminary_markdown = supplemented_markdown
                    retryable_failures = []
            if not retryable_failures:
                break
            if attempt == 1:
                raise ValueError("; ".join(retryable_failures))
            revision_hints = [*base_revision_hints, *retryable_failures]
        previous_version = ctx.core.get_current_plan()
        updated_plan = apply_section_updates(current_plan_doc, revision_result.updates)
        updated_plan = self._preserve_localized_refinement_detail(
            plan=updated_plan,
            current_plan=current_plan_doc,
            requirements=ctx.requirements,
        )
        preliminary_markdown = render_markdown(updated_plan)
        final_validation = self._author.validate_refinement_result(
            plan_content=preliminary_markdown,
            repo_understanding=revision_repo_understanding,
            verified_paths_extra=verified_paths,
            requirements_text=ctx.requirements,
        )
        if final_validation.retryable:
            raise ValueError("; ".join(final_validation.failure_messages))
        previous_graph = decision_graph_from_json(getattr(previous_version, "decision_graph_json", None))
        candidate_graph = decision_graph_from_plan(
            open_questions=updated_plan.open_questions,
            plan_content=preliminary_markdown,
        )
        reconciled_graph = merge_decision_graphs(
            candidate_graph,
            previous_graph,
            carry_forward_unresolved=True,
        )
        canonical_open_questions = self._open_questions_from_graph(reconciled_graph)
        if updated_plan.open_questions != canonical_open_questions:
            updated_plan = apply_section_updates(updated_plan, {"open_questions": canonical_open_questions})
        updated_markdown = render_markdown(updated_plan)
        changed_sections = sorted(
            section
            for section in {*revision_result.updates.keys(), "open_questions"}
            if str(getattr(current_plan_doc, section, "")) != str(getattr(updated_plan, section, ""))
        )
        version = ctx.core.save_plan_version(
            updated_markdown,
            round_number=ctx.round_number,
            plan_document=updated_plan,
            changed_sections=changed_sections,
        )
        self._emit_plan_artifacts(
            ctx=ctx,
            version_id=version.id,
            plan_document=updated_plan,
            plan_content=updated_markdown,
            previous_graph_json=getattr(previous_version, "decision_graph_json", None),
        )
        ctx.state.plan_markdown = updated_markdown
        architecture_changed = any(section in {"architecture", "files_changed"} for section in changed_sections)
        ctx.state.architecture_change_count = int(ctx.state.architecture_change_count) + int(architecture_changed)
        ctx.state.architecture_change_rounds.append(architecture_changed)
        if len(ctx.state.architecture_change_rounds) > 8:
            ctx.state.architecture_change_rounds = ctx.state.architecture_change_rounds[-8:]
        summary_bits: list[str] = []
        if review_result.primary_issue:
            summary_bits.append(f"Primary issue: {review_result.primary_issue}")
        if changed_sections:
            summary_bits.append(f"Updated: {', '.join(changed_sections)}")
        await emit_tool(
            "apply_critique",
            "done",
            stage="author",
            duration_ms=round((time.perf_counter() - revise_started) * 1000),
            query=" | ".join(summary_bits) if summary_bits else None,
        )
        ctx.core.add_turn(
            "author",
            (
                f"Updated sections: {', '.join(changed_sections) if changed_sections else '(none)'}\n\n"
                f"Problem understanding: {revision_result.problem_understanding}\n"
                f"Review prediction: {revision_result.review_prediction}"
            ),
            round_number=ctx.round_number,
        )
        return updated_markdown, changed_sections, revision_result

    async def validation_review(
        self,
        *,
        ctx: PlanningRoundContext,
        updated_markdown: str,
        emit_tool: Callable[..., Any],
    ) -> ReviewResult:
        snapshot = ctx.core.transition_and_snapshot(
            "refining", phase_message="Validating changes", allow_round_stability=True
        )
        await self._emit_event(ctx.event_callback, snapshot, ctx.session_id)
        manifesto_result: ManifestoCheckResult = self._manifesto_checker.validate(
            plan_content=updated_markdown,
            constraints=ctx.state.constraints,
        )
        if manifesto_result.violations:
            await self._emit_event(
                ctx.event_callback,
                {
                    "type": "warning",
                    "message": ("Manifesto check violations: " + "; ".join(manifesto_result.violations[:4])),
                },
                ctx.session_id,
            )
        await emit_tool("review_validation", "running", stage="reviewer")
        validation_started = time.perf_counter()
        blocks = self._repo_memory(ctx.state)
        validation_payload = await self._critic.run_design_review(
            requirements=ctx.requirements,
            plan_content=updated_markdown,
            manifesto=ctx.state.manifesto,
            architecture=blocks.get("architecture", ""),
            design_record=json.dumps(self._design_record_payload(ctx.state.design_record) or {}, indent=2),
            modules=blocks.get("modules", ""),
            patterns=blocks.get("patterns", ""),
            constraints=ctx.state.constraints,
            prior_critique=ctx.issue_tracker.distilled_context(),
            model_override=ctx.selected_critic_model,
            fallback_model_override=self._critic_fallback_model(ctx),
            session_id=ctx.session_id,
            round_number=ctx.round_number,
            mode="validation",
        )
        if not isinstance(validation_payload, ReviewResult):
            raise RuntimeError("Expected ReviewResult from validation phase")
        validation_review = validation_payload
        validation_review.constraint_violations = self._confirmed_constraint_violations(
            plan_content=updated_markdown,
            constraints=ctx.state.constraints,
            candidate_violations=validation_review.constraint_violations,
        )
        await emit_tool(
            "review_validation",
            "done",
            stage="reviewer",
            duration_ms=round((time.perf_counter() - validation_started) * 1000),
        )
        current_plan = ctx.core.get_current_plan()
        current_decision_graph_json = getattr(current_plan, "decision_graph_json", None)
        for issue_id in validation_review.resolved_issues:
            ctx.issue_tracker.resolve_issue(issue_id, ctx.round_number)
        validation_primary_id = ""
        if validation_review.primary_issue and validation_review.primary_issue.strip():
            validation_primary_id = ctx.issue_tracker.add_issue(
                validation_review.primary_issue,
                ctx.round_number,
                severity=review_issue_severity("validation_primary_issue"),  # type: ignore[arg-type]
                source="validation",
                issue_type=review_issue_type(
                    validation_review.primary_issue,
                    issue_kind="validation_primary_issue",
                ),
            ).id
            if validation_primary_id:
                await self._link_issue_to_decisions(
                    ctx=ctx,
                    issue_id=validation_primary_id,
                    issue_text=validation_review.primary_issue,
                    decision_graph_json=current_decision_graph_json,
                    issue_kind="validation_primary_issue",
                )
        for item in [*validation_review.blocking_issues, *validation_review.architectural_concerns]:
            if item.strip():
                kind = (
                    "validation_blocking_issue"
                    if item in validation_review.blocking_issues
                    else "validation_architectural_concern"
                )
                issue = ctx.issue_tracker.add_issue(
                    item,
                    ctx.round_number,
                    severity=review_issue_severity(kind),  # type: ignore[arg-type]
                    source="validation",
                    issue_type=review_issue_type(item, issue_kind=kind),  # type: ignore[arg-type]
                )
                if issue.id:
                    await self._link_issue_to_decisions(
                        ctx=ctx,
                        issue_id=issue.id,
                        issue_text=item,
                        decision_graph_json=current_decision_graph_json,
                        issue_kind=kind,
                    )
                if validation_primary_id and issue.id:
                    ctx.issue_tracker.add_edge(validation_primary_id, issue.id, "causes")
        if self._causality_extractor is not None:
            self._causality_extractor.extract_edges(
                graph=ctx.issue_tracker,
                review=validation_review,
                round_number=ctx.round_number,
            )
        return validation_review

    async def convergence_check(
        self,
        *,
        ctx: PlanningRoundContext,
        updated_markdown: str,
        validation_review: ReviewResult,
        emit_tool: Callable[..., Any],
    ) -> tuple[ImplementabilityResult, ConvergenceResult]:
        implementability = ImplementabilityResult(
            implementable=True,
            missing_details=[],
            implementation_risks=[],
            suggested_additions=[],
            prose="Skipped - prerequisites not met",
        )
        score_history = [*ctx.state.review_score_history, float(validation_review.design_quality_score)]
        pre_impl_decision = await self._convergence_reasoner.decide(
            ReasoningContext(
                signals=ConvergenceSignals(
                    round_number=ctx.round_number,
                    design_quality_score=float(validation_review.design_quality_score),
                    review_complete=bool(validation_review.review_complete),
                    blocking_issue_count=len(validation_review.blocking_issues),
                    architectural_concern_count=len(validation_review.architectural_concerns),
                    has_primary_issue=bool(validation_review.primary_issue),
                    constraint_violation_count=len(validation_review.constraint_violations),
                    root_open_issue_count=len(ctx.issue_tracker.root_open_issues()),
                    unresolved_dependency_chains=int(ctx.issue_tracker.unresolved_dependency_chains()),
                    architecture_change_rounds=list(ctx.state.architecture_change_rounds),
                    review_score_history=list(ctx.state.review_score_history),
                    open_issue_history=list(ctx.state.open_issue_history),
                    implementable=True,
                )
            )
        )
        would_converge = pre_impl_decision.converged
        if would_converge:
            await emit_tool("implementability_check", "running", stage="reviewer")
            impl_started = time.perf_counter()
            blocks = self._repo_memory(ctx.state)
            impl_payload = await self._critic.run_design_review(
                requirements=ctx.requirements,
                plan_content=updated_markdown,
                manifesto=ctx.state.manifesto,
                architecture=blocks.get("architecture", ""),
                design_record=json.dumps(self._design_record_payload(ctx.state.design_record) or {}, indent=2),
                modules=blocks.get("modules", ""),
                patterns=blocks.get("patterns", ""),
                constraints=ctx.state.constraints,
                prior_critique=ctx.issue_tracker.distilled_context(),
                model_override=ctx.selected_critic_model,
                fallback_model_override=self._critic_fallback_model(ctx),
                session_id=ctx.session_id,
                round_number=ctx.round_number,
                mode="implementability",
            )
            if isinstance(impl_payload, ImplementabilityResult):
                implementability = impl_payload
            await emit_tool(
                "implementability_check",
                "done",
                stage="reviewer",
                duration_ms=round((time.perf_counter() - impl_started) * 1000),
            )
            if not implementability.implementable:
                for detail in implementability.missing_details:
                    ctx.issue_tracker.add_issue(
                        detail,
                        ctx.round_number,
                        severity=review_issue_severity("implementability_detail"),  # type: ignore[arg-type]
                    )

        open_issue_count = len(ctx.issue_tracker.open_issues())
        root_open_issue_count = len(ctx.issue_tracker.root_open_issues())
        issue_history = [*ctx.state.open_issue_history, open_issue_count]
        ctx.state.review_score_history = score_history[-8:]
        ctx.state.open_issue_history = issue_history[-8:]
        convergence_decision = await self._convergence_reasoner.decide(
            ReasoningContext(
                signals=ConvergenceSignals(
                    round_number=ctx.round_number,
                    design_quality_score=float(validation_review.design_quality_score),
                    review_complete=bool(validation_review.review_complete),
                    blocking_issue_count=len(validation_review.blocking_issues),
                    architectural_concern_count=len(validation_review.architectural_concerns),
                    has_primary_issue=bool(validation_review.primary_issue),
                    constraint_violation_count=len(validation_review.constraint_violations),
                    root_open_issue_count=root_open_issue_count,
                    unresolved_dependency_chains=int(ctx.issue_tracker.unresolved_dependency_chains()),
                    architecture_change_rounds=list(ctx.state.architecture_change_rounds),
                    review_score_history=score_history[:-1],
                    open_issue_history=issue_history[:-1],
                    implementable=bool(implementability.implementable),
                )
            )
        )
        converged = convergence_decision.converged
        convergence = ConvergenceResult(
            converged=converged,
            reason=convergence_decision.rationale or ("review_complete" if converged else "review_open_issues"),
            change_pct=0.0,
            regression=None,
            major_issues=root_open_issue_count,
        )
        return implementability, convergence
