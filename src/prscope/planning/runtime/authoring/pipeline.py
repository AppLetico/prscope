from __future__ import annotations

import logging
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

from ..tools import extract_file_references
from .models import AttemptContext, AuthorResult, EvidenceBundle, ValidationResult

logger = logging.getLogger(__name__)

_DEFAULT_DRAFT_LOOP_BUDGET_MS = 7000
_MODERATE_DRAFT_LOOP_BUDGET_MS = 16000
_COMPLEX_DRAFT_LOOP_BUDGET_MS = 28000


class AuthorPlannerPipeline:
    def __init__(
        self,
        *,
        tool_executor: Any,
        scan_repo_candidates: Callable[..., Any],
        explore_repo: Callable[..., Any],
        classify_complexity: Callable[..., Any],
        draft_plan: Callable[..., Awaitable[str]],
        validate_draft: Callable[..., Any],
        self_review_draft: Callable[..., Awaitable[list[str]]] | None = None,
        event_emitter: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        self._tool_executor = tool_executor
        self._scan_repo_candidates = scan_repo_candidates
        self._explore_repo = explore_repo
        self._classify_complexity = classify_complexity
        self._draft_plan = draft_plan
        self._validate_draft = validate_draft
        self._self_review_draft = self_review_draft
        self._emit = event_emitter

    @staticmethod
    def _extract_symbol_names(content: str) -> list[str]:
        patterns = (
            r"^\s*(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\b",
            r"^\s*export\s+(?:async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)\b",
        )
        symbols: list[str] = []
        seen: set[str] = set()
        for pattern in patterns:
            for match in re.findall(pattern, content, flags=re.MULTILINE):
                name = str(match).strip()
                if name and name not in seen:
                    seen.add(name)
                    symbols.append(name)
        return symbols

    @staticmethod
    def _extract_imported_helpers(content: str) -> list[str]:
        helpers: list[str] = []
        seen: set[str] = set()
        for imported_block in re.findall(
            r"import\s*\{([^}]+)\}\s*from\s*[\"'][^\"']+[\"']",
            content,
            flags=re.MULTILINE | re.DOTALL,
        ):
            for raw_name in imported_block.split(","):
                name = raw_name.strip()
                if not name:
                    continue
                if " as " in name:
                    name = name.split(" as ", 1)[0].strip()
                if name and name not in seen:
                    seen.add(name)
                    helpers.append(name)
        for imported_block in re.findall(r"from\s+[^\n]+\s+import\s+([A-Za-z0-9_,\s]+)", content):
            for raw_name in imported_block.split(","):
                name = raw_name.strip()
                if name and name not in seen:
                    seen.add(name)
                    helpers.append(name)
        return helpers

    @staticmethod
    def _extract_routes_or_helpers(content: str) -> list[str]:
        entries: list[str] = []
        seen: set[str] = set()
        for method, route in re.findall(r"@app\.(get|post|put|delete|patch)\([\"']([^\"']+)[\"']\)", content):
            label = f"{method.upper()} {route}"
            if label not in seen:
                seen.add(label)
                entries.append(label)
        for route in re.findall(r"[\"'](/api/[^\"']+)[\"']", content):
            label = f"ROUTE {route}"
            if label not in seen:
                seen.add(label)
                entries.append(label)
        for helper in AuthorPlannerPipeline._extract_imported_helpers(content):
            if any(token in helper.lower() for token in ("get", "list", "fetch", "load", "export", "download", "snapshot", "diagnostic")):
                if helper not in seen:
                    seen.add(helper)
                    entries.append(helper)
        for name in AuthorPlannerPipeline._extract_symbol_names(content):
            if any(token in name.lower() for token in ("get", "list", "fetch", "load", "export", "download", "snapshot", "diagnostic")):
                if name not in seen:
                    seen.add(name)
                    entries.append(name)
        return entries

    def _evidence_source_content(self, path: str, content: str, requirements: str) -> str:
        normalized_path = str(path).strip()
        if not normalized_path:
            return content
        lowered_path = normalized_path.lower()
        lowered_requirements = str(requirements or "").lower()
        should_read_header = lowered_path.endswith(("planningview.tsx", "actionbar.tsx", "planpanel.tsx"))
        should_read_api = lowered_path.endswith("/lib/api.ts") and any(
            token in lowered_requirements for token in ("export", "download", "snapshot", "diagnostic")
        )
        if not should_read_header and not should_read_api:
            return content
        try:
            payload = self._tool_executor.read_file(
                normalized_path,
                max_lines=320 if should_read_api else 120,
            )
        except Exception:  # noqa: BLE001
            return content
        enriched = str(payload.get("content", "") or "").strip()
        return enriched or content

    def _build_evidence_bundle(self, repo_understanding: Any, requirements: str) -> EvidenceBundle:
        lower_requirements = str(requirements or "").lower()

        def _path_priority(path: str) -> tuple[int, str]:
            lowered = str(path).lower()
            score = 0
            if "actionbar" in lower_requirements and "actionbar" in lowered:
                score += 8
            if ("planningview" in lower_requirements or "planning page" in lower_requirements) and "planningview" in lowered:
                score += 7
            if lowered.endswith("/lib/api.ts"):
                score += 6
            if "planpanel" in lower_requirements and "planpanel" in lowered:
                score += 3
            if lowered.endswith("planningview.test.ts"):
                score += 2
            return (-score, lowered)

        relevant_files: list[str] = []
        source_candidates = [
            str(path).strip()
            for path in getattr(repo_understanding, "relevant_modules", [])
            if str(path).strip()
        ]
        test_candidates = [
            str(path).strip()
            for path in getattr(repo_understanding, "relevant_tests", [])
            if str(path).strip()
        ]
        extra_candidates = [
            str(path).strip()
            for path in list(getattr(repo_understanding, "file_contents", {}).keys()) + list(getattr(repo_understanding, "entrypoints", []))
            if str(path).strip()
        ]
        source_candidates = sorted(source_candidates, key=_path_priority)
        test_candidates = sorted(test_candidates, key=_path_priority)
        extra_candidates = sorted(extra_candidates, key=_path_priority)
        for group in (source_candidates, test_candidates, extra_candidates):
            for path in group:
                if path and path not in relevant_files:
                    relevant_files.append(path)
        existing_components: list[str] = []
        existing_routes_or_helpers: list[str] = []
        file_contents = getattr(repo_understanding, "file_contents", {})
        for path in relevant_files[:12]:
            content = self._evidence_source_content(path, str(file_contents.get(path, "") or ""), requirements)
            self_symbols = AuthorPlannerPipeline._extract_symbol_names(content)
            for symbol in self_symbols:
                if symbol not in existing_components:
                    existing_components.append(symbol)
            for helper in AuthorPlannerPipeline._extract_imported_helpers(content):
                if any(token in helper.lower() for token in ("session", "snapshot", "diagnostic", "export", "download")):
                    if helper not in existing_components:
                        existing_components.append(helper)
            for route_or_helper in AuthorPlannerPipeline._extract_routes_or_helpers(content):
                if route_or_helper not in existing_routes_or_helpers:
                    existing_routes_or_helpers.append(route_or_helper)
            if "diagnostic" in lower_requirements and "get_session_diagnostics" in self_symbols:
                if "get_session_diagnostics" not in existing_routes_or_helpers:
                    existing_routes_or_helpers.append("get_session_diagnostics")
            if "snapshot" in lower_requirements and "getSessionSnapshot" in content and "getSessionSnapshot" not in existing_routes_or_helpers:
                existing_routes_or_helpers.append("getSessionSnapshot")
        related_modules: list[str] = []
        for path in source_candidates:
            if path not in relevant_files and path not in related_modules:
                related_modules.append(path)
        evidence_notes: list[str] = []
        architecture_summary = str(getattr(repo_understanding, "architecture_summary", "") or "").strip()
        if architecture_summary:
            evidence_notes.append(architecture_summary)
        if "planpanel" in lower_requirements and any("PlanPanel.tsx" in path for path in relevant_files):
            evidence_notes.append("Existing PlanPanel health presentation should remain intact unless requirements say otherwise.")
        for risk in getattr(repo_understanding, "risks", [])[:3]:
            note = str(risk).strip()
            if note:
                evidence_notes.append(note)
        return EvidenceBundle(
            relevant_files=tuple(relevant_files[:12]),
            existing_components=tuple(existing_components[:12]),
            test_targets=tuple(test_candidates[:6]),
            related_modules=tuple(related_modules[:8]),
            existing_routes_or_helpers=tuple(existing_routes_or_helpers[:12]),
            evidence_notes=tuple(evidence_notes[:6]),
        )

    @staticmethod
    def _coerce_validation_result(raw: Any) -> ValidationResult:
        if isinstance(raw, ValidationResult):
            return raw
        if isinstance(raw, list):
            return ValidationResult(
                failure_messages=tuple(str(item).strip() for item in raw if str(item).strip()),
                reason_codes=tuple(sorted({str(item).strip() for item in raw if str(item).strip()})),
                retryable=bool(raw),
                failure_count=len([item for item in raw if str(item).strip()]),
            )
        return ValidationResult.success()

    @staticmethod
    def _should_replace_best(best_result: ValidationResult | None, current_result: ValidationResult) -> bool:
        if best_result is None:
            return True
        return current_result.failure_count <= best_result.failure_count

    @staticmethod
    def _deterministic_revision_hints(validation_result: ValidationResult) -> tuple[str, ...]:
        hints: list[str] = []
        for failure in validation_result.failure_messages:
            text = str(failure).strip()
            if not text:
                continue
            if text.startswith("missing test target reference; reference one of:"):
                candidates = text.split(":", 1)[1].strip()
                hints.append(f"Reference at least one exact verified test target in the plan: {candidates}.")
            elif text.startswith("missing explicit helper reuse reference for "):
                hints.append(text.replace("missing explicit helper reuse reference for ", "Mention exact verified helper reuse for "))
            elif text.startswith("replace unverified path "):
                hints.append(text)
        return tuple(dict.fromkeys(hints))

    @staticmethod
    def _insert_test_target_into_files_changed(plan_content: str, test_target: str) -> str:
        if not test_target or test_target in plan_content:
            return plan_content
        pattern = re.compile(r"(^##\s+Files Changed\b.*?)(?=^##\s+|\Z)", re.IGNORECASE | re.MULTILINE | re.DOTALL)
        match = pattern.search(plan_content)
        if not match:
            return plan_content
        section = match.group(1)
        bullet = f"- `{test_target}`: Add or extend regression coverage for the requested change.\n"
        if bullet.strip() in section:
            return plan_content
        updated_section = section.rstrip() + "\n" + bullet
        return plan_content[: match.start(1)] + updated_section + plan_content[match.end(1) :]

    @staticmethod
    def _insert_path_into_files_changed(plan_content: str, path: str, rationale: str) -> str:
        if not path or path in plan_content:
            return plan_content
        pattern = re.compile(r"(^##\s+Files Changed\b.*?)(?=^##\s+|\Z)", re.IGNORECASE | re.MULTILINE | re.DOTALL)
        match = pattern.search(plan_content)
        if not match:
            return plan_content
        section = match.group(1)
        bullet = f"- `{path}`: {rationale}\n"
        if bullet.strip() in section:
            return plan_content
        updated_section = section.rstrip() + "\n" + bullet
        return plan_content[: match.start(1)] + updated_section + plan_content[match.end(1) :]

    @staticmethod
    def _strip_localized_scope_drift_lines(plan_content: str) -> str:
        filtered_lines = [
            line
            for line in plan_content.splitlines()
            if not any(token in line.lower() for token in ("observability", "telemetry", "logging", "monitoring"))
        ]
        repaired = "\n".join(filtered_lines).strip()
        return repaired + "\n" if repaired else plan_content

    def _deterministic_plan_repairs(
        self,
        *,
        plan_content: str,
        validation_result: ValidationResult,
        repo_understanding: RepoUnderstanding,
        evidence_bundle: EvidenceBundle,
        min_grounding_ratio: float,
        grounding_paths: set[str],
        requirements: str,
    ) -> tuple[str, ValidationResult]:
        repaired = plan_content
        if "missing_tests" in validation_result.reason_codes and evidence_bundle.test_targets:
            repaired = self._insert_test_target_into_files_changed(repaired, evidence_bundle.test_targets[0])
        if "localized_scope_drift" in validation_result.reason_codes:
            repaired = self._strip_localized_scope_drift_lines(repaired)
        helper_owner_path = "src/prscope/web/frontend/src/lib/api.ts"
        helper_tokens = ("exportSession", "downloadFile", "getSessionSnapshot")
        if (
            helper_owner_path in evidence_bundle.relevant_files
            and helper_owner_path not in repaired
            and any(token in repaired for token in helper_tokens)
        ):
            repaired = self._insert_path_into_files_changed(
                repaired,
                helper_owner_path,
                "Preserve the existing frontend API helper wiring for the reused export/snapshot helpers.",
            )
        if repaired == plan_content:
            return plan_content, validation_result
        repaired_validation = self._validate_draft(
            plan_content=repaired,
            repo_understanding=repo_understanding,
            draft_phase="planner",
            min_grounding_ratio=min_grounding_ratio,
            verified_paths_extra=grounding_paths,
            requirements_text=requirements,
        )
        if repaired_validation.failure_count <= validation_result.failure_count:
            return repaired, repaired_validation
        return plan_content, validation_result

    @staticmethod
    def _draft_loop_budget_ms_for_complexity(complexity: str) -> int:
        normalized = str(complexity or "").strip().lower()
        if normalized == "simple":
            return _DEFAULT_DRAFT_LOOP_BUDGET_MS
        if normalized == "moderate":
            return _MODERATE_DRAFT_LOOP_BUDGET_MS
        return _COMPLEX_DRAFT_LOOP_BUDGET_MS

    async def _emit_progress(
        self,
        *,
        stage: str,
        step: str,
        **extra: Any,
    ) -> None:
        if self._emit is None:
            return
        payload: dict[str, Any] = {"type": "setup_progress", "step": step, "draft_stage": stage}
        payload.update(extra)
        await self._emit(payload)

    async def run(
        self,
        *,
        requirements: str,
        min_grounding_ratio: float | None,
        grounding_paths: set[str] | None,
        model_override: str | None,
        rejection_counts: dict[str, int],
        rejection_reasons: list[dict[str, str]],
        timeout_seconds_override: int | Callable[[], int] | None,
    ) -> AuthorResult:
        pipeline_start = time.perf_counter()
        mental_model = ""
        if callable(self._tool_executor.memory_block_callback):
            try:
                mental_payload = self._tool_executor.memory_block_callback("mental_model")
                mental_model = str(mental_payload.get("content", "")).strip()
            except Exception:  # noqa: BLE001
                mental_model = ""

        seeded_paths = {path for path in (grounding_paths or set()) if str(path).strip()}
        scan_step = (
            "Draft: reusing discovery evidence..." if seeded_paths else "Draft: scanning repository candidates..."
        )
        await self._emit_progress(stage="planner_scan", step=scan_step)
        t0 = time.perf_counter()
        candidates = self._scan_repo_candidates(mental_model=mental_model, seed_paths=seeded_paths or None)
        t1 = time.perf_counter()
        await self._emit_progress(stage="planner_explore", step="Draft: reading the most relevant files...")
        repo_understanding = self._explore_repo(
            requirements=requirements,
            candidates=candidates,
            mental_model=mental_model,
        )
        t2 = time.perf_counter()
        await self._emit_progress(stage="planner_classify", step="Draft: sizing implementation complexity...")
        complexity = self._classify_complexity(
            requirements=requirements,
            repo_understanding=repo_understanding,
        )
        t3 = time.perf_counter()
        evidence_bundle = self._build_evidence_bundle(repo_understanding, requirements)

        await self._emit_progress(
            stage="planner_draft",
            step="Draft: writing the first implementation plan...",
            complexity=complexity,
        )
        t4 = time.perf_counter()
        max_attempts = 1 if complexity == "simple" else 2
        attempt_count = 0
        self_review_used = False
        stability_stop = False
        stability_reason_codes: tuple[str, ...] = ()
        previous_signature: frozenset[str] | None = None
        previous_failures: tuple[str, ...] = ()
        revision_hints: tuple[str, ...] = ()
        draft_redraft_reason_codes: list[str] = []
        best_plan_content = ""
        best_validation_result: ValidationResult | None = None
        draft_elapsed = 0.0
        redraft_elapsed = 0.0
        loop_budget_ms = self._draft_loop_budget_ms_for_complexity(complexity)
        final_validation_result = ValidationResult.success()
        retry_exception_message: str | None = None

        for attempt_number in range(1, max_attempts + 1):
            attempt_count = attempt_number
            attempt_context = AttemptContext(
                attempt_number=attempt_number,
                previous_failures=previous_failures,
                revision_hints=revision_hints,
                elapsed_ms=round((time.perf_counter() - t4) * 1000),
            )
            if attempt_number > 1:
                await self._emit_progress(
                    stage="planner_redraft",
                    step="Draft: revising outline to satisfy planner guardrails...",
                    failures=list(previous_failures),
                    attempt=attempt_number,
                )
            attempt_started = time.perf_counter()
            try:
                plan_content = await self._draft_plan(
                    requirements=requirements,
                    repo_understanding=repo_understanding,
                    evidence_bundle=evidence_bundle,
                    attempt_context=attempt_context,
                    architecture=None,
                    draft_phase="planner",
                    model_override=model_override,
                    revision_hints=list(revision_hints),
                    timeout_seconds_override=timeout_seconds_override,
                )
            except Exception as exc:  # noqa: BLE001
                if best_plan_content:
                    retry_exception_message = f"attempt {attempt_number} failed after best draft: {exc}"
                    logger.warning("planner_pipeline retry_failed attempt=%s error=%s", attempt_number, exc)
                    break
                raise
            attempt_elapsed = time.perf_counter() - attempt_started
            if attempt_number == 1:
                draft_elapsed = attempt_elapsed
            else:
                redraft_elapsed += attempt_elapsed
            validation_result = self._coerce_validation_result(
                self._validate_draft(
                    plan_content=plan_content,
                    repo_understanding=repo_understanding,
                    draft_phase="planner",
                    min_grounding_ratio=min_grounding_ratio,
                    verified_paths_extra=set(grounding_paths or set()),
                    requirements_text=requirements,
                )
            )
            final_validation_result = validation_result
            draft_redraft_reason_codes.extend(validation_result.reason_codes)
            if self._should_replace_best(best_validation_result, validation_result):
                best_plan_content = plan_content
                best_validation_result = validation_result
            if validation_result.ok:
                break
            logger.warning(
                "planner_pipeline redraft_requested attempt=%s failures=%s",
                attempt_number,
                "; ".join(validation_result.failure_messages),
            )
            if previous_signature is not None and validation_result.normalized_signature == previous_signature:
                stability_stop = True
                stability_reason_codes = validation_result.reason_codes
                break
            if not validation_result.retryable:
                break
            if attempt_number >= max_attempts:
                break
            if ((time.perf_counter() - t4) * 1000) > loop_budget_ms:
                break
            if self._self_review_draft is not None:
                try:
                    hints = await self._self_review_draft(
                        requirements=requirements,
                        plan_content=plan_content,
                        evidence_bundle=evidence_bundle,
                        validation_result=validation_result,
                        attempt_context=attempt_context,
                        model_override=model_override,
                        timeout_seconds_override=timeout_seconds_override,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("planner_pipeline self_review_failed attempt=%s error=%s", attempt_number, exc)
                    hints = []
                merged_hints = [
                    *(str(hint).strip() for hint in hints if str(hint).strip()),
                    *self._deterministic_revision_hints(validation_result),
                ]
                revision_hints = tuple(dict.fromkeys(hint for hint in merged_hints if hint))[:4]
                self_review_used = self_review_used or bool(revision_hints)
            else:
                revision_hints = tuple(
                    dict.fromkeys(
                        [*validation_result.failure_messages[:4], *self._deterministic_revision_hints(validation_result)]
                    )
                )[:4]
            previous_signature = validation_result.normalized_signature
            previous_failures = validation_result.reason_codes

        plan_content = best_plan_content or plan_content
        final_validation_result = best_validation_result or final_validation_result
        plan_content, final_validation_result = self._deterministic_plan_repairs(
            plan_content=plan_content,
            validation_result=final_validation_result,
            repo_understanding=repo_understanding,
            evidence_bundle=evidence_bundle,
            min_grounding_ratio=min_grounding_ratio,
            grounding_paths=grounding_paths or set(),
            requirements=requirements,
        )
        if final_validation_result.failure_messages:
            rejection_reasons.append(
                {
                    "reason": "COMPLETION_CONSTRAINT",
                    "details": "; ".join(final_validation_result.failure_messages),
                }
            )
            logger.warning(
                "planner_pipeline skip_redraft failures=%s",
                "; ".join(final_validation_result.failure_messages),
            )

        total = time.perf_counter() - pipeline_start
        logger.info(
            "planner_pipeline total=%.1fs scan=%.2fs explore=%.2fs classify=%.2fs "
            "arch=%.2fs draft=%.2fs redraft=%.2fs complexity=%s loop_budget_ms=%s attempts=%s self_review=%s "
            "evidence_files=%s evidence_tests=%s stability_stop=%s retry_codes=%s",
            total,
            t1 - t0,
            t2 - t1,
            t3 - t2,
            0.0,
            draft_elapsed,
            redraft_elapsed,
            complexity,
            loop_budget_ms,
            attempt_count,
            self_review_used,
            len(evidence_bundle.relevant_files),
            len(evidence_bundle.test_targets),
            stability_stop,
            ",".join(final_validation_result.reason_codes),
        )
        referenced = extract_file_references(plan_content)
        verified_paths = (
            set(repo_understanding.file_contents.keys())
            | set(repo_understanding.entrypoints)
            | set(repo_understanding.core_modules)
            | set(repo_understanding.relevant_modules)
            | set(repo_understanding.relevant_tests)
            | set(grounding_paths or set())
            | set(self._tool_executor.read_history.keys())
        )
        return AuthorResult(
            plan=plan_content,
            unverified_references=referenced - verified_paths,
            accessed_paths=verified_paths,
            design_record=None,
            rejection_counts=rejection_counts,
            rejection_reasons=rejection_reasons,
            draft_diagnostics={
                "evidence_bundle_files_count": len(evidence_bundle.relevant_files),
                "evidence_bundle_test_targets_count": len(evidence_bundle.test_targets),
                "author_internal_attempts": attempt_count,
                "author_self_review_used": self_review_used,
                "draft_redraft_reason_codes": list(dict.fromkeys(draft_redraft_reason_codes)),
                "quality_gate_failures": list(final_validation_result.failure_messages),
                "initial_draft_total_ms": round((time.perf_counter() - t4) * 1000),
                "draft_loop_budget_ms": loop_budget_ms,
                "retry_exception": retry_exception_message,
                "stability_stop": stability_stop,
                "stability_reason_codes": list(stability_reason_codes),
            },
        )
