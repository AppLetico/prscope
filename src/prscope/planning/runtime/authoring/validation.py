from __future__ import annotations

import difflib
import re
from typing import Any, Literal

from ..tools import extract_file_references
from .models import ValidationResult
from .discovery import (
    is_localized_frontend_request,
    is_entrypoint_like,
    is_non_trivial_source,
    is_test_or_config,
    path_tokens,
    requirements_keywords,
)

_RETRYABLE_REASON_CODES = frozenset(
    {
        "grounding_failure",
        "unknown_file_reference",
        "missing_sections",
        "missing_tests",
        "missing_helper_reuse",
        "localized_scope_drift",
    }
)

_SECTION_FAILURE_NAMES = frozenset(
    {
        "title",
        "summary",
        "goals",
        "non_goals",
        "changes",
        "files_changed",
        "todos_in_order",
        "architecture",
        "mermaid_diagram",
        "implementation_steps",
        "test_strategy",
        "rollback_plan",
        "example_code_snippets",
        "open_questions",
        "design_decision_records",
        "user_stories",
        "mermaid_content",
        "example_code_fence",
        "ordered_todos",
    }
)


class AuthorValidationService:
    def __init__(self, tool_executor: Any) -> None:
        self.tool_executor = tool_executor

    @staticmethod
    def _suggest_verified_path(unverified_path: str, verified_paths: set[str]) -> str | None:
        normalized = str(unverified_path).strip()
        if not normalized:
            return None
        candidates = difflib.get_close_matches(normalized, sorted(verified_paths), n=1, cutoff=0.55)
        if candidates:
            return candidates[0]
        target_name = normalized.rsplit("/", 1)[-1]
        for candidate in sorted(verified_paths):
            if candidate.endswith(target_name):
                return candidate
        return None

    def _helper_mentions_from_repo(self, repo_understanding: Any) -> list[str]:
        file_contents = dict(getattr(repo_understanding, "file_contents", {}) or {})
        candidate_paths = [
            str(path).strip()
            for path in (
                list(getattr(repo_understanding, "relevant_modules", []))
                + list(getattr(repo_understanding, "entrypoints", []))
            )
            if str(path).strip()
        ]
        for path in candidate_paths[:6]:
            lowered = path.lower()
            should_read = lowered.endswith(("planningview.tsx", "actionbar.tsx", "planpanel.tsx", "/lib/api.ts"))
            if not should_read:
                continue
            existing = str(file_contents.get(path, "") or "").strip()
            if existing and any(
                token in existing.lower() for token in ("getsessionsnapshot", "exportsession", "downloadfile")
            ):
                continue
            try:
                payload = self.tool_executor.read_file(path, max_lines=320 if lowered.endswith("/lib/api.ts") else 120)
            except Exception:  # noqa: BLE001
                continue
            enriched = str(payload.get("content", "") or "").strip()
            if enriched:
                file_contents[path] = enriched
        names: list[str] = []
        seen: set[str] = set()
        for content in file_contents.values():
            text = str(content or "")
            for match in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", text):
                lowered = match.lower()
                if not any(token in lowered for token in ("snapshot", "diagnostic", "export", "download")):
                    continue
                if lowered in seen:
                    continue
                seen.add(lowered)
                names.append(match)
        return names

    @staticmethod
    def extract_section(content: str, heading: str) -> str:
        pattern = re.compile(
            rf"^##\s+{re.escape(heading)}\b(.*?)(?=^##\s+|\Z)",
            re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        match = pattern.search(content)
        return match.group(1).strip() if match else ""

    def explorer_gate_failures(self, requirements_text: str) -> list[str]:
        read_history = self.tool_executor.read_history
        read_paths = set(read_history.keys())
        failures: list[str] = []
        if len(read_paths) < 3:
            failures.append(f"need at least 3 unique `read_file` calls (currently {len(read_paths)})")
        if not any(is_non_trivial_source(path) for path in read_paths):
            failures.append("need at least 1 non-trivial source/config file read")
        keywords = requirements_keywords(requirements_text)
        if keywords and not any(path_tokens(path) & keywords for path in read_paths):
            failures.append("need at least 1 requirement-relevant file read")
        if keywords and not any(
            (path_tokens(path) & keywords)
            and (int(meta.get("line_count", 0)) >= 20 or int(meta.get("file_size_bytes", 0)) >= 1000)
            for path, meta in read_history.items()
        ):
            failures.append("need at least 1 requirement-relevant substantive read")
        if not any(
            int(meta.get("line_count", 0)) >= 30 or int(meta.get("file_size_bytes", 0)) >= 1500
            for meta in read_history.values()
        ):
            failures.append("need at least 1 substantive read (>=30 lines OR >=1.5KB)")
        if not any(is_entrypoint_like(path) for path in read_paths):
            failures.append("need at least 1 entrypoint/runtime file read")
        if not any(is_test_or_config(path) for path in read_paths):
            failures.append("need at least 1 test or config file read")
        return failures

    def grounding_failures(
        self,
        plan_content: str,
        verified_paths: set[str],
        min_grounding_ratio: float,
        draft_phase: Literal["planner", "refiner"],
    ) -> tuple[list[str], set[str], float]:
        referenced = extract_file_references(plan_content)
        if not referenced:
            return [], set(), 1.0
        unverified = referenced - verified_paths
        grounding_ratio = (len(referenced) - len(unverified)) / float(len(referenced))
        failures: list[str] = []
        if grounding_ratio < min_grounding_ratio:
            failures.append(f"grounding ratio {grounding_ratio:.2f} below required {min_grounding_ratio:.2f}")
        if draft_phase == "refiner":
            files_changed = extract_file_references(self.extract_section(plan_content, "Files Changed"))
            implementation = extract_file_references(self.extract_section(plan_content, "Implementation Steps"))
            missing_impl_refs = sorted(files_changed - implementation)
            if missing_impl_refs:
                failures.append(
                    "Files Changed entries missing from Implementation Steps: " + ", ".join(missing_impl_refs)
                )
        return failures, unverified, grounding_ratio

    @staticmethod
    def incremental_grounding_failures(
        previous_plan_content: str,
        updated_plan_content: str,
        verified_paths: set[str],
    ) -> list[str]:
        previous_refs = extract_file_references(previous_plan_content)
        updated_refs = extract_file_references(updated_plan_content)
        new_refs = updated_refs - previous_refs
        unverified_new_refs = sorted(new_refs - verified_paths)
        if not unverified_new_refs:
            return []
        return [
            "revision introduced unverified file references: " + ", ".join(unverified_new_refs),
        ]

    @staticmethod
    def phase_failures(plan_content: str, draft_phase: Literal["planner", "refiner"]) -> list[str]:
        if draft_phase != "planner":
            return []
        failures: list[str] = []
        fence_count = len(re.findall(r"```", plan_content))
        if fence_count > 2:
            failures.append("planner draft has too many code fences")
        if re.search(r"^##\s+Implementation\s+Steps\b", plan_content, re.IGNORECASE | re.MULTILINE):
            failures.append("planner draft must not include Implementation Steps section")
        if re.search(r"^##\s+Test\s+Strategy\b", plan_content, re.IGNORECASE | re.MULTILINE):
            failures.append("planner draft must not include Test Strategy section")
        if re.search(r"^##\s+Rollback\s+Plan\b", plan_content, re.IGNORECASE | re.MULTILINE):
            failures.append("planner draft must not include Rollback Plan section")
        numbered_items = re.findall(
            r"^\s*\d+\.\s+.*\b(modify|add|update|delete|replace|rename|refactor)\b.*$",
            plan_content,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if len(numbered_items) > 5:
            failures.append("planner draft contains detailed implementation step list")
        return failures

    def completion_failures(self, plan_content: str) -> list[str]:
        failures: list[str] = []
        lower = plan_content.lower()
        if re.search(r"\b(todo|tbd|placeholder)\b", lower):
            failures.append("final draft contains TODO/TBD/placeholder markers")
        required_non_empty = [
            "Goals",
            "Non-Goals",
            "Files Changed",
            "Architecture",
            "Implementation Steps",
            "Test Strategy",
            "Rollback Plan",
        ]
        for heading in required_non_empty:
            if not self.extract_section(plan_content, heading):
                failures.append(f"required section is empty: {heading}")
        files_changed = extract_file_references(self.extract_section(plan_content, "Files Changed"))
        if not files_changed:
            failures.append("Files Changed section is empty")
        total_references = extract_file_references(plan_content)
        if len(files_changed) == 1 and len(total_references) > 1:
            failures.append("under-scoped draft: one file in Files Changed but multiple referenced files")
        implementation = self.extract_section(plan_content, "Implementation Steps").lower()
        if implementation and not any(
            token in implementation for token in ("interface", "signature", "contract", "api shape")
        ):
            failures.append("Implementation Steps missing interface/signature impact notes")
        test_strategy = self.extract_section(plan_content, "Test Strategy").lower()
        if test_strategy and not any(
            token in test_strategy
            for token in ("assert", "expects", "status code", "error path", "failure", "regression")
        ):
            failures.append("Test Strategy lacks concrete assertions or failure-path checks")
        rollback = self.extract_section(plan_content, "Rollback Plan").lower()
        if rollback and not any(token in rollback for token in ("trigger", "if ", "on ", "when ")):
            failures.append("Rollback Plan missing rollback trigger conditions")
        if rollback and not any(token in rollback for token in ("revert", "disable", "restore", "rollback action")):
            failures.append("Rollback Plan missing explicit rollback actions")
        architecture = self.extract_section(plan_content, "Architecture").lower()
        if architecture and not any(
            token in architecture for token in ("metric", "log", "alert", "observe", "monitor")
        ):
            failures.append("Architecture missing observability/monitoring specifics")
        return failures

    @staticmethod
    def localized_ui_scope_failures(plan_content: str, requirements_text: str | None) -> list[str]:
        requirements = str(requirements_text or "")
        if not is_localized_frontend_request(requirements):
            return []
        lowered_requirements = requirements.lower()
        if any(token in lowered_requirements for token in ("observability", "telemetry", "logging", "monitoring")):
            return []
        lowered_plan = plan_content.lower()
        if any(token in lowered_plan for token in ("observability", "telemetry", "logging")):
            return [
                "localized UI/API draft introduced observability or telemetry scope not present in requirements; "
                "remove logging/telemetry/observability wording and keep the plan focused on existing UI wiring, "
                "API helper reuse, and tests"
            ]
        return []

    @staticmethod
    def missing_test_target_failures(
        plan_content: str,
        repo_understanding: Any,
        requirements_text: str | None,
    ) -> list[str]:
        requirements = str(requirements_text or "").lower()
        if not any(token in requirements for token in ("test", "tests", "coverage", "regression")):
            return []
        relevant_tests = [
            str(path).strip()
            for path in getattr(repo_understanding, "relevant_tests", [])
            if str(path).strip()
        ]
        if not relevant_tests:
            return []
        referenced = extract_file_references(plan_content)
        if any(path in referenced for path in relevant_tests):
            return []
        candidates = ", ".join(relevant_tests[:3])
        return [f"missing test target reference; reference one of: {candidates}"]

    def missing_helper_reuse_failures(
        self,
        plan_content: str,
        repo_understanding: Any,
        requirements_text: str | None,
    ) -> list[str]:
        requirements = str(requirements_text or "").lower()
        if not is_localized_frontend_request(requirements):
            return []
        desired_tokens = tuple(token for token in ("snapshot", "export", "download") if token in requirements)
        if not desired_tokens:
            return []
        lowered_plan = str(plan_content or "").lower()
        failures: list[str] = []
        helper_names = self._helper_mentions_from_repo(repo_understanding)
        for token in desired_tokens:
            matching_helpers = [name for name in helper_names if token in name.lower()]
            if not matching_helpers:
                continue
            if any(name.lower() in lowered_plan for name in matching_helpers):
                continue
            candidates = ", ".join(matching_helpers[:3])
            failures.append(f"missing explicit helper reuse reference for {token}; mention one of: {candidates}")
        return failures

    @staticmethod
    def missing_required_sections(plan_content: str, draft_phase: Literal["planner", "refiner"]) -> list[str]:
        missing: list[str] = []
        section_patterns = {
            "title": r"^#\s+.+",
            "summary": r"^##\s+summary\b",
            "goals": r"^##\s+goals\b",
            "non_goals": r"^##\s+non-goals\b",
            "changes": r"^##\s+changes\b",
            "files_changed": r"^##\s+files\s+changed\b",
            "todos_in_order": r"^##\s+to-?dos?\s+in\s+order\b",
            "architecture": r"^##\s+architecture\b",
            "mermaid_diagram": r"^##\s+mermaid\s+diagram\b",
            "implementation_steps": r"^##\s+implementation\s+steps\b",
            "test_strategy": r"^##\s+test\s+strategy\b",
            "rollback_plan": r"^##\s+rollback\s+plan\b",
            "example_code_snippets": r"^##\s+example\s+code\s+snippets\b",
            "open_questions": r"^##\s+open\s+questions\b",
            "design_decision_records": r"^##\s+design\s+decision\s+records\b",
            "user_stories": r"^##\s+user\s+stories\b",
        }
        if draft_phase == "planner":
            planner_required = {"title", "goals", "non_goals", "files_changed", "architecture"}
            for section_name, pattern in section_patterns.items():
                if section_name not in planner_required:
                    continue
                if not re.search(pattern, plan_content, re.IGNORECASE | re.MULTILINE):
                    missing.append(section_name)
            return missing
        for section_name, pattern in section_patterns.items():
            if not re.search(pattern, plan_content, re.IGNORECASE | re.MULTILINE):
                missing.append(section_name)

        has_mermaid_block = "```mermaid" in plan_content.lower()
        has_explicit_mermaid_waiver = (
            "mermaid diagram is unnecessary" in plan_content.lower()
            or "no mermaid diagram needed" in plan_content.lower()
        )
        if not has_mermaid_block and not has_explicit_mermaid_waiver:
            missing.append("mermaid_content")
        if draft_phase == "refiner":
            has_any_code_fence = bool(re.search(r"```[a-zA-Z0-9_-]*\n", plan_content))
            if not has_any_code_fence:
                missing.append("example_code_fence")
        has_ordered_todos = bool(re.search(r"^\s*1\.\s+.+", plan_content, re.MULTILINE))
        if not has_ordered_todos:
            missing.append("ordered_todos")
        return missing

    def validate_draft(
        self,
        *,
        plan_content: str,
        repo_understanding: Any,
        draft_phase: Literal["planner", "refiner"] = "refiner",
        min_grounding_ratio: float | None = None,
        verified_paths_extra: set[str] | None = None,
        requirements_text: str | None = None,
    ) -> list[str]:
        return list(
            self.validate_draft_result(
                plan_content=plan_content,
                repo_understanding=repo_understanding,
                draft_phase=draft_phase,
                min_grounding_ratio=min_grounding_ratio,
                verified_paths_extra=verified_paths_extra,
                requirements_text=requirements_text,
            ).failure_messages
        )

    @staticmethod
    def _normalize_failure_messages(failures: list[str]) -> tuple[str, ...]:
        normalized: list[str] = []
        seen: set[str] = set()
        for failure in failures:
            text = str(failure or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(text)
        return tuple(normalized)

    @staticmethod
    def _reason_code_for_failure(failure: str) -> str:
        normalized = str(failure or "").strip().lower()
        if not normalized:
            return "validation_failure"
        if normalized in _SECTION_FAILURE_NAMES:
            return "missing_sections"
        if normalized.startswith("grounding ratio"):
            return "grounding_failure"
        if normalized.startswith("unknown file references:"):
            return "unknown_file_reference"
        if normalized.startswith("replace unverified path "):
            return "unknown_file_reference"
        if normalized.startswith("required section is empty:"):
            return "missing_sections"
        if "files changed section is empty" in normalized:
            return "missing_sections"
        if "planner draft must not include" in normalized:
            return "missing_sections"
        if "planner draft has too many code fences" in normalized:
            return "missing_sections"
        if "planner draft contains detailed implementation step list" in normalized:
            return "missing_sections"
        if "test strategy" in normalized or "missing test" in normalized:
            return "missing_tests"
        if normalized.startswith("missing explicit helper reuse reference for "):
            return "missing_helper_reuse"
        if "localized ui/api draft introduced" in normalized:
            return "localized_scope_drift"
        return "validation_failure"

    @classmethod
    def build_validation_result(cls, failures: list[str]) -> ValidationResult:
        normalized_failures = cls._normalize_failure_messages(failures)
        if not normalized_failures:
            return ValidationResult.success()
        reason_codes = tuple(sorted({cls._reason_code_for_failure(failure) for failure in normalized_failures}))
        return ValidationResult(
            failure_messages=normalized_failures,
            reason_codes=reason_codes,
            retryable=any(code in _RETRYABLE_REASON_CODES for code in reason_codes),
            failure_count=len(normalized_failures),
        )

    def validate_draft_result(
        self,
        *,
        plan_content: str,
        repo_understanding: Any,
        draft_phase: Literal["planner", "refiner"] = "refiner",
        min_grounding_ratio: float | None = None,
        verified_paths_extra: set[str] | None = None,
        requirements_text: str | None = None,
    ) -> ValidationResult:
        failures: list[str] = []
        failures.extend(self.phase_failures(plan_content, draft_phase=draft_phase))
        failures.extend(self.missing_required_sections(plan_content, draft_phase=draft_phase))
        if draft_phase == "refiner":
            failures.extend(self.completion_failures(plan_content))
        failures.extend(self.localized_ui_scope_failures(plan_content, requirements_text))
        failures.extend(self.missing_test_target_failures(plan_content, repo_understanding, requirements_text))
        failures.extend(self.missing_helper_reuse_failures(plan_content, repo_understanding, requirements_text))
        if min_grounding_ratio is not None:
            verified_paths = (
                set(repo_understanding.file_contents.keys())
                | set(repo_understanding.entrypoints)
                | set(repo_understanding.core_modules)
                | set(repo_understanding.relevant_modules)
                | set(repo_understanding.relevant_tests)
                | set(verified_paths_extra or set())
            )
            grounding, _, _ = self.grounding_failures(
                plan_content=plan_content,
                verified_paths=verified_paths,
                min_grounding_ratio=min_grounding_ratio,
                draft_phase=draft_phase,
            )
            failures.extend(grounding)
            referenced = extract_file_references(plan_content)
            unverified = sorted(referenced - verified_paths)
            if unverified:
                failures.append("unknown file references: " + ", ".join(unverified[:6]))
                for path in unverified[:4]:
                    suggested = self._suggest_verified_path(path, verified_paths)
                    if suggested and suggested != path:
                        failures.append(f"replace unverified path `{path}` with `{suggested}`")
        return self.build_validation_result(failures)
