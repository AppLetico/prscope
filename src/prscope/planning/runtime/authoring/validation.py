from __future__ import annotations

import re
from typing import Any, Literal

from ..tools import extract_file_references
from .discovery import (
    is_entrypoint_like,
    is_non_trivial_source,
    is_test_or_config,
    path_tokens,
    requirements_keywords,
)


class AuthorValidationService:
    def __init__(self, tool_executor: Any) -> None:
        self.tool_executor = tool_executor

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
    ) -> list[str]:
        failures: list[str] = []
        failures.extend(self.phase_failures(plan_content, draft_phase=draft_phase))
        failures.extend(self.missing_required_sections(plan_content, draft_phase=draft_phase))
        if draft_phase == "refiner":
            failures.extend(self.completion_failures(plan_content))
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
        return failures
