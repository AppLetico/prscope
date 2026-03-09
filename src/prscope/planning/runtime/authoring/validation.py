from __future__ import annotations

import difflib
import re
from typing import Any, Literal

from ..tools import extract_file_references
from .discovery import (
    is_entrypoint_like,
    is_localized_frontend_request,
    is_non_trivial_source,
    is_test_or_config,
    path_tokens,
    requirements_keywords,
)
from .models import ValidationResult

_RETRYABLE_REASON_CODES = frozenset(
    {
        "grounding_failure",
        "unknown_file_reference",
        "missing_sections",
        "missing_tests",
        "missing_helper_reuse",
        "localized_scope_drift",
        "missing_localized_backend_grounding",
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


def localized_request_explicit_payload_change(requirements_text: str | None) -> bool:
    lowered_requirements = str(requirements_text or "").lower()
    mentions_payload_terms = any(
        token in lowered_requirements
        for token in ("payload", "response", "serialization", "serializer", "shape", "contract")
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
    return not any(phrase in lowered_requirements for phrase in conditional_only_phrases)


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
                + list(getattr(repo_understanding, "core_modules", []))
                + list(getattr(repo_understanding, "entrypoints", []))
                + list(file_contents.keys())
            )
            if str(path).strip()
        ]
        prioritized_paths = list(dict.fromkeys(candidate_paths))
        prioritized_paths.sort(
            key=lambda path: (
                0 if path.lower().endswith("/lib/api.ts") else 1,
                0
                if path.lower().endswith(("planningview.tsx", "actionbar.tsx", "planpanel.tsx", "/lib/api.ts"))
                else 1,
            )
        )
        for path in prioritized_paths[:8]:
            lowered = path.lower()
            should_read = lowered.endswith(("planningview.tsx", "actionbar.tsx", "planpanel.tsx", "/lib/api.ts"))
            if not should_read:
                continue
            existing = str(file_contents.get(path, "") or "").strip()
            if existing and all(
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
                if match[0].isupper():
                    continue
                if not lowered.startswith(("get", "export", "download")):
                    continue
                is_helper_like = "_" in match or any(ch.isupper() for ch in match[1:])
                if not is_helper_like:
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
    def _prioritized_frontend_tests(paths: list[str]) -> list[str]:
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
    def localized_ui_scope_failures(plan_content: str, requirements_text: str | None) -> list[str]:
        requirements = str(requirements_text or "")
        if not is_localized_frontend_request(requirements):
            return []
        lowered_requirements = requirements.lower()
        actionable_sections = "\n".join(
            section
            for section in (
                AuthorValidationService.extract_section(plan_content, "Summary"),
                AuthorValidationService.extract_section(plan_content, "Changes"),
                AuthorValidationService.extract_section(plan_content, "Files Changed"),
                AuthorValidationService.extract_section(plan_content, "Architecture"),
                AuthorValidationService.extract_section(plan_content, "Implementation Steps"),
                AuthorValidationService.extract_section(plan_content, "Test Strategy"),
            )
            if section
        ).lower()
        failures: list[str] = []
        open_questions_section = AuthorValidationService.extract_section(plan_content, "Open Questions").lower()
        if not any(token in lowered_requirements for token in ("observability", "telemetry", "logging", "monitoring")):
            if any(token in actionable_sections for token in ("observability", "telemetry", "logging")) or any(
                token in open_questions_section
                for token in (
                    "logging",
                    "telemetry",
                    "observability",
                    "monitoring",
                    "log the export",
                    "logging the export",
                )
            ):
                failures.append(
                    "localized UI/API draft introduced observability or telemetry scope not present in requirements; "
                    "remove logging/telemetry/observability wording and keep the plan focused on existing UI wiring, "
                    "API helper reuse, and tests"
                )
        forbids_frontend_state_abstractions = any(
            phrase in lowered_requirements
            for phrase in (
                "do not introduce new hooks",
                "do not introduce hooks",
                "avoid new hooks",
                "do not introduce new contexts",
                "avoid new contexts",
                "do not introduce shared state",
                "shared state layers",
                "do not introduce background polling",
                "background polling",
                "keep the change localized",
            )
        )
        if forbids_frontend_state_abstractions:
            forbidden_patterns = (
                r"\b(new|dedicated)\s+hooks?\b",
                r"\bhooks?/contexts?\b",
                r"\buseState\b",
                r"\buseEffect\b",
                r"\buseRef\b",
                r"\buseCallback\b",
                r"\buseMemo\b",
                r"\bisExporting\b",
                r"\blastExportResult\b",
                r"\bcontext\s+provider\b",
                r"\blocal state\b",
                r"\bshared state\b",
                r"\bcentralized state\b",
                r"\bsingle state object\b",
                r"\bui state object\b",
                r"\bstate management\b",
                r"\bseparation of state and logic\b",
                r"\bseparation of ui and business logic\b",
                r"\bmanage concurrency\b",
                r"\bconcurrency proactively\b",
                r"\bconcurrent(?:ly)?\s+exports?\b",
                r"\bbackground polling\b",
                r"\bpolling\b",
                r"\bself-contained state\b",
                r"\breact hook patterns?\b",
                r"\bstate initialization\b",
                r"\bmount tracking\b",
                r"\bunmount(?:ed)?\b",
                r"\bunmount guard\b",
                r"\bisMountedRef\b",
                r"\bPromise\.race\b",
                r"\bsetTimeout\b",
                r"\btimeout recovery\b",
                r"\bdouble-click prevention\b",
                r"\btype\s+[A-Z][A-Za-z0-9_]*\s*=",
            )
            if any(re.search(pattern, actionable_sections, re.IGNORECASE) for pattern in forbidden_patterns):
                failures.append(
                    "localized UI/API draft introduced frontend state or polling abstractions not present in requirements; "
                    "remove hooks/contexts/shared-state/state-management/polling wording and keep the plan focused on direct component wiring, "
                    "existing helper reuse, PlanPanel compatibility, and tests"
                )
        explicit_payload_change_requested = localized_request_explicit_payload_change(requirements)
        if not explicit_payload_change_requested:
            backend_contract_patterns = (
                r"\bget_session\b",
                r"\bgetsession\b",
                r"\bplanningsession\b",
                r"\bis_exporting\b",
                r"\blast_export_at\b",
                r"\blast_export_result\b",
                r"\bbackend api responses?\b",
                r"\bbackend api contract\b",
                r"\bapi contract changes?\b",
                r"\bapi model regression test\b",
                r"\bendpoint response\b.*\bnew fields?\b",
                r"\binclude new fields?\b",
                r"\bsync(?:ing)? with the backend'?s status\b",
                r"\btests/test_web_api_models\.py\b",
            )
            if any(re.search(pattern, actionable_sections, re.IGNORECASE) for pattern in backend_contract_patterns):
                failures.append(
                    "localized UI/API draft introduced backend contract or session-state changes not present in requirements; "
                    "do not invent new response fields or backend status plumbing unless the requirements explicitly require a payload/response change"
                )
        ownership_sections = "\n".join(
            section
            for section in (
                AuthorValidationService.extract_section(plan_content, "Changes"),
                AuthorValidationService.extract_section(plan_content, "Files Changed"),
                AuthorValidationService.extract_section(plan_content, "Architecture"),
                AuthorValidationService.extract_section(plan_content, "Implementation Steps"),
            )
            if section
        ).lower()
        mentions_planningview_owner = any(
            phrase in ownership_sections
            for phrase in (
                "planningview maintains export state",
                "planningview owns export state",
                "state lives in planningview",
                "planningview passes",
                "passed to planpanel as props",
                "pass the necessary props to `planpanel`",
                "pass the necessary props to planpanel",
                "props from planningview",
            )
        )
        mentions_planpanel_internal_owner = any(
            phrase in ownership_sections
            for phrase in (
                "planpanel wraps it internally",
                "planpanel internal state",
                "keep its internal state",
                "state lives in planpanel",
                "managed within planpanel",
                "planpanel manages export state",
                "planpanel component will be updated to keep its internal state",
                "reducing reliance on external props",
                "planpanel operates independently",
                "maintain minimal external dependencies",
                "internal state for visibility of the last export result",
                "internal state for export status",
                "maintain its internal state",
                "only require essential props from planningview",
                "responsible for displaying the last export result",
                "responsible for displaying the last export result, enabling or disabling the export action",
            )
        ) or any(
            re.search(pattern, ownership_sections, re.IGNORECASE)
            for pattern in (
                r"planpanel[^.]{0,120}\binternal state\b",
                r"planpanel[^.]{0,120}\boperate(?:s)? independently\b",
                r"planpanel[^.]{0,160}\bresponsible for\b",
                r"planpanel[^.]{0,160}\breduc(?:e|ing) the dependency on planningview\b",
            )
        )
        if mentions_planningview_owner and mentions_planpanel_internal_owner:
            failures.append(
                "localized UI/API draft leaves export-state ownership ambiguous between `PlanningView.tsx` and `PlanPanel.tsx`; "
                "choose one owner and keep Files Changed, Architecture, and Implementation Steps consistent about whether state is passed as props or managed inside PlanPanel"
            )
        preserves_planpanel_behavior = (
            "preserve current planpanel behavior" in lowered_requirements
            or "preserve planpanel behavior" in lowered_requirements
        )
        if preserves_planpanel_behavior and mentions_planpanel_internal_owner:
            failures.append(
                "localized UI/API draft shifts established export-state ownership into `PlanPanel.tsx` even though the request says to preserve current PlanPanel behavior; "
                "keep the existing `PlanningView.tsx`-owned wiring and limit PlanPanel changes to localized rendering/button behavior"
            )
        asks_for_latest_result = any(
            phrase in lowered_requirements
            for phrase in (
                "show the last export result",
                "show the latest export result",
                "show the last result",
                "show the latest result",
            )
        )
        mentions_format_requirements = any(
            phrase in lowered_requirements
            for phrase in (
                "format",
                "formatted",
                "plain text",
                "success message",
                "error detail",
                "toast",
                "banner",
            )
        )
        speculative_result_display_patterns = (
            r"(specific format|format or details|plain text sufficient)",
            r"(what|which).{0,50}(details?|format).{0,80}(export result|latest result|last result)",
            r"(success messages?|error details?)",
        )
        if (
            asks_for_latest_result
            and not mentions_format_requirements
            and any(
                re.search(pattern, open_questions_section, re.IGNORECASE)
                for pattern in speculative_result_display_patterns
            )
        ):
            failures.append(
                "localized UI/API draft turned a simple result-display request into speculative formatting open questions; "
                "default to a concise success/failure result presentation unless the requirements explicitly demand richer formatting"
            )
        return failures

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
            str(path).strip() for path in getattr(repo_understanding, "relevant_tests", []) if str(path).strip()
        ]
        if not relevant_tests:
            return []
        relevant_tests = sorted(relevant_tests)
        referenced = extract_file_references(plan_content)
        if any(path in referenced for path in relevant_tests):
            failures: list[str] = []
        else:
            candidates = ", ".join(relevant_tests[:3])
            failures = [f"missing test target reference; reference one of: {candidates}"]
        if is_localized_frontend_request(requirements):
            frontend_tests = AuthorValidationService._prioritized_frontend_tests(
                [
                    path
                    for path in relevant_tests
                    if "/frontend/" in path and re.search(r"\.test\.(?:[jt]sx?)$", path, re.IGNORECASE)
                ]
            )
            if frontend_tests and not any(path in referenced for path in frontend_tests):
                failures.append(
                    "localized frontend UI change should reference a frontend regression target; "
                    f"mention `{frontend_tests[0]}`"
                )
        return failures

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
            camel_case_helpers = [name for name in matching_helpers if "_" not in name]
            if camel_case_helpers:
                matching_helpers = camel_case_helpers
            if any(name.lower() in lowered_plan for name in matching_helpers):
                continue
            candidates = ", ".join(matching_helpers[:3])
            failures.append(f"missing explicit helper reuse reference for {token}; mention one of: {candidates}")
        return failures

    @staticmethod
    def localized_backend_grounding_failures(
        plan_content: str,
        repo_understanding: Any,
        requirements_text: str | None,
    ) -> list[str]:
        requirements = str(requirements_text or "").lower()
        if not is_localized_frontend_request(requirements):
            return []
        if not localized_request_explicit_payload_change(requirements_text):
            return []
        verified_backend_paths = {
            str(path).strip()
            for path in (
                list(getattr(repo_understanding, "entrypoints", []))
                + list(getattr(repo_understanding, "core_modules", []))
                + list(getattr(repo_understanding, "relevant_modules", []))
            )
            if str(path).strip().endswith("/web/api.py")
        }
        if not verified_backend_paths:
            return []
        lowered_plan = str(plan_content or "").lower()
        backend_change_markers = (
            "payload",
            "response",
            "serialization",
            "serializer",
            "shape tweak",
            "shape change",
            "format change",
            "field change",
            "backend payload",
            "api response",
        )
        mentions_backend_adjustment = any(marker in lowered_plan for marker in backend_change_markers)
        mentions_open_question = "open questions" in lowered_plan and any(
            marker in lowered_plan for marker in ("payload", "response", "format")
        )
        if not (mentions_backend_adjustment or mentions_open_question):
            return []
        referenced = extract_file_references(plan_content)
        missing: list[str] = []
        if not any(path in referenced for path in verified_backend_paths):
            target = sorted(verified_backend_paths)[0]
            missing.append(
                f"localized backend payload/response change must reference the existing API path; mention `{target}`"
            )
        relevant_api_tests = [
            str(path).strip()
            for path in getattr(repo_understanding, "relevant_tests", [])
            if str(path).strip().endswith("test_web_api_models.py")
        ]
        if relevant_api_tests and not any(path in referenced for path in relevant_api_tests):
            target = relevant_api_tests[0]
            missing.append(
                f"localized backend payload/response change must reference the API model regression target; mention `{target}`"
            )
        return missing

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
        if normalized.startswith("under-scoped draft:"):
            return "missing_sections"
        if normalized.startswith("files changed entries missing from implementation steps:"):
            return "grounding_failure"
        if "test strategy" in normalized or "missing test" in normalized:
            return "missing_tests"
        if normalized.startswith("localized frontend ui change should reference a frontend regression target;"):
            return "missing_tests"
        if normalized.startswith("missing explicit helper reuse reference for "):
            return "missing_helper_reuse"
        if "localized ui/api draft introduced" in normalized:
            return "localized_scope_drift"
        if normalized.startswith("localized backend payload/response change must reference "):
            return "missing_localized_backend_grounding"
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
        failures.extend(self.localized_backend_grounding_failures(plan_content, repo_understanding, requirements_text))
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

    def validate_refinement_result(
        self,
        *,
        plan_content: str,
        repo_understanding: Any,
        verified_paths_extra: set[str] | None = None,
        requirements_text: str | None = None,
        min_grounding_ratio: float | None = None,
    ) -> ValidationResult:
        failures: list[str] = []
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
        implementation = extract_file_references(self.extract_section(plan_content, "Implementation Steps"))
        missing_impl_refs = sorted(files_changed - implementation)
        if missing_impl_refs:
            failures.append("Files Changed entries missing from Implementation Steps: " + ", ".join(missing_impl_refs))
        failures.extend(self.localized_ui_scope_failures(plan_content, requirements_text))
        failures.extend(self.missing_test_target_failures(plan_content, repo_understanding, requirements_text))
        failures.extend(self.missing_helper_reuse_failures(plan_content, repo_understanding, requirements_text))
        failures.extend(self.localized_backend_grounding_failures(plan_content, repo_understanding, requirements_text))
        if min_grounding_ratio is not None:
            verified_paths = (
                set(getattr(repo_understanding, "file_contents", {}).keys())
                | set(getattr(repo_understanding, "entrypoints", []))
                | set(getattr(repo_understanding, "core_modules", []))
                | set(getattr(repo_understanding, "relevant_modules", []))
                | set(getattr(repo_understanding, "relevant_tests", []))
                | set(verified_paths_extra or set())
            )
            grounding, _, _ = self.grounding_failures(
                plan_content=plan_content,
                verified_paths=verified_paths,
                min_grounding_ratio=min_grounding_ratio,
                draft_phase="refiner",
            )
            failures.extend(grounding)
        return self.build_validation_result(failures)
