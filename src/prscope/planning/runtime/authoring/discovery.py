from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any, Literal

from .models import RepoCandidates, RepoUnderstanding

REQUIREMENT_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "into",
    "from",
    "your",
    "about",
    "have",
    "will",
    "just",
    "need",
    "want",
    "make",
    "more",
    "less",
    "only",
}

LOW_SIGNAL_GREP_KEYWORDS = {
    "endpoint",
    "endpoints",
    "feature",
    "features",
    "handler",
    "handlers",
    "route",
    "routes",
    "tests",
    "test",
    "change",
    "changes",
}

NON_TRIVIAL_EXTENSIONS = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".go",
    ".rs",
    ".java",
    ".rb",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".sql",
    ".sh",
}

TRIVIAL_FILENAMES = {"readme", "license", ".gitignore"}
IGNORED_SCAN_DIRS = {
    ".git",
    ".prscope",
    ".venv",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "venv",
}


def requirements_keywords(text: str) -> set[str]:
    tokens = re.split(r"[^a-z0-9]+", text.lower())
    return {token for token in tokens if len(token) >= 3 and token not in REQUIREMENT_STOPWORDS}


def is_localized_frontend_request(text: str) -> bool:
    lowered = str(text or "").lower()
    frontend_markers = (
        "frontend",
        "react",
        "ui",
        "component",
        "button",
        "actionbar",
        "planpanel",
        "planningview",
        "planning page",
    )
    return any(marker in lowered for marker in frontend_markers)


def path_tokens(path: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", path.lower()) if token}


def is_non_trivial_source(path: str) -> bool:
    lower = path.lower()
    base = lower.rsplit("/", 1)[-1]
    stem = base.split(".", 1)[0]
    if stem in TRIVIAL_FILENAMES or lower == ".prscope/manifesto.md":
        return False
    dot = base.rfind(".")
    ext = base[dot:] if dot >= 0 else ""
    return ext in NON_TRIVIAL_EXTENSIONS


def is_entrypoint_like(path: str) -> bool:
    lower = path.lower()
    base = lower.rsplit("/", 1)[-1]
    return (
        base in {"main.py", "app.py", "server.py", "api.py", "index.ts", "index.tsx", "index.js"}
        or "/cli" in lower
        or "/cmd/" in lower
        or "/bin/" in lower
        or "/server/" in lower
        or "/api/" in lower
    )


def is_test_or_config(path: str) -> bool:
    lower = path.lower()
    base = lower.rsplit("/", 1)[-1]
    if any(token in base for token in (".test.", ".spec.")):
        return True
    if lower.startswith("tests/") or lower.startswith("test/") or "/tests/" in lower or "/test/" in lower:
        return True
    return base in {
        "pyproject.toml",
        "package.json",
        "package-lock.json",
        "tsconfig.json",
        "vite.config.ts",
        "dockerfile",
        "docker-compose.yml",
        ".github/workflows/ci.yml",
    } or lower.endswith((".yml", ".yaml", ".toml", ".json"))


def extract_paths_from_mental_model(mental_model: str) -> set[str]:
    if not mental_model:
        return set()
    candidates = re.findall(r"`([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)`", mental_model)
    return {path.strip() for path in candidates if path.strip()}


def requirement_search_patterns(text: str) -> list[str]:
    route_literals = re.findall(r"/[A-Za-z0-9_./:-]+", text)
    patterns: list[str] = [re.escape(route.strip()) for route in route_literals if route.strip()]
    keywords = [
        token
        for token in sorted(requirements_keywords(text), key=lambda item: (-len(item), item))
        if token not in LOW_SIGNAL_GREP_KEYWORDS
    ]
    for keyword in keywords[:4]:
        patterns.append(rf"\b{re.escape(keyword)}\b")
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(patterns))


def grep_match_priority(line: str) -> int:
    lowered = str(line or "").strip().lower()
    if not lowered:
        return 0
    if re.search(r"@(app|router)\.(get|post|put|patch|delete)\(", lowered):
        return 6
    if re.search(r"\bclient\.(get|post|put|patch|delete)\(", lowered):
        return 5
    if re.search(r"^\s*(async\s+def|def)\s+\w+", lowered):
        return 4
    if lowered.startswith(("assert ", "response = ", "return ")):
        return 3
    if lowered.startswith(("#", '"', "'")):
        return 1
    return 2


class AuthorDiscoveryService:
    def __init__(self, tool_executor: Any):
        self.tool_executor = tool_executor

    def scan_repo_candidates(
        self,
        *,
        mental_model: str | None = None,
        seed_paths: set[str] | None = None,
        max_entries_per_dir: int = 250,
    ) -> RepoCandidates:
        visited_dirs: set[str] = set()
        discovered_files: set[str] = set()
        queue: list[str] = []

        if seed_paths:
            for raw_path in seed_paths:
                normalized = str(raw_path or "").strip().replace("\\", "/").strip("/")
                if not normalized or normalized == ".":
                    continue
                base = normalized.rsplit("/", 1)[-1]
                if "." in base:
                    discovered_files.add(normalized)
                else:
                    queue.append(normalized)
        if not discovered_files and not queue:
            queue.append(".")

        while queue:
            current = queue.pop(0)
            if current in visited_dirs:
                continue
            visited_dirs.add(current)
            try:
                listing = self.tool_executor.list_files(path=current, max_entries=max_entries_per_dir)
            except Exception:  # noqa: BLE001
                continue
            for entry in listing.get("entries", []):
                entry_path = str(entry.get("path", "")).strip()
                if not entry_path:
                    continue
                entry_name = entry_path.rsplit("/", 1)[-1].lower()
                if entry_name.startswith(".") and entry_name not in {".github"}:
                    continue
                if str(entry.get("type", "")) == "dir":
                    if entry_name in IGNORED_SCAN_DIRS:
                        continue
                    queue.append(entry_path)
                else:
                    discovered_files.add(entry_path)

        mental_model_paths = extract_paths_from_mental_model(mental_model or "")
        discovered_files.update(mental_model_paths)

        entrypoints = sorted(path for path in discovered_files if is_entrypoint_like(path))
        source_modules = sorted(
            path for path in discovered_files if is_non_trivial_source(path) and not is_test_or_config(path)
        )
        tests_and_config = sorted(path for path in discovered_files if is_test_or_config(path))
        return RepoCandidates(
            entrypoints=entrypoints,
            source_modules=source_modules,
            tests_and_config=tests_and_config,
            all_paths=sorted(discovered_files),
        )

    def explore_repo(
        self,
        *,
        requirements: str,
        candidates: RepoCandidates,
        mental_model: str | None = None,
        max_file_reads: int = 5,
    ) -> RepoUnderstanding:
        seeded_paths = extract_paths_from_mental_model(mental_model or "")
        req_keywords = requirements_keywords(requirements)
        localized_frontend = is_localized_frontend_request(requirements)
        needs_backend_contract_test = bool(
            localized_frontend and {"payload", "response", "serialization", "serializer", "shape"} & req_keywords
        )
        if localized_frontend and max_file_reads < 6:
            max_file_reads = 6
        if needs_backend_contract_test and max_file_reads < 7:
            max_file_reads = 7
        search_roots: list[str | None] = []
        top_level_roots = []
        for path in [*candidates.source_modules, *candidates.tests_and_config]:
            root = str(path).split("/", 1)[0]
            if root and root not in top_level_roots:
                top_level_roots.append(root)
        for preferred in ("src", "tests", "test"):
            if preferred in top_level_roots:
                search_roots.append(preferred)
        search_roots.extend(root for root in top_level_roots if root not in {"src", "tests", "test"})
        if not search_roots:
            search_roots = [None]
        grep_hits_by_path: dict[str, list[tuple[int, str]]] = {}
        for pattern in requirement_search_patterns(requirements):
            for root in search_roots:
                try:
                    payload = self.tool_executor.grep_code(pattern=pattern, path=root, max_results=40)
                except Exception:  # noqa: BLE001
                    continue
                for match in payload.get("results", []):
                    path = str(match.get("path", "")).strip()
                    if not path:
                        continue
                    try:
                        line_number = int(match.get("line", 0) or 0)
                    except (TypeError, ValueError):
                        line_number = 0
                    grep_hits_by_path.setdefault(path, [])
                    if line_number > 0:
                        grep_hits_by_path[path].append((line_number, str(match.get("text", ""))))
        score_by_path: dict[str, int] = {}
        scored_paths: list[tuple[int, str]] = []

        def _skip_for_localized_frontend(path: str) -> bool:
            lower_path = path.lower()
            return localized_frontend and ("/planning/runtime/" in lower_path or "/authoring/" in lower_path)

        for path in candidates.all_paths:
            tokens = path_tokens(path)
            lower_path = path.lower()
            overlap = len(tokens & req_keywords)
            score = overlap
            if path in candidates.entrypoints:
                score += 2
            if path in candidates.source_modules:
                score += 1
            if path in seeded_paths:
                score += 2
            if path in grep_hits_by_path:
                score += 5 + min(3, len(grep_hits_by_path[path]))
            if ("test" in req_keywords or "tests" in req_keywords) and path in candidates.tests_and_config:
                score += 2
            if localized_frontend:
                if "/web/frontend/" in lower_path:
                    score += 4
                elif not lower_path.endswith("/web/api.py"):
                    score -= 3
                if any(marker in lower_path for marker in ("actionbar", "planningview", "planpanel")):
                    score += 3
                if lower_path.endswith("/lib/api.ts"):
                    score += 3
                if "actionbar" in req_keywords and "actionbar" in tokens:
                    score += 6
                if "planningview" in req_keywords and "planningview" in lower_path:
                    score += 8
                if "planpanel" in req_keywords and "planpanel" in lower_path:
                    score += 6
                if "planning" in req_keywords and "page" in req_keywords and "planningview" in lower_path:
                    score += 6
                if any(token in req_keywords for token in ("diagnostics", "snapshot", "health")) and any(
                    marker in lower_path for marker in ("planningview", "planpanel")
                ):
                    score += 6
                if "export" in req_keywords and lower_path.endswith("/lib/api.ts"):
                    score += 5
                if "endpoint" in req_keywords and lower_path.endswith("/web/api.py"):
                    score += 4
                if {"payload", "response", "serialization", "serializer", "shape"} & req_keywords:
                    if lower_path.endswith("/web/api.py"):
                        score += 6
                    if path in candidates.tests_and_config and "web_api_models" in lower_path:
                        score += 5
                if path in candidates.tests_and_config and "planningview" in lower_path:
                    score += 4
                if path in candidates.tests_and_config and any(
                    marker in lower_path for marker in ("planningview", "actionbar", "planpanel")
                ):
                    score += 3
                if (
                    path in candidates.tests_and_config
                    and "decisiongraphrender" in lower_path
                    and not ({"decision", "graph"} & req_keywords)
                ):
                    score -= 6
                if "sessionlist" in lower_path and "sessionlist" not in req_keywords:
                    score -= 8
                if lower_path.endswith("/lib/markdown.ts") and not (
                    {"render", "renderer", "preview", "format"} & req_keywords
                ):
                    score -= 6
                if "chatpanel" in lower_path and "chat" not in req_keywords:
                    score -= 8
                if "/planning/runtime/" in lower_path or "/authoring/" in lower_path:
                    score -= 15
            score_by_path[path] = score
            scored_paths.append((score, path))
        scored_paths.sort(key=lambda item: (-item[0], item[1]))
        wants_tests = "test" in req_keywords or "tests" in req_keywords
        source_candidates = [
            path
            for _, path in scored_paths
            if path in candidates.source_modules
            and not _skip_for_localized_frontend(path)
            and (not localized_frontend or score_by_path.get(path, 0) > 0)
        ]
        test_candidates = [path for _, path in scored_paths if path in candidates.tests_and_config]
        selected: list[str] = []
        if source_candidates:
            reserved_for_test = 0
            if wants_tests and test_candidates and max_file_reads > 1:
                reserved_for_test = 1
                if needs_backend_contract_test and len(test_candidates) > 1 and max_file_reads > 2:
                    reserved_for_test = 2
            source_budget = max(1, max_file_reads - reserved_for_test)
            selected.extend(source_candidates[:source_budget])
        if wants_tests and test_candidates:
            source_token_pool: set[str] = set()
            for path in selected or source_candidates[:2]:
                source_token_pool |= path_tokens(path)
            selected_has_web_api = any(path.lower().endswith("/web/api.py") for path in selected)
            ranked_tests = sorted(
                test_candidates,
                key=lambda path: (
                    -(
                        1
                        if needs_backend_contract_test and selected_has_web_api and "web_api_models" in path.lower()
                        else 0
                    ),
                    -len(path_tokens(path) & source_token_pool),
                    -(score_by_path.get(path, 0)),
                    path,
                ),
            )
            for path in ranked_tests:
                if path not in selected:
                    selected.append(path)
                if len(selected) >= max(1, max_file_reads):
                    break
        for _, path in scored_paths:
            if _skip_for_localized_frontend(path):
                continue
            if localized_frontend and score_by_path.get(path, 0) <= 0:
                continue
            if path not in selected:
                selected.append(path)
            if len(selected) >= max(1, max_file_reads):
                break

        file_contents: dict[str, str] = {}
        for path in selected:
            try:
                match_lines = grep_hits_by_path.get(path, [])
                if match_lines:
                    focus_line, _ = max(match_lines, key=lambda item: (grep_match_priority(item[1]), item[0]))
                    payload = self.tool_executor.read_file(path=path, around_line=focus_line, radius=60)
                else:
                    payload = self.tool_executor.read_file(path=path, max_lines=140)
                text = str(payload.get("content", "")).strip()
                if text:
                    file_contents[path] = text
            except Exception:  # noqa: BLE001
                continue

        relevant_modules = [path for path in selected if path in candidates.source_modules]
        relevant_tests = [path for path in selected if path in candidates.tests_and_config]
        evidence_notes: list[str] = []
        for path in selected:
            matches = grep_hits_by_path.get(path, [])
            if not matches:
                continue
            focus_line, focus_text = max(matches, key=lambda item: (grep_match_priority(item[1]), item[0]))
            if grep_match_priority(focus_text) >= 4:
                evidence_notes.append(f"{path}:{focus_line} {focus_text}")
            if len(evidence_notes) >= 3:
                break
        architecture_summary = (
            "Mental model seeded exploration with deterministic candidate scanning."
            if mental_model
            else "Deterministic repository scan based on entrypoints/source/test-config heuristics plus keyword/route grep hits."
        )
        if evidence_notes:
            architecture_summary += " Existing matching evidence: " + " | ".join(evidence_notes)
        risks: list[str] = []
        if not file_contents:
            risks.append("Unable to read candidate files during exploration")

        return RepoUnderstanding(
            entrypoints=candidates.entrypoints[:20],
            core_modules=candidates.source_modules[:30],
            relevant_modules=relevant_modules[:20],
            relevant_tests=relevant_tests[:20],
            architecture_summary=architecture_summary,
            risks=risks,
            file_contents=file_contents,
            from_mental_model=bool(mental_model),
        )

    def classify_complexity(
        self,
        *,
        requirements: str,
        repo_understanding: RepoUnderstanding,
    ) -> Literal["simple", "moderate", "complex"]:
        tokens = [token for token in re.split(r"[^a-z0-9]+", requirements.lower()) if token]
        uniq_tokens = {token for token in tokens if len(token) >= 4}
        path_mentions = re.findall(r"[A-Za-z0-9_./-]+\.[A-Za-z0-9]+", requirements)
        module_count = len(repo_understanding.relevant_modules)
        architecture_terms = {
            "architecture",
            "migration",
            "orchestration",
            "state",
            "pipeline",
            "scalability",
            "concurrency",
        }
        architecture_hits = sum(1 for token in uniq_tokens if token in architecture_terms)
        score = min(len(uniq_tokens), 12) + (3 * len(path_mentions)) + (2 * module_count) + (4 * architecture_hits)
        if architecture_hits >= 2 or (score >= 24 and len(path_mentions) >= 1):
            return "complex"
        if score >= 14 or module_count >= 3:
            return "moderate"
        return "simple"


class AuthorDesignService:
    def __init__(
        self,
        stage_runner: Any,
        extract_first_json_object: Callable[[str], tuple[str, str]],
    ) -> None:
        self.stage_runner = stage_runner
        self.extract_first_json_object = extract_first_json_object

    async def design_architecture(
        self,
        *,
        requirements: str,
        repo_understanding: RepoUnderstanding,
        model_override: str | None = None,
        timeout_seconds_override: int | Callable[[], int] | None = None,
    ) -> Any:
        from .models import ArchitectureDesign

        messages = [
            {
                "role": "system",
                "content": (
                    "You are designing architecture for an implementation plan.\n"
                    "Return JSON with fields:\n"
                    "- problem_summary: str\n"
                    "- proposed_components: list[str]\n"
                    "- responsibilities: object {component: responsibility}\n"
                    "- data_flow: str\n"
                    "- integration_points: list[str]\n"
                    "- alternatives_considered: list[str]\n"
                    "- chosen_design: str\n"
                    "- simplification_opportunities: list[str]\n"
                    "Prefer the simplest architecture that satisfies requirements."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## Requirements\n{requirements}\n\n"
                    f"## Repo Understanding\n{json.dumps(repo_understanding.__dict__, indent=2)}"
                ),
            },
        ]
        raw = await self.stage_runner.run_stage(
            "architecture_design",
            messages,
            allow_tools=False,
            max_output_tokens=1800,
            model_override=model_override,
            timeout_seconds_override=timeout_seconds_override,
        )
        json_text, _ = self.extract_first_json_object(raw)
        payload = json.loads(json_text)
        return ArchitectureDesign(
            problem_summary=str(payload.get("problem_summary", "")),
            proposed_components=[str(item) for item in payload.get("proposed_components", [])],
            responsibilities={str(k): str(v) for k, v in payload.get("responsibilities", {}).items()},
            data_flow=str(payload.get("data_flow", "")),
            integration_points=[str(item) for item in payload.get("integration_points", [])],
            alternatives_considered=[str(item) for item in payload.get("alternatives_considered", [])],
            chosen_design=str(payload.get("chosen_design", "")),
            simplification_opportunities=[str(item) for item in payload.get("simplification_opportunities", [])],
        )

    @staticmethod
    def design_record_from_architecture(architecture: Any) -> Any:
        from .models import DesignRecord

        return DesignRecord(
            problem_summary=architecture.problem_summary,
            constraints=[],
            architecture=architecture.data_flow,
            alternatives_considered=list(architecture.alternatives_considered),
            tradeoffs=[],
            chosen_design=architecture.chosen_design,
            assumptions=[],
            potential_failure_modes=[],
        )
