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
        base in {"main.py", "app.py", "server.py", "index.ts", "index.tsx", "index.js"}
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
        source_modules = sorted(path for path in discovered_files if is_non_trivial_source(path))
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
        scored_paths: list[tuple[int, str]] = []
        for path in candidates.all_paths:
            tokens = path_tokens(path)
            overlap = len(tokens & req_keywords)
            score = overlap
            if path in candidates.entrypoints:
                score += 2
            if path in candidates.source_modules:
                score += 1
            if path in seeded_paths:
                score += 2
            scored_paths.append((score, path))
        scored_paths.sort(key=lambda item: (-item[0], item[1]))
        selected = [path for _, path in scored_paths[: max(1, max_file_reads)]]

        file_contents: dict[str, str] = {}
        for path in selected:
            try:
                payload = self.tool_executor.read_file(path=path, max_lines=140)
                text = str(payload.get("content", "")).strip()
                if text:
                    file_contents[path] = text
            except Exception:  # noqa: BLE001
                continue

        relevant_modules = [path for path in selected if path in candidates.source_modules]
        relevant_tests = [path for path in selected if path in candidates.tests_and_config]
        architecture_summary = (
            "Mental model seeded exploration with deterministic candidate scanning."
            if mental_model
            else "Deterministic repository scan based on entrypoints/source/test-config heuristics."
        )
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
