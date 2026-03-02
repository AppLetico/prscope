"""
Structured planning memory and manifesto constraints.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .config import PlanningConfig, RepoProfile
from .planning.scanners import ScannerBackend, get_scanner

logger = logging.getLogger(__name__)

MEMORY_BLOCKS = ("architecture", "modules", "patterns", "entrypoints")

MANIFESTO_STARTER = """# Project Manifesto

## Vision
What this project is and why it exists.

## Engineering Principles
- Principle 1

## Non-Negotiable Constraints
- Constraint 1

## Strategic Direction
Where the project is heading in the next 6-12 months.

## Machine-Readable Constraints
# Optional V2 stub:
# extends: org-default
constraints:
  - id: C-001
    text: "No synchronous I/O on the main thread"
    severity: hard
  - id: C-002
    text: "All public APIs must have type annotations"
    severity: hard
"""


@dataclass
class ParsedConstraint:
    id: str
    text: str
    severity: str = "hard"
    optional: bool = False
    extends: str | None = None


class MemoryStore:
    """Builds and loads structured memory blocks for planning."""

    def __init__(
        self,
        repo: RepoProfile,
        config: PlanningConfig,
        scanner: ScannerBackend | None = None,
    ):
        self.repo = repo
        self.config = config
        self.memory_dir = repo.memory_dir
        self.meta_path = self.memory_dir / "_meta.json"
        self.manifesto_path = repo.resolved_manifesto
        self._scanner: ScannerBackend = scanner or get_scanner(config.scanner)

    def ensure_manifesto(self) -> Path:
        self.manifesto_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.manifesto_path.exists():
            self.manifesto_path.write_text(MANIFESTO_STARTER, encoding="utf-8")
        return self.manifesto_path

    def load_manifesto(self) -> str:
        if not self.manifesto_path.exists():
            return ""
        return self.manifesto_path.read_text(encoding="utf-8")

    def load_block(self, name: str) -> str:
        if name == "context":
            context_path = self.memory_dir / "context.md"
            if not context_path.exists():
                return ""
            return context_path.read_text(encoding="utf-8")
        path = self.memory_dir / f"{name}.md"
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def load_constraints(self, manifesto_path: Path | None = None) -> list[ParsedConstraint]:
        path = manifesto_path or self.manifesto_path
        if not path.exists():
            return []

        raw = path.read_text(encoding="utf-8")
        yaml_block = self._extract_machine_constraints(raw)
        if not yaml_block:
            return []

        try:
            data = yaml.safe_load(yaml_block) or {}
        except yaml.YAMLError:
            logger.warning("Failed to parse manifesto machine-readable constraints.")
            return []

        if not isinstance(data, dict):
            return []

        extends_value = data.get("extends")
        if extends_value:
            logger.warning(
                "Manifesto 'extends: %s' not resolved (V2 feature). Parent constraints omitted.",
                extends_value,
            )

        constraints_raw = data.get("constraints", [])
        parsed: list[ParsedConstraint] = []
        if not isinstance(constraints_raw, list):
            return parsed

        for item in constraints_raw:
            if not isinstance(item, dict):
                continue
            constraint_id = str(item.get("id", "")).strip()
            text = str(item.get("text", "")).strip()
            if not constraint_id or not text:
                continue
            severity = str(item.get("severity", "hard")).strip().lower()
            if severity not in {"hard", "soft"}:
                severity = "hard"
            parsed.append(
                ParsedConstraint(
                    id=constraint_id,
                    text=text,
                    severity=severity,
                    optional=bool(item.get("optional", False)),
                    extends=str(extends_value) if extends_value else None,
                )
            )

        return parsed

    def _extract_machine_constraints(self, markdown: str) -> str:
        lines = markdown.splitlines()
        extends_line: str | None = None
        capture_index: int | None = None

        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("extends:"):
                extends_line = stripped
            elif stripped.startswith("#") and "extends:" in stripped:
                extends_line = stripped.lstrip("#").strip()
            if stripped == "constraints:":
                capture_index = idx
                break

        if capture_index is None:
            return ""

        captured: list[str] = []
        if extends_line:
            captured.append(extends_line)
        captured.append("constraints:")

        for line in lines[capture_index + 1 :]:
            if not line.strip():
                captured.append(line)
                continue
            if line.startswith(" ") or line.startswith("\t") or line.startswith("-"):
                captured.append(line)
                continue
            if line.strip().startswith("#"):
                captured.append(line)
                continue
            break

        return "\n".join(captured).strip()

    def _read_meta(self) -> dict[str, Any]:
        if not self.meta_path.exists():
            return {}
        try:
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _write_meta(self, git_sha: str) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "git_sha": git_sha,
            "built_at": datetime.utcnow().isoformat() + "Z",
            "repo_path": str(self.repo.resolved_path),
        }
        self.meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def should_rebuild(self, git_sha: str, rebuild: bool = False) -> bool:
        if rebuild:
            return True
        meta = self._read_meta()
        if meta.get("git_sha") != git_sha:
            return True
        return not all((self.memory_dir / f"{block}.md").exists() for block in MEMORY_BLOCKS)

    async def ensure_memory(self, profile: dict[str, Any], rebuild: bool = False) -> None:
        git_sha = str(profile.get("git_sha", "unknown"))
        if not self.should_rebuild(git_sha, rebuild=rebuild):
            return
        await self.build_all_blocks(profile)
        self._write_meta(git_sha)

    async def build_all_blocks(self, profile: dict[str, Any]) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        scanner_name = getattr(self._scanner, "name", "grep")

        if scanner_name in ("repomap", "repomix"):
            # Non-grep backends produce a single rich context string directly —
            # no LLM call needed.  Write it as a single "context" block and
            # create lightweight stubs for the other expected blocks so the
            # rest of the system keeps working.
            context = await asyncio.to_thread(
                self._scanner.build_context, self.repo.resolved_path, profile
            )
            (self.memory_dir / "context.md").write_text(context, encoding="utf-8")
            # Stubs so load_block() never returns empty string for expected blocks
            stub = f"*See context.md for full {scanner_name} output.*"
            for block in MEMORY_BLOCKS:
                block_path = self.memory_dir / f"{block}.md"
                if not block_path.exists():
                    block_path.write_text(stub, encoding="utf-8")
            return

        # Default grep path: scanner builds a text profile, then LLM summarises
        # it into structured memory blocks.
        scanner_context = await asyncio.to_thread(
            self._scanner.build_context, self.repo.resolved_path, profile
        )
        semaphore = asyncio.Semaphore(max(self.config.memory_concurrency, 1))
        prompts = self._build_prompts(profile, scanner_context=scanner_context)

        async def build_one(block_name: str, prompt: str) -> None:
            async with semaphore:
                content = await self._complete(prompt)
                (self.memory_dir / f"{block_name}.md").write_text(content, encoding="utf-8")

        await asyncio.gather(*(build_one(name, prompt) for name, prompt in prompts.items()))

    def load_all_blocks(self) -> dict[str, str]:
        scanner_name = getattr(self._scanner, "name", "grep")
        if scanner_name in ("repomap", "repomix"):
            # Return the single rich context block for non-grep backends
            context_path = self.memory_dir / "context.md"
            if context_path.exists():
                return {"context": context_path.read_text(encoding="utf-8")}
        return {name: self.load_block(name) for name in MEMORY_BLOCKS}

    def _build_prompts(
        self, profile: dict[str, Any], scanner_context: str | None = None
    ) -> dict[str, str]:
        files = profile.get("file_tree", {}).get("files", [])
        dirs = profile.get("file_tree", {}).get("directories", [])
        extensions = profile.get("file_tree", {}).get("extensions", {})
        readme = profile.get("readme") or ""
        import_stats = profile.get("import_stats", {})

        if scanner_context:
            # Use the rich scanner output as the base for LLM prompts
            summary = scanner_context
        else:
            summary = (
                f"Repo: {self.repo.name}\n"
                f"Path: {self.repo.resolved_path}\n"
                f"Total files: {profile.get('file_tree', {}).get('total_files', 0)}\n"
                f"Extensions: {json.dumps(extensions, indent=2)}\n"
                f"Import stats: {json.dumps(import_stats, indent=2)}\n"
                f"Directories sample: {dirs[:100]}\n"
                f"Files sample: {files[:200]}\n"
                f"README:\n{readme[:3000]}\n"
            )

        return {
            "architecture": (
                "Write a concise architecture overview in markdown.\n\n"
                f"{summary}"
            ),
            "modules": (
                "Write a module index in markdown with key directories and responsibilities.\n\n"
                f"{summary}"
            ),
            "patterns": (
                "Write coding and design patterns in markdown inferred from repo structure.\n\n"
                f"{summary}"
            ),
            "entrypoints": (
                "Write operational entrypoints and execution paths in markdown.\n\n"
                f"{summary}"
            ),
        }

    async def _complete(self, prompt: str) -> str:
        try:
            import litellm
        except ImportError:
            return self._fallback_summary(prompt)

        def _run_completion() -> str:
            response = litellm.completion(
                model=self.config.author_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You summarize repositories for planning memory. Be concise and factual.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1200,
            )
            content = response.choices[0].message.content
            return str(content or "").strip()

        try:
            return await asyncio.to_thread(_run_completion)
        except Exception:
            return self._fallback_summary(prompt)

    def _fallback_summary(self, prompt: str) -> str:
        title = "Memory Block"
        first_line = prompt.splitlines()[0] if prompt else ""
        if "architecture" in prompt.lower():
            title = "Architecture"
        elif "module" in prompt.lower():
            title = "Modules"
        elif "pattern" in prompt.lower():
            title = "Patterns"
        elif "entrypoint" in prompt.lower():
            title = "Entrypoints"
        return f"# {title}\n\nAuto-generated fallback summary.\n\n{first_line}\n"


def constraints_by_id(constraints: list[ParsedConstraint]) -> dict[str, ParsedConstraint]:
    return {c.id: c for c in constraints}


def dump_constraints(constraints: list[ParsedConstraint]) -> str:
    if not constraints:
        return "[]"
    return json.dumps([asdict(c) for c in constraints], indent=2)
