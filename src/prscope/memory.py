"""
Structured planning memory and manifesto constraints.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import yaml

from .config import PlanningConfig, RepoProfile
from .model_catalog import litellm_model_name, model_provider
from .planning.scanners import ScannerBackend, get_scanner
from .pricing import MODEL_CONTEXT_WINDOWS, estimate_cost_usd

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
    evidence_keywords: list[str] = field(default_factory=list)


def load_skills(skills_dir: Path, max_chars: int) -> str:
    """Load repo-local skill files with deterministic, boundary-safe truncation."""
    if not skills_dir.exists():
        logger.info("[skills] Loaded 0 skills (dir not found)")
        return ""

    files = sorted(skills_dir.glob("*.md"))
    if not files:
        logger.info("[skills] Loaded 0 skills")
        return ""

    blocks: list[str] = []
    total_chars = 0

    for file in files:
        content = file.read_text(encoding="utf-8").strip()
        block = f"### {file.name}\n\n{content}\n"
        if total_chars == 0 and len(block) > max_chars:
            logger.warning(
                "[skills] Skipped %s - exceeds skills_max_chars=%d",
                file.name,
                max_chars,
            )
            continue
        if total_chars + len(block) > max_chars:
            break
        blocks.append(block)
        total_chars += len(block)

    trimmed = len(files) - len(blocks)
    logger.info(
        "[skills] Loaded %d skills (%d chars)%s",
        len(blocks),
        total_chars,
        f" - trimmed {trimmed} of {len(files)} due to skills_max_chars={max_chars}" if trimmed else "",
    )

    result = "\n---\n".join(blocks)
    if trimmed:
        result += "\n\n... (truncated due to token budget)"
    return result


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
            raw_keywords = item.get("evidence_keywords", [])
            evidence_keywords: list[str] = (
                [str(k).strip() for k in raw_keywords if str(k).strip()] if isinstance(raw_keywords, list) else []
            )
            parsed.append(
                ParsedConstraint(
                    id=constraint_id,
                    text=text,
                    severity=severity,
                    optional=bool(item.get("optional", False)),
                    extends=str(extends_value) if extends_value else None,
                    evidence_keywords=evidence_keywords,
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
        all_blocks = [*MEMORY_BLOCKS, "mental_model"]
        return not all((self.memory_dir / f"{block}.md").exists() for block in all_blocks)

    async def ensure_memory(
        self,
        profile: dict[str, Any],
        rebuild: bool = False,
        progress_callback: Callable[[str], Awaitable[None]] | None = None,
        event_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        git_sha = str(profile.get("git_sha", "unknown"))
        if not self.should_rebuild(git_sha, rebuild=rebuild):
            if progress_callback:
                await progress_callback("Using cached codebase memory.")
            return
        await self.build_all_blocks(
            profile,
            progress_callback=progress_callback,
            event_callback=event_callback,
        )
        self._write_meta(git_sha)
        if progress_callback:
            await progress_callback("Ready.")

    async def build_all_blocks(
        self,
        profile: dict[str, Any],
        progress_callback: Callable[[str], Awaitable[None]] | None = None,
        event_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        scanner_name = getattr(self._scanner, "name", "grep")

        if scanner_name in ("repomap", "repomix"):
            # Non-grep backends produce a single rich context string directly —
            # no LLM call needed.  Write it as a single "context" block and
            # create lightweight stubs for the other expected blocks so the
            # rest of the system keeps working.
            if progress_callback:
                await progress_callback("Scanning codebase...")
            context = await asyncio.to_thread(self._scanner.build_context, self.repo.resolved_path, profile)
            if progress_callback:
                await progress_callback("Building context...")
            (self.memory_dir / "context.md").write_text(context, encoding="utf-8")
            # Stubs so load_block() never returns empty string for expected blocks
            stub = f"*See context.md for full {scanner_name} output.*"
            for block in MEMORY_BLOCKS:
                block_path = self.memory_dir / f"{block}.md"
                if not block_path.exists():
                    block_path.write_text(stub, encoding="utf-8")
            await self._build_mental_model(
                progress_callback=progress_callback,
                event_callback=event_callback,
            )
            return

        # Default grep path: scanner builds a text profile, then LLM summarises
        # it into structured memory blocks.
        if progress_callback:
            await progress_callback("Scanning codebase...")
        scanner_context = await asyncio.to_thread(self._scanner.build_context, self.repo.resolved_path, profile)
        semaphore = asyncio.Semaphore(max(self.config.memory_concurrency, 1))
        prompts = self._build_prompts(profile, scanner_context=scanner_context)

        async def build_one(block_name: str, prompt: str) -> None:
            async with semaphore:
                if progress_callback:
                    await progress_callback(f"Building {block_name} summary...")
                content = await self._complete(prompt, block_name=block_name, event_callback=event_callback)
                (self.memory_dir / f"{block_name}.md").write_text(content, encoding="utf-8")

        await asyncio.gather(*(build_one(name, prompt) for name, prompt in prompts.items()))
        await self._build_mental_model(
            progress_callback=progress_callback,
            event_callback=event_callback,
        )

    async def _build_mental_model(
        self,
        progress_callback: Callable[[str], Awaitable[None]] | None = None,
        event_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        """Synthesize a mental model from base memory blocks."""
        if progress_callback:
            await progress_callback("Synthesizing repository mental model...")
        blocks = {name: self.load_block(name) for name in MEMORY_BLOCKS}
        combined = "\n\n".join(
            f"## {name.title()}\n{content}"
            for name, content in blocks.items()
            if content and not content.startswith("*See context.md")
        )
        if not combined.strip():
            (self.memory_dir / "mental_model.md").write_text(
                "# Repository Mental Model\n\nNo base memory blocks available.",
                encoding="utf-8",
            )
            return
        prompt = (
            "Synthesize a concise repository mental model from the following memory blocks.\n"
            "Focus on: key abstractions, dependency flow between modules, critical paths,\n"
            "and the most important files an engineer should read first.\n"
            "Output markdown with sections: Key Abstractions, Dependency Flow, "
            "Critical Paths, Essential Files.\n"
            "Be concise — this will be used as a quick-reference guide for planning.\n\n"
            f"{combined}"
        )
        content = await self._complete(prompt, block_name="mental_model", event_callback=event_callback)
        (self.memory_dir / "mental_model.md").write_text(content, encoding="utf-8")

    def load_all_blocks(self) -> dict[str, str]:
        scanner_name = getattr(self._scanner, "name", "grep")
        if scanner_name in ("repomap", "repomix"):
            result: dict[str, str] = {}
            context_path = self.memory_dir / "context.md"
            if context_path.exists():
                result["context"] = context_path.read_text(encoding="utf-8")
            mental_model = self.load_block("mental_model")
            if mental_model:
                result["mental_model"] = mental_model
            return result
        blocks = {name: self.load_block(name) for name in MEMORY_BLOCKS}
        mental_model = self.load_block("mental_model")
        if mental_model:
            blocks["mental_model"] = mental_model
        return blocks

    def _build_prompts(self, profile: dict[str, Any], scanner_context: str | None = None) -> dict[str, str]:
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
            "architecture": (f"Write a concise architecture overview in markdown.\n\n{summary}"),
            "modules": (f"Write a module index in markdown with key directories and responsibilities.\n\n{summary}"),
            "patterns": (f"Write coding and design patterns in markdown inferred from repo structure.\n\n{summary}"),
            "entrypoints": (f"Write operational entrypoints and execution paths in markdown.\n\n{summary}"),
        }

    async def _complete(
        self,
        prompt: str,
        block_name: str,
        event_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> str:
        try:
            import litellm
        except ImportError as exc:
            raise RuntimeError(f"Memory generation unavailable for block '{block_name}'") from exc

        def _run_completion() -> Any:
            return litellm.completion(
                model=litellm_model_name(self.config.memory_model),
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

        try:
            llm_started = asyncio.get_running_loop().time()
            response = await asyncio.to_thread(_run_completion)
            llm_elapsed_ms = (asyncio.get_running_loop().time() - llm_started) * 1000.0
            content = response.choices[0].message.content
            usage_obj = getattr(response, "usage", None)
            prompt_tokens = int(getattr(usage_obj, "prompt_tokens", None) or getattr(usage_obj, "input_tokens", 0) or 0)
            completion_tokens = int(
                getattr(usage_obj, "completion_tokens", None) or getattr(usage_obj, "output_tokens", 0) or 0
            )
            model = self.config.memory_model
            context_window = MODEL_CONTEXT_WINDOWS.get(model)
            estimate = estimate_cost_usd(model, prompt_tokens, completion_tokens)
            logger = logging.getLogger(__name__)
            logger.info(
                "memory llm_call block=%s model=%s prompt=%s completion=%s cost=$%.6f latency=%.0fms",
                block_name,
                model,
                prompt_tokens,
                completion_tokens,
                estimate.total_cost_usd,
                llm_elapsed_ms,
            )
            usage_event = {
                "type": "token_usage",
                "session_stage": "memory",
                "memory_block": block_name,
                "model": model,
                "model_provider": model_provider(model),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "call_cost_usd": estimate.total_cost_usd,
                "llm_call_latency_ms": round(llm_elapsed_ms, 2),
                "context_window_tokens": context_window,
                "context_usage_ratio": (
                    round(float(prompt_tokens) / float(context_window), 4)
                    if context_window and context_window > 0
                    else None
                ),
            }
            if event_callback is not None:
                await event_callback(usage_event)
            return str(content or "").strip()
        except Exception as exc:
            raise RuntimeError(f"Memory generation failed for block '{block_name}'") from exc


def constraints_by_id(constraints: list[ParsedConstraint]) -> dict[str, ParsedConstraint]:
    return {c.id: c for c in constraints}


def dump_constraints(constraints: list[ParsedConstraint]) -> str:
    if not constraints:
        return "[]"
    return json.dumps([asdict(c) for c in constraints], indent=2)
