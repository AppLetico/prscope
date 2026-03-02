"""
Planning runtime tool definitions and sandboxed execution.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

BLOCKED_PATTERNS = {".env", "id_rsa", "credentials", "token", ".secret"}
IGNORED_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".prscope"}

# File extensions that are always binary/non-text — skip entirely in search
BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico",
    ".pdf", ".zip", ".gz", ".tar", ".bz2", ".xz", ".7z",
    ".db", ".sqlite", ".sqlite3", ".db-wal", ".db-shm",
    ".wasm", ".dylib", ".so", ".dll", ".exe", ".bin",
    ".ttf", ".woff", ".woff2", ".eot",
    ".mp4", ".mp3", ".mov", ".avi", ".wav",
    ".lock",  # package lock files (huge, unreadable)
}

# Max chars for a single tool result injected into the LLM conversation
TOOL_RESULT_MAX_CHARS = 8_000


class ToolSafetyError(RuntimeError):
    """Raised when a tool invocation violates sandbox policy."""


CODEBASE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories under a path in the repository",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "max_entries": {"type": "integer"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file in the repository",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_lines": {"type": "integer"},
                    "start_line": {"type": "integer"},
                    "around_line": {"type": "integer"},
                    "radius": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_code",
            "description": "Search the repository for a regex pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "max_results": {"type": "integer"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_clarification",
            "description": "Ask the user a clarification question and pause progression",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "context": {"type": "string"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_memory_block",
            "description": "Fetch an on-demand planning memory block by key",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    },
]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


class ToolExecutor:
    """Sandboxed file-system tools for planning."""

    def __init__(
        self,
        repo_root: Path,
        clarification_callback: Callable[[str, str], list[str]] | None = None,
        memory_block_callback: Callable[[str], dict[str, Any]] | None = None,
    ):
        self.repo_root = repo_root.resolve()
        self.accessed_paths: set[str] = set()
        self.read_history: dict[str, dict[str, int]] = {}
        self.clarification_callback = clarification_callback
        self.memory_block_callback = memory_block_callback
        self.session_id: str | None = None
        self.artifact_root = self.repo_root / ".prscope" / "tool-results"
        self._last_ttl_cleanup_at: datetime | None = None

    def set_session(self, session_id: str) -> None:
        self.session_id = session_id

    def maybe_cleanup_artifacts(self, max_age_days: int = 7) -> None:
        now = datetime.now(timezone.utc)
        if self._last_ttl_cleanup_at and (now - self._last_ttl_cleanup_at) < timedelta(hours=24):
            return
        self._last_ttl_cleanup_at = now
        if not self.artifact_root.exists():
            return
        cutoff = now - timedelta(days=max_age_days)
        for session_dir in self.artifact_root.iterdir():
            if not session_dir.is_dir():
                continue
            modified = datetime.fromtimestamp(session_dir.stat().st_mtime, timezone.utc)
            if modified < cutoff:
                for child in session_dir.glob("*"):
                    if child.is_file():
                        child.unlink(missing_ok=True)
                session_dir.rmdir()

    def delete_session_artifacts(self, session_id: str) -> None:
        session_dir = self.artifact_root / session_id
        if not session_dir.exists():
            return
        for child in session_dir.glob("*"):
            if child.is_file():
                child.unlink(missing_ok=True)
        session_dir.rmdir()

    def _safe_path(self, raw_path: str | None) -> Path:
        if not raw_path:
            return self.repo_root
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = (self.repo_root / candidate).resolve()
        else:
            candidate = candidate.resolve()

        try:
            candidate.relative_to(self.repo_root)
        except ValueError as exc:
            raise ToolSafetyError(f"Path escapes repo root: {raw_path}") from exc

        lower_name = candidate.name.lower()
        if any(token in lower_name for token in BLOCKED_PATTERNS):
            raise ToolSafetyError(f"Blocked sensitive file: {raw_path}")
        return candidate

    def list_files(self, path: str | None = None, max_entries: int = 200) -> dict[str, Any]:
        safe = self._safe_path(path)
        if not safe.exists() or not safe.is_dir():
            raise ToolSafetyError(f"Directory not found: {path or '.'}")
        entries = []
        for child in sorted(safe.iterdir(), key=lambda p: p.name)[:max_entries]:
            rel = str(child.relative_to(self.repo_root))
            entries.append({"path": rel, "type": "dir" if child.is_dir() else "file"})
            self.accessed_paths.add(rel)
        return {"path": str(safe.relative_to(self.repo_root)), "entries": entries}

    def read_file(
        self,
        path: str,
        max_lines: int = 200,
        start_line: int | None = None,
        around_line: int | None = None,
        radius: int = 80,
    ) -> dict[str, Any]:
        safe = self._safe_path(path)
        if not safe.exists() or not safe.is_file():
            raise ToolSafetyError(f"File not found: {path}")
        if safe.suffix.lower() in BINARY_EXTENSIONS:
            raise ToolSafetyError(f"Binary file not readable as text: {path}")
        raw_text = safe.read_text(encoding="utf-8", errors="ignore")
        lines = raw_text.splitlines()
        if around_line is not None:
            focus = max(1, int(around_line))
            win = max(1, int(radius))
            start_idx = max(0, focus - win - 1)
            end_idx = min(len(lines), focus + win)
        elif start_line is not None:
            start_idx = max(0, int(start_line) - 1)
            end_idx = min(len(lines), start_idx + max(1, int(max_lines)))
        else:
            start_idx = 0
            end_idx = min(len(lines), max(1, int(max_lines)))
        snippet = lines[start_idx:end_idx]
        rel = str(safe.relative_to(self.repo_root))
        self.accessed_paths.add(rel)
        self.read_history[rel] = {
            "line_count": len(lines),
            "file_size_bytes": len(raw_text.encode("utf-8")),
        }
        return {
            "path": rel,
            "truncated": start_idx > 0 or end_idx < len(lines),
            "line_count": len(lines),
            "file_size_bytes": self.read_history[rel]["file_size_bytes"],
            "start_line": start_idx + 1,
            "end_line": end_idx,
            "content": "\n".join(snippet),
        }

    def grep_code(
        self,
        pattern: str,
        path: str | None = None,
        max_results: int = 40,
    ) -> dict[str, Any]:
        root = self._safe_path(path)
        if root.is_file():
            candidates = [root]
        else:
            candidates = []
            for file_path in root.rglob("*"):
                if not file_path.is_file():
                    continue
                try:
                    rel_parts = file_path.relative_to(self.repo_root).parts
                except ValueError:
                    continue
                if any(part in IGNORED_DIRS for part in rel_parts):
                    continue
                candidates.append(file_path)

        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ToolSafetyError(f"Invalid regex pattern: {pattern}") from exc

        matches: list[dict[str, Any]] = []
        for file_path in candidates:
            if len(matches) >= max_results:
                break
            # Skip binary/non-text files entirely
            if file_path.suffix.lower() in BINARY_EXTENSIONS:
                continue
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            for line_num, line in enumerate(content.splitlines(), start=1):
                if regex.search(line):
                    rel = str(file_path.relative_to(self.repo_root))
                    self.accessed_paths.add(rel)
                    # Truncate individual lines to avoid binary/minified blowup
                    matches.append({"path": rel, "line": line_num, "text": line.strip()[:500]})
                    if len(matches) >= max_results:
                        break

        return {"pattern": pattern, "results": matches, "count": len(matches)}

    @staticmethod
    def _parse_tool_call(raw_call: Any) -> ToolCall:
        call_id = getattr(raw_call, "id", None) or raw_call.get("id", "tool-call")
        func = getattr(raw_call, "function", None) or raw_call.get("function", {})
        name = getattr(func, "name", None) or func.get("name")
        raw_args = getattr(func, "arguments", None) or func.get("arguments", "{}")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}
        return ToolCall(id=call_id, name=name or "", arguments=args)

    def execute(self, raw_tool_call: Any) -> dict[str, Any]:
        parsed = self._parse_tool_call(raw_tool_call)
        if parsed.name == "list_files":
            result = self.list_files(
                path=parsed.arguments.get("path"),
                max_entries=int(parsed.arguments.get("max_entries", 200)),
            )
        elif parsed.name == "read_file":
            result = self.read_file(
                path=str(parsed.arguments.get("path", "")),
                max_lines=int(parsed.arguments.get("max_lines", 200)),
                start_line=(
                    int(parsed.arguments.get("start_line"))
                    if parsed.arguments.get("start_line") is not None
                    else None
                ),
                around_line=(
                    int(parsed.arguments.get("around_line"))
                    if parsed.arguments.get("around_line") is not None
                    else None
                ),
                radius=int(parsed.arguments.get("radius", 80)),
            )
        elif parsed.name == "grep_code":
            result = self.grep_code(
                pattern=str(parsed.arguments.get("pattern", "")),
                path=parsed.arguments.get("path"),
                max_results=int(parsed.arguments.get("max_results", 40)),
            )
        elif parsed.name == "ask_clarification":
            question = str(parsed.arguments.get("question", "")).strip()
            context = str(parsed.arguments.get("context", "")).strip()
            if not question:
                raise ToolSafetyError("ask_clarification requires non-empty question")
            if self.clarification_callback is None:
                result = {
                    "question": question,
                    "context": context,
                    "answers": [],
                    "timed_out": True,
                }
            else:
                answers = self.clarification_callback(question, context)
                result = {
                    "question": question,
                    "context": context,
                    "answers": answers,
                    "timed_out": len(answers) == 0,
                }
        elif parsed.name == "get_memory_block":
            key = str(parsed.arguments.get("key", "")).strip().lower()
            if not key:
                raise ToolSafetyError("get_memory_block requires non-empty key")
            if self.memory_block_callback is None:
                raise ToolSafetyError("get_memory_block unavailable in this context")
            result = self.memory_block_callback(key)
        else:
            raise ToolSafetyError(f"Unknown tool: {parsed.name}")

        return {
            "tool_call_id": parsed.id,
            "name": parsed.name,
            "result": self._format_result_payload(parsed.id, parsed.name, result),
        }

    @staticmethod
    def _smart_truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        marker = "\n...[truncated]...\n"
        half = max(1, (max_chars - len(marker)) // 2)
        return text[:half] + marker + text[-half:]

    def _write_artifact(self, call_id: str, payload: dict[str, Any]) -> str:
        safe_call_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", call_id) or "tool-call"
        session = self.session_id or "unknown-session"
        target_dir = self.artifact_root / session
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{safe_call_id}.json"
        file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(file_path.relative_to(self.repo_root))

    @staticmethod
    def _result_summary(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        if tool_name == "grep_code":
            results = payload.get("results", [])
            summary["match_count"] = int(payload.get("count", len(results)))
            if isinstance(results, list):
                paths = []
                for item in results:
                    path = str(item.get("path", "")).strip() if isinstance(item, dict) else ""
                    if path and path not in paths:
                        paths.append(path)
                    if len(paths) >= 5:
                        break
                summary["top_matches"] = paths
        elif tool_name == "read_file":
            summary["line_count"] = int(payload.get("line_count", 0) or 0)
            summary["path"] = str(payload.get("path", ""))
        elif tool_name == "list_files":
            entries = payload.get("entries", [])
            summary["entry_count"] = len(entries) if isinstance(entries, list) else 0
            summary["path"] = str(payload.get("path", ""))
        return summary

    def _format_result_payload(self, call_id: str, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(payload, ensure_ascii=False)
        if len(encoded) <= TOOL_RESULT_MAX_CHARS:
            return {
                "result": payload,
                "note": "Tool result injected. If referencing these files, inspect them explicitly.",
            }
        stored_at = self._write_artifact(call_id, payload)
        summary = self._result_summary(tool_name, payload)
        return {
            "stored_at": stored_at,
            "truncated": True,
            **summary,
            "note": "Tool result stored as artifact. Read it with read_file before referencing files.",
        }


def extract_file_references(text: str) -> set[str]:
    refs = set(re.findall(r"`([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)`", text))
    return {ref for ref in refs if "/" in ref}
