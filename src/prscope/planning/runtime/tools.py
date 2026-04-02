"""
Planning runtime tool definitions and sandboxed execution.
"""

from __future__ import annotations

import glob as glob_std
import json
import logging
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from ...config import PlanningToolsConfig
from .ripgrep_search import grep_code_python, grep_code_ripgrep, resolve_grep_backend

logger = logging.getLogger(__name__)

BLOCKED_PATTERNS = {".env", "id_rsa", "credentials", "token", ".secret"}
IGNORED_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".prscope"}

# File extensions that are always binary/non-text — skip entirely in search
BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".ico",
    ".pdf",
    ".zip",
    ".gz",
    ".tar",
    ".bz2",
    ".xz",
    ".7z",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".db-wal",
    ".db-shm",
    ".wasm",
    ".dylib",
    ".so",
    ".dll",
    ".exe",
    ".bin",
    ".ttf",
    ".woff",
    ".woff2",
    ".eot",
    ".mp4",
    ".mp3",
    ".mov",
    ".avi",
    ".wav",
    ".lock",  # package lock files (huge, unreadable)
}


class ToolSafetyError(RuntimeError):
    """Raised when a tool invocation violates sandbox policy."""


def _normalize_repo_rel(rel: str) -> str:
    if not rel or str(rel).strip() in {".", ""}:
        return "."
    return Path(str(rel)).as_posix().strip("/")


def _rel_under_prefix(rel_norm: str, prefix_norm: str) -> bool:
    if prefix_norm in {".", ""}:
        return True
    if rel_norm == prefix_norm:
        return True
    return rel_norm.startswith(prefix_norm + "/")


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
            "description": (
                "Search the repository for a regex pattern. Uses ripgrep when available "
                "(planning.tools.grep_backend) for speed and .gitignore-aware matching; "
                "falls back to a Python scan otherwise. "
                "Use output_mode 'files_with_matches' to list only paths (smaller payload); "
                "default 'content' returns matching lines for evidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "max_results": {"type": "integer"},
                    "head_limit": {"type": "integer"},
                    "glob": {
                        "type": "string",
                        "description": 'Optional rg --glob filter (e.g. "*.py", "*.{ts,tsx}")',
                    },
                    "type": {
                        "type": "string",
                        "description": "Optional ripgrep --type (e.g. py, rust, js)",
                    },
                    "case_insensitive": {"type": "boolean"},
                    "output_mode": {
                        "type": "string",
                        "enum": ["content", "files_with_matches"],
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob_files",
            "description": (
                "Find files by glob pattern under a directory (recursive patterns like **/*.py). "
                "Complements list_files (single-directory listing) for wide file discovery."
            ),
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
            "description": "Ask the user a clarification question that CANNOT be answered by scanning the codebase. Only call this AFTER you have used list_files, glob_files, read_file, and grep_code to exhaust relevant file-based evidence. Never ask about directory structure, file locations, test frameworks, or naming conventions without first calling list_files('.') to inspect the root.",
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
        tools_config: PlanningToolsConfig | None = None,
    ):
        self.repo_root = repo_root.resolve()
        self.tools_config = tools_config or PlanningToolsConfig()
        self.accessed_paths: set[str] = set()
        self.read_history: dict[str, dict[str, int]] = {}
        self._access_lock = threading.Lock()
        self.clarification_callback = clarification_callback
        self.memory_block_callback = memory_block_callback
        self.session_id: str | None = None
        self.artifact_root = self.repo_root / ".prscope" / "tool-results"
        self._last_ttl_cleanup_at: datetime | None = None

    def set_session(self, session_id: str) -> None:
        # Avoid cross-session evidence leakage while preserving continuity
        # within the same session across discovery -> drafting transitions.
        if self.session_id is not None and self.session_id != session_id:
            with self._access_lock:
                self.accessed_paths.clear()
                self.read_history.clear()
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

    def _repo_rel_posix(self, safe: Path) -> str:
        rel = safe.relative_to(self.repo_root).as_posix()
        if rel in {".", ""}:
            return "."
        return rel

    def _enforce_path_allowlist(self, tool_name: str, rel_posix: str) -> None:
        cfg = self.tools_config.path_allowlist
        if tool_name not in cfg:
            return
        rules = cfg[tool_name]
        if not rules:
            raise ToolSafetyError(f"Tool {tool_name} is disabled by planning.tools.path_allowlist (empty prefix list)")
        rel_norm = _normalize_repo_rel(rel_posix)
        for raw in rules:
            pnorm = _normalize_repo_rel(str(raw))
            if _rel_under_prefix(rel_norm, pnorm):
                return
        raise ToolSafetyError(
            f"Path not allowed for {tool_name} by planning.tools.path_allowlist: {rel_posix!r} "
            f"(allowed prefixes: {rules!r})"
        )

    def list_files(self, path: str | None = None, max_entries: int = 200) -> dict[str, Any]:
        safe = self._safe_path(path)
        self._enforce_path_allowlist("list_files", self._repo_rel_posix(safe))
        if not safe.exists() or not safe.is_dir():
            raise ToolSafetyError(f"Directory not found: {path or '.'}")
        entries = []
        for child in sorted(safe.iterdir(), key=lambda p: p.name)[:max_entries]:
            rel = str(child.relative_to(self.repo_root))
            entries.append({"path": rel, "type": "dir" if child.is_dir() else "file"})
            with self._access_lock:
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
        self._enforce_path_allowlist("read_file", self._repo_rel_posix(safe))
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
        file_size_bytes = len(raw_text.encode("utf-8"))
        with self._access_lock:
            self.accessed_paths.add(rel)
            self.read_history[rel] = {
                "line_count": len(lines),
                "file_size_bytes": file_size_bytes,
            }
        return {
            "path": rel,
            "truncated": start_idx > 0 or end_idx < len(lines),
            "line_count": len(lines),
            "file_size_bytes": file_size_bytes,
            "start_line": start_idx + 1,
            "end_line": end_idx,
            "content": "\n".join(snippet),
        }

    def glob_files(
        self,
        pattern: str,
        path: str | None = None,
        max_results: int | None = None,
    ) -> dict[str, Any]:
        """Find files matching a glob pattern under a sandboxed directory."""
        max_r = max_results if max_results is not None else self.tools_config.glob_max_results
        base = self._safe_path(path)
        self._enforce_path_allowlist("glob_files", self._repo_rel_posix(base))
        if not base.is_dir():
            raise ToolSafetyError(f"Directory not found: {path or '.'}")
        matches = glob_std.glob(str(base / pattern), recursive=True)
        out: list[str] = []
        truncated = False
        for abs_path in sorted(matches):
            if len(out) >= max_r:
                truncated = True
                break
            p = Path(abs_path)
            if not p.is_file():
                continue
            try:
                rel = str(p.relative_to(self.repo_root))
            except ValueError:
                continue
            if any(part in IGNORED_DIRS for part in Path(rel).parts):
                continue
            if p.suffix.lower() in BINARY_EXTENSIONS:
                continue
            with self._access_lock:
                self.accessed_paths.add(rel)
            out.append(rel)
        return {
            "pattern": pattern,
            "path": str(base.relative_to(self.repo_root)),
            "results": out,
            "count": len(out),
            "truncated": truncated,
        }

    def grep_code(
        self,
        pattern: str,
        path: str | None = None,
        max_results: int = 40,
        *,
        head_limit: int | None = None,
        glob: str | None = None,
        type_tag: str | None = None,
        case_insensitive: bool = False,
        output_mode: str = "content",
    ) -> dict[str, Any]:
        if output_mode not in {"content", "files_with_matches"}:
            raise ToolSafetyError(f"Invalid output_mode: {output_mode}")
        limit = int(head_limit) if head_limit is not None else int(max_results)
        if limit < 1:
            limit = 1

        root = self._safe_path(path)
        self._enforce_path_allowlist("grep_code", self._repo_rel_posix(root))
        cfg = self.tools_config
        backend = resolve_grep_backend(cfg.grep_backend, cfg.ripgrep_command)

        matches: list[dict[str, Any]] = []
        used_backend = backend
        rg_note: str | None = None

        def _as_files_with_matches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            seen: set[str] = set()
            slim: list[dict[str, Any]] = []
            for item in rows:
                pth = str(item.get("path", "")).strip()
                if pth and pth not in seen:
                    seen.add(pth)
                    slim.append({"path": pth, "line": 0, "text": ""})
            return slim

        if backend == "ripgrep":
            results, err = grep_code_ripgrep(
                repo_root=self.repo_root,
                search_root=root,
                pattern=pattern,
                max_results=limit,
                command=cfg.ripgrep_command,
                timeout_seconds=cfg.ripgrep_timeout_seconds,
                max_columns=cfg.ripgrep_max_columns,
                respect_ignore_files=cfg.ripgrep_respect_ignore_files,
                output_mode=output_mode,
                glob=glob,
                type_tag=type_tag,
                case_insensitive=case_insensitive,
            )
            if err:
                logger.warning("grep_code ripgrep: %s — using Python fallback", err)
                rg_note = err
                used_backend = "python"
                try:
                    matches = grep_code_python(
                        repo_root=self.repo_root,
                        search_root=root,
                        pattern=pattern,
                        max_results=limit,
                        ignored_dir_parts=frozenset(IGNORED_DIRS),
                        binary_suffixes=frozenset(BINARY_EXTENSIONS),
                        case_insensitive=case_insensitive,
                    )
                except ValueError as exc:
                    raise ToolSafetyError(str(exc)) from exc
                if output_mode == "files_with_matches":
                    matches = _as_files_with_matches(matches)
                if glob or type_tag:
                    extra = "Python fallback ignores glob/type filters."
                    rg_note = f"{rg_note} {extra}" if rg_note else extra
            else:
                matches = results
        else:
            if glob or type_tag:
                rg_note = (
                    "Python grep ignores glob/type filters; install ripgrep or set planning.tools.grep_backend=ripgrep."
                )
            try:
                matches = grep_code_python(
                    repo_root=self.repo_root,
                    search_root=root,
                    pattern=pattern,
                    max_results=limit,
                    ignored_dir_parts=frozenset(IGNORED_DIRS),
                    binary_suffixes=frozenset(BINARY_EXTENSIONS),
                    case_insensitive=case_insensitive,
                )
            except ValueError as exc:
                raise ToolSafetyError(str(exc)) from exc
            if output_mode == "files_with_matches":
                matches = _as_files_with_matches(matches)

        with self._access_lock:
            for item in matches:
                pth = str(item.get("path", "")).strip()
                if pth:
                    self.accessed_paths.add(pth)

        payload: dict[str, Any] = {
            "pattern": pattern,
            "output_mode": output_mode,
            "results": matches,
            "count": len(matches),
            "grep_backend": used_backend,
        }
        if rg_note:
            payload["note"] = rg_note
        return payload

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
                    int(parsed.arguments.get("start_line")) if parsed.arguments.get("start_line") is not None else None
                ),
                around_line=(
                    int(parsed.arguments.get("around_line"))
                    if parsed.arguments.get("around_line") is not None
                    else None
                ),
                radius=int(parsed.arguments.get("radius", 80)),
            )
        elif parsed.name == "grep_code":
            args = parsed.arguments
            hl = args.get("head_limit")
            om = str(args.get("output_mode") or "content").strip().lower()
            if om not in {"content", "files_with_matches"}:
                om = "content"
            result = self.grep_code(
                pattern=str(args.get("pattern", "")),
                path=args.get("path"),
                max_results=int(args.get("max_results", 40)),
                head_limit=int(hl) if hl is not None else None,
                glob=str(args.get("glob") or "").strip() or None,
                type_tag=str(args.get("type") or "").strip() or None,
                case_insensitive=bool(args.get("case_insensitive", False)),
                output_mode=om,
            )
        elif parsed.name == "glob_files":
            gr = parsed.arguments.get("max_results")
            result = self.glob_files(
                pattern=str(parsed.arguments.get("pattern", "")),
                path=parsed.arguments.get("path"),
                max_results=int(gr) if gr is not None else None,
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
        elif tool_name == "glob_files":
            results = payload.get("results", [])
            summary["match_count"] = int(payload.get("count", len(results)))
            summary["truncated"] = bool(payload.get("truncated", False))
            if isinstance(results, list):
                summary["top_matches"] = [str(p) for p in results[:5]]
        return summary

    def _format_result_payload(self, call_id: str, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(payload, ensure_ascii=False)
        max_chars = max(1024, int(self.tools_config.tool_result_max_chars))
        if len(encoded) <= max_chars:
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
