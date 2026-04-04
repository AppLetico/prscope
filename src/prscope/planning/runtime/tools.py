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
from .read_file_range import read_file_slice
from .ripgrep_search import grep_code_python, grep_code_ripgrep, resolve_grep_backend
from .tool_arg_coerce import (
    clamp_int,
    coerce_bool,
    coerce_int_optional,
    coerce_non_negative_int,
    coerce_positive_int,
)

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
            "description": (
                "Read a text file in the repository. By default reads from the start of the file for up to "
                "`max_lines` (default 200). For large files, prefer windowed reads: set `around_line` to a line "
                "number of interest (e.g. from grep) and `radius` for lines before/after; or set `start_line` "
                "with `max_lines` to read sequential chunks."
            ),
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
                "default 'content' returns matching lines for evidence. "
                "Optional 'context' adds lines before/after each match (ripgrep only). "
                "Optional 'offset' skips the first N rows for pagination."
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
                    "context": {
                        "type": "integer",
                        "description": "Lines of context before/after each match (ripgrep -C). Python fallback ignores context.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Skip the first N result rows (matches + context lines) before applying head_limit/max_results.",
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
                "Uses Python glob semantics: brace expansion is NOT supported (patterns like "
                "'**/*.{py,ts}' match literally and usually return zero files). Use **/*.py, "
                "**/*.ts, or separate glob_files calls per extension. "
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
        st = safe.stat()
        if st.st_size > int(self.tools_config.read_file_max_file_bytes):
            raise ToolSafetyError(
                f"File too large ({st.st_size} bytes; max planning.tools.read_file_max_file_bytes="
                f"{self.tools_config.read_file_max_file_bytes}). "
                "Use start_line or around_line for a window, or grep_code to locate content."
            )
        rel = str(safe.relative_to(self.repo_root))
        result = read_file_slice(
            safe,
            max_lines=max_lines,
            start_line=start_line,
            around_line=around_line,
            radius=radius,
            fast_path_max_bytes=int(self.tools_config.read_file_fast_path_max_bytes),
        )
        cap = max(1, int(self.tools_config.read_file_max_output_chars))
        content = result.content
        content_truncated = False
        note: str | None = None
        if len(content) > cap:
            content = self._smart_truncate(content, cap)
            content_truncated = True
            note = (
                "content was truncated to read_file_max_output_chars; narrow max_lines/radius or use grep_code "
                "for very long single lines."
            )
        with self._access_lock:
            self.accessed_paths.add(rel)
            self.read_history[rel] = {
                "line_count": result.line_count,
                "file_size_bytes": result.file_size_bytes,
            }
        out: dict[str, Any] = {
            "path": rel,
            "truncated": result.truncated,
            "line_count": result.line_count,
            "file_size_bytes": result.file_size_bytes,
            "start_line": result.start_line,
            "end_line": result.end_line,
            "content": content,
        }
        if content_truncated:
            out["content_truncated"] = True
        if note:
            out["note"] = note
        return out

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
        brace_note: str | None = None
        if "{" in pattern and "," in pattern:
            brace_note = (
                "Python glob does not expand brace groups (e.g. *.{py,go}). "
                "Use **/*.py, **/*.go, or separate glob_files calls."
            )
        payload: dict[str, Any] = {
            "pattern": pattern,
            "path": str(base.relative_to(self.repo_root)),
            "results": out,
            "count": len(out),
            "truncated": truncated,
        }
        if brace_note:
            payload["note"] = brace_note
        return payload

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
        context: int | None = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        if output_mode not in {"content", "files_with_matches"}:
            raise ToolSafetyError(f"Invalid output_mode: {output_mode}")
        limit = int(head_limit) if head_limit is not None else int(max_results)
        if limit < 1:
            limit = 1
        ctx = max(0, int(context)) if context is not None else 0
        off = max(0, int(offset))

        root = self._safe_path(path)
        self._enforce_path_allowlist("grep_code", self._repo_rel_posix(root))
        cfg = self.tools_config
        backend = resolve_grep_backend(cfg.grep_backend, cfg.ripgrep_command)

        matches: list[dict[str, Any]] = []
        used_backend = backend
        rg_note: str | None = None
        result_truncated = False
        # Python scan: content mode needs enough rows for offset+page; path mode needs many rows before dedupe.
        _py_match_cap = min(off + limit + 1, 100_000)
        _py_path_collect_cap = 100_000

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
            results, err, truncated_rg = grep_code_ripgrep(
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
                context_lines=ctx,
                offset=off,
            )
            if err:
                logger.warning("grep_code ripgrep: %s — using Python fallback", err)
                rg_note = err
                used_backend = "python"
                try:
                    if output_mode == "files_with_matches":
                        matches_full = grep_code_python(
                            repo_root=self.repo_root,
                            search_root=root,
                            pattern=pattern,
                            max_results=_py_path_collect_cap,
                            ignored_dir_parts=frozenset(IGNORED_DIRS),
                            binary_suffixes=frozenset(BINARY_EXTENSIONS),
                            case_insensitive=case_insensitive,
                        )
                        matches_full = _as_files_with_matches(matches_full)
                        result_truncated = len(matches_full) > off + limit or len(matches_full) >= _py_path_collect_cap
                    else:
                        matches_full = grep_code_python(
                            repo_root=self.repo_root,
                            search_root=root,
                            pattern=pattern,
                            max_results=_py_match_cap,
                            ignored_dir_parts=frozenset(IGNORED_DIRS),
                            binary_suffixes=frozenset(BINARY_EXTENSIONS),
                            case_insensitive=case_insensitive,
                        )
                        result_truncated = len(matches_full) >= _py_match_cap
                except ValueError as exc:
                    raise ToolSafetyError(str(exc)) from exc
                matches = matches_full[off : off + limit]
                if ctx > 0 or off > 0:
                    extra = (
                        "Python fallback: 'context' is ignored; "
                        "see docs for offset semantics (content vs files_with_matches)."
                    )
                    rg_note = f"{rg_note} {extra}" if rg_note else extra
                if glob or type_tag:
                    extra = "Python fallback ignores glob/type filters."
                    rg_note = f"{rg_note} {extra}" if rg_note else extra
            else:
                matches = results
                result_truncated = truncated_rg
        else:
            if glob or type_tag:
                rg_note = (
                    "Python grep ignores glob/type filters; install ripgrep or set planning.tools.grep_backend=ripgrep."
                )
            try:
                if output_mode == "files_with_matches":
                    matches_full = grep_code_python(
                        repo_root=self.repo_root,
                        search_root=root,
                        pattern=pattern,
                        max_results=_py_path_collect_cap,
                        ignored_dir_parts=frozenset(IGNORED_DIRS),
                        binary_suffixes=frozenset(BINARY_EXTENSIONS),
                        case_insensitive=case_insensitive,
                    )
                    matches_full = _as_files_with_matches(matches_full)
                    result_truncated = len(matches_full) > off + limit or len(matches_full) >= _py_path_collect_cap
                else:
                    matches_full = grep_code_python(
                        repo_root=self.repo_root,
                        search_root=root,
                        pattern=pattern,
                        max_results=_py_match_cap,
                        ignored_dir_parts=frozenset(IGNORED_DIRS),
                        binary_suffixes=frozenset(BINARY_EXTENSIONS),
                        case_insensitive=case_insensitive,
                    )
                    result_truncated = len(matches_full) >= _py_match_cap
            except ValueError as exc:
                raise ToolSafetyError(str(exc)) from exc
            matches = matches_full[off : off + limit]
            if ctx > 0:
                extra = "Python grep ignores 'context' (use ripgrep for -C)."
                rg_note = f"{rg_note} {extra}" if rg_note else extra

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
        if result_truncated:
            payload["truncated"] = True
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
                max_entries=coerce_positive_int(parsed.arguments.get("max_entries"), default=200),
            )
        elif parsed.name == "read_file":
            rad = coerce_positive_int(parsed.arguments.get("radius"), default=80, minimum=1)
            rad = clamp_int(rad, 1, 500)
            result = self.read_file(
                path=str(parsed.arguments.get("path", "")),
                max_lines=coerce_positive_int(parsed.arguments.get("max_lines"), default=200, minimum=1),
                start_line=coerce_int_optional(parsed.arguments.get("start_line")),
                around_line=coerce_int_optional(parsed.arguments.get("around_line")),
                radius=rad,
            )
        elif parsed.name == "grep_code":
            args = parsed.arguments
            hl = args.get("head_limit")
            om = str(args.get("output_mode") or "content").strip().lower()
            if om not in {"content", "files_with_matches"}:
                om = "content"
            ctx = args.get("context")
            ctx_i = coerce_int_optional(ctx)
            if ctx_i is not None:
                ctx_i = clamp_int(ctx_i, 0, 200)
            ofs = coerce_non_negative_int(args.get("offset"), default=0)
            result = self.grep_code(
                pattern=str(args.get("pattern", "")),
                path=args.get("path"),
                max_results=coerce_positive_int(args.get("max_results"), default=40, minimum=1),
                head_limit=coerce_int_optional(hl),
                glob=str(args.get("glob") or "").strip() or None,
                type_tag=str(args.get("type") or "").strip() or None,
                case_insensitive=coerce_bool(args.get("case_insensitive"), default=False),
                output_mode=om,
                context=ctx_i,
                offset=ofs,
            )
        elif parsed.name == "glob_files":
            gr = parsed.arguments.get("max_results")
            if gr is None:
                gr_coerced: int | None = None
            else:
                gr_coerced = coerce_int_optional(gr)
            result = self.glob_files(
                pattern=str(parsed.arguments.get("pattern", "")),
                path=parsed.arguments.get("path"),
                max_results=gr_coerced,
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

    def _effective_tool_result_max_chars(self, tool_name: str) -> int:
        override = self.tools_config.tool_result_max_chars_by_tool.get(tool_name)
        if override is not None:
            return max(1024, int(override))
        return max(1024, int(self.tools_config.tool_result_max_chars))

    def _format_result_payload(self, call_id: str, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(payload, ensure_ascii=False)
        max_chars = self._effective_tool_result_max_chars(tool_name)
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
            "note": (
                f"Tool result stored as artifact at repo-relative path `{stored_at}`. "
                "Call read_file with that exact path to load the full JSON before citing file contents."
            ),
        }


def extract_file_references(text: str) -> set[str]:
    refs = set(re.findall(r"`([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)`", text))
    return {ref for ref in refs if "/" in ref}
