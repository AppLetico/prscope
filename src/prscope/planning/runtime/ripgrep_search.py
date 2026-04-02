"""
Ripgrep-backed codebase search with a pure-Python fallback.

Mirrors common agent-harness behavior (bounded output, ignore files, timeouts)
without requiring ripgrep at install time.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

VCS_GLOB_EXCLUDES = (".git", ".svn", ".hg", ".bzr", ".jj")


def ripgrep_binary_available(command: str = "rg") -> bool:
    return shutil.which(command) is not None


def grep_code_python(
    *,
    repo_root: Path,
    search_root: Path,
    pattern: str,
    max_results: int,
    ignored_dir_parts: frozenset[str],
    binary_suffixes: frozenset[str],
    case_insensitive: bool = False,
) -> list[dict[str, Any]]:
    """Line-oriented regex search (legacy behavior)."""
    if search_root.is_file():
        candidates = [search_root]
    else:
        candidates = []
        for file_path in search_root.rglob("*"):
            if not file_path.is_file():
                continue
            try:
                rel_parts = file_path.relative_to(repo_root).parts
            except ValueError:
                continue
            if any(part in ignored_dir_parts for part in rel_parts):
                continue
            candidates.append(file_path)

    flags = re.IGNORECASE if case_insensitive else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        raise ValueError(f"Invalid regex pattern: {pattern}") from exc

    matches: list[dict[str, Any]] = []
    for file_path in candidates:
        if len(matches) >= max_results:
            break
        if file_path.suffix.lower() in binary_suffixes:
            continue
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        for line_num, line in enumerate(content.splitlines(), start=1):
            if regex.search(line):
                rel = str(file_path.relative_to(repo_root))
                matches.append({"path": rel, "line": line_num, "text": line.strip()[:500]})
                if len(matches) >= max_results:
                    break

    return matches


def _path_text(path_obj: Any) -> str:
    if isinstance(path_obj, dict):
        return str(path_obj.get("text", "") or "")
    if path_obj is None:
        return ""
    return str(path_obj)


def normalize_repo_relative_path(path: str) -> str:
    """Strip leading ./ from ripgrep output so paths match list_files/read_file conventions."""
    p = str(path).strip().replace("\\", "/")
    if p.startswith("./"):
        return p[2:]
    return p


def grep_code_ripgrep(
    *,
    repo_root: Path,
    search_root: Path,
    pattern: str,
    max_results: int,
    command: str,
    timeout_seconds: float,
    max_columns: int,
    respect_ignore_files: bool,
    output_mode: str,
    glob: str | None,
    type_tag: str | None,
    case_insensitive: bool,
) -> tuple[list[dict[str, Any]], str | None]:
    """
    Run ripgrep in the repo sandbox. Returns (results, error_message).

    error_message is set on timeout or rg failure (non-0/1 exit).
    """
    try:
        rel_root = search_root.resolve().relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError("search_root must be under repo_root") from exc

    rel_arg = "." if rel_root == Path(".") else str(rel_root)

    args: list[str] = [
        command,
        "--max-columns",
        str(max_columns),
        "--max-columns-preview",
    ]

    for d in VCS_GLOB_EXCLUDES:
        args.extend(["--glob", f"!{d}/**"])

    if not respect_ignore_files:
        args.append("--no-ignore")

    if case_insensitive:
        args.append("-i")

    if glob:
        for raw in glob.replace(",", " ").split():
            g = raw.strip()
            if g:
                args.extend(["--glob", g])

    if type_tag:
        args.extend(["--type", type_tag])

    if output_mode == "files_with_matches":
        args.append("-l")
    else:
        args.append("--json")

    if pattern.startswith("-"):
        args.extend(["-e", pattern])
    else:
        args.append("-e")
        args.append(pattern)

    args.append(rel_arg)

    try:
        proc = subprocess.run(
            args,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return [], f"ripgrep timed out after {timeout_seconds}s"

    if proc.returncode not in (0, 1):
        err = (proc.stderr or proc.stdout or "").strip()
        return [], f"ripgrep failed (exit {proc.returncode}): {err[:500]}"

    if output_mode == "files_with_matches":
        results: list[dict[str, Any]] = []
        for raw in (proc.stdout or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            rel = normalize_repo_relative_path(line.replace("\\", "/"))
            results.append({"path": rel, "line": 0, "text": ""})
            if len(results) >= max_results:
                break
        return results, None

    results = []
    for json_line in (proc.stdout or "").splitlines():
        if len(results) >= max_results:
            break
        line = json_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") != "match":
            continue
        data = obj.get("data") or {}
        path_raw = data.get("path")
        rel = normalize_repo_relative_path(_path_text(path_raw))
        if not rel:
            continue
        line_number = int(data.get("line_number") or 0)
        lines_obj = data.get("lines") or {}
        text = str(lines_obj.get("text", "") if isinstance(lines_obj, dict) else "").rstrip("\n")
        if len(text) > max_columns:
            text = text[:max_columns]
        results.append({"path": rel, "line": line_number, "text": text.strip()[:500]})

    return results, None


def resolve_grep_backend(requested: str, command: str) -> str:
    """Return 'ripgrep' or 'python' based on config and PATH."""
    if requested == "python":
        return "python"
    if requested == "ripgrep":
        if not ripgrep_binary_available(command):
            logger.warning("planning.tools.grep_backend=ripgrep but %s not found; using Python fallback", command)
            return "python"
        return "ripgrep"
    # auto
    return "ripgrep" if ripgrep_binary_available(command) else "python"
