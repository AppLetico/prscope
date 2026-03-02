"""
RepoMapScanner — uses aider's tree-sitter based RepoMap.

Produces a structured symbol map of the codebase (classes, functions,
interfaces) extracted via tree-sitter without needing an LLM call.
This is the same technique Cursor uses to understand code structure.

Install:  pip install aider-chat
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Binary/generated extensions to exclude from the map
_SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".pdf", ".zip", ".gz", ".tar", ".bz2", ".wasm", ".dylib",
    ".so", ".dll", ".exe", ".bin", ".ttf", ".woff", ".woff2",
    ".db", ".sqlite", ".sqlite3", ".db-wal", ".db-shm",
    ".mp4", ".mp3", ".mov", ".wav",
    ".lock", ".min.js", ".min.css",
}
_SKIP_DIRS = {
    ".git", "node_modules", ".venv", "venv", "__pycache__",
    ".prscope", "dist", "build", ".next", "coverage",
}


class RepoMapScanner:
    """
    Uses aider's RepoMap (tree-sitter) to extract a structural symbol map.

    Gives the LLM accurate knowledge of functions, classes, and their
    relationships — significantly better than a file-tree dump for large
    TypeScript/Python/Go codebases.
    """

    name = "repomap"

    def is_available(self) -> bool:
        try:
            from aider.repomap import RepoMap  # noqa: F401
            return True
        except ImportError:
            return False

    def build_context(self, repo_path: Path, profile: dict[str, Any]) -> str:
        try:
            from aider.repomap import RepoMap
        except ImportError:
            logger.warning(
                "RepoMapScanner: aider-chat not installed. "
                "Run: pip install aider-chat  — falling back to file-tree summary."
            )
            from .grep import GrepScanner
            return GrepScanner().build_context(repo_path, profile)

        # Collect all candidate source files
        source_files: list[str] = []
        for path in repo_path.rglob("*"):
            if not path.is_file():
                continue
            # Skip ignored dirs
            parts = path.relative_to(repo_path).parts
            if any(part in _SKIP_DIRS for part in parts):
                continue
            # Skip binary extensions
            if path.suffix.lower() in _SKIP_EXTENSIONS:
                continue
            source_files.append(str(path))

        if not source_files:
            logger.warning("RepoMapScanner: no source files found.")
            from .grep import GrepScanner
            return GrepScanner().build_context(repo_path, profile)

        try:
            rm = RepoMap(
                root=str(repo_path),
                main_model=None,  # no LLM needed — uses tree-sitter only
                io=None,
                verbose=False,
            )
            repo_map_text = rm.get_repo_map(
                chat_files=[],
                other_files=source_files,
            )
        except Exception as exc:
            logger.warning("RepoMapScanner: RepoMap failed (%s) — falling back.", exc)
            from .grep import GrepScanner
            return GrepScanner().build_context(repo_path, profile)

        if not repo_map_text or not repo_map_text.strip():
            from .grep import GrepScanner
            return GrepScanner().build_context(repo_path, profile)

        # Prepend a header and cap size
        header = f"# Repository Symbol Map: {repo_path.name}\n\n"
        context = header + repo_map_text
        # Hard cap at 12 000 chars to stay well within prompt budget
        if len(context) > 12_000:
            context = context[:12_000] + "\n\n*(map truncated)*"

        return context
