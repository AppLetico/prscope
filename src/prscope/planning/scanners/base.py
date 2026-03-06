"""
ScannerBackend protocol — all codebase scanner backends implement this.

A scanner's job is to produce a *context string* describing the repository
structure.  That string is injected into the LLM system prompt as static
memory before any planning conversation starts.

Dynamic tool calls (list_files / read_file / grep_code) are handled
separately by ToolExecutor and are NOT the scanner's responsibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ScannerBackend(Protocol):
    """Protocol every scanner backend must satisfy."""

    #: Short identifier used in config (e.g. "grep", "repomap", "repomix")
    name: str

    def is_available(self) -> bool:
        """Return True if this backend's dependencies are installed/accessible."""
        ...

    def build_context(self, repo_path: Path, profile: dict[str, Any]) -> str:
        """
        Produce a codebase context string for injection into the LLM system prompt.

        Args:
            repo_path: Absolute path to the repository root.
            profile:   Pre-built profile dict from prscope.profile.build_profile()
                       (file tree, readme, import stats, git sha, etc.).
                       Backends may use or ignore this.

        Returns:
            A markdown string describing the repository.  Should be under
            ~8 000 characters to avoid prompt bloat.
        """
        ...
