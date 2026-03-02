"""
GrepScanner — default scanner backend.

Builds context from the repo's file tree + README using pure Python/regex.
No extra dependencies required.  This matches the pre-scanner behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class GrepScanner:
    name = "grep"

    def is_available(self) -> bool:
        return True  # always available — only stdlib used

    def build_context(self, repo_path: Path, profile: dict[str, Any]) -> str:
        file_tree = profile.get("file_tree", {})
        files: list[str] = file_tree.get("files", [])
        dirs: list[str] = file_tree.get("directories", [])
        extensions: dict[str, int] = file_tree.get("extensions", {})
        readme: str = (profile.get("readme") or "")[:2000]
        import_stats: dict[str, Any] = profile.get("import_stats", {})

        top_ext = sorted(extensions.items(), key=lambda x: x[1], reverse=True)[:10]
        ext_summary = ", ".join(f"{ext}({n})" for ext, n in top_ext)

        top_imports = sorted(import_stats.items(), key=lambda x: x[1], reverse=True)[:15]
        dep_summary = ", ".join(f"{pkg}({n})" for pkg, n in top_imports)

        lines: list[str] = [
            f"# Repository: {repo_path.name}",
            f"**Path:** `{repo_path}`",
            f"**Total files:** {file_tree.get('total_files', len(files))}",
            f"**Top extensions:** {ext_summary or 'n/a'}",
            f"**Top dependencies:** {dep_summary or 'n/a'}",
            "",
            "## Directory Structure",
        ]
        for d in dirs[:60]:
            lines.append(f"- `{d}/`")
        if len(dirs) > 60:
            lines.append(f"- *(+{len(dirs) - 60} more)*")

        ranked_files = sorted(files, key=self._rank_file_path, reverse=True)

        lines += ["", "## Source Files (sample)"]
        for f in ranked_files[:120]:
            lines.append(f"- `{f}`")
        if len(ranked_files) > 120:
            lines.append(f"- *(+{len(ranked_files) - 120} more)*")

        if readme.strip():
            lines += ["", "## README", readme]

        return "\n".join(lines)

    @staticmethod
    def _rank_file_path(path: str) -> int:
        lower = path.lower()
        score = 0
        if lower.startswith(("src/", "app/", "server/", "backend/", "api/")):
            score += 5
        if any(token in lower for token in ("/test/", "/tests/", ".test.", ".spec.")):
            score += 3
        if lower.endswith((".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java")):
            score += 4
        if lower.endswith((".yml", ".yaml", ".toml", ".json")):
            score += 1
        if any(token in lower for token in ("/dist/", "/build/", "/node_modules/", "/coverage/")):
            score -= 8
        score -= max(0, lower.count("/") - 3)
        return score
