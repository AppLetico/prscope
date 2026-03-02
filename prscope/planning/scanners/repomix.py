"""
RepomixScanner — uses the repomix CLI to pack the repo into a single context.

repomix packs all source files into one structured text optimised for LLMs,
respecting .gitignore, filtering binaries, and adding file separators.

Install:  npm install -g repomix    (or use npx — no install needed)
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Max characters to pass to the LLM (~3-4k tokens)
_MAX_CHARS = 10_000


class RepomixScanner:
    """
    Shells out to the repomix CLI to produce a full-repo context dump.

    repomix handles binary filtering, .gitignore respect, and outputs a
    clean structured text that LLMs understand very well.

    Falls back to GrepScanner if repomix/npx is not installed.
    """

    name = "repomix"

    def is_available(self) -> bool:
        return shutil.which("repomix") is not None or shutil.which("npx") is not None

    def build_context(self, repo_path: Path, profile: dict[str, Any]) -> str:
        if not self.is_available():
            logger.warning(
                "RepomixScanner: neither repomix nor npx found. "
                "Install with: npm install -g repomix  — falling back."
            )
            from .grep import GrepScanner
            return GrepScanner().build_context(repo_path, profile)

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            out_path = Path(tmp.name)

        try:
            # Prefer global repomix; fall back to npx (downloads on demand)
            cmd_base = ["repomix"] if shutil.which("repomix") else ["npx", "--yes", "repomix"]
            cmd = [
                *cmd_base,
                "--output", str(out_path),
                "--style", "plain",      # plain text, not XML
                "--no-file-summary",     # skip the summary header block
                str(repo_path),
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(repo_path),
            )
            if result.returncode != 0:
                logger.warning(
                    "RepomixScanner: repomix exited %d: %s — falling back.",
                    result.returncode,
                    result.stderr[:500],
                )
                from .grep import GrepScanner
                return GrepScanner().build_context(repo_path, profile)

            raw = out_path.read_text(encoding="utf-8", errors="ignore")

        except subprocess.TimeoutExpired:
            logger.warning("RepomixScanner: timed out — falling back.")
            from .grep import GrepScanner
            return GrepScanner().build_context(repo_path, profile)
        except Exception as exc:
            logger.warning("RepomixScanner: unexpected error (%s) — falling back.", exc)
            from .grep import GrepScanner
            return GrepScanner().build_context(repo_path, profile)
        finally:
            out_path.unlink(missing_ok=True)

        if not raw.strip():
            from .grep import GrepScanner
            return GrepScanner().build_context(repo_path, profile)

        # Prepend header and cap to avoid prompt bloat
        header = f"# Repository Contents: {repo_path.name}\n*(packed by repomix)*\n\n"
        context = header + raw
        if len(context) > _MAX_CHARS:
            context = context[:_MAX_CHARS] + "\n\n*(output truncated — increase repomix token budget if needed)*"

        return context
