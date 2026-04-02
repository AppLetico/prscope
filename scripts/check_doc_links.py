#!/usr/bin/env python3
"""
Verify relative markdown links from docs/*.md and top-level *.md point to existing files.

Usage: python scripts/check_doc_links.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def _targets(md_path: Path) -> list[tuple[str, Path]]:
    text = md_path.read_text(encoding="utf-8")
    out: list[tuple[str, Path]] = []
    for m in LINK_RE.finditer(text):
        raw = m.group(1).strip()
        if not raw or raw.startswith(("#", "http://", "https://", "mailto:")):
            continue
        # Strip title / fragment
        path_part = raw.split()[0] if raw else raw
        path_part = path_part.split("#", 1)[0]
        if not path_part:
            continue
        if path_part.startswith("/"):
            resolved = (ROOT / path_part.lstrip("/")).resolve()
        elif path_part.startswith("src/") or path_part.startswith("./src/"):
            resolved = (ROOT / path_part.lstrip("./")).resolve()
        else:
            resolved = (md_path.parent / path_part).resolve()
        out.append((raw, resolved))
    return out


def main() -> int:
    md_files = list(ROOT.glob("docs/**/*.md")) + [
        ROOT / "README.md",
        ROOT / "AGENTS.md",
        ROOT / "ARCHITECTURE.md",
        ROOT / "CONTRIBUTING.md",
    ]
    errors: list[str] = []
    for md in sorted({p.resolve() for p in md_files if p.is_file()}):
        for raw, resolved in _targets(md):
            if resolved.exists():
                continue
            rel = md.relative_to(ROOT)
            tgt = resolved.relative_to(ROOT) if resolved.is_relative_to(ROOT) else resolved
            errors.append(f"{rel}: broken link target {raw!r} -> {tgt}")
    if errors:
        print("Doc link check failed:", file=sys.stderr)
        for line in errors:
            print(f"  {line}", file=sys.stderr)
        return 1
    print(f"OK: checked links in {len(md_files)} markdown files (unique roots).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
