"""
Prior-critique compression helpers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


CONSTRAINT_ID_RE = re.compile(r"\b[A-Z]{1,6}-\d{1,4}\b")


def extract_constraint_ids(text: str) -> list[str]:
    return sorted(set(CONSTRAINT_ID_RE.findall(text)))


@dataclass
class CritiqueCompressor:
    max_recent_chars: int = 5000
    max_summary_chars: int = 1200

    def summarize(self, rounds: list[str]) -> str:
        if not rounds:
            return ""
        if len(rounds) == 1:
            return rounds[0][: self.max_recent_chars]

        latest = rounds[-1][: self.max_recent_chars]
        older = rounds[:-1]
        snippets: list[str] = []
        for index, critique in enumerate(older, start=1):
            ids = extract_constraint_ids(critique)
            first_line = critique.strip().splitlines()[0] if critique.strip() else "no summary"
            snippets.append(f"Round {index}: ids={ids or 'none'}; summary={first_line[:160]}")
        summary = "\n".join(snippets)
        summary = summary[: self.max_summary_chars]
        return f"Older critique summary:\n{summary}\n\nLatest critique:\n{latest}"

