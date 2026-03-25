#!/usr/bin/env python3
"""
Summarize failed_gate (and related) counts from prscope DEBUG lines containing convergence_gate JSON.

Usage:
  python scripts/summarize_convergence_logs.py [path-to-log]
  grep convergence_gate server.log | python scripts/summarize_convergence_logs.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

# Loguru may single-quote; try JSON first, then a loose failed_gate extract.
_FAILED_GATE_RE = re.compile(r'"failed_gate"\s*:\s*"([^"]*)"')


def _extract_json_objects(line: str) -> list[str]:
    """Best-effort: find {...} spans that parse as JSON."""
    out: list[str] = []
    start = 0
    while start < len(line):
        i = line.find("{", start)
        if i < 0:
            break
        depth = 0
        for j in range(i, len(line)):
            c = line[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    chunk = line[i : j + 1]
                    try:
                        json.loads(chunk)
                    except json.JSONDecodeError:
                        start = j + 1
                        break
                    out.append(chunk)
                    start = j + 1
                    break
        else:
            break
    return out


def _failed_gate_from_line(line: str) -> str | None:
    for chunk in _extract_json_objects(line):
        try:
            data = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "failed_gate" in data:
            fg = data.get("failed_gate")
            return str(fg) if fg is not None else None
    m = _FAILED_GATE_RE.search(line)
    return m.group(1) if m else None


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    if paths:
        lines: list[str] = []
        for p in paths:
            lines.extend(p.read_text(encoding="utf-8", errors="replace").splitlines())
    else:
        lines = sys.stdin.read().splitlines()

    gates = Counter()
    converged_true = 0
    converged_false = 0
    for line in lines:
        if "convergence_gate" not in line:
            continue
        for chunk in _extract_json_objects(line):
            try:
                data = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            if "failed_gate" not in data and "converged" not in data:
                continue
            if data.get("converged") is True:
                converged_true += 1
            elif data.get("converged") is False:
                converged_false += 1
            fg = data.get("failed_gate")
            if fg is not None:
                gates[str(fg)] += 1
            break
        else:
            fg = _failed_gate_from_line(line)
            if fg:
                gates[fg] += 1

    print("convergence_gate lines (parsed):")
    print(f"  converged=true:  {converged_true}")
    print(f"  converged=false: {converged_false}")
    print("failed_gate counts:")
    if not gates:
        print("  (none — enable DEBUG for prscope.planning or point at a log file that contains convergence_gate JSON)")
    else:
        for k, v in gates.most_common():
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
