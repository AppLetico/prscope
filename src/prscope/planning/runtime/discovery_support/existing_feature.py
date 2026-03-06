from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable
from typing import Any, Callable

from .models import Evidence, FeatureIntent
from .signals import INTENT_STOP_WORDS


def parse_evidence_reference(line: str) -> tuple[str, int] | None:
    match = re.search(r"`([^`:]+):(\d+)`", str(line))
    if not match:
        return None
    path = match.group(1).strip()
    line_num = int(match.group(2))
    if not path or line_num <= 0:
        return None
    return path, line_num


def format_evidence_line(item: Evidence) -> str:
    path = str(item.path or "").strip()
    snippet = str(item.snippet or "").strip()
    if not path:
        return ""
    if int(item.line or 0) > 0:
        return f"`{path}:{int(item.line)}` {snippet[:120]}".strip()
    return f"`{path}` {snippet[:120]}".strip()


def summarize_endpoint_snippet(snippet: str) -> str | None:
    if not snippet.strip():
        return None
    route_match = re.search(
        r"@(app|router)\.(get|post|put|patch|delete)\(([^)]*)\)",
        snippet,
        flags=re.IGNORECASE,
    )
    handler_match = re.search(r"^\s*(async\s+def|def)\s+([a-zA-Z0-9_]+)\(", snippet, flags=re.MULTILINE)
    return_match = re.search(r"^\s*return\s+(.+)$", snippet, flags=re.MULTILINE)
    details: list[str] = []
    if route_match:
        verb = route_match.group(2).upper()
        route_expr = route_match.group(3).strip()
        details.append(f"Detected endpoint shape: `{verb} {route_expr}`")
    if handler_match:
        details.append(f"Handler function: `{handler_match.group(2)}`")
    if return_match:
        details.append(f"Observed return behavior: `{return_match.group(1).strip()[:120]}`")
    return "\n".join(f"- {item}" for item in details) if details else None


def functional_summary_from_snippet(snippet: str) -> str | None:
    if not snippet.strip():
        return None
    route_match = re.search(
        r"@(app|router)\.(get|post|put|patch|delete)\(([^)]*)\)",
        snippet,
        flags=re.IGNORECASE,
    )
    return_match = re.search(r"^\s*return\s+(.+)$", snippet, flags=re.MULTILINE)
    if route_match and return_match:
        verb = route_match.group(2).upper()
        route_expr = route_match.group(3).strip().strip("'\"")
        return_expr = return_match.group(1).strip()
        return f"This route already exists: `{verb} {route_expr}` currently returns `{return_expr[:120]}`."
    return None


def existing_feature_evidence_lines(
    insights: dict[str, Any],
    route_file_score: Callable[[str], int],
) -> list[str]:
    lines = [str(line).strip() for line in insights.get("matched_evidence", []) if str(line).strip()]
    if lines:
        keywords = [
            str(keyword).strip().lower() for keyword in insights.get("feature_keywords", []) if str(keyword).strip()
        ]
        generic_tokens = INTENT_STOP_WORDS | {
            "api",
            "route",
            "routes",
            "handler",
            "handlers",
            "create",
            "add",
            "build",
            "implement",
            "review",
            "current",
            "behavior",
        }
        ranked_keywords = [kw for kw in keywords if len(kw) >= 3 and kw not in generic_tokens]

        scored_lines: list[tuple[int, str, bool]] = []
        for line in lines:
            lowered = line.lower()
            parsed = parse_evidence_reference(line)
            path_score = route_file_score(parsed[0]) if parsed else 0
            keyword_hits = sum(1 for kw in ranked_keywords if kw in lowered)
            has_route_shape = bool(
                re.search(
                    r"@(?:app|router)\.(?:get|post|put|patch|delete)\(|\b(?:app|router)\.(?:get|post|put|patch|delete)\(",
                    lowered,
                )
            )
            if parsed and path_score <= 0 and keyword_hits == 0 and not has_route_shape:
                continue
            score = max(path_score, 0) + keyword_hits * 3 + (2 if parsed else 0) + (2 if has_route_shape else 0)
            if score <= 0 and parsed is None:
                continue
            is_runtime_evidence = bool(parsed and path_score > 0) or has_route_shape
            scored_lines.append((score, line, is_runtime_evidence))

        if scored_lines:
            runtime_available = any(is_runtime for _, _, is_runtime in scored_lines)
            filtered = [item for item in scored_lines if item[2]] if runtime_available else scored_lines
            filtered.sort(key=lambda item: (-item[0], item[1]))
            return [line for _, line, _ in filtered[:5]]

    paths = [str(path).strip() for path in insights.get("matched_paths", []) if str(path).strip()]
    feature_label = str(insights.get("feature_label", "feature")).strip() or "feature"
    if paths:
        ranked_paths = sorted(paths, key=lambda path: (-route_file_score(path), path))
        top_path = ranked_paths[0]
        return [f"Found existing {feature_label} in `{top_path}`"]
    return [f"Found existing {feature_label} in the codebase"]


def merge_feature_evidence(
    insights_by_session: dict[str, dict[str, Any]],
    *,
    session_id: str,
    feature: FeatureIntent,
    candidate_paths: list[str],
    evidence_lines: list[str] | None = None,
) -> None:
    insights = insights_by_session.setdefault(
        session_id,
        {"existing_feature": False, "feature_label": feature.label, "matched_paths": [], "matched_evidence": []},
    )
    existing_paths = {str(path).strip() for path in insights.get("matched_paths", []) if str(path).strip()}
    existing_evidence = {str(line).strip() for line in insights.get("matched_evidence", []) if str(line).strip()}
    for path in candidate_paths:
        normalized = str(path).strip()
        if normalized:
            existing_paths.add(normalized)
    for line in evidence_lines or []:
        normalized_line = str(line).strip()
        if normalized_line:
            existing_evidence.add(normalized_line)
    insights["matched_paths"] = sorted(existing_paths)
    insights["matched_evidence"] = sorted(existing_evidence)
    insights["existing_feature"] = bool(existing_paths)
    insights["feature_label"] = feature.label
    insights["feature_keywords"] = list(feature.keywords)
    insights["feature_patterns"] = list(feature.patterns)


async def build_existing_endpoint_deep_summary(
    *,
    evidence_lines: list[str],
    emit: Callable[[dict[str, Any]], Awaitable[None]],
    read_file: Callable[..., dict[str, Any]],
) -> tuple[str | None, str | None]:
    for line in evidence_lines:
        parsed = parse_evidence_reference(line)
        if parsed is None:
            continue
        path, anchor_line = parsed
        await emit(
            {
                "type": "tool_call",
                "name": "read_file",
                "session_stage": "discovery",
                "path": path,
                "query": f"around:{anchor_line}",
            }
        )
        started = asyncio.get_running_loop().time()
        try:
            payload = await asyncio.to_thread(
                read_file,
                path,
                120,
                None,
                anchor_line,
                24,
            )
        except Exception:
            continue
        finally:
            elapsed = (asyncio.get_running_loop().time() - started) * 1000.0
            await emit(
                {
                    "type": "tool_result",
                    "name": "read_file",
                    "session_stage": "discovery",
                    "duration_ms": round(elapsed, 2),
                }
            )
        snippet = str(payload.get("content", ""))
        summary = summarize_endpoint_snippet(snippet)
        functional = functional_summary_from_snippet(snippet)
        if summary or functional:
            return summary, functional
    return None, None


async def build_existing_feature_enhancement_summary(
    *,
    insights: dict[str, Any],
    feature_label: str,
    requested_improvement: str | None,
    route_file_score: Callable[[str], int],
    deep_summary_loader: Callable[[], Awaitable[tuple[str | None, str | None]]],
) -> str:
    evidence_lines = existing_feature_evidence_lines(insights, route_file_score)
    matched_paths = [str(path).strip() for path in insights.get("matched_paths", []) if str(path).strip()]
    primary_impl = next(
        (path for path in sorted(matched_paths, key=lambda p: (-route_file_score(p), p)) if path),
        None,
    )
    evidence_block = "\n".join(f"- {line}" for line in evidence_lines[:5]) or (
        f"- Found existing {feature_label} implementation in the codebase"
    )
    deep_summary, functional_summary = await deep_summary_loader()
    summary_lines = [
        f"Enhance the existing {feature_label} implementation instead of creating a duplicate.",
        (
            f"Requested improvement: {requested_improvement.strip()}"
            if requested_improvement and requested_improvement.strip()
            else "Requested improvement: Propose targeted enhancements grounded in the current implementation."
        ),
    ]
    if primary_impl:
        summary_lines.append(f"Primary implementation file: `{primary_impl}`")
    if functional_summary:
        summary_lines.append(f"Functional overview: {functional_summary}")
    if deep_summary:
        summary_lines.append("Implementation notes:")
        summary_lines.append(deep_summary)
    summary_lines.append("Grounding evidence:")
    summary_lines.append(evidence_block)
    return "\n".join(summary_lines)
