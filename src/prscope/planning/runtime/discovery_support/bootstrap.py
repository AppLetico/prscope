from __future__ import annotations

import asyncio
from typing import Any

from .existing_feature import format_evidence_line
from .models import Evidence, FeatureIntent
from .signals import BOOTSTRAP_ROUTE_REGEX


class DiscoveryBootstrapService:
    def __init__(self, manager: Any):
        self._manager = manager

    async def run_bootstrap_tool(
        self,
        *,
        tool_name: str,
        path: str | None = None,
        pattern: str | None = None,
        max_entries: int = 120,
        max_results: int = 80,
    ) -> dict[str, Any] | None:
        await self._manager._emit(
            {
                "type": "tool_call",
                "name": tool_name,
                "session_stage": "discovery",
                "path": path,
                "query": pattern,
            }
        )
        started = asyncio.get_running_loop().time()
        try:
            if tool_name == "list_files":
                result = self._manager.tool_executor.list_files(path=path, max_entries=max_entries)
            elif tool_name == "read_file":
                if not path:
                    return None
                result = self._manager.tool_executor.read_file(path=path, max_lines=220)
            elif tool_name == "grep_code":
                if not pattern:
                    return None
                result = self._manager.tool_executor.grep_code(
                    pattern=pattern,
                    path=path,
                    max_results=max_results,
                )
            else:
                return None
            return result
        except Exception:
            return None
        finally:
            elapsed = (asyncio.get_running_loop().time() - started) * 1000.0
            await self._manager._emit(
                {
                    "type": "tool_result",
                    "name": tool_name,
                    "session_stage": "discovery",
                    "duration_ms": round(elapsed, 2),
                }
            )

    async def maybe_read_more_context(self, path: str, line_number: int) -> str:
        payload = await self._manager._run_bootstrap_tool(tool_name="read_file", path=path)
        if not payload or not isinstance(payload, dict):
            return ""
        content = str(payload.get("content", ""))
        if not content:
            return ""
        if line_number <= 0:
            return content
        lines = content.splitlines()
        if line_number > len(lines):
            return content
        start = max(0, line_number - 6)
        end = min(len(lines), line_number + 5)
        return "\n".join(lines[start:end])

    async def ingest_feature_evidence_from_tool(
        self,
        *,
        session_id: str,
        feature: FeatureIntent | None,
        tool_name: str,
        parsed_args: dict[str, Any],
        tool_result_payload: dict[str, Any],
    ) -> list[Evidence]:
        if tool_name not in {"grep_code", "read_file"} or feature is None:
            return []
        patterns = feature.compiled_patterns
        payload = tool_result_payload.get("result")
        if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
            payload = payload.get("result")
        if not isinstance(payload, dict):
            payload = {}
        evidence_list: list[Evidence] = []
        candidate_paths: set[str] = set()
        if tool_name == "grep_code":
            results = payload.get("results", [])
            if isinstance(results, list):
                for item in results:
                    if not isinstance(item, dict):
                        continue
                    path = str(item.get("path", "")).strip()
                    text = str(item.get("text", "")).strip()
                    line_num = int(item.get("line", 0) or 0)
                    if not path:
                        continue
                    matched = any(pattern.search(text) for pattern in patterns)
                    snippet = text
                    if not matched and snippet.count("\n") < 2:
                        expanded = await self._manager._maybe_read_more_context(path, line_num)
                        if expanded and any(pattern.search(expanded) for pattern in patterns):
                            matched = True
                            snippet = expanded
                    if not matched:
                        continue
                    confidence = self._manager._location_score(path)
                    evidence_list.append(
                        Evidence(
                            path=path,
                            snippet=snippet[:240],
                            confidence=confidence,
                            line=line_num if line_num > 0 else 0,
                        )
                    )
                    candidate_paths.add(path)
        else:
            path = str(parsed_args.get("path") or payload.get("path") or "").strip()
            content = str(payload.get("content", "")).strip()
            if path and any(pattern.search(content) for pattern in patterns):
                confidence = self._manager._location_score(path, file_line_count=len(content.splitlines()))
                evidence_list.append(Evidence(path=path, snippet=content[:240], confidence=confidence, line=0))
                candidate_paths.add(path)
        aggregated = self._manager._aggregate_evidence(evidence_list)
        if candidate_paths:
            evidence_lines = [
                formatted for formatted in (format_evidence_line(item) for item in aggregated[:5]) if formatted
            ]
            self._manager._merge_feature_evidence(
                session_id=session_id,
                feature=feature,
                candidate_paths=sorted(candidate_paths),
                evidence_lines=evidence_lines,
            )
        return aggregated

    @staticmethod
    def extract_feature_evidence_from_content(
        path: str,
        content: str,
        feature: FeatureIntent,
        limit: int = 3,
    ) -> list[str]:
        evidence: list[str] = []
        patterns = feature.compiled_patterns
        for idx, raw_line in enumerate((content or "").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            if any(pattern.search(line) for pattern in patterns):
                evidence.append(f"`{path}:{idx}` {line[:120]}")
                if len(evidence) >= limit:
                    break
        return evidence

    async def verify_feature_in_candidate_files(
        self,
        *,
        session_id: str,
        feature: FeatureIntent,
        candidate_paths: list[str],
    ) -> list[Evidence]:
        patterns = feature.compiled_patterns
        evidence_list: list[Evidence] = []
        for path in candidate_paths:
            normalized = str(path or "").strip()
            if not normalized or self._manager._route_file_score(normalized) <= 0:
                continue
            payload = await self._manager._run_bootstrap_tool(
                tool_name="grep_code",
                path=normalized,
                pattern="|".join(feature.patterns),
                max_results=12,
            )
            if not payload or not isinstance(payload, dict):
                continue
            results = payload.get("results", [])
            if not isinstance(results, list):
                continue
            for item in results[:12]:
                if not isinstance(item, dict):
                    continue
                item_path = str(item.get("path", normalized)).strip()
                line_num = int(item.get("line", 0) or 0)
                text = str(item.get("text", "")).strip()
                if not item_path:
                    continue
                matched = any(pattern.search(text) for pattern in patterns)
                snippet = text
                if not matched and snippet.count("\n") < 2:
                    expanded = await self._manager._maybe_read_more_context(item_path, line_num)
                    if expanded and any(pattern.search(expanded) for pattern in patterns):
                        matched = True
                        snippet = expanded
                if not matched:
                    continue
                evidence_list.append(
                    Evidence(
                        path=item_path,
                        snippet=snippet[:240],
                        confidence=self._manager._location_score(item_path),
                        line=line_num if line_num > 0 else 0,
                    )
                )
        aggregated = self._manager._aggregate_evidence(evidence_list)
        if aggregated:
            self._manager._merge_feature_evidence(
                session_id=session_id,
                feature=feature,
                candidate_paths=[item.path for item in aggregated],
                evidence_lines=[
                    formatted for formatted in (format_evidence_line(item) for item in aggregated[:5]) if formatted
                ],
            )
        return aggregated

    async def build_first_turn_bootstrap_context(
        self,
        session_id: str,
        conversation: list[dict[str, Any]],
        turn_count: int,
    ) -> tuple[str, str | None]:
        if not hasattr(self._manager, "bootstrap_insights_by_session"):
            self._manager.bootstrap_insights_by_session = {}
        if turn_count != 1:
            return "", None
        if not hasattr(self._manager, "tool_executor"):
            return "", None
        user_message = self._manager._latest_user_message(conversation)
        if not self._manager._should_bootstrap_scan(user_message):
            return "", None
        feature = self._manager._extract_feature_intent(user_message)
        previous_insights = self._manager.bootstrap_insights_by_session.get(session_id, {})
        prev_label = str(previous_insights.get("feature_label", "")).strip()
        if prev_label and feature and prev_label != feature.label:
            self._manager.bootstrap_insights_by_session[session_id] = {}
        else:
            self._manager.bootstrap_insights_by_session.pop(session_id, None)

        context_lines: list[str] = [
            "## Bootstrap Scan Evidence (automatic first-turn preflight)",
            "Use this before asking implementation clarification questions.",
        ]
        inferred_framework: str | None = None

        root_listing = await self._manager._run_bootstrap_tool(tool_name="list_files", path=".", max_entries=120)
        if root_listing and isinstance(root_listing.get("entries"), list):
            top_level = [str(item.get("path", "")) for item in root_listing["entries"][:20]]
            visible = ", ".join(path for path in top_level if path) or "(none)"
            context_lines.append(f"Top-level entries: {visible}")
        root_entries = root_listing.get("entries", []) if isinstance(root_listing, dict) else []
        candidate_dirs = self._manager._select_scan_directories(root_entries if isinstance(root_entries, list) else [])

        for candidate in candidate_dirs:
            listed = await self._manager._run_bootstrap_tool(tool_name="list_files", path=candidate, max_entries=80)
            if listed and isinstance(listed.get("entries"), list):
                sample = [str(item.get("path", "")) for item in listed["entries"][:10]]
                if sample:
                    context_lines.append(f"- `{candidate}` sample: {', '.join(sample)}")

        bootstrap_patterns = [BOOTSTRAP_ROUTE_REGEX.pattern]
        if feature is not None:
            bootstrap_patterns.extend(feature.patterns)
        endpoint_matches = await self._manager._run_bootstrap_tool(
            tool_name="grep_code",
            pattern="|".join(pattern for pattern in bootstrap_patterns if pattern),
            max_results=120,
        )
        if endpoint_matches and isinstance(endpoint_matches.get("results"), list):
            matches = [item for item in endpoint_matches["results"] if isinstance(item, dict)]
            signal_index = self._manager._build_signal_index(matches)
            inferred_framework = self._manager._detect_framework(signal_index)
            signal_scores = self._manager._detect_code_signals(signal_index)
            architecture = self._manager._detect_architecture(signal_scores)
            evidence_list: list[Evidence] = []
            if feature is not None:
                feature_patterns = feature.compiled_patterns
                for item in matches:
                    path = str(item.get("path", "")).strip()
                    line = int(item.get("line", 0) or 0)
                    text = str(item.get("text", "")).strip()
                    if not path or not any(pattern.search(text) for pattern in feature_patterns):
                        continue
                    evidence_list.append(
                        Evidence(
                            path=path,
                            snippet=text[:240],
                            confidence=self._manager._location_score(path, file_line_count=line if line > 0 else None),
                            line=line if line > 0 else 0,
                        )
                    )
            evidence_list = self._manager._aggregate_evidence(evidence_list)
            if feature is not None and not evidence_list:
                candidate_paths = sorted(
                    {str(item.get("path", "")).strip() for item in matches if str(item.get("path", "")).strip()}
                )
                evidence_list = await self._manager._verify_feature_in_candidate_files(
                    session_id=session_id,
                    feature=feature,
                    candidate_paths=candidate_paths[:12],
                )
            if feature is not None:
                self._manager._merge_feature_evidence(
                    session_id=session_id,
                    feature=feature,
                    candidate_paths=[item.path for item in evidence_list],
                    evidence_lines=[
                        formatted
                        for formatted in (format_evidence_line(item) for item in evidence_list[:6])
                        if formatted
                    ],
                )
                insights = self._manager.bootstrap_insights_by_session.setdefault(session_id, {})
                insights["architecture"] = architecture
                insights["signal_scores"] = signal_scores
                insights["best_path"] = evidence_list[0].path if evidence_list else None
            context_lines.append("Endpoint/API-related code matches:")
            context_lines.append(self._manager._format_grep_matches(matches, limit=16))
            if inferred_framework:
                context_lines.append(
                    "Inferred backend framework from code scan: "
                    f"{inferred_framework}. Do not ask which backend framework is used."
                )
            if architecture:
                context_lines.append(f"Inferred repository architecture: {architecture}")
            if signal_scores:
                context_lines.append(f"Detected code signals: {signal_scores}")

        if len(context_lines) <= 2:
            return "", inferred_framework
        return "\n".join(context_lines)[:3500], inferred_framework
