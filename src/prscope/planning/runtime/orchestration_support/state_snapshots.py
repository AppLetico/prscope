from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any


class RuntimeStateSnapshots:
    def __init__(self, runtime: Any, schema_version: int):
        self._runtime = runtime
        self._schema_version = schema_version

    @staticmethod
    def as_serializable(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "__dataclass_fields__"):
            return asdict(value)
        return value

    def persist_state_snapshot(self, session_id: str) -> None:
        state = self._runtime._states.get(session_id)
        if state is None:
            return
        sessions_dir = self._runtime.repo.resolved_path / ".prscope" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        issue_entries: list[dict[str, Any]] = []
        issue_graph: dict[str, Any] = {
            "nodes": [],
            "edges": [],
            "duplicate_alias": {},
            "summary": {
                "open_total": 0,
                "root_open": 0,
                "resolved_total": 0,
                "open_major": 0,
                "open_minor": 0,
                "open_info": 0,
                "unresolved_dependency_chains": 0,
            },
        }
        if hasattr(state.issue_tracker, "open_issue_dicts") and hasattr(state.issue_tracker, "graph_snapshot"):
            issue_entries = state.issue_tracker.open_issue_dicts()
            snapshot = state.issue_tracker.graph_snapshot()
            if isinstance(snapshot, dict):
                issue_graph = snapshot
        payload: dict[str, Any] = {
            "schema_version": self._schema_version,
            "session_id": state.session_id,
            "requirements": state.requirements,
            "repo_memory_keys": sorted(list(state.repo_memory.keys())),
            "manifesto_excerpt": (state.manifesto or "")[:2000],
            "constraints": [asdict(constraint) for constraint in state.constraints],
            "plan_markdown": state.plan_markdown,
            "design_record": self.as_serializable(state.design_record),
            "review": self.as_serializable(state.review),
            "constraint_eval": self.as_serializable(state.constraint_eval),
            "revision_round": state.revision_round,
            "architecture_change_rounds": state.architecture_change_rounds[-8:],
            "review_score_history": state.review_score_history[-8:],
            "open_issue_history": state.open_issue_history[-8:],
            "open_issues": issue_entries,
            "issue_graph": issue_graph,
            "accessed_paths": sorted(state.accessed_paths),
            "session_cost_usd": state.session_cost_usd,
            "round_cost_usd": state.round_cost_usd,
            "max_prompt_tokens": state.max_prompt_tokens,
            "author_prompt_tokens": state.author_prompt_tokens,
            "author_completion_tokens": state.author_completion_tokens,
            "critic_prompt_tokens": state.critic_prompt_tokens,
            "critic_completion_tokens": state.critic_completion_tokens,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        snapshot_path = sessions_dir / f"{session_id}.json"
        snapshot_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def read_state_snapshot(self, session_id: str) -> dict[str, Any] | None:
        snapshot_path = self._runtime.repo.resolved_path / ".prscope" / "sessions" / f"{session_id}.json"
        if not snapshot_path.exists():
            return None
        try:
            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def list_state_snapshots(self) -> list[dict[str, Any]]:
        snapshots_dir = self._runtime.repo.resolved_path / ".prscope" / "sessions"
        if not snapshots_dir.exists():
            return []
        items: list[dict[str, Any]] = []
        for path in sorted(snapshots_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            items.append(
                {
                    "session_id": str(payload.get("session_id", path.stem)),
                    "updated_at": str(payload.get("updated_at", "")),
                    "path": str(path),
                }
            )
        return items
