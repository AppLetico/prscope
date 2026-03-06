"""
Tool event persistence helpers for planning runtime.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ....store import Store
    from ...core import PlanningCore


class ToolEventStateManager:
    def __init__(
        self,
        *,
        store: Store,
        core_resolver: Callable[[str], PlanningCore],
        max_active_tool_calls: int,
        max_completed_tool_call_groups: int,
    ) -> None:
        self._store = store
        self._core_resolver = core_resolver
        self._max_active_tool_calls = max_active_tool_calls
        self._max_completed_tool_call_groups = max_completed_tool_call_groups

    def persist_event(
        self,
        session_id: str,
        event_type: str,
        event: dict[str, Any],
    ) -> dict[str, Any] | None:
        core: PlanningCore = self._core_resolver(session_id)
        session = core.get_session()

        active: list[dict[str, Any]] = []
        if session.active_tool_calls_json:
            try:
                parsed = json.loads(session.active_tool_calls_json)
                if isinstance(parsed, list):
                    active = [item for item in parsed if isinstance(item, dict)]
            except json.JSONDecodeError:
                active = []

        completed_groups: list[dict[str, Any]] = []
        if getattr(session, "completed_tool_call_groups_json", None):
            try:
                parsed_completed = json.loads(str(session.completed_tool_call_groups_json))
                if isinstance(parsed_completed, list):
                    for group in parsed_completed:
                        if isinstance(group, dict) and "tools" in group:
                            completed_groups.append(group)
                        elif isinstance(group, list):
                            completed_groups.append(
                                {
                                    "sequence": 0,
                                    "created_at": "",
                                    "tools": [e for e in group if isinstance(e, dict)],
                                }
                            )
            except json.JSONDecodeError:
                completed_groups = []

        now_iso = datetime.now(timezone.utc).isoformat()
        original_type = str(event.get("type", "")).strip().lower()
        is_start = original_type == "tool_call" or (
            event_type == "tool_update" and str(event.get("status", "")) == "running"
        )
        is_done = original_type == "tool_result" or (
            event_type == "tool_update" and str(event.get("status", "")) == "done"
        )

        if is_start:
            call_id = str(event.get("call_id", ""))
            if not call_id:
                call_id = f"{int(time.time() * 1000)}-{len(active)}"
            active.append(
                {
                    "id": call_id,
                    "call_id": call_id,
                    "name": str(event.get("name", "tool")),
                    "path": event.get("path"),
                    "query": event.get("query"),
                    "session_stage": event.get("session_stage"),
                    "status": "running",
                    "created_at": now_iso,
                }
            )
        elif is_done:
            name = str(event.get("name", "tool"))
            stage = str(event.get("session_stage", ""))
            call_id = str(event.get("call_id", ""))
            matched = False
            for idx in range(len(active) - 1, -1, -1):
                item = active[idx]
                if call_id and str(item.get("call_id", "")) == call_id:
                    item["status"] = "done"
                    item["duration_ms"] = event.get("duration_ms")
                    matched = True
                    break
                if (
                    not call_id
                    and str(item.get("name", "")) == name
                    and str(item.get("session_stage", "")) == stage
                    and str(item.get("status", "")) == "running"
                ):
                    item["status"] = "done"
                    item["duration_ms"] = event.get("duration_ms")
                    matched = True
                    break
            if not matched:
                if not call_id:
                    call_id = f"{int(time.time() * 1000)}-{len(active)}"
                active.append(
                    {
                        "id": call_id,
                        "call_id": call_id,
                        "name": name,
                        "session_stage": event.get("session_stage"),
                        "status": "done",
                        "duration_ms": event.get("duration_ms"),
                        "created_at": now_iso,
                    }
                )
        elif event_type == "complete":
            finalized_group = [
                {**item, "status": "done"} if str(item.get("status", "")) == "running" else item
                for item in active
                if isinstance(item, dict)
            ]
            if finalized_group:
                seq = self._store.increment_event_seq(session_id)
                group_obj: dict[str, Any] = {
                    "sequence": seq,
                    "created_at": now_iso,
                    "tools": finalized_group,
                }
                completed_groups.append(group_obj)
                if len(completed_groups) > self._max_completed_tool_call_groups:
                    completed_groups = completed_groups[-self._max_completed_tool_call_groups :]
            active = []
        else:
            return None

        active.sort(key=lambda item: str(item.get("created_at", "")))
        if len(active) > self._max_active_tool_calls:
            active = active[-self._max_active_tool_calls :]

        # Persist completed groups BEFORE snapshot so the emitted session_state
        # is authoritative and includes the just-completed group.
        if event_type == "complete":
            self._store.update_planning_session(
                session_id,
                _bypass_protection=True,
                completed_tool_call_groups_json=json.dumps(completed_groups),
            )

        snapshot = core.transition_and_snapshot(
            session.status,
            phase_message=session.phase_message,
            active_tool_calls_json=json.dumps(active),
            allow_round_stability=True,
        )
        return snapshot
