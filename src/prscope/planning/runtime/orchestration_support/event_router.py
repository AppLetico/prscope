from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from ..events import AnalyticsEmitter, apply_token_usage_event


class RuntimeEventRouter:
    def __init__(self, runtime: Any):
        self._runtime = runtime

    def next_version(self, session_id: str) -> int:
        v = self._runtime._session_version.get(session_id, 0) + 1
        self._runtime._session_version[session_id] = v
        return v

    def persist_tool_event_state(
        self,
        session_id: str,
        event_type: str,
        event: dict[str, Any],
    ) -> dict[str, Any] | None:
        return self._runtime._tool_event_state.persist_event(session_id=session_id, event_type=event_type, event=event)

    async def emit_event(self, callback: Any | None, event: dict[str, Any], session_id: str) -> None:
        emitter = AnalyticsEmitter(callback)
        event_type = str(event.get("type", "")).strip().lower()

        async def emit_versioned(payload: dict[str, Any]) -> None:
            version = self.next_version(session_id)
            await emitter.emit({**payload, "session_version": version})

        if event_type in {"tool_call", "tool_result", "tool_update", "complete"}:
            persist_type = event_type
            if event_type in {"tool_call", "tool_result"}:
                persist_type = "tool_update"
            snapshot = self.persist_tool_event_state(session_id, persist_type, event)
            if snapshot is not None:
                await emit_versioned(snapshot)

        if event.get("type") == "token_usage":
            event = apply_token_usage_event(
                state=self._runtime._state(session_id),
                store=self._runtime.store,
                session_id=session_id,
                event=event,
            )

        if event_type in {"tool_call", "tool_result"}:
            now_iso = datetime.now(timezone.utc).isoformat()
            call_id = str(event.get("call_id", ""))
            if not call_id:
                call_id = f"{int(time.time() * 1000)}-{event.get('name', 'tool')}"
            tool_entry: dict[str, Any] = {
                "call_id": call_id,
                "name": str(event.get("name", "tool")),
                "sessionStage": event.get("session_stage"),
                "path": event.get("path"),
                "query": event.get("query"),
                "status": "done" if event_type == "tool_result" else "running",
                "created_at": now_iso,
            }
            if event_type == "tool_result":
                tool_entry["durationMs"] = event.get("duration_ms")
            await emit_versioned(
                {
                    "type": "tool_update",
                    "tool": tool_entry,
                }
            )
        else:
            await emit_versioned(event)
