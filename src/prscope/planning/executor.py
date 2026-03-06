from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ..store import PlanningCommand, Store


@dataclass
class HandlerResult:
    metadata: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CommandContext:
    store: Store
    session_id: str
    command: str
    command_id: str
    payload: dict[str, Any]
    command_row: PlanningCommand


class CommandConflictError(RuntimeError):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(detail or reason)
        self.reason = reason


async def execute_command(
    *,
    store: Store,
    session_id: str,
    command: str,
    command_id: str,
    payload: dict[str, Any],
    lease_seconds: int,
    heartbeat_seconds: int,
    allowed_commands_by_status: dict[str, set[str]],
    handler: Callable[[CommandContext], Awaitable[HandlerResult]],
    build_snapshot: Callable[[str], dict[str, Any]],
    emit_event: Callable[[str, dict[str, Any]], Awaitable[None]],
) -> dict[str, Any]:
    payload_json = json.dumps(payload)
    reserved, replay, active, reason = store.reserve_planning_command(
        session_id=session_id,
        command=command,
        command_id=command_id,
        payload_json=payload_json,
        lease_seconds=lease_seconds,
        allowed_commands_by_status=allowed_commands_by_status,
    )
    if replay is not None:
        replay_payload: dict[str, Any] = {}
        if replay.result_snapshot_json:
            try:
                parsed = json.loads(replay.result_snapshot_json)
            except json.JSONDecodeError:
                parsed = {}
            if isinstance(parsed, dict):
                replay_payload = parsed
        if not replay_payload:
            replay_payload = build_snapshot(session_id)
        replay_payload["idempotent_replay"] = True
        replay_payload["reason"] = "duplicate_command"
        return replay_payload
    if reason == "session_not_found":
        raise ValueError("session not found")
    if reason == "invalid_status":
        raise CommandConflictError("invalid_status")
    if reason == "processing_lock" or active is not None:
        raise CommandConflictError("processing_lock")
    if reserved is None:
        raise RuntimeError("failed to reserve command")

    stop_lease = asyncio.Event()
    lease_task: asyncio.Task[None] | None = None

    async def _heartbeat() -> None:
        while not stop_lease.is_set():
            await asyncio.sleep(max(1, heartbeat_seconds))
            if stop_lease.is_set():
                break
            store.renew_planning_command_lease(reserved.id, lease_seconds=lease_seconds)

    if heartbeat_seconds > 0:
        lease_task = asyncio.create_task(_heartbeat())

    result = HandlerResult()
    try:
        result = await handler(
            CommandContext(
                store=store,
                session_id=session_id,
                command=command,
                command_id=command_id,
                payload=payload,
                command_row=reserved,
            )
        )
    except Exception as exc:
        store.fail_planning_command(reserved.id, reason=str(exc))
        try:
            await emit_event(session_id, build_snapshot(session_id))
            await emit_event(session_id, {"type": "error", "message": str(exc)})
        except Exception:
            # Preserve the original command failure even if recovery emission fails.
            pass
        raise
    finally:
        stop_lease.set()
        if lease_task is not None:
            lease_task.cancel()
            try:
                await lease_task
            except asyncio.CancelledError:
                pass

    command_row = store.begin_planning_command_finalize(reserved.id)
    if command_row is None or command_row.status != "finalizing":
        # Cancel/timeout raced in; do not commit late command completion.
        snapshot = build_snapshot(session_id)
        snapshot["reason"] = "cancelled"
        return snapshot

    snapshot = build_snapshot(session_id)
    response = dict(snapshot)
    if result.metadata:
        response.update(result.metadata)
    store.complete_planning_command(
        command_row_id=reserved.id,
        result_snapshot_json=json.dumps(response),
    )
    await emit_event(session_id, snapshot)
    for event in result.events:
        await emit_event(session_id, event)
    return response
