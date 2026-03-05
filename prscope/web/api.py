"""
FastAPI transport layer for prscope planning runtime.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ..config import PrscopeConfig, get_repo_root
from ..model_catalog import get_model, list_models
from ..planning.core import (
    ApprovalBlockedError,
    InvalidCommandError,
    InvalidTransitionError,
    PlanningCore,
)
from ..planning.executor import CommandConflictError, CommandContext, HandlerResult, execute_command
from ..planning.runtime import PlanningRuntime
from ..store import Store
from .events import SessionEventEmitter

COMMAND_LEASE_SECONDS = 120
COMMAND_HEARTBEAT_SECONDS = 60


class StartSessionRequest(BaseModel):
    mode: str = Field(pattern="^(requirements|pr|chat)$")
    repo: Optional[str] = None
    requirements: Optional[str] = None
    upstream_repo: Optional[str] = None
    pr_number: Optional[int] = None
    rebuild_memory: bool = False
    author_model: Optional[str] = None
    critic_model: Optional[str] = None


class MessageRequest(BaseModel):
    message: str
    command_id: str
    repo: Optional[str] = None
    author_model: Optional[str] = None
    critic_model: Optional[str] = None


class RoundRequest(BaseModel):
    user_input: Optional[str] = None
    command_id: str
    repo: Optional[str] = None
    author_model: Optional[str] = None
    critic_model: Optional[str] = None


class ApproveRequest(BaseModel):
    command_id: str
    repo: Optional[str] = None
    unverified_references: list[str] = Field(default_factory=list)


class ExportRequest(BaseModel):
    command_id: Optional[str] = None
    repo: Optional[str] = None


class CommandRequest(BaseModel):
    command: str
    command_id: str
    repo: Optional[str] = None
    user_input: Optional[str] = None
    message: Optional[str] = None
    author_model: Optional[str] = None
    critic_model: Optional[str] = None
    unverified_references: list[str] = Field(default_factory=list)


class ClarifyRequest(BaseModel):
    answers: list[str] = Field(default_factory=list)
    repo: Optional[str] = None


class RuntimeRegistry:
    """Caches runtime instances so locks survive across requests."""

    def __init__(self, emitter: SessionEventEmitter):
        self._runtimes: dict[str, PlanningRuntime] = {}
        self._stores: dict[str, Store] = {}
        self._emitter = emitter
        self._session_timing: dict[str, dict[str, Any]] = {}

    def _key(self, repo_name: str) -> str:
        return repo_name

    @staticmethod
    def _default_session_timing() -> dict[str, Any]:
        return {
            "initial_draft_started_at_unix_s": None,
            "initial_draft_elapsed_s": None,
            "initial_draft_completed_at_unix_s": None,
            "initial_draft_failed": False,
            "warnings_total": 0,
            "errors_total": 0,
            "author_call_timeouts": 0,
            "author_fallback_warnings": 0,
            "llm_calls_total": 0,
            "llm_call_latency_ms_total": 0.0,
            "llm_call_latency_ms_max": 0.0,
            "tool_calls_total": 0,
            "tool_exec_latency_ms_total": 0.0,
            "tool_exec_latency_ms_max": 0.0,
            "first_plan_saved_at_unix_s": None,
            "first_plan_round": None,
        }

    def _ensure_session_timing(self, session_id: str) -> dict[str, Any]:
        timing = self._session_timing.get(session_id)
        if timing is None:
            timing = self._default_session_timing()
            self._session_timing[session_id] = timing
        return timing

    def note_event(self, session_id: str, event: dict[str, Any]) -> None:
        etype = str(event.get("type", "")).strip().lower()
        if not etype:
            return
        timing = self._ensure_session_timing(session_id)
        if etype == "warning":
            timing["warnings_total"] = int(timing.get("warnings_total", 0)) + 1
            message = str(event.get("message", "")).lower()
            if "author call timeout on" in message:
                timing["author_call_timeouts"] = int(timing.get("author_call_timeouts", 0)) + 1
            if "trying fallback model" in message or "author fallback used" in message:
                timing["author_fallback_warnings"] = int(timing.get("author_fallback_warnings", 0)) + 1
            return
        if etype == "token_usage":
            timing["llm_calls_total"] = int(timing.get("llm_calls_total", 0)) + 1
            latency = float(event.get("llm_call_latency_ms", 0.0) or 0.0)
            timing["llm_call_latency_ms_total"] = float(timing.get("llm_call_latency_ms_total", 0.0)) + latency
            timing["llm_call_latency_ms_max"] = max(
                float(timing.get("llm_call_latency_ms_max", 0.0)),
                latency,
            )
            return
        if etype == "tool_result":
            timing["tool_calls_total"] = int(timing.get("tool_calls_total", 0)) + 1
            latency = float(event.get("duration_ms", 0.0) or 0.0)
            timing["tool_exec_latency_ms_total"] = float(timing.get("tool_exec_latency_ms_total", 0.0)) + latency
            timing["tool_exec_latency_ms_max"] = max(
                float(timing.get("tool_exec_latency_ms_max", 0.0)),
                latency,
            )
            return
        if etype == "plan_ready":
            if timing.get("first_plan_saved_at_unix_s") is None:
                saved_at = event.get("saved_at_unix_s")
                try:
                    saved_at_value = float(saved_at if saved_at is not None else time.time())
                except (TypeError, ValueError):
                    saved_at_value = time.time()
                timing["first_plan_saved_at_unix_s"] = saved_at_value
                try:
                    timing["first_plan_round"] = int(event.get("round", 0) or 0)
                except (TypeError, ValueError):
                    timing["first_plan_round"] = 0
            return
        if etype == "error":
            timing["errors_total"] = int(timing.get("errors_total", 0)) + 1

    def _config_root(self) -> Path:
        forced_root = os.environ.get("PRSCOPE_CONFIG_ROOT")
        if forced_root:
            return Path(forced_root).expanduser().resolve()
        return get_repo_root()

    def get_runtime(self, repo_name: Optional[str] = None) -> tuple[PrscopeConfig, Store, PlanningRuntime]:
        config = PrscopeConfig.load(self._config_root())
        repo = config.resolve_repo(repo_name, cwd=Path.cwd())
        key = self._key(repo.name)
        runtime = self._runtimes.get(key)
        store = self._stores.get(key)
        if runtime is None or store is None:
            store = Store()
            runtime = PlanningRuntime(store=store, config=config, repo=repo)
            self._stores[key] = store
            self._runtimes[key] = runtime
        return config, store, runtime

    def cleanup_session(self, session_id: str) -> None:
        self._emitter.cleanup(session_id)
        self._session_timing.pop(session_id, None)
        for runtime in self._runtimes.values():
            runtime.cleanup_session_resources(session_id)


def _session_to_dict(session: Any) -> dict[str, Any]:
    data = asdict(session)
    pending_questions: Any = None
    if data.get("pending_questions_json"):
        try:
            parsed = json.loads(str(data["pending_questions_json"]))
            if isinstance(parsed, list):
                pending_questions = parsed
        except json.JSONDecodeError:
            pending_questions = None
    active_tool_calls: list[dict[str, Any]] = []
    if data.get("active_tool_calls_json"):
        try:
            parsed = json.loads(str(data["active_tool_calls_json"]))
            if isinstance(parsed, list):
                active_tool_calls = [item for item in parsed if isinstance(item, dict)]
        except json.JSONDecodeError:
            active_tool_calls = []
    completed_tool_call_groups: list[list[dict[str, Any]]] = []
    if data.get("completed_tool_call_groups_json"):
        try:
            parsed = json.loads(str(data["completed_tool_call_groups_json"]))
            if isinstance(parsed, list):
                completed_tool_call_groups = [
                    [entry for entry in group if isinstance(entry, dict)] for group in parsed if isinstance(group, list)
                ]
        except json.JSONDecodeError:
            completed_tool_call_groups = []
    data["pending_questions"] = pending_questions
    data["phase_message"] = data.get("phase_message")
    data["is_processing"] = bool(data.get("is_processing", 0))
    data["active_tool_calls"] = active_tool_calls
    data["completed_tool_call_groups"] = completed_tool_call_groups
    return data


def _turn_to_dict(turn: Any) -> dict[str, Any]:
    return asdict(turn)


def _version_to_dict(version: Any) -> dict[str, Any]:
    return asdict(version)


def create_app() -> FastAPI:
    emitter = SessionEventEmitter()
    registry = RuntimeRegistry(emitter)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Executor-mode startup reconciliation:
        # fail any expired running commands so lock state is coherent.
        try:
            store = Store()
            expired = store.fail_expired_running_planning_commands()
            if expired:
                logger.warning("startup reconciliation failed expired commands={}", expired)
        except Exception as exc:  # noqa: BLE001
            logger.warning("startup reconciliation skipped: {}", exc)
        try:
            yield
        finally:
            pass

    app = FastAPI(title="prscope-web", version="0.1.0", lifespan=lifespan)

    def _runtime_for(repo_name: Optional[str]) -> tuple[PrscopeConfig, Store, PlanningRuntime]:
        try:
            return registry.get_runtime(repo_name=repo_name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _repo_for_session(session_id: str, explicit_repo: Optional[str]) -> str:
        if explicit_repo:
            return explicit_repo
        session = Store().get_planning_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        return session.repo_name

    def _validated_model_or_400(model_id: Optional[str], role: str) -> Optional[str]:
        if not model_id:
            return None
        model = get_model(model_id)
        if model is None:
            raise HTTPException(
                status_code=400,
                detail=(f"Invalid {role} model '{model_id}'. Pick a model from /api/models."),
            )
        if not bool(model.get("available")):
            reason = str(model.get("unavailable_reason") or "provider key unavailable")
            raise HTTPException(
                status_code=400,
                detail=f"Selected {role} model '{model_id}' is unavailable: {reason}",
            )
        return model_id

    COMMAND_MATRIX: dict[str, set[str]] = {
        "draft": {"message", "reset"},
        "refining": {"run_round", "reset", "message"},
        "converged": {"run_round", "approve", "reset", "message"},
        "approved": {"export", "reset"},
        "error": {"reset"},
    }

    def _allowed_commands_for(session: Any, store: Store) -> list[str]:
        active = store.get_live_running_planning_command(session.id)
        if active is not None:
            return ["cancel"]
        return sorted(COMMAND_MATRIX.get(session.status, set()))

    def _build_session_snapshot(session_id: str, store: Store) -> dict[str, Any]:
        session = store.get_planning_session(session_id)
        if session is None:
            raise ValueError("session not found")
        active = store.get_live_running_planning_command(session_id)
        payload = _session_to_dict(session)
        return {
            "type": "session_state",
            "v": 1,
            "status": payload.get("status"),
            "phase_message": payload.get("phase_message"),
            "is_processing": bool(active is not None),
            "current_round": int(payload.get("current_round", 0) or 0),
            "pending_questions": payload.get("pending_questions"),
            "active_tool_calls": payload.get("active_tool_calls", []),
            "completed_tool_call_groups": payload.get("completed_tool_call_groups", []),
            "allowed_commands": _allowed_commands_for(session, store),
            "active_command": (
                {
                    "command_id": active.command_id,
                    "command": active.command,
                    "started_at": active.started_at,
                    "lease_expires_at": active.lease_expires_at,
                }
                if active is not None
                else None
            ),
        }

    async def _emit_session_event(session_id: str, event: dict[str, Any]) -> None:
        registry.note_event(session_id, event)
        await emitter.emit(session_id, event)

    async def _execute_registered_command(
        *,
        session_id: str,
        command: str,
        command_id: str,
        repo_name: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        config, store, runtime = _runtime_for(repo_name=repo_name)

        async def _run_round(ctx: CommandContext) -> HandlerResult:
            session = ctx.store.get_planning_session(ctx.session_id)
            if session is None:
                raise ValueError("session not found")
            author_model = _validated_model_or_400(
                payload.get("author_model") or session.author_model,
                "author",
            )
            critic_model = _validated_model_or_400(
                payload.get("critic_model") or session.critic_model,
                "critic",
            )

            async def callback(event: dict[str, Any]) -> None:
                enriched = dict(event)
                enriched.setdefault("command_id", command_id)
                await _emit_session_event(ctx.session_id, enriched)

            await runtime.run_adversarial_round(
                session_id=ctx.session_id,
                user_input=payload.get("user_input"),
                author_model_override=author_model,
                critic_model_override=critic_model,
                event_callback=callback,
            )
            return HandlerResult()

        async def _message(ctx: CommandContext) -> HandlerResult:
            session = ctx.store.get_planning_session(ctx.session_id)
            if session is None:
                raise ValueError("session not found")
            message = str(payload.get("message") or "").strip()
            if not message:
                raise CommandConflictError("invalid_status", "message is required")
            author_model = _validated_model_or_400(
                payload.get("author_model") or session.author_model,
                "author",
            )
            critic_model = _validated_model_or_400(
                payload.get("critic_model") or session.critic_model,
                "critic",
            )

            async def callback(event: dict[str, Any]) -> None:
                enriched = dict(event)
                enriched.setdefault("command_id", command_id)
                await _emit_session_event(ctx.session_id, enriched)

            if session.status in {"refining", "converged"}:
                reply = await runtime.chat_with_author(
                    session_id=ctx.session_id,
                    user_message=message,
                    author_model_override=author_model,
                    event_callback=callback,
                )
                return HandlerResult(metadata={"accepted": True, "mode": "author_chat", "reply": reply})
            if session.status not in {"draft", "refining", "converged"}:
                raise CommandConflictError("invalid_status", "message is only allowed in draft/refining/converged")

            result = await runtime.handle_discovery_turn(
                session_id=ctx.session_id,
                user_message=message,
                author_model_override=author_model,
                critic_model_override=critic_model,
                event_callback=callback,
                defer_initial_draft=True,
            )
            if result.complete:
                draft_requirements = result.summary or message

                async def _continue_discovery_draft_task() -> None:
                    try:
                        await runtime.continue_discovery_draft(
                            session_id=ctx.session_id,
                            requirements=draft_requirements,
                            author_model_override=author_model,
                            event_callback=callback,
                        )
                    except Exception as exc:  # noqa: BLE001
                        await callback({"type": "error", "message": f"Initial draft failed: {exc}"})

                task = asyncio.create_task(_continue_discovery_draft_task())
                task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            return HandlerResult(metadata={"result": asdict(result)})

        async def _approve(ctx: CommandContext) -> HandlerResult:
            runtime.approve(
                session_id=ctx.session_id,
                unverified_references=set(payload.get("unverified_references") or []),
            )
            return HandlerResult(metadata={"approved": True})

        async def _export(ctx: CommandContext) -> HandlerResult:
            paths = await asyncio.to_thread(runtime.export, ctx.session_id)
            return HandlerResult(
                metadata={
                    "files": [
                        {
                            "name": "prd.md",
                            "kind": "prd",
                            "url": f"/api/sessions/{ctx.session_id}/download/prd",
                        },
                        {
                            "name": "rfc.md",
                            "kind": "rfc",
                            "url": f"/api/sessions/{ctx.session_id}/download/rfc",
                        },
                        {
                            "name": "conversation.md",
                            "kind": "conversation",
                            "url": f"/api/sessions/{ctx.session_id}/download/conversation",
                        },
                    ],
                    "paths": {k: str(v) for k, v in paths.items()},
                }
            )

        async def _reset(ctx: CommandContext) -> HandlerResult:
            session = ctx.store.get_planning_session(ctx.session_id)
            if session is None:
                raise ValueError("session not found")
            ctx.store.update_planning_session(
                ctx.session_id,
                _bypass_protection=True,
                status="draft",
                phase_message=None,
                is_processing=0,
                processing_started_at=None,
                current_round=0,
                current_command_id=None,
            )
            return HandlerResult(metadata={"reset": True})

        async def _cancel(ctx: CommandContext) -> HandlerResult:
            cancelled = ctx.store.cancel_active_planning_command(ctx.session_id)
            if cancelled is None:
                raise CommandConflictError("invalid_status", "No running command to cancel")
            session = ctx.store.get_planning_session(ctx.session_id)
            if session is not None and bool(session.is_processing):
                core = PlanningCore(store=ctx.store, session_id=ctx.session_id, config=config.planning)
                snapshot = core.transition_and_snapshot(
                    session.status,
                    phase_message=None,
                    allow_round_stability=(session.status == "refining"),
                )
                await _emit_session_event(ctx.session_id, snapshot)
            return HandlerResult(metadata={"cancelled": True})

        handlers: dict[str, Any] = {
            "run_round": _run_round,
            "message": _message,
            "approve": _approve,
            "export": _export,
            "reset": _reset,
            "cancel": _cancel,
        }
        handler = handlers.get(command)
        if handler is None:
            raise HTTPException(status_code=400, detail={"reason": "unknown_command", "command": command})
        try:
            return await execute_command(
                store=store,
                session_id=session_id,
                command=command,
                command_id=command_id,
                payload=payload,
                lease_seconds=COMMAND_LEASE_SECONDS,
                heartbeat_seconds=COMMAND_HEARTBEAT_SECONDS,
                allowed_commands_by_status=COMMAND_MATRIX,
                handler=handler,
                build_snapshot=lambda sid: _build_session_snapshot(sid, store),
                emit_event=_emit_session_event,
            )
        except (InvalidTransitionError, InvalidCommandError, ApprovalBlockedError) as exc:
            session = store.get_planning_session(session_id)
            detail: dict[str, Any] = {"reason": "invalid_status", "detail": str(exc)}
            if session is not None:
                detail.update(
                    {
                        "status": session.status,
                        "phase_message": session.phase_message,
                        "allowed_commands": _allowed_commands_for(session, store),
                    }
                )
            raise HTTPException(status_code=409, detail=detail) from exc
        except CommandConflictError as exc:
            session = store.get_planning_session(session_id)
            detail: dict[str, Any] = {"reason": exc.reason}
            if session is not None:
                detail.update(
                    {
                        "status": session.status,
                        "phase_message": session.phase_message,
                        "allowed_commands": _allowed_commands_for(session, store),
                    }
                )
            raise HTTPException(status_code=409, detail=detail) from exc

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy"}

    @app.get("/api/models")
    async def get_models() -> dict[str, Any]:
        return {"items": list_models()}

    @app.get("/api/repos")
    async def list_repos() -> dict[str, Any]:
        config = PrscopeConfig.load(registry._config_root())
        repos = config.list_repos()
        cwd_path = Path.cwd()
        return {
            "cwd": {
                "name": cwd_path.name,
                "path": str(cwd_path),
            },
            "items": [
                {
                    "name": repo.name,
                    "path": str(repo.resolved_path),
                }
                for repo in repos
            ],
        }

    @app.get("/api/sessions")
    async def list_sessions(
        repo: Optional[str] = Query(default=None),
        status: Optional[str] = Query(default=None),
    ) -> dict[str, Any]:
        # When launched outside a repo (e.g. from ~), still allow listing global sessions.
        if repo is None:
            store = Store()
            sessions = store.list_planning_sessions(status=status, limit=200)
            return {"items": [_session_to_dict(s) for s in sessions]}
        _, store, _ = _runtime_for(repo_name=repo)
        sessions = store.list_planning_sessions(repo_name=repo, status=status, limit=200)
        return {"items": [_session_to_dict(s) for s in sessions]}

    @app.post("/api/sessions")
    async def create_session(payload: StartSessionRequest) -> dict[str, Any]:
        request_started = time.perf_counter()
        try:
            runtime_started = time.perf_counter()
            _, _, runtime = _runtime_for(repo_name=payload.repo)
            runtime_elapsed = time.perf_counter() - runtime_started
            author_model = _validated_model_or_400(payload.author_model, "author")
            critic_model = _validated_model_or_400(payload.critic_model, "critic")
            if payload.mode == "requirements":
                if not payload.requirements:
                    raise HTTPException(status_code=400, detail="requirements is required for mode=requirements")
                create_started = time.perf_counter()
                session = await runtime.create_requirements_session(
                    requirements=payload.requirements,
                    author_model=author_model,
                    critic_model=critic_model,
                    rebuild_memory=payload.rebuild_memory,
                )
                create_elapsed = time.perf_counter() - create_started

                async def callback(event: dict[str, Any]) -> None:
                    enriched = dict(event)
                    enriched.setdefault("command_id", f"initial-draft:{session.id}")
                    registry.note_event(session.id, enriched)
                    await emitter.emit(session.id, enriched)

                draft_started = time.perf_counter()
                draft_started_wall = time.time()
                timing = registry._ensure_session_timing(session.id)
                timing.update(
                    {
                        "initial_draft_started_at_unix_s": draft_started_wall,
                        "initial_draft_elapsed_s": None,
                        "initial_draft_completed_at_unix_s": None,
                        "initial_draft_failed": False,
                    }
                )

                async def _run_initial_draft_task() -> None:
                    try:
                        await runtime.continue_requirements_draft(
                            session_id=session.id,
                            requirements=payload.requirements or "",
                            author_model_override=author_model,
                            event_callback=callback,
                            rebuild_memory=payload.rebuild_memory,
                        )
                        logger.info(
                            "api.sessions draft complete session_id={} elapsed_s={:.3f}",
                            session.id,
                            time.perf_counter() - draft_started,
                        )
                        timing = registry._ensure_session_timing(session.id)
                        timing.update(
                            {
                                "initial_draft_started_at_unix_s": draft_started_wall,
                                "initial_draft_elapsed_s": round(time.perf_counter() - draft_started, 3),
                                "initial_draft_completed_at_unix_s": time.time(),
                                "initial_draft_failed": False,
                            }
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.error(
                            "api.sessions draft failed session_id={} elapsed_s={:.3f} error={}",
                            session.id,
                            time.perf_counter() - draft_started,
                            exc,
                        )
                        timing = registry._ensure_session_timing(session.id)
                        timing.update(
                            {
                                "initial_draft_started_at_unix_s": draft_started_wall,
                                "initial_draft_elapsed_s": round(time.perf_counter() - draft_started, 3),
                                "initial_draft_completed_at_unix_s": time.time(),
                                "initial_draft_failed": True,
                            }
                        )
                        raise

                task = asyncio.create_task(_run_initial_draft_task())
                task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
                logger.info(
                    "api.sessions created mode=requirements session_id={} runtime_s={:.3f} create_s={:.3f} total_s={:.3f}",
                    session.id,
                    runtime_elapsed,
                    create_elapsed,
                    time.perf_counter() - request_started,
                )
                return {"session": _session_to_dict(session)}
            if payload.mode == "pr":
                if not payload.upstream_repo or payload.pr_number is None:
                    raise HTTPException(
                        status_code=400,
                        detail="upstream_repo and pr_number are required for mode=pr",
                    )
                session = await runtime.start_from_pr(
                    upstream_repo=payload.upstream_repo,
                    pr_number=payload.pr_number,
                    author_model=author_model,
                    critic_model=critic_model,
                    rebuild_memory=payload.rebuild_memory,
                )
                logger.info(
                    "api.sessions created mode=pr session_id={} runtime_s={:.3f} total_s={:.3f}",
                    session.id,
                    runtime_elapsed,
                    time.perf_counter() - request_started,
                )
                return {"session": _session_to_dict(session)}
            session, opening = await runtime.start_from_chat(
                author_model=author_model,
                critic_model=critic_model,
                rebuild_memory=payload.rebuild_memory,
            )

            # Chat session starts as draft; memory build runs in background, progress via SSE.
            async def emit_setup_event(event: dict[str, Any]) -> None:
                registry.note_event(session.id, event)
                await emitter.emit(session.id, event)

            async def run_chat_setup() -> None:
                try:
                    await runtime.continue_chat_setup(
                        session.id,
                        rebuild_memory=payload.rebuild_memory,
                        event_callback=emit_setup_event,
                    )
                    logger.info(
                        "api.sessions chat setup complete session_id=%s",
                        session.id,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("api.sessions chat setup failed session_id=%s", session.id)
                    await emit_setup_event({"type": "error", "message": str(exc)})

            asyncio.create_task(run_chat_setup())
            logger.info(
                "api.sessions created mode=chat session_id={} runtime_s={:.3f} total_s={:.3f}",
                session.id,
                runtime_elapsed,
                time.perf_counter() - request_started,
            )
            return {"session": _session_to_dict(session), "opening": opening}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/sessions/{session_id}")
    async def get_session(
        session_id: str,
        repo: Optional[str] = Query(default=None),
        lightweight: bool = Query(default=False),
    ) -> dict[str, Any]:
        if repo is None:
            store = Store()
        else:
            _, store, _ = _runtime_for(repo_name=repo)
        session = store.get_planning_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        versions = store.get_plan_versions(session_id, limit=200)
        current_plan = versions[0] if versions else None
        timing = registry._session_timing.get(session_id)
        if lightweight:
            return {
                "session": _session_to_dict(session),
                "current_plan": _version_to_dict(current_plan) if current_plan else None,
                "draft_timing": timing,
            }
        turns = store.get_planning_turns(session_id)
        round_metrics_rows = store.get_round_metrics(session_id)
        score_by_round = {v.round: v.convergence_score for v in versions}  # O(n) lookup
        return {
            "session": _session_to_dict(session),
            "conversation": [_turn_to_dict(t) for t in turns],
            "plan_versions": [_version_to_dict(v) for v in versions],
            "current_plan": _version_to_dict(current_plan) if current_plan else None,
            "draft_timing": timing,
            "round_metrics": [
                {
                    "round": m.round,
                    "major_issues": m.major_issues,
                    "minor_issues": m.minor_issues,
                    "critic_confidence": m.critic_confidence,
                    "convergence_score": score_by_round.get(m.round),
                    "call_cost_usd": m.call_cost_usd,
                }
                for m in round_metrics_rows
            ],
            "tool_summary": {
                "recent_tool_calls": [
                    t.content[:240]
                    for t in turns
                    if t.role in {"author", "critic"} and ("tool" in t.content.lower() or "search_" in t.content)
                ][:20]
            },
        }

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str, repo: Optional[str] = Query(default=None)) -> dict[str, Any]:
        if repo is None:
            store = Store()
        else:
            _, store, _ = _runtime_for(repo_name=repo)
        deleted = store.delete_planning_session(session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="session not found")
        registry.cleanup_session(session_id)
        return {"deleted": True}

    @app.post("/api/sessions/{session_id}/stop")
    async def stop_session(session_id: str, repo: Optional[str] = Query(default=None)) -> dict[str, Any]:
        repo_name = _repo_for_session(session_id, repo)
        payload = await _execute_registered_command(
            session_id=session_id,
            command="cancel",
            command_id=f"cancel-{uuid4()}",
            repo_name=repo_name,
            payload={},
        )
        return {"stopped": bool(payload.get("cancelled", False)), "snapshot": payload}

    @app.post("/api/sessions/{session_id}/command")
    async def command_session(session_id: str, payload: CommandRequest) -> dict[str, Any]:
        repo_name = _repo_for_session(session_id, payload.repo)
        command = str(payload.command or "").strip()
        if not command:
            raise HTTPException(status_code=400, detail={"reason": "unknown_command"})
        return await _execute_registered_command(
            session_id=session_id,
            command=command,
            command_id=payload.command_id,
            repo_name=repo_name,
            payload={
                "user_input": payload.user_input,
                "message": payload.message,
                "author_model": payload.author_model,
                "critic_model": payload.critic_model,
                "unverified_references": payload.unverified_references,
            },
        )

    @app.post("/api/sessions/{session_id}/message")
    async def send_message(session_id: str, payload: MessageRequest) -> dict[str, Any]:
        repo_name = _repo_for_session(session_id, payload.repo)
        return await _execute_registered_command(
            session_id=session_id,
            command="message",
            command_id=payload.command_id,
            repo_name=repo_name,
            payload={
                "message": payload.message,
                "author_model": payload.author_model,
                "critic_model": payload.critic_model,
            },
        )

    @app.post("/api/sessions/{session_id}/round")
    async def run_round(session_id: str, payload: RoundRequest) -> dict[str, Any]:
        repo_name = _repo_for_session(session_id, payload.repo)
        result = await _execute_registered_command(
            session_id=session_id,
            command="run_round",
            command_id=payload.command_id,
            repo_name=repo_name,
            payload={
                "user_input": payload.user_input,
                "author_model": payload.author_model,
                "critic_model": payload.critic_model,
            },
        )
        return result

    @app.post("/api/sessions/{session_id}/clarify")
    async def submit_clarification(session_id: str, payload: ClarifyRequest) -> dict[str, Any]:
        repo_name = _repo_for_session(session_id, payload.repo)
        _, _, runtime = _runtime_for(repo_name=repo_name)
        runtime.provide_clarification(session_id, payload.answers)
        return {"ok": True}

    @app.post("/api/sessions/{session_id}/approve")
    async def approve_session(session_id: str, payload: ApproveRequest) -> dict[str, Any]:
        repo_name = _repo_for_session(session_id, payload.repo)
        return await _execute_registered_command(
            session_id=session_id,
            command="approve",
            command_id=payload.command_id,
            repo_name=repo_name,
            payload={"unverified_references": payload.unverified_references},
        )

    @app.post("/api/sessions/{session_id}/export")
    async def export_session(session_id: str, payload: ExportRequest) -> dict[str, Any]:
        repo_name = _repo_for_session(session_id, payload.repo)
        command_id = payload.command_id or f"export-{uuid4()}"
        return await _execute_registered_command(
            session_id=session_id,
            command="export",
            command_id=command_id,
            repo_name=repo_name,
            payload={},
        )

    @app.get("/api/sessions/{session_id}/download/{kind}")
    async def download_export(
        session_id: str,
        kind: str,
        repo: Optional[str] = Query(default=None),
    ) -> StreamingResponse:
        repo_name = _repo_for_session(session_id, repo)
        _, _, runtime = _runtime_for(repo_name=repo_name)
        try:
            paths = await asyncio.to_thread(runtime.export, session_id)
        except InvalidTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        path = paths.get(kind)
        if path is None or not path.exists():
            raise HTTPException(status_code=404, detail="file not found")

        filename = path.name

        def iter_file() -> Any:
            with path.open("rb") as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    yield chunk

        return StreamingResponse(
            iter_file(),
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/sessions/{session_id}/diff")
    async def get_diff(
        session_id: str,
        round_number: Optional[int] = Query(default=None),
        repo: Optional[str] = Query(default=None),
    ) -> dict[str, Any]:
        repo_name = _repo_for_session(session_id, repo)
        _, _, runtime = _runtime_for(repo_name=repo_name)
        diff = runtime.plan_diff(session_id, round_number=round_number)
        return {"diff": diff}

    @app.get("/api/sessions/{session_id}/events")
    async def stream_events(session_id: str) -> EventSourceResponse:
        async def event_generator() -> Any:
            try:
                session = Store().get_planning_session(session_id)
                if session is not None:
                    snapshot = _build_session_snapshot(session_id, Store())
                    yield {
                        "event": "session_state",
                        "data": json.dumps({k: v for k, v in snapshot.items() if k != "type"}),
                    }
                async for event in emitter.subscribe(session_id):
                    etype = str(event.get("type", "message"))
                    payload = {k: v for k, v in event.items() if k != "type"}
                    yield {
                        "event": etype,
                        "data": json.dumps(payload),
                    }
            finally:
                for runtime in registry._runtimes.values():
                    runtime.abort_clarification(session_id)

        return EventSourceResponse(event_generator())

    return app
