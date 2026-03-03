"""
FastAPI transport layer for prscope planning runtime.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from loguru import logger

from ..config import PrscopeConfig, get_repo_root
from ..model_catalog import get_model, list_models
from ..planning.core import ApprovalBlockedError, InvalidTransitionError
from ..planning.runtime import PlanningRuntime
from ..store import Store
from .events import SessionEventEmitter


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
    repo: Optional[str] = None
    author_model: Optional[str] = None
    critic_model: Optional[str] = None


class RoundRequest(BaseModel):
    user_input: Optional[str] = None
    repo: Optional[str] = None
    author_model: Optional[str] = None
    critic_model: Optional[str] = None


class ApproveRequest(BaseModel):
    repo: Optional[str] = None
    unverified_references: list[str] = Field(default_factory=list)


class ExportRequest(BaseModel):
    repo: Optional[str] = None


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
                timing["author_fallback_warnings"] = int(
                    timing.get("author_fallback_warnings", 0)
                ) + 1
            return
        if etype == "token_usage":
            timing["llm_calls_total"] = int(timing.get("llm_calls_total", 0)) + 1
            latency = float(event.get("llm_call_latency_ms", 0.0) or 0.0)
            timing["llm_call_latency_ms_total"] = float(
                timing.get("llm_call_latency_ms_total", 0.0)
            ) + latency
            timing["llm_call_latency_ms_max"] = max(
                float(timing.get("llm_call_latency_ms_max", 0.0)),
                latency,
            )
            return
        if etype == "tool_result":
            timing["tool_calls_total"] = int(timing.get("tool_calls_total", 0)) + 1
            latency = float(event.get("duration_ms", 0.0) or 0.0)
            timing["tool_exec_latency_ms_total"] = float(
                timing.get("tool_exec_latency_ms_total", 0.0)
            ) + latency
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
    return asdict(session)


def _turn_to_dict(turn: Any) -> dict[str, Any]:
    return asdict(turn)


def _version_to_dict(version: Any) -> dict[str, Any]:
    return asdict(version)


def create_app() -> FastAPI:
    app = FastAPI(title="prscope-web", version="0.1.0")
    emitter = SessionEventEmitter()
    registry = RuntimeRegistry(emitter)

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
                detail=(
                    f"Invalid {role} model '{model_id}'. "
                    "Pick a model from /api/models."
                ),
            )
        if not bool(model.get("available")):
            reason = str(model.get("unavailable_reason") or "provider key unavailable")
            raise HTTPException(
                status_code=400,
                detail=f"Selected {role} model '{model_id}' is unavailable: {reason}",
            )
        return model_id

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
            ]
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
                    registry.note_event(session.id, event)
                    await emitter.emit(session.id, event)

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

                task = asyncio.create_task(
                    _run_initial_draft_task()
                )
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
            # Chat session starts as "preparing"; memory build runs in background, progress via SSE
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
        return {
            "session": _session_to_dict(session),
            "conversation": [_turn_to_dict(t) for t in turns],
            "plan_versions": [_version_to_dict(v) for v in versions],
            "current_plan": _version_to_dict(current_plan) if current_plan else None,
            "draft_timing": timing,
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

    @app.post("/api/sessions/{session_id}/message")
    async def send_message(session_id: str, payload: MessageRequest) -> dict[str, Any]:
        try:
            repo_name = _repo_for_session(session_id, payload.repo)
            _, _, runtime = _runtime_for(repo_name=repo_name)
            session = Store().get_planning_session(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="session not found")
            author_model = _validated_model_or_400(
                payload.author_model or session.author_model,
                "author",
            )
            critic_model = _validated_model_or_400(
                payload.critic_model or session.critic_model,
                "critic",
            )

            async def callback(event: dict[str, Any]) -> None:
                registry.note_event(session_id, event)
                await emitter.emit(session_id, event)

            result = await runtime.handle_discovery_turn(
                session_id=session_id,
                user_message=payload.message,
                author_model_override=author_model,
                critic_model_override=critic_model,
                event_callback=callback,
                defer_initial_draft=True,
            )
            logger.info(
                "api.sessions discovery turn session_id={} complete={} questions={}",
                session_id,
                result.complete,
                len(result.questions),
            )
            if result.complete:
                draft_requirements = result.summary or payload.message
                logger.info(
                    "api.sessions discovery complete session_id={} scheduling initial draft",
                    session_id,
                )

                async def _continue_discovery_draft_task() -> None:
                    try:
                        await runtime.continue_discovery_draft(
                            session_id=session_id,
                            requirements=draft_requirements,
                            author_model_override=author_model,
                            event_callback=callback,
                        )
                        logger.info(
                            "api.sessions discovery draft complete session_id={}",
                            session_id,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.exception(
                            "api.sessions discovery draft failed session_id=%s",
                            session_id,
                        )
                        await callback({"type": "error", "message": f"Initial draft failed: {exc}"})

                task = asyncio.create_task(_continue_discovery_draft_task())
                task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            return {"result": asdict(result)}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/sessions/{session_id}/round")
    async def run_round(session_id: str, payload: RoundRequest) -> dict[str, Any]:
        repo_name = _repo_for_session(session_id, payload.repo)
        _, _, runtime = _runtime_for(repo_name=repo_name)
        session = Store().get_planning_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        author_model = _validated_model_or_400(
            payload.author_model or session.author_model,
            "author",
        )
        critic_model = _validated_model_or_400(
            payload.critic_model or session.critic_model,
            "critic",
        )

        async def callback(event: dict[str, Any]) -> None:
            registry.note_event(session_id, event)
            await emitter.emit(session_id, event)

        try:
            critic, author, convergence = await runtime.run_adversarial_round(
                session_id=session_id,
                user_input=payload.user_input,
                author_model_override=author_model,
                critic_model_override=critic_model,
                event_callback=callback,
            )
        except InvalidTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "critic": asdict(critic),
            "author": asdict(author),
            "convergence": asdict(convergence),
        }

    @app.post("/api/sessions/{session_id}/clarify")
    async def submit_clarification(session_id: str, payload: ClarifyRequest) -> dict[str, Any]:
        repo_name = _repo_for_session(session_id, payload.repo)
        _, _, runtime = _runtime_for(repo_name=repo_name)
        runtime.provide_clarification(session_id, payload.answers)
        return {"ok": True}

    @app.post("/api/sessions/{session_id}/approve")
    async def approve_session(session_id: str, payload: ApproveRequest) -> dict[str, Any]:
        repo_name = _repo_for_session(session_id, payload.repo)
        _, _, runtime = _runtime_for(repo_name=repo_name)
        try:
            runtime.approve(
                session_id=session_id,
                unverified_references=set(payload.unverified_references),
            )
        except ApprovalBlockedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except InvalidTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"approved": True}

    @app.post("/api/sessions/{session_id}/export")
    async def export_session(session_id: str, payload: ExportRequest) -> dict[str, Any]:
        repo_name = _repo_for_session(session_id, payload.repo)
        _, _, runtime = _runtime_for(repo_name=repo_name)
        try:
            paths = await asyncio.to_thread(runtime.export, session_id)
        except InvalidTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "files": [
                {"name": "prd.md", "kind": "prd", "url": f"/api/sessions/{session_id}/download/prd"},
                {"name": "rfc.md", "kind": "rfc", "url": f"/api/sessions/{session_id}/download/rfc"},
                {
                    "name": "conversation.md",
                    "kind": "conversation",
                    "url": f"/api/sessions/{session_id}/download/conversation",
                },
            ],
            "paths": {k: str(v) for k, v in paths.items()},
        }

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

