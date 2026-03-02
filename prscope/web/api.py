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


class MessageRequest(BaseModel):
    message: str
    repo: Optional[str] = None


class RoundRequest(BaseModel):
    user_input: Optional[str] = None
    repo: Optional[str] = None


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

    def _key(self, repo_name: str) -> str:
        return repo_name

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

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
            if payload.mode == "requirements":
                if not payload.requirements:
                    raise HTTPException(status_code=400, detail="requirements is required for mode=requirements")
                create_started = time.perf_counter()
                session = await runtime.create_requirements_session(
                    requirements=payload.requirements,
                    rebuild_memory=payload.rebuild_memory,
                )
                create_elapsed = time.perf_counter() - create_started

                async def callback(event: dict[str, Any]) -> None:
                    await emitter.emit(session.id, event)

                draft_started = time.perf_counter()

                async def _run_initial_draft_task() -> None:
                    try:
                        await runtime.continue_requirements_draft(
                            session_id=session.id,
                            requirements=payload.requirements or "",
                            event_callback=callback,
                            rebuild_memory=payload.rebuild_memory,
                        )
                        logger.info(
                            "api.sessions draft complete session_id={} elapsed_s={:.3f}",
                            session.id,
                            time.perf_counter() - draft_started,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.error(
                            "api.sessions draft failed session_id={} elapsed_s={:.3f} error={}",
                            session.id,
                            time.perf_counter() - draft_started,
                            exc,
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
                    rebuild_memory=payload.rebuild_memory,
                )
                logger.info(
                    "api.sessions created mode=pr session_id={} runtime_s={:.3f} total_s={:.3f}",
                    session.id,
                    runtime_elapsed,
                    time.perf_counter() - request_started,
                )
                return {"session": _session_to_dict(session)}
            session, opening = await runtime.start_from_chat(rebuild_memory=payload.rebuild_memory)
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
    async def get_session(session_id: str, repo: Optional[str] = Query(default=None)) -> dict[str, Any]:
        if repo is None:
            store = Store()
        else:
            _, store, _ = _runtime_for(repo_name=repo)
        session = store.get_planning_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        turns = store.get_planning_turns(session_id)
        versions = store.get_plan_versions(session_id, limit=200)
        current_plan = versions[0] if versions else None
        return {
            "session": _session_to_dict(session),
            "conversation": [_turn_to_dict(t) for t in turns],
            "plan_versions": [_version_to_dict(v) for v in versions],
            "current_plan": _version_to_dict(current_plan) if current_plan else None,
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

            async def callback(event: dict[str, Any]) -> None:
                await emitter.emit(session_id, event)

            result = await runtime.handle_discovery_turn(
                session_id=session_id,
                user_message=payload.message,
                event_callback=callback,
            )
            return {"result": asdict(result)}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/sessions/{session_id}/round")
    async def run_round(session_id: str, payload: RoundRequest) -> dict[str, Any]:
        repo_name = _repo_for_session(session_id, payload.repo)
        _, _, runtime = _runtime_for(repo_name=repo_name)

        async def callback(event: dict[str, Any]) -> None:
            await emitter.emit(session_id, event)

        try:
            critic, author, convergence = await runtime.run_adversarial_round(
                session_id=session_id,
                user_input=payload.user_input,
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

