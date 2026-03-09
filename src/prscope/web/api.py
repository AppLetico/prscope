"""
FastAPI transport layer for prscope planning runtime.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
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
from ..planning.runtime.review import build_impact_view
from ..store import Store
from .events import SessionEventEmitter

COMMAND_LEASE_SECONDS = 120
COMMAND_HEARTBEAT_SECONDS = 60


def _to_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _summarize_background_failure(action: str, exc: Exception) -> str:
    detail = " ".join(str(exc).split()).strip() or exc.__class__.__name__
    detail = detail[:237].rstrip() + "..." if len(detail) > 240 else detail
    return f"{action} failed: {detail}"


async def _surface_background_session_failure(
    *,
    session_id: str,
    store: Store,
    action: str,
    exc: Exception,
    emit_event: Callable[[str, dict[str, Any]], Awaitable[None]],
    build_snapshot: Callable[[str, Store], dict[str, Any]],
) -> None:
    phase_message = _summarize_background_failure(action, exc)
    persist_error: Exception | None = None
    for attempt in range(5):
        try:
            store.update_planning_session(
                session_id,
                _bypass_protection=True,
                status="error",
                phase_message=phase_message,
                is_processing=0,
                processing_started_at=None,
                pending_questions_json=None,
                active_tool_calls_json=json.dumps([]),
            )
            persist_error = None
            break
        except sqlite3.OperationalError as locked_exc:
            persist_error = locked_exc
            if "locked" not in str(locked_exc).lower() or attempt == 4:
                break
            await asyncio.sleep(0.05)
        except Exception as unexpected_exc:  # noqa: BLE001
            persist_error = unexpected_exc
            break
    if persist_error is not None:
        logger.error(
            "api.sessions failed surfacing background error state session_id={} action={} error={}",
            session_id,
            action,
            persist_error,
        )
        return
    await emit_event(session_id, build_snapshot(session_id, store))
    await emit_event(session_id, {"type": "error", "message": phase_message})


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
    plan_version_id: Optional[int] = None
    followup_id: Optional[str] = None
    followup_answer: Optional[str] = None
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
            "memory_prep_started_at_unix_s": None,
            "memory_prep_elapsed_s": None,
            "memory_prep_completed_at_unix_s": None,
            "memory_prep_failed": False,
            "chat_setup_started_at_unix_s": None,
            "chat_setup_elapsed_s": None,
            "chat_setup_completed_at_unix_s": None,
            "chat_setup_failed": False,
            "discovery_started_at_unix_s": None,
            "discovery_elapsed_s": None,
            "discovery_completed_at_unix_s": None,
            "discovery_failed": False,
            "discovery_llm_calls": 0,
            "discovery_llm_latency_ms_total": 0.0,
            "discovery_llm_latency_ms_max": 0.0,
            "discovery_tool_calls": 0,
            "discovery_tool_latency_ms_total": 0.0,
            "discovery_tool_latency_ms_max": 0.0,
            "initial_draft_started_at_unix_s": None,
            "initial_draft_elapsed_s": None,
            "initial_draft_completed_at_unix_s": None,
            "initial_draft_failed": False,
            "draft_phase": None,
            "draft_phase_started_at_unix_s": None,
            "draft_memory_elapsed_s": None,
            "draft_planner_elapsed_s": None,
            "draft_cache_hit": None,
            "draft_failed_stage": None,
            "draft_author_calls": 0,
            "draft_author_latency_ms_total": 0.0,
            "draft_author_latency_ms_max": 0.0,
            "draft_redraft_count": 0,
            "draft_complexity": None,
            "evidence_bundle_files_count": 0,
            "evidence_bundle_test_targets_count": 0,
            "author_internal_attempts": 0,
            "author_self_review_used": False,
            "draft_redraft_reason_codes": [],
            "quality_gate_failures": [],
            "initial_draft_total_ms": None,
            "draft_loop_budget_ms": None,
            "stability_stop": False,
            "stability_reason_codes": [],
            "initial_draft_model": None,
            "initial_draft_provider": None,
            "initial_draft_fallback_model": None,
            "author_refine_model": None,
            "author_refine_provider": None,
            "author_refine_fallback_model": None,
            "critic_review_model": None,
            "critic_review_provider": None,
            "critic_review_fallback_model": None,
            "memory_model": None,
            "memory_provider": None,
            "discovery_model": None,
            "discovery_provider": None,
            "author_refine_json_retries": 0,
            "author_refine_fallbacks": 0,
            "author_refine_last_failure_category": None,
            "critic_review_json_retries": 0,
            "critic_review_fallbacks": 0,
            "critic_review_last_failure_category": None,
            "warnings_total": 0,
            "errors_total": 0,
            "author_call_timeouts": 0,
            "author_fallback_warnings": 0,
            "routing_decisions_total": 0,
            "routing_heuristic_decisions": 0,
            "routing_model_decisions": 0,
            "routing_fallback_decisions": 0,
            "route_author_chat_total": 0,
            "route_lightweight_refine_total": 0,
            "route_full_refine_total": 0,
            "route_existing_feature_total": 0,
            "refinement_turns_total": 0,
            "investigation_trigger_total": 0,
            "investigation_skip_total": 0,
            "investigation_trigger_reason_last": None,
            "investigation_trigger_reason_counts": {},
            "investigation_trigger_rate": 0.0,
            "refinement_turn_tokens_total": 0,
            "average_refinement_turn_tokens": 0.0,
            "options_memo_deferred": True,
            "plan_patch_evaluation_status": "deferred_until_post_benchmark",
            "llm_calls_total": 0,
            "llm_call_latency_ms_total": 0.0,
            "llm_call_latency_ms_max": 0.0,
            "tool_calls_total": 0,
            "tool_exec_latency_ms_total": 0.0,
            "tool_exec_latency_ms_max": 0.0,
            "first_plan_saved_at_unix_s": None,
            "first_plan_round": None,
            "timing_updated_at_unix_s": None,
        }

    def _find_store_for_session(self, session_id: str) -> Store | None:
        for store in self._stores.values():
            try:
                if store.get_planning_session(session_id) is not None:
                    return store
            except Exception:  # noqa: BLE001
                continue
        return None

    def _load_persisted_session_timing(self, session_id: str, store: Store | None = None) -> dict[str, Any]:
        selected_store = store or self._find_store_for_session(session_id)
        base = self._default_session_timing()
        if selected_store is None:
            return base
        session = selected_store.get_planning_session(session_id)
        if session is None:
            return base
        raw = getattr(session, "diagnostics_json", None)
        if not raw:
            return base
        try:
            parsed = json.loads(str(raw))
        except json.JSONDecodeError:
            return base
        if not isinstance(parsed, dict):
            return base
        base.update(parsed)
        return base

    def _persist_session_timing(self, session_id: str, timing: dict[str, Any], store: Store | None = None) -> None:
        selected_store = store or self._find_store_for_session(session_id)
        if selected_store is None:
            return
        selected_store.update_planning_session(
            session_id,
            diagnostics_json=json.dumps(timing, separators=(",", ":")),
        )

    def get_session_timing_with_source(self, session_id: str, store: Store | None = None) -> tuple[dict[str, Any], str]:
        timing = self._session_timing.get(session_id)
        if timing is not None:
            return timing, "live_memory"
        selected_store = store or self._find_store_for_session(session_id)
        if selected_store is not None:
            session = selected_store.get_planning_session(session_id)
            if session is not None and getattr(session, "diagnostics_json", None):
                timing = self._load_persisted_session_timing(session_id, store=selected_store)
                self._session_timing[session_id] = timing
                return timing, "persisted_session"
        timing = self._default_session_timing()
        self._session_timing[session_id] = timing
        return timing, "default"

    def get_session_timing(self, session_id: str, store: Store | None = None) -> dict[str, Any] | None:
        timing, _source = self.get_session_timing_with_source(session_id, store=store)
        return timing

    def _ensure_session_timing(self, session_id: str, store: Store | None = None) -> dict[str, Any]:
        timing = self.get_session_timing(session_id, store=store)
        if timing is None:
            timing = self._default_session_timing()
            self._session_timing[session_id] = timing
        return timing

    def note_event(self, session_id: str, event: dict[str, Any], *, store: Store | None = None) -> None:
        etype = str(event.get("type", "")).strip().lower()
        if not etype:
            return
        timing = self._ensure_session_timing(session_id, store=store)
        changed = False
        if etype == "warning":
            timing["warnings_total"] = int(timing.get("warnings_total", 0)) + 1
            changed = True
            message = str(event.get("message", "")).lower()
            if "author call timeout on" in message:
                timing["author_call_timeouts"] = int(timing.get("author_call_timeouts", 0)) + 1
                changed = True
            if "trying fallback model" in message or "author fallback used" in message:
                timing["author_fallback_warnings"] = int(timing.get("author_fallback_warnings", 0)) + 1
                changed = True
        elif etype == "token_usage":
            timing["llm_calls_total"] = int(timing.get("llm_calls_total", 0)) + 1
            latency = float(event.get("llm_call_latency_ms", 0.0) or 0.0)
            prompt_tokens = int(event.get("prompt_tokens", 0) or 0)
            completion_tokens = int(event.get("completion_tokens", 0) or 0)
            timing["llm_call_latency_ms_total"] = float(timing.get("llm_call_latency_ms_total", 0.0)) + latency
            timing["llm_call_latency_ms_max"] = max(
                float(timing.get("llm_call_latency_ms_max", 0.0)),
                latency,
            )
            selected_store = store or self._find_store_for_session(session_id)
            current_session = selected_store.get_planning_session(session_id) if selected_store is not None else None
            current_status = str(getattr(current_session, "status", "")).strip().lower()
            if current_status in {"refining", "converged"}:
                timing["refinement_turn_tokens_total"] = (
                    int(timing.get("refinement_turn_tokens_total", 0)) + prompt_tokens + completion_tokens
                )
                turns_total = int(timing.get("refinement_turns_total", 0) or 0)
                timing["average_refinement_turn_tokens"] = (
                    round(float(timing["refinement_turn_tokens_total"]) / float(turns_total), 2)
                    if turns_total > 0
                    else 0.0
                )
            session_stage = str(event.get("session_stage", "")).strip().lower()
            if session_stage == "author":
                timing["draft_author_calls"] = int(timing.get("draft_author_calls", 0)) + 1
                timing["draft_author_latency_ms_total"] = (
                    float(timing.get("draft_author_latency_ms_total", 0.0)) + latency
                )
                timing["draft_author_latency_ms_max"] = max(
                    float(timing.get("draft_author_latency_ms_max", 0.0)),
                    latency,
                )
            elif session_stage == "discovery":
                timing["discovery_model"] = str(event.get("model", "")).strip() or timing.get("discovery_model")
                timing["discovery_provider"] = str(event.get("model_provider", "")).strip() or timing.get(
                    "discovery_provider"
                )
                timing["discovery_llm_calls"] = int(timing.get("discovery_llm_calls", 0)) + 1
                timing["discovery_llm_latency_ms_total"] = (
                    float(timing.get("discovery_llm_latency_ms_total", 0.0)) + latency
                )
                timing["discovery_llm_latency_ms_max"] = max(
                    float(timing.get("discovery_llm_latency_ms_max", 0.0)),
                    latency,
                )
            elif session_stage == "memory":
                timing["memory_model"] = str(event.get("model", "")).strip() or timing.get("memory_model")
                timing["memory_provider"] = str(event.get("model_provider", "")).strip() or timing.get(
                    "memory_provider"
                )
            changed = True
        elif etype == "setup_progress":
            draft_stage = str(event.get("draft_stage", "")).strip()
            if draft_stage:
                now = time.time()
                timing["draft_phase"] = draft_stage
                timing["draft_phase_started_at_unix_s"] = now
                if draft_stage == "memory":
                    if timing.get("memory_prep_started_at_unix_s") is None:
                        timing["memory_prep_started_at_unix_s"] = now
                        timing["memory_prep_elapsed_s"] = None
                        timing["memory_prep_completed_at_unix_s"] = None
                        timing["memory_prep_failed"] = False
                elif draft_stage in {"planner", "planner_draft", "planner_complete"}:
                    memory_started = timing.get("memory_prep_started_at_unix_s")
                    if timing.get("memory_prep_completed_at_unix_s") is None:
                        timing["memory_prep_completed_at_unix_s"] = now
                    if event.get("memory_elapsed_s") is not None:
                        timing["memory_prep_elapsed_s"] = float(event.get("memory_elapsed_s") or 0.0)
                    elif memory_started is not None:
                        try:
                            timing["memory_prep_elapsed_s"] = round(now - float(memory_started), 3)
                        except (TypeError, ValueError):
                            timing["memory_prep_elapsed_s"] = None
                    timing["memory_prep_failed"] = False
                    if timing.get("initial_draft_started_at_unix_s") is None:
                        timing["initial_draft_started_at_unix_s"] = now
                        timing["initial_draft_elapsed_s"] = None
                        timing["initial_draft_completed_at_unix_s"] = None
                        timing["initial_draft_failed"] = False
                if event.get("memory_elapsed_s") is not None:
                    timing["draft_memory_elapsed_s"] = float(event.get("memory_elapsed_s") or 0.0)
                if event.get("planner_elapsed_s") is not None:
                    timing["draft_planner_elapsed_s"] = float(event.get("planner_elapsed_s") or 0.0)
                if event.get("cache_hit") is not None:
                    timing["draft_cache_hit"] = bool(event.get("cache_hit"))
                if event.get("complexity") is not None:
                    timing["draft_complexity"] = str(event.get("complexity"))
                if draft_stage == "planner_redraft":
                    timing["draft_redraft_count"] = int(timing.get("draft_redraft_count", 0)) + 1
                changed = True
        elif etype == "tool_result":
            timing["tool_calls_total"] = int(timing.get("tool_calls_total", 0)) + 1
            latency = float(event.get("duration_ms", 0.0) or 0.0)
            timing["tool_exec_latency_ms_total"] = float(timing.get("tool_exec_latency_ms_total", 0.0)) + latency
            timing["tool_exec_latency_ms_max"] = max(
                float(timing.get("tool_exec_latency_ms_max", 0.0)),
                latency,
            )
            if str(event.get("session_stage", "")).strip().lower() == "discovery":
                timing["discovery_tool_calls"] = int(timing.get("discovery_tool_calls", 0)) + 1
                timing["discovery_tool_latency_ms_total"] = (
                    float(timing.get("discovery_tool_latency_ms_total", 0.0)) + latency
                )
                timing["discovery_tool_latency_ms_max"] = max(
                    float(timing.get("discovery_tool_latency_ms_max", 0.0)),
                    latency,
                )
            changed = True
        elif etype == "tool_update":
            tool = event.get("tool")
            if isinstance(tool, dict) and str(tool.get("status", "")).strip().lower() == "done":
                timing["tool_calls_total"] = int(timing.get("tool_calls_total", 0)) + 1
                latency = float(tool.get("durationMs", 0.0) or 0.0)
                timing["tool_exec_latency_ms_total"] = float(timing.get("tool_exec_latency_ms_total", 0.0)) + latency
                timing["tool_exec_latency_ms_max"] = max(
                    float(timing.get("tool_exec_latency_ms_max", 0.0)),
                    latency,
                )
                if str(tool.get("sessionStage", "")).strip().lower() == "discovery":
                    timing["discovery_tool_calls"] = int(timing.get("discovery_tool_calls", 0)) + 1
                    timing["discovery_tool_latency_ms_total"] = (
                        float(timing.get("discovery_tool_latency_ms_total", 0.0)) + latency
                    )
                    timing["discovery_tool_latency_ms_max"] = max(
                        float(timing.get("discovery_tool_latency_ms_max", 0.0)),
                        latency,
                    )
                changed = True
        elif etype == "routing_decision":
            timing["routing_decisions_total"] = int(timing.get("routing_decisions_total", 0)) + 1
            source = str(event.get("source", "")).strip().lower()
            route = str(event.get("route", "")).strip().lower()
            if source == "heuristic":
                timing["routing_heuristic_decisions"] = int(timing.get("routing_heuristic_decisions", 0)) + 1
            elif source == "model":
                timing["routing_model_decisions"] = int(timing.get("routing_model_decisions", 0)) + 1
            elif source == "fallback":
                timing["routing_fallback_decisions"] = int(timing.get("routing_fallback_decisions", 0)) + 1
            if route == "author_chat":
                timing["route_author_chat_total"] = int(timing.get("route_author_chat_total", 0)) + 1
            elif "lightweight" in route:
                timing["route_lightweight_refine_total"] = int(timing.get("route_lightweight_refine_total", 0)) + 1
            elif "full_refine" in route:
                timing["route_full_refine_total"] = int(timing.get("route_full_refine_total", 0)) + 1
            elif "existing_feature" in route or "proposal_review" in route or "revision_input" in route:
                timing["route_existing_feature_total"] = int(timing.get("route_existing_feature_total", 0)) + 1
            changed = True
        elif etype == "refinement_investigation":
            used = bool(event.get("used"))
            reason = str(event.get("trigger_reason", "")).strip()
            count_as_turn = bool(event.get("count_as_turn", True))
            if count_as_turn:
                timing["refinement_turns_total"] = int(timing.get("refinement_turns_total", 0)) + 1
            if used:
                timing["investigation_trigger_total"] = int(timing.get("investigation_trigger_total", 0)) + 1
                if reason:
                    reason_counts = timing.get("investigation_trigger_reason_counts")
                    if not isinstance(reason_counts, dict):
                        reason_counts = {}
                    reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
                    timing["investigation_trigger_reason_counts"] = reason_counts
                    timing["investigation_trigger_reason_last"] = reason
            else:
                if count_as_turn:
                    timing["investigation_skip_total"] = int(timing.get("investigation_skip_total", 0)) + 1
            turns_total = int(timing.get("refinement_turns_total", 0) or 0)
            trigger_total = int(timing.get("investigation_trigger_total", 0) or 0)
            timing["investigation_trigger_rate"] = (
                round(float(trigger_total) / float(turns_total), 4) if turns_total > 0 else 0.0
            )
            timing["average_refinement_turn_tokens"] = (
                round(float(timing.get("refinement_turn_tokens_total", 0) or 0) / float(turns_total), 2)
                if turns_total > 0
                else 0.0
            )
            changed = True
        elif etype == "phase_timing":
            stage = str(event.get("session_stage", "")).strip().lower()
            state = str(event.get("state", "")).strip().lower()
            now = time.time()
            elapsed_ms = event.get("elapsed_ms")
            elapsed_s: float | None = None
            if elapsed_ms is not None:
                try:
                    elapsed_s = round(float(elapsed_ms) / 1000.0, 3)
                except (TypeError, ValueError):
                    elapsed_s = None
            if stage == "chat_setup":
                if state == "start":
                    timing["chat_setup_started_at_unix_s"] = now
                    timing["chat_setup_elapsed_s"] = None
                    timing["chat_setup_completed_at_unix_s"] = None
                    timing["chat_setup_failed"] = False
                elif state == "complete":
                    timing["chat_setup_completed_at_unix_s"] = now
                    timing["chat_setup_elapsed_s"] = elapsed_s
                    timing["chat_setup_failed"] = False
                elif state == "failed":
                    timing["chat_setup_completed_at_unix_s"] = now
                    timing["chat_setup_elapsed_s"] = elapsed_s
                    timing["chat_setup_failed"] = True
                changed = True
            elif stage == "discovery":
                if state == "start":
                    timing["discovery_started_at_unix_s"] = now
                    timing["discovery_elapsed_s"] = None
                    timing["discovery_completed_at_unix_s"] = None
                    timing["discovery_failed"] = False
                elif state == "complete":
                    timing["discovery_completed_at_unix_s"] = now
                    timing["discovery_elapsed_s"] = elapsed_s
                    timing["discovery_failed"] = False
                elif state == "failed":
                    timing["discovery_completed_at_unix_s"] = now
                    timing["discovery_elapsed_s"] = elapsed_s
                    timing["discovery_failed"] = True
                changed = True
            elif stage == "initial_draft":
                if state == "start":
                    timing["initial_draft_started_at_unix_s"] = now
                    timing["initial_draft_elapsed_s"] = None
                    timing["initial_draft_completed_at_unix_s"] = None
                    timing["initial_draft_failed"] = False
                elif state == "complete":
                    timing["initial_draft_completed_at_unix_s"] = now
                    timing["initial_draft_elapsed_s"] = elapsed_s
                    timing["initial_draft_failed"] = False
                    if event.get("memory_elapsed_s") is not None:
                        timing["draft_memory_elapsed_s"] = float(event.get("memory_elapsed_s") or 0.0)
                    if event.get("planner_elapsed_s") is not None:
                        timing["draft_planner_elapsed_s"] = float(event.get("planner_elapsed_s") or 0.0)
                    if event.get("cache_hit") is not None:
                        timing["draft_cache_hit"] = bool(event.get("cache_hit"))
                    if timing.get("first_plan_saved_at_unix_s") is None:
                        timing["first_plan_saved_at_unix_s"] = now
                        timing["first_plan_round"] = 0
                elif state == "failed":
                    timing["initial_draft_completed_at_unix_s"] = now
                    timing["initial_draft_elapsed_s"] = elapsed_s
                    timing["initial_draft_failed"] = True
                    if event.get("failed_stage") is not None:
                        timing["draft_failed_stage"] = str(event.get("failed_stage"))
                changed = True
        elif etype == "plan_ready":
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
                changed = True
            draft_started = timing.get("initial_draft_started_at_unix_s")
            if draft_started is not None:
                try:
                    saved_at_value = float(
                        event.get("saved_at_unix_s") if event.get("saved_at_unix_s") is not None else time.time()
                    )
                    timing["initial_draft_completed_at_unix_s"] = saved_at_value
                    timing["initial_draft_elapsed_s"] = round(
                        float(timing.get("draft_planner_elapsed_s") or (saved_at_value - float(draft_started))), 3
                    )
                    timing["initial_draft_failed"] = False
                    changed = True
                except (TypeError, ValueError):
                    pass
        elif etype == "error":
            timing["errors_total"] = int(timing.get("errors_total", 0)) + 1
            if timing.get("first_plan_saved_at_unix_s") is None and timing.get("draft_phase"):
                timing["draft_failed_stage"] = timing.get("draft_phase")
            now = time.time()
            if str(timing.get("draft_phase", "")).strip() == "memory":
                started = timing.get("memory_prep_started_at_unix_s")
                timing["memory_prep_completed_at_unix_s"] = now
                if started is not None:
                    try:
                        timing["memory_prep_elapsed_s"] = round(now - float(started), 3)
                    except (TypeError, ValueError):
                        timing["memory_prep_elapsed_s"] = timing.get("memory_prep_elapsed_s")
                timing["memory_prep_failed"] = True
            elif timing.get("initial_draft_started_at_unix_s") is not None:
                started = timing.get("initial_draft_started_at_unix_s")
                timing["initial_draft_completed_at_unix_s"] = now
                if started is not None:
                    try:
                        timing["initial_draft_elapsed_s"] = round(
                            float(timing.get("draft_planner_elapsed_s") or (now - float(started))), 3
                        )
                    except (TypeError, ValueError):
                        timing["initial_draft_elapsed_s"] = timing.get("initial_draft_elapsed_s")
                timing["initial_draft_failed"] = True
            changed = True
        elif etype == "draft_diagnostics":
            int_fields = (
                "evidence_bundle_files_count",
                "evidence_bundle_test_targets_count",
                "author_internal_attempts",
                "initial_draft_total_ms",
                "draft_loop_budget_ms",
            )
            bool_fields = ("author_self_review_used", "stability_stop")
            list_fields = ("draft_redraft_reason_codes", "quality_gate_failures", "stability_reason_codes")
            for field_name in int_fields:
                raw_value = event.get(field_name)
                if raw_value is None:
                    continue
                try:
                    timing[field_name] = int(raw_value)
                    changed = True
                except (TypeError, ValueError):
                    continue
            for field_name in bool_fields:
                raw_value = event.get(field_name)
                if raw_value is None:
                    continue
                timing[field_name] = bool(raw_value)
                changed = True
            for field_name in list_fields:
                raw_items = event.get(field_name)
                if not isinstance(raw_items, list):
                    continue
                timing[field_name] = [str(item).strip() for item in raw_items if str(item).strip()]
                changed = True
        elif etype == "model_selection":
            stage = str(event.get("model_stage", "")).strip().lower()
            if stage in {"initial_draft", "author_refine", "critic_review", "memory", "discovery"}:
                timing[f"{stage}_model"] = str(event.get("model", "")).strip() or None
                timing[f"{stage}_provider"] = str(event.get("provider", "")).strip() or None
                fallback_model = str(event.get("fallback_model", "")).strip()
                if fallback_model:
                    timing[f"{stage}_fallback_model"] = fallback_model
                changed = True
        elif etype == "structured_output_retry":
            stage = str(event.get("model_stage", "")).strip().lower()
            if stage in {"author_refine", "critic_review"}:
                retry_key = f"{stage}_json_retries"
                timing[retry_key] = int(timing.get(retry_key, 0)) + 1
                failure_category = str(event.get("failure_category", "")).strip()
                if failure_category:
                    timing[f"{stage}_last_failure_category"] = failure_category
                changed = True
        elif etype == "structured_output_fallback":
            stage = str(event.get("model_stage", "")).strip().lower()
            if stage in {"author_refine", "critic_review"}:
                fallback_key = f"{stage}_fallbacks"
                timing[fallback_key] = int(timing.get(fallback_key, 0)) + 1
                fallback_model = str(event.get("fallback_model", "")).strip()
                if fallback_model:
                    timing[f"{stage}_fallback_model"] = fallback_model
                failure_category = str(event.get("failure_category", "")).strip()
                if failure_category:
                    timing[f"{stage}_last_failure_category"] = failure_category
                changed = True

        if changed:
            timing["timing_updated_at_unix_s"] = time.time()
            self._persist_session_timing(session_id, timing, store=store)

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
    completed_tool_call_groups: list[dict[str, Any]] = []
    if data.get("completed_tool_call_groups_json"):
        try:
            parsed = json.loads(str(data["completed_tool_call_groups_json"]))
            if isinstance(parsed, list):
                for idx, group in enumerate(parsed):
                    if isinstance(group, dict) and "tools" in group:
                        completed_tool_call_groups.append(group)
                    elif isinstance(group, list):
                        tools = [e for e in group if isinstance(e, dict)]
                        first_ts = ""
                        if tools:
                            first_ts = str(tools[0].get("created_at", ""))
                        completed_tool_call_groups.append(
                            {
                                "sequence": idx,
                                "created_at": first_ts,
                                "tools": tools,
                            }
                        )
        except json.JSONDecodeError:
            completed_tool_call_groups = []
    data["pending_questions"] = pending_questions
    data["phase_message"] = data.get("phase_message")
    data["is_processing"] = bool(data.get("is_processing", 0))
    data["active_tool_calls"] = active_tool_calls
    data["completed_tool_call_groups"] = completed_tool_call_groups
    return data


def _coerce_stale_processing_payload(
    *,
    session: Any,
    payload: dict[str, Any],
    store: Store,
) -> dict[str, Any]:
    """Normalize stale runtime state when no live command exists.

    If a command lease has expired, the session row can still indicate processing
    with running tools. For read endpoints, reconcile that view by treating
    lingering running tools as done and clearing active processing state.
    """
    live = store.get_live_running_planning_command(session.id)
    if live is not None:
        return payload

    active_tools = payload.get("active_tool_calls")
    normalized_active: list[dict[str, Any]] = []
    had_running = False
    had_done = False
    if isinstance(active_tools, list):
        for item in active_tools:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status", ""))
            if status == "running":
                had_running = True
                normalized_active.append({**item, "status": "done"})
            else:
                if status == "done":
                    had_done = True
                normalized_active.append(item)

    # Chat setup runs outside the registered command lease flow. During that
    # window we still want readers to see "processing" so the setup UI renders.
    phase_message = str(payload.get("phase_message", ""))
    is_chat_setup_phase = str(payload.get("status", "")) == "draft" and phase_message.startswith(
        "Preparing codebase memory"
    )
    is_initial_draft_phase = str(payload.get("status", "")) == "draft" and phase_message.startswith(
        "Building initial plan draft"
    )
    should_clear_processing = had_running or (
        bool(payload.get("is_processing")) and not is_chat_setup_phase and not is_initial_draft_phase
    )
    if should_clear_processing:
        payload["is_processing"] = False
        payload["phase_message"] = None

    should_finalize_active_group = bool(normalized_active) and (had_running or had_done)
    if should_finalize_active_group:
        existing_groups = payload.get("completed_tool_call_groups")
        groups: list[dict[str, Any]] = []
        if isinstance(existing_groups, list):
            groups = [g for g in existing_groups if isinstance(g, dict)]
        next_seq = max((_to_int(group.get("sequence"), 0) for group in groups), default=0) + 1
        groups.append(
            {
                "sequence": next_seq,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "tools": normalized_active,
            }
        )
        payload["completed_tool_call_groups"] = groups
        payload["active_tool_calls"] = []
    else:
        payload["active_tool_calls"] = normalized_active

    return payload


def _turn_to_dict(turn: Any) -> dict[str, Any]:
    return asdict(turn)


def _version_to_dict(version: Any) -> dict[str, Any]:
    data = asdict(version)
    for raw_key, parsed_key in (("decision_graph_json", "decision_graph"), ("followups_json", "followups")):
        parsed: Any = None
        raw = data.get(raw_key)
        if raw:
            try:
                payload = json.loads(str(raw))
                if isinstance(payload, dict):
                    parsed = payload
            except json.JSONDecodeError:
                parsed = None
        data[parsed_key] = parsed
    return data


def create_app() -> FastAPI:
    from .server import setup_logging

    setup_logging()
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
        "draft": {"message", "reset", "export"},
        "refining": {"run_round", "reset", "message", "followup_answer", "export"},
        "converged": {"run_round", "approve", "reset", "message", "followup_answer", "export"},
        "approved": {"export", "reset"},
        "error": {"reset"},
    }

    def _allowed_commands_for(session: Any, store: Store) -> list[str]:
        active = store.get_live_running_planning_command(session.id)
        if active is not None:
            return ["cancel"]
        return sorted(COMMAND_MATRIX.get(session.status, set()))

    def _parse_iso_unix_s(value: str | None) -> float | None:
        if not value:
            return None
        try:
            normalized = value.replace("Z", "+00:00")
            return datetime.fromisoformat(normalized).timestamp()
        except ValueError:
            return None

    def _command_diagnostics_payload(command: Any) -> dict[str, Any]:
        started_unix_s = _parse_iso_unix_s(command.started_at)
        completed_unix_s = _parse_iso_unix_s(command.completed_at)
        duration_s: float | None = None
        if started_unix_s is not None and completed_unix_s is not None:
            duration_s = max(0.0, round(completed_unix_s - started_unix_s, 3))
        return {
            "id": command.id,
            "command_id": command.command_id,
            "command": command.command,
            "status": command.status,
            "attempt_count": command.attempt_count,
            "started_at": command.started_at,
            "completed_at": command.completed_at,
            "duration_s": duration_s,
            "lease_expires_at": command.lease_expires_at,
            "last_error": command.last_error,
            "created_at": command.created_at,
        }

    def _build_session_snapshot(session_id: str, store: Store) -> dict[str, Any]:
        store.fail_expired_running_planning_commands()
        session = store.get_planning_session(session_id)
        if session is None:
            raise ValueError("session not found")
        active = store.get_live_running_planning_command(session_id)
        payload = _coerce_stale_processing_payload(
            session=session,
            payload=_session_to_dict(session),
            store=store,
        )
        return {
            "type": "session_state",
            "v": 1,
            "status": payload.get("status"),
            "phase_message": payload.get("phase_message"),
            "is_processing": bool(payload.get("is_processing")) if active is None else True,
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
                mode, reply = await runtime.handle_refinement_message(
                    session_id=ctx.session_id,
                    user_message=message,
                    author_model_override=author_model,
                    critic_model_override=critic_model,
                    event_callback=callback,
                )
                return HandlerResult(metadata={"accepted": True, "mode": mode, "reply": reply})
            if session.status not in {"draft", "refining", "converged"}:
                raise CommandConflictError("invalid_status", "message is only allowed in draft/refining/converged")

            result = await runtime.handle_discovery_turn(
                session_id=ctx.session_id,
                user_message=message,
                author_model_override=author_model,
                critic_model_override=critic_model,
                event_callback=callback,
            )
            return HandlerResult(metadata={"result": asdict(result)})

        async def _approve(ctx: CommandContext) -> HandlerResult:
            runtime.approve(
                session_id=ctx.session_id,
                unverified_references=set(payload.get("unverified_references") or []),
            )
            return HandlerResult(metadata={"approved": True})

        async def _followup_answer(ctx: CommandContext) -> HandlerResult:
            session = ctx.store.get_planning_session(ctx.session_id)
            if session is None:
                raise ValueError("session not found")
            current_plan = ctx.store.get_plan_versions(ctx.session_id, limit=1)
            current = current_plan[0] if current_plan else None
            if current is None or current.id is None:
                raise CommandConflictError("invalid_status", "No current plan version is available")
            plan_version_id = payload.get("plan_version_id")
            if int(plan_version_id or 0) != int(current.id):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "reason": "stale_plan_version",
                        "detail": "This follow-up belongs to an older plan version. Refresh and answer the latest question.",
                    },
                )
            followup_id = str(payload.get("followup_id") or "").strip()
            followup_answer = str(payload.get("followup_answer") or "").strip()
            if not followup_id or not followup_answer:
                raise CommandConflictError("invalid_status", "followup_id and followup_answer are required")
            target_sections: list[str] = []
            if current.followups_json:
                try:
                    followups_payload = json.loads(current.followups_json)
                except json.JSONDecodeError:
                    followups_payload = {}
                questions_payload = followups_payload.get("questions", [])
                if isinstance(questions_payload, list):
                    for item in questions_payload:
                        if not isinstance(item, dict):
                            continue
                        if str(item.get("id", "")) == followup_id:
                            raw_sections = item.get("target_sections")
                            if isinstance(raw_sections, list):
                                target_sections = [str(section) for section in raw_sections if str(section).strip()]
                            break

            author_model = _validated_model_or_400(
                payload.get("author_model") or session.author_model,
                "author",
            )

            async def callback(event: dict[str, Any]) -> None:
                enriched = dict(event)
                enriched.setdefault("command_id", command_id)
                await _emit_session_event(ctx.session_id, enriched)

            mode, reply = await runtime.apply_followup_answer(
                session_id=ctx.session_id,
                followup_id=followup_id,
                followup_answer=followup_answer,
                target_sections=target_sections,
                author_model_override=author_model,
                event_callback=callback,
            )
            return HandlerResult(metadata={"accepted": True, "mode": mode, "reply": reply})

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
            ctx.store.clear_planning_session_history(ctx.session_id)
            ctx.store.update_planning_session(
                ctx.session_id,
                _bypass_protection=True,
                status="draft",
                phase_message=None,
                is_processing=0,
                processing_started_at=None,
                current_round=0,
                pending_questions_json=None,
                active_tool_calls_json=json.dumps([]),
                completed_tool_call_groups_json=json.dumps([]),
                diagnostics_json=None,
                clarifications_log_json=json.dumps([]),
                session_total_cost_usd=0.0,
                max_prompt_tokens=None,
                confidence_trend=None,
                converged_early=0,
                last_commands_json=json.dumps({}),
                current_command_id=None,
            )
            registry._session_timing.pop(ctx.session_id, None)
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
            "followup_answer": _followup_answer,
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
            logger.info(
                "command conflict session_id={} command={} reason=invalid_status detail={}",
                session_id,
                command,
                str(exc),
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
            logger.info(
                "command conflict session_id={} command={} reason={} status={} phase_message={}",
                session_id,
                command,
                exc.reason,
                detail.get("status", ""),
                detail.get("phase_message", ""),
            )
            raise HTTPException(status_code=409, detail=detail) from exc
        except ValueError as exc:
            message = str(exc)
            if message == "session not found":
                raise HTTPException(status_code=404, detail="session not found") from exc
            session = store.get_planning_session(session_id)
            if session is not None:
                try:
                    _, _, runtime = _runtime_for(repo_name=session.repo_name)
                    runtime.persist_state_snapshot(session_id)
                except Exception:  # noqa: BLE001
                    logger.debug("failed to persist snapshot for command failure session_id={}", session_id)
            detail: dict[str, Any] = {"reason": "command_failed", "detail": message}
            if session is not None:
                detail.update(
                    {
                        "status": session.status,
                        "phase_message": session.phase_message,
                        "allowed_commands": _allowed_commands_for(session, store),
                    }
                )
            logger.info(
                "command conflict session_id={} command={} reason=command_failed detail={}",
                session_id,
                command,
                message,
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
            _, store, runtime = _runtime_for(repo_name=payload.repo)
            runtime_elapsed = time.perf_counter() - runtime_started
            author_model = _validated_model_or_400(
                payload.author_model or runtime.planning_config.author_model,
                "author",
            )
            critic_model = _validated_model_or_400(
                payload.critic_model or runtime.planning_config.critic_model,
                "critic",
            )
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
                    registry.note_event(session.id, enriched, store=store)
                    await emitter.emit(session.id, enriched)

                draft_started = time.perf_counter()
                timing = registry._ensure_session_timing(session.id, store=store)
                timing.update(
                    {
                        "memory_prep_started_at_unix_s": None,
                        "memory_prep_elapsed_s": None,
                        "memory_prep_completed_at_unix_s": None,
                        "memory_prep_failed": False,
                        "initial_draft_started_at_unix_s": None,
                        "initial_draft_elapsed_s": None,
                        "initial_draft_completed_at_unix_s": None,
                        "initial_draft_failed": False,
                        "timing_updated_at_unix_s": time.time(),
                    }
                )
                registry._persist_session_timing(session.id, timing, store=store)

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
                        timing = registry._ensure_session_timing(session.id, store=store)
                        timing.update(
                            {
                                "timing_updated_at_unix_s": time.time(),
                            }
                        )
                        registry._persist_session_timing(session.id, timing, store=store)
                    except Exception as exc:  # noqa: BLE001
                        logger.error(
                            "api.sessions draft failed session_id={} elapsed_s={:.3f} error={}",
                            session.id,
                            time.perf_counter() - draft_started,
                            exc,
                        )
                        await _surface_background_session_failure(
                            session_id=session.id,
                            store=store,
                            action="Initial draft",
                            exc=exc,
                            emit_event=_emit_session_event,
                            build_snapshot=_build_session_snapshot,
                        )
                        try:
                            timing = registry._ensure_session_timing(session.id, store=store)
                            timing.update(
                                {
                                    "initial_draft_failed": True,
                                    "timing_updated_at_unix_s": time.time(),
                                }
                            )
                            registry._persist_session_timing(session.id, timing, store=store)
                        except Exception as timing_exc:  # noqa: BLE001
                            logger.warning(
                                "api.sessions draft failure timing persistence skipped session_id={} error={}",
                                session.id,
                                timing_exc,
                            )

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
                buffered_events: list[dict[str, Any]] = []

                async def pr_event_callback(event: dict[str, Any]) -> None:
                    buffered_events.append(event)

                session = await runtime.start_from_pr(
                    upstream_repo=payload.upstream_repo,
                    pr_number=payload.pr_number,
                    author_model=author_model,
                    critic_model=critic_model,
                    rebuild_memory=payload.rebuild_memory,
                    event_callback=pr_event_callback,
                )
                for ev in buffered_events:
                    registry.note_event(session.id, ev, store=store)
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
                event_type = str(event.get("type", ""))
                if event_type == "setup_progress":
                    logger.info(
                        "api.sessions chat setup progress session_id={} step={}",
                        session.id,
                        str(event.get("step", "")),
                    )
                elif event_type == "discovery_ready":
                    logger.info(
                        "api.sessions chat setup event session_id={} type=discovery_ready",
                        session.id,
                    )
                registry.note_event(session.id, event, store=store)
                await emitter.emit(session.id, event)

            async def run_chat_setup() -> None:
                try:
                    await runtime.continue_chat_setup(
                        session.id,
                        rebuild_memory=payload.rebuild_memory,
                        event_callback=emit_setup_event,
                    )
                    logger.info(
                        "api.sessions chat setup complete session_id={}",
                        session.id,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("api.sessions chat setup failed session_id={}", session.id)
                    await _surface_background_session_failure(
                        session_id=session.id,
                        store=store,
                        action="Chat setup",
                        exc=exc,
                        emit_event=_emit_session_event,
                        build_snapshot=_build_session_snapshot,
                    )

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
        store.fail_expired_running_planning_commands()
        session = store.get_planning_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        session_payload = _coerce_stale_processing_payload(
            session=session,
            payload=_session_to_dict(session),
            store=store,
        )
        versions = store.get_plan_versions(session_id, limit=200)
        current_plan = versions[0] if versions else None
        current_plan_payload = _version_to_dict(current_plan) if current_plan else None
        previous_plan_payload = _version_to_dict(versions[1]) if len(versions) > 1 else None
        timing, timing_source = registry.get_session_timing_with_source(session_id, store=store)
        if lightweight:
            return {
                "session": session_payload,
                "current_plan": current_plan_payload,
                "draft_timing": timing,
                "draft_timing_source": timing_source,
            }
        turns = store.get_planning_turns(session_id)
        round_metrics_rows = store.get_round_metrics(session_id)
        score_by_round = {v.round: v.convergence_score for v in versions}  # O(n) lookup
        issue_graph_summary: dict[str, Any] | None = None
        impact_view: dict[str, Any] | None = None
        try:
            _, _, runtime = _runtime_for(repo_name=session.repo_name)
            snapshot_payload = runtime.current_state_snapshot(session_id)
            if isinstance(snapshot_payload, dict):
                graph = snapshot_payload.get("issue_graph")
                if isinstance(graph, dict):
                    summary = graph.get("summary")
                    if isinstance(summary, dict):
                        issue_graph_summary = summary
                    impact_view = build_impact_view(
                        decision_graph=(
                            current_plan_payload.get("decision_graph")
                            if isinstance(current_plan_payload, dict)
                            else None
                        ),
                        issue_graph=graph,
                        previous_decision_graph=(
                            previous_plan_payload.get("decision_graph")
                            if isinstance(previous_plan_payload, dict)
                            else None
                        ),
                    )
        except Exception:  # noqa: BLE001
            issue_graph_summary = None
            impact_view = None
        return {
            "session": session_payload,
            "conversation": [_turn_to_dict(t) for t in turns],
            "plan_versions": [_version_to_dict(v) for v in versions],
            "current_plan": current_plan_payload,
            "impact_view": impact_view,
            "draft_timing": timing,
            "draft_timing_source": timing_source,
            "round_metrics": [
                {
                    "round": m.round,
                    "major_issues": m.major_issues,
                    "minor_issues": m.minor_issues,
                    "critic_confidence": m.critic_confidence,
                    "convergence_score": score_by_round.get(m.round),
                    "call_cost_usd": m.call_cost_usd,
                    "issue_graph_summary": issue_graph_summary,
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

    @app.get("/api/sessions/{session_id}/diagnostics")
    async def get_session_diagnostics(
        session_id: str,
        repo: Optional[str] = Query(default=None),
        limit: int = Query(default=20, ge=1, le=200),
    ) -> dict[str, Any]:
        repo_name = _repo_for_session(session_id, repo)
        _, store, _ = _runtime_for(repo_name=repo_name)
        session = store.get_planning_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")

        active = store.get_live_running_planning_command(session_id)
        commands = store.list_planning_commands(session_id, limit=limit)
        session_payload = _session_to_dict(session)
        groups = session_payload.get("completed_tool_call_groups", [])
        draft_tools: list[dict[str, Any]] = []
        if isinstance(groups, list):
            for group in groups:
                if not isinstance(group, dict):
                    continue
                for tool in group.get("tools", []):
                    if not isinstance(tool, dict):
                        continue
                    if str(tool.get("name", "")) == "draft_plan":
                        draft_tools.append(
                            {
                                "sequence": int(group.get("sequence", 0) or 0),
                                "created_at": group.get("created_at"),
                                "status": tool.get("status"),
                                "duration_ms": tool.get("duration_ms"),
                                "session_stage": tool.get("session_stage"),
                            }
                        )
        timing, timing_source = registry.get_session_timing_with_source(session_id, store=store)
        return {
            "session_id": session_id,
            "repo_name": repo_name,
            "status": session.status,
            "current_round": session.current_round,
            "is_processing": bool(session.is_processing),
            "phase_message": session.phase_message,
            "active_command": _command_diagnostics_payload(active) if active is not None else None,
            "recent_commands": [_command_diagnostics_payload(command) for command in commands],
            "draft_timing": timing,
            "draft_timing_source": timing_source,
            "draft_tool_runs": draft_tools,
        }

    @app.get("/api/sessions/{session_id}/snapshot")
    async def get_session_snapshot(
        session_id: str,
        repo: Optional[str] = Query(default=None),
    ) -> dict[str, Any]:
        repo_name = _repo_for_session(session_id, repo)
        _, _, runtime = _runtime_for(repo_name=repo_name)
        payload = runtime.current_state_snapshot(session_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="snapshot not found")
        return {"snapshot": payload}

    @app.get("/api/snapshots")
    async def list_session_snapshots(repo: Optional[str] = Query(default=None)) -> dict[str, Any]:
        _, _, runtime = _runtime_for(repo_name=repo)
        return {"items": runtime.list_state_snapshots()}

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
                "plan_version_id": payload.plan_version_id,
                "followup_id": payload.followup_id,
                "followup_answer": payload.followup_answer,
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

        return EventSourceResponse(event_generator(), ping=20)

    return app
