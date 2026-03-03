"""
Repeatable prompt benchmark runner for planning startup performance/quality.

Usage:
    python -m prscope.benchmark --base-url http://127.0.0.1:8443 --repo prscope
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener

import yaml

DEFAULT_PROMPTS = [
    "Add a lightweight /health endpoint and tests for it.",
    "Add request ID propagation in server middleware and include it in structured logs, with tests.",
    "Add rate limiting for auth endpoints with config knobs and observability metrics.",
    "Refactor env/config loading to validate required vars at startup and fail fast with clear errors.",
    "Add a new admin-only diagnostics endpoint and ensure role checks plus negative-path tests.",
]

DEFAULT_PROMPTS_FILE = Path("benchmarks/prompts.json")
DEFAULT_OUTPUT_DIR = Path("benchmarks/results")

REQUIRED_SECTIONS = [
    "## Summary",
    "## Goals",
    "## Non-Goals",
    "## Changes",
    "## Files Changed",
    "## To-dos In Order",
    "## Architecture",
    "## Mermaid Diagram",
    "## Implementation Steps",
    "## Test Strategy",
    "## Rollback Plan",
]

GENERIC_MARKERS = [
    "placeholder",
    "todo",
    "tbd",
    "example_change",
    "src/path/to/module.py",
]

NO_PROXY_OPENER = build_opener(ProxyHandler({}))


def _redact_home_path(raw_path: str | Path) -> str:
    raw = str(raw_path)
    home = str(Path.home().resolve())
    if raw == home:
        return "~"
    prefix = f"{home}/"
    if raw.startswith(prefix):
        return f"~/{raw[len(prefix) :]}"
    return raw


@dataclass
class PromptRun:
    prompt_index: int
    prompt: str
    session_id: str
    create_elapsed_s: float
    time_to_plan_s: int | None
    server_initial_draft_elapsed_s: float | None
    server_first_plan_elapsed_s: float | None
    client_detect_gap_s: float | None
    client_detect_gap_from_server_first_plan_s: float | None
    timed_out: bool
    fallback: bool
    status: str
    content_len: int
    ref_count: int
    section_score: int
    generic_markers: int
    quality_score: float
    poll_attempts: int
    poll_timeouts: int
    avg_poll_latency_ms: float | None
    max_poll_latency_ms: float | None
    poll_transport_latency_ms_total: float
    server_llm_calls_total: int
    server_llm_latency_ms_total: float
    server_tool_calls_total: int
    server_tool_exec_latency_ms_total: float
    author_call_timeouts: int
    author_fallback_warnings: int
    runtime_warnings_total: int
    runtime_errors_total: int
    error: str | None


def _http_json(method: str, url: str, body: dict[str, Any] | None = None, timeout: int = 30) -> dict[str, Any]:
    raw = json.dumps(body).encode("utf-8") if body is not None else None
    request = Request(
        url,
        data=raw,
        method=method,
        headers={"Content-Type": "application/json", "Connection": "close"},
    )
    with NO_PROXY_OPENER.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _log_line(log_path: Path, payload: dict[str, Any]) -> None:
    line = json.dumps(payload, ensure_ascii=True)
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _safe_delete_session(base_url: str, repo: str, session_id: str, timeout: int = 3) -> tuple[bool, str]:
    try:
        request = Request(
            f"{base_url}/api/sessions/{session_id}?repo={quote(repo)}",
            method="DELETE",
            headers={"Connection": "close"},
        )
        with NO_PROXY_OPENER.open(request, timeout=timeout):
            return True, "deleted"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _wait_server(base_url: str, timeout_seconds: int = 20) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            _http_json("GET", f"{base_url}/api/sessions", timeout=2)
            return
        except Exception:
            time.sleep(0.4)
    raise RuntimeError(f"Server not ready at {base_url} after {timeout_seconds}s")


def _extract_file_refs(plan: str) -> int:
    import re

    refs = set(re.findall(r"`([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)`", plan))
    return len(refs)


def _section_score(plan: str) -> int:
    lower = plan.lower()
    return sum(1 for section in REQUIRED_SECTIONS if section.lower() in lower)


def _generic_marker_count(plan: str) -> int:
    lower = plan.lower()
    return sum(1 for marker in GENERIC_MARKERS if marker in lower)


def _quality_score(plan: str, fallback: bool) -> float:
    if not plan:
        return 0.0
    section_ratio = _section_score(plan) / float(len(REQUIRED_SECTIONS))
    refs = min(1.0, _extract_file_refs(plan) / 8.0)
    generic_penalty = min(0.4, _generic_marker_count(plan) * 0.1)
    score = (section_ratio * 0.55) + (refs * 0.45) - generic_penalty
    if fallback:
        score -= 0.35
    return round(max(0.0, min(1.0, score)), 4)


def _load_prompts(path: Path | None) -> list[str]:
    if path is None:
        if DEFAULT_PROMPTS_FILE.exists():
            return _load_prompts(DEFAULT_PROMPTS_FILE.resolve())
        return list(DEFAULT_PROMPTS)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError("Prompts file must be a JSON array of strings")
    prompts = [item.strip() for item in data if item.strip()]
    if not prompts:
        raise ValueError("Prompts file is empty")
    return prompts


def _load_config_metadata(config_root: Path) -> dict[str, Any]:
    path = config_root / "prscope.yml"
    if not path.exists():
        return {
            "config_root": _redact_home_path(config_root),
            "config_path": _redact_home_path(path),
            "found": False,
        }
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    planning = data.get("planning", {}) if isinstance(data, dict) else {}
    return {
        "config_root": _redact_home_path(config_root),
        "config_path": _redact_home_path(path),
        "found": True,
        "scanner": planning.get("scanner", "grep"),
        "author_model": planning.get("author_model"),
        "critic_model": planning.get("critic_model"),
    }


def _summary(results: list[PromptRun]) -> dict[str, Any]:
    finished = [r for r in results if r.time_to_plan_s is not None]
    avg_time = statistics.mean(r.time_to_plan_s for r in finished) if finished else None
    avg_create = statistics.mean(r.create_elapsed_s for r in results) if results else None
    avg_quality = statistics.mean(r.quality_score for r in results) if results else None
    avg_poll_latency_ms = (
        statistics.mean(r.avg_poll_latency_ms for r in results if r.avg_poll_latency_ms is not None)
        if results
        else None
    )
    avg_server_draft_elapsed_s = (
        statistics.mean(
            r.server_initial_draft_elapsed_s for r in results if r.server_initial_draft_elapsed_s is not None
        )
        if results
        else None
    )
    avg_client_detect_gap_s = (
        statistics.mean(r.client_detect_gap_s for r in results if r.client_detect_gap_s is not None)
        if results
        else None
    )
    slowest = max(
        (r for r in finished),
        key=lambda item: item.time_to_plan_s or 0,
        default=None,
    )
    return {
        "runs": len(results),
        "non_fallback_runs": sum(1 for r in results if not r.fallback),
        "fallback_runs": sum(1 for r in results if r.fallback),
        "timed_out_runs": sum(1 for r in results if r.timed_out),
        "error_runs": sum(1 for r in results if bool(r.error)),
        "avg_create_elapsed_s": round(avg_create, 3) if avg_create is not None else None,
        "avg_time_to_plan_s": round(avg_time, 3) if avg_time is not None else None,
        "max_time_to_plan_s": max((r.time_to_plan_s for r in finished), default=None),
        "avg_quality_score": round(avg_quality, 4) if avg_quality is not None else None,
        "total_poll_timeouts": sum(r.poll_timeouts for r in results),
        "total_author_call_timeouts": sum(r.author_call_timeouts for r in results),
        "total_author_fallback_warnings": sum(r.author_fallback_warnings for r in results),
        "total_runtime_warnings": sum(r.runtime_warnings_total for r in results),
        "total_runtime_errors": sum(r.runtime_errors_total for r in results),
        "runs_with_author_timeouts": sum(1 for r in results if r.author_call_timeouts > 0),
        "total_poll_transport_latency_ms": round(
            sum(r.poll_transport_latency_ms_total for r in results),
            2,
        ),
        "total_server_llm_calls": sum(r.server_llm_calls_total for r in results),
        "total_server_llm_latency_ms": round(
            sum(r.server_llm_latency_ms_total for r in results),
            2,
        ),
        "total_server_tool_calls": sum(r.server_tool_calls_total for r in results),
        "total_server_tool_exec_latency_ms": round(
            sum(r.server_tool_exec_latency_ms_total for r in results),
            2,
        ),
        "avg_poll_latency_ms": round(avg_poll_latency_ms, 2) if avg_poll_latency_ms is not None else None,
        "avg_server_initial_draft_elapsed_s": (
            round(avg_server_draft_elapsed_s, 3) if avg_server_draft_elapsed_s is not None else None
        ),
        "avg_server_first_plan_elapsed_s": (
            round(
                statistics.mean(
                    r.server_first_plan_elapsed_s for r in results if r.server_first_plan_elapsed_s is not None
                ),
                3,
            )
            if any(r.server_first_plan_elapsed_s is not None for r in results)
            else None
        ),
        "avg_client_detect_gap_s": round(avg_client_detect_gap_s, 3) if avg_client_detect_gap_s is not None else None,
        "avg_client_detect_gap_from_server_first_plan_s": (
            round(
                statistics.mean(
                    r.client_detect_gap_from_server_first_plan_s
                    for r in results
                    if r.client_detect_gap_from_server_first_plan_s is not None
                ),
                3,
            )
            if any(r.client_detect_gap_from_server_first_plan_s is not None for r in results)
            else None
        ),
        "slowest_prompt_index": slowest.prompt_index if slowest is not None else None,
        "slowest_prompt_time_to_plan_s": slowest.time_to_plan_s if slowest is not None else None,
    }


def _is_better(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    # Ranking: fewer fallback runs, faster avg time to plan, higher quality.
    c_fallback = int(candidate.get("summary", {}).get("fallback_runs", 9999))
    i_fallback = int(incumbent.get("summary", {}).get("fallback_runs", 9999))
    if c_fallback != i_fallback:
        return c_fallback < i_fallback
    c_time = candidate.get("summary", {}).get("avg_time_to_plan_s")
    i_time = incumbent.get("summary", {}).get("avg_time_to_plan_s")
    if c_time is not None and i_time is not None and c_time != i_time:
        return c_time < i_time
    c_quality = float(candidate.get("summary", {}).get("avg_quality_score", 0.0) or 0.0)
    i_quality = float(incumbent.get("summary", {}).get("avg_quality_score", 0.0) or 0.0)
    return c_quality > i_quality


def run_health_check(
    *,
    base_url: str,
    repo: str,
    create_timeout_seconds: int,
    log_path: Path,
) -> dict[str, Any]:
    _wait_server(base_url)
    checks: list[dict[str, Any]] = []

    t0 = time.time()
    _http_json("GET", f"{base_url}/api/sessions", timeout=3)
    checks.append({"check": "list_sessions", "ok": True, "elapsed_s": round(time.time() - t0, 3)})
    _log_line(log_path, {"event": "health_check", "check": "list_sessions", "ok": True})

    create_started = time.time()
    created = _http_json(
        "POST",
        f"{base_url}/api/sessions",
        body={"mode": "chat", "repo": repo, "rebuild_memory": False},
        timeout=max(1, int(create_timeout_seconds)),
    )
    create_elapsed = time.time() - create_started
    session_id = str(created.get("session", {}).get("id", ""))
    checks.append(
        {
            "check": "create_chat_session",
            "ok": bool(session_id),
            "elapsed_s": round(create_elapsed, 3),
            "session_id": session_id,
        }
    )
    _log_line(
        log_path,
        {
            "event": "health_check",
            "check": "create_chat_session",
            "ok": bool(session_id),
            "elapsed_s": round(create_elapsed, 3),
            "session_id": session_id,
        },
    )

    deleted, delete_msg = _safe_delete_session(base_url, repo, session_id) if session_id else (False, "no_session_id")
    checks.append({"check": "delete_chat_session", "ok": deleted, "detail": delete_msg})
    _log_line(
        log_path,
        {
            "event": "health_check",
            "check": "delete_chat_session",
            "ok": deleted,
            "detail": delete_msg,
            "session_id": session_id,
        },
    )

    ok = all(bool(item.get("ok")) for item in checks)
    return {"ok": ok, "checks": checks}


def run_benchmark(
    *,
    base_url: str,
    repo: str,
    prompts: list[str],
    create_timeout_seconds: int,
    poll_timeout_seconds: int,
    poll_request_timeout_seconds: int,
    max_consecutive_poll_timeouts: int,
    poll_unhealthy_window_seconds: int,
    max_consecutive_create_failures: int,
    stop_on_first_problem: bool,
    max_suite_seconds: int | None,
    log_path: Path,
) -> list[PromptRun]:
    suite_started = time.time()
    runs: list[PromptRun] = []
    consecutive_create_failures = 0
    for idx, prompt in enumerate(prompts, start=1):
        if max_suite_seconds is not None and (time.time() - suite_started) >= max_suite_seconds:
            _log_line(
                log_path,
                {
                    "event": "suite_timeout",
                    "prompt_index": idx,
                    "prompt_total": len(prompts),
                    "elapsed_s": round(time.time() - suite_started, 3),
                    "max_suite_seconds": max_suite_seconds,
                },
            )
            break
        _log_line(
            log_path,
            {
                "event": "prompt_start",
                "prompt_index": idx,
                "prompt_total": len(prompts),
                "prompt": prompt,
            },
        )
        t0 = time.time()
        create_timeout = max(1, int(create_timeout_seconds))
        if max_suite_seconds is not None:
            remaining_suite = max_suite_seconds - (time.time() - suite_started)
            if remaining_suite <= 0:
                _log_line(
                    log_path,
                    {
                        "event": "suite_timeout",
                        "prompt_index": idx,
                        "prompt_total": len(prompts),
                        "elapsed_s": round(time.time() - suite_started, 3),
                        "max_suite_seconds": max_suite_seconds,
                    },
                )
                break
            create_timeout = min(create_timeout, max(1, int(remaining_suite)))
        try:
            created = _http_json(
                "POST",
                f"{base_url}/api/sessions",
                body={
                    "mode": "requirements",
                    "repo": repo,
                    "requirements": prompt,
                    "rebuild_memory": False,
                },
                timeout=create_timeout,
            )
            create_elapsed = time.time() - t0
            session_id = str(created["session"]["id"])
        except Exception as exc:  # noqa: BLE001
            create_elapsed = time.time() - t0
            consecutive_create_failures += 1
            _log_line(
                log_path,
                {
                    "event": "prompt_create_failed",
                    "prompt_index": idx,
                    "prompt_total": len(prompts),
                    "create_elapsed_s": round(create_elapsed, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                    "consecutive_create_failures": consecutive_create_failures,
                },
            )
            runs.append(
                PromptRun(
                    prompt_index=idx,
                    prompt=prompt,
                    session_id="",
                    create_elapsed_s=round(create_elapsed, 3),
                    time_to_plan_s=None,
                    server_initial_draft_elapsed_s=None,
                    server_first_plan_elapsed_s=None,
                    client_detect_gap_s=None,
                    client_detect_gap_from_server_first_plan_s=None,
                    timed_out=False,
                    fallback=False,
                    status="create_failed",
                    content_len=0,
                    ref_count=0,
                    section_score=0,
                    generic_markers=0,
                    quality_score=0.0,
                    poll_attempts=0,
                    poll_timeouts=0,
                    avg_poll_latency_ms=None,
                    max_poll_latency_ms=None,
                    poll_transport_latency_ms_total=0.0,
                    server_llm_calls_total=0,
                    server_llm_latency_ms_total=0.0,
                    server_tool_calls_total=0,
                    server_tool_exec_latency_ms_total=0.0,
                    author_call_timeouts=0,
                    author_fallback_warnings=0,
                    runtime_warnings_total=0,
                    runtime_errors_total=0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            if consecutive_create_failures >= max(1, int(max_consecutive_create_failures)):
                _log_line(
                    log_path,
                    {
                        "event": "suite_stop",
                        "reason": "consecutive_create_failures",
                        "consecutive_create_failures": consecutive_create_failures,
                        "max_consecutive_create_failures": max_consecutive_create_failures,
                    },
                )
                break
            if stop_on_first_problem:
                _log_line(
                    log_path,
                    {
                        "event": "suite_stop",
                        "reason": "stop_on_first_problem",
                        "prompt_index": idx,
                        "problem": "create_failed",
                    },
                )
                break
            continue
        consecutive_create_failures = 0
        _log_line(
            log_path,
            {
                "event": "prompt_created",
                "prompt_index": idx,
                "prompt_total": len(prompts),
                "session_id": session_id,
                "create_elapsed_s": round(create_elapsed, 3),
            },
        )

        deadline = time.time() + poll_timeout_seconds
        status = "unknown"
        content = ""
        found_after: int | None = None
        started = time.time()
        poll_attempts = 0
        poll_timeouts = 0
        consecutive_poll_timeouts = 0
        poll_latencies_ms: list[float] = []
        prompt_error: str | None = None
        last_successful_poll_at = time.time()
        server_initial_draft_elapsed_s: float | None = None
        server_first_plan_saved_at_unix_s: float | None = None
        server_initial_draft_started_at_unix_s: float | None = None
        found_at_unix_s: float | None = None
        author_call_timeouts = 0
        author_fallback_warnings = 0
        runtime_warnings_total = 0
        runtime_errors_total = 0
        server_llm_calls_total = 0
        server_llm_latency_ms_total = 0.0
        server_tool_calls_total = 0
        server_tool_exec_latency_ms_total = 0.0

        while time.time() < deadline:
            if max_suite_seconds is not None:
                remaining_suite = max_suite_seconds - (time.time() - suite_started)
                if remaining_suite <= 0:
                    prompt_error = "suite_timeout_during_prompt"
                    _log_line(
                        log_path,
                        {
                            "event": "suite_timeout",
                            "prompt_index": idx,
                            "prompt_total": len(prompts),
                            "elapsed_s": round(time.time() - suite_started, 3),
                            "max_suite_seconds": max_suite_seconds,
                        },
                    )
                    break
            poll_attempts += 1
            poll_t0 = time.time()
            per_request_timeout = max(1, int(poll_request_timeout_seconds))
            if max_suite_seconds is not None:
                per_request_timeout = min(per_request_timeout, max(1, int(remaining_suite)))
            try:
                state = _http_json(
                    "GET",
                    f"{base_url}/api/sessions/{session_id}?repo={quote(repo)}&lightweight=true",
                    timeout=per_request_timeout,
                )
                poll_latency_ms = (time.time() - poll_t0) * 1000.0
                poll_latencies_ms.append(poll_latency_ms)
                consecutive_poll_timeouts = 0
                last_successful_poll_at = time.time()
                status = str(state.get("session", {}).get("status", "unknown"))
                current_plan = state.get("current_plan") or {}
                content = str(current_plan.get("plan_content", "") or "").strip()
                draft_timing = state.get("draft_timing") or {}
                if isinstance(draft_timing, dict):
                    started_value = draft_timing.get("initial_draft_started_at_unix_s")
                    if started_value is not None:
                        try:
                            server_initial_draft_started_at_unix_s = float(started_value)
                        except (TypeError, ValueError):
                            pass
                    elapsed_value = draft_timing.get("initial_draft_elapsed_s")
                    if elapsed_value is not None:
                        try:
                            server_initial_draft_elapsed_s = float(elapsed_value)
                        except (TypeError, ValueError):
                            pass
                    saved_at_value = draft_timing.get("first_plan_saved_at_unix_s")
                    if saved_at_value is not None:
                        try:
                            server_first_plan_saved_at_unix_s = float(saved_at_value)
                        except (TypeError, ValueError):
                            pass
                    try:
                        author_call_timeouts = int(draft_timing.get("author_call_timeouts", 0) or 0)
                    except (TypeError, ValueError):
                        author_call_timeouts = 0
                    try:
                        author_fallback_warnings = int(draft_timing.get("author_fallback_warnings", 0) or 0)
                    except (TypeError, ValueError):
                        author_fallback_warnings = 0
                    try:
                        runtime_warnings_total = int(draft_timing.get("warnings_total", 0) or 0)
                    except (TypeError, ValueError):
                        runtime_warnings_total = 0
                    try:
                        runtime_errors_total = int(draft_timing.get("errors_total", 0) or 0)
                    except (TypeError, ValueError):
                        runtime_errors_total = 0
                    try:
                        server_llm_calls_total = int(draft_timing.get("llm_calls_total", 0) or 0)
                    except (TypeError, ValueError):
                        server_llm_calls_total = 0
                    try:
                        server_llm_latency_ms_total = float(draft_timing.get("llm_call_latency_ms_total", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        server_llm_latency_ms_total = 0.0
                    try:
                        server_tool_calls_total = int(draft_timing.get("tool_calls_total", 0) or 0)
                    except (TypeError, ValueError):
                        server_tool_calls_total = 0
                    try:
                        server_tool_exec_latency_ms_total = float(
                            draft_timing.get("tool_exec_latency_ms_total", 0.0) or 0.0
                        )
                    except (TypeError, ValueError):
                        server_tool_exec_latency_ms_total = 0.0
                _log_line(
                    log_path,
                    {
                        "event": "prompt_poll",
                        "prompt_index": idx,
                        "session_id": session_id,
                        "attempt": poll_attempts,
                        "latency_ms": round(poll_latency_ms, 2),
                        "status": status,
                        "has_plan": bool(content),
                        "author_call_timeouts": author_call_timeouts,
                    },
                )
                if content:
                    found_after = int(time.time() - started)
                    found_at_unix_s = time.time()
                    break
            except (TimeoutError, URLError, OSError):
                # Retry once with a short timeout so transport hiccups do not add
                # multi-second blocking to first-plan detection.
                recovered = False
                retry_timeout = max(2, min(4, per_request_timeout))
                try:
                    retry_t0 = time.time()
                    state = _http_json(
                        "GET",
                        f"{base_url}/api/sessions/{session_id}?repo={quote(repo)}&lightweight=true",
                        timeout=retry_timeout,
                    )
                    poll_latency_ms = (time.time() - retry_t0) * 1000.0
                    poll_latencies_ms.append(poll_latency_ms)
                    consecutive_poll_timeouts = 0
                    last_successful_poll_at = time.time()
                    status = str(state.get("session", {}).get("status", "unknown"))
                    current_plan = state.get("current_plan") or {}
                    content = str(current_plan.get("plan_content", "") or "").strip()
                    draft_timing = state.get("draft_timing") or {}
                    if isinstance(draft_timing, dict):
                        started_value = draft_timing.get("initial_draft_started_at_unix_s")
                        if started_value is not None:
                            try:
                                server_initial_draft_started_at_unix_s = float(started_value)
                            except (TypeError, ValueError):
                                pass
                        elapsed_value = draft_timing.get("initial_draft_elapsed_s")
                        if elapsed_value is not None:
                            try:
                                server_initial_draft_elapsed_s = float(elapsed_value)
                            except (TypeError, ValueError):
                                pass
                        saved_at_value = draft_timing.get("first_plan_saved_at_unix_s")
                        if saved_at_value is not None:
                            try:
                                server_first_plan_saved_at_unix_s = float(saved_at_value)
                            except (TypeError, ValueError):
                                pass
                        try:
                            author_call_timeouts = int(draft_timing.get("author_call_timeouts", 0) or 0)
                        except (TypeError, ValueError):
                            author_call_timeouts = 0
                        try:
                            author_fallback_warnings = int(draft_timing.get("author_fallback_warnings", 0) or 0)
                        except (TypeError, ValueError):
                            author_fallback_warnings = 0
                        try:
                            runtime_warnings_total = int(draft_timing.get("warnings_total", 0) or 0)
                        except (TypeError, ValueError):
                            runtime_warnings_total = 0
                        try:
                            runtime_errors_total = int(draft_timing.get("errors_total", 0) or 0)
                        except (TypeError, ValueError):
                            runtime_errors_total = 0
                        try:
                            server_llm_calls_total = int(draft_timing.get("llm_calls_total", 0) or 0)
                        except (TypeError, ValueError):
                            server_llm_calls_total = 0
                        try:
                            server_llm_latency_ms_total = float(
                                draft_timing.get("llm_call_latency_ms_total", 0.0) or 0.0
                            )
                        except (TypeError, ValueError):
                            server_llm_latency_ms_total = 0.0
                        try:
                            server_tool_calls_total = int(draft_timing.get("tool_calls_total", 0) or 0)
                        except (TypeError, ValueError):
                            server_tool_calls_total = 0
                        try:
                            server_tool_exec_latency_ms_total = float(
                                draft_timing.get("tool_exec_latency_ms_total", 0.0) or 0.0
                            )
                        except (TypeError, ValueError):
                            server_tool_exec_latency_ms_total = 0.0
                    _log_line(
                        log_path,
                        {
                            "event": "prompt_poll_recovered",
                            "prompt_index": idx,
                            "session_id": session_id,
                            "attempt": poll_attempts,
                            "retry_timeout_seconds": retry_timeout,
                            "latency_ms": round(poll_latency_ms, 2),
                            "status": status,
                            "has_plan": bool(content),
                            "author_call_timeouts": author_call_timeouts,
                        },
                    )
                    if content:
                        found_after = int(time.time() - started)
                        found_at_unix_s = time.time()
                        break
                    recovered = True
                except (TimeoutError, URLError, OSError):
                    recovered = False
                if recovered:
                    continue
                poll_timeouts += 1
                consecutive_poll_timeouts += 1
                _log_line(
                    log_path,
                    {
                        "event": "prompt_poll_timeout",
                        "prompt_index": idx,
                        "session_id": session_id,
                        "attempt": poll_attempts,
                        "elapsed_s": round(time.time() - started, 3),
                        "request_timeout_seconds": per_request_timeout,
                        "retry_timeout_seconds": retry_timeout,
                    },
                )
                if consecutive_poll_timeouts >= max(1, int(max_consecutive_poll_timeouts)):
                    unhealthy_window = time.time() - last_successful_poll_at
                    if unhealthy_window < max(1, int(poll_unhealthy_window_seconds)):
                        continue
                    _log_line(
                        log_path,
                        {
                            "event": "prompt_poll_unhealthy",
                            "prompt_index": idx,
                            "session_id": session_id,
                            "consecutive_poll_timeouts": consecutive_poll_timeouts,
                            "request_timeout_seconds": poll_request_timeout_seconds,
                            "unhealthy_window_seconds": round(unhealthy_window, 3),
                            "required_window_seconds": poll_unhealthy_window_seconds,
                        },
                    )
                    # Do not abort early here; keep polling until poll_timeout_seconds.
                    # This avoids false "halted" runs when the server is slow but still progressing.
                    continue
            # Keep detect-latency low while drafting; back off slightly once refining.
            time.sleep(0.25 if status in {"drafting", "discovering", "discovery"} else 0.5)

        lower = content.lower()
        fallback = "fallback plan because the authoring model was unavailable" in lower
        timed_out = found_after is None
        if timed_out:
            _log_line(
                log_path,
                {
                    "event": "prompt_timeout",
                    "prompt_index": idx,
                    "session_id": session_id,
                    "poll_timeout_seconds": poll_timeout_seconds,
                    "status": status,
                    "poll_attempts": poll_attempts,
                    "poll_timeouts": poll_timeouts,
                },
            )
        run = PromptRun(
            prompt_index=idx,
            prompt=prompt,
            session_id=session_id,
            create_elapsed_s=round(create_elapsed, 3),
            time_to_plan_s=found_after,
            server_initial_draft_elapsed_s=server_initial_draft_elapsed_s,
            server_first_plan_elapsed_s=(
                round(
                    max(
                        0.0,
                        server_first_plan_saved_at_unix_s - server_initial_draft_started_at_unix_s,
                    ),
                    3,
                )
                if (
                    server_first_plan_saved_at_unix_s is not None and server_initial_draft_started_at_unix_s is not None
                )
                else None
            ),
            client_detect_gap_s=(
                round(max(0.0, found_after - server_initial_draft_elapsed_s), 3)
                if found_after is not None and server_initial_draft_elapsed_s is not None
                else None
            ),
            client_detect_gap_from_server_first_plan_s=(
                round(max(0.0, found_at_unix_s - server_first_plan_saved_at_unix_s), 3)
                if found_at_unix_s is not None and server_first_plan_saved_at_unix_s is not None
                else None
            ),
            timed_out=timed_out,
            fallback=fallback,
            status=status,
            content_len=len(content),
            ref_count=_extract_file_refs(content),
            section_score=_section_score(content),
            generic_markers=_generic_marker_count(content),
            quality_score=_quality_score(content, fallback),
            poll_attempts=poll_attempts,
            poll_timeouts=poll_timeouts,
            avg_poll_latency_ms=round(statistics.mean(poll_latencies_ms), 2) if poll_latencies_ms else None,
            max_poll_latency_ms=round(max(poll_latencies_ms), 2) if poll_latencies_ms else None,
            poll_transport_latency_ms_total=round(sum(poll_latencies_ms), 2),
            server_llm_calls_total=server_llm_calls_total,
            server_llm_latency_ms_total=round(server_llm_latency_ms_total, 2),
            server_tool_calls_total=server_tool_calls_total,
            server_tool_exec_latency_ms_total=round(server_tool_exec_latency_ms_total, 2),
            author_call_timeouts=author_call_timeouts,
            author_fallback_warnings=author_fallback_warnings,
            runtime_warnings_total=runtime_warnings_total,
            runtime_errors_total=runtime_errors_total,
            error=prompt_error,
        )
        _log_line(
            log_path,
            {
                "event": "prompt_complete",
                "prompt_index": idx,
                "session_id": session_id,
                "time_to_plan_s": run.time_to_plan_s,
                "server_initial_draft_elapsed_s": run.server_initial_draft_elapsed_s,
                "server_first_plan_elapsed_s": run.server_first_plan_elapsed_s,
                "client_detect_gap_s": run.client_detect_gap_s,
                "client_detect_gap_from_server_first_plan_s": (run.client_detect_gap_from_server_first_plan_s),
                "timed_out": run.timed_out,
                "fallback": run.fallback,
                "quality_score": run.quality_score,
                "poll_attempts": run.poll_attempts,
                "poll_timeouts": run.poll_timeouts,
                "author_call_timeouts": run.author_call_timeouts,
                "author_fallback_warnings": run.author_fallback_warnings,
                "runtime_warnings_total": run.runtime_warnings_total,
                "runtime_errors_total": run.runtime_errors_total,
                "poll_transport_latency_ms_total": run.poll_transport_latency_ms_total,
                "server_llm_calls_total": run.server_llm_calls_total,
                "server_llm_latency_ms_total": run.server_llm_latency_ms_total,
                "server_tool_calls_total": run.server_tool_calls_total,
                "server_tool_exec_latency_ms_total": run.server_tool_exec_latency_ms_total,
            },
        )
        runs.append(run)
        if run.timed_out or run.fallback or bool(run.error):
            deleted, delete_msg = _safe_delete_session(base_url, repo, session_id)
            _log_line(
                log_path,
                {
                    "event": "problem_session_cleanup",
                    "prompt_index": idx,
                    "session_id": session_id,
                    "deleted": deleted,
                    "detail": delete_msg,
                },
            )
            if stop_on_first_problem:
                _log_line(
                    log_path,
                    {
                        "event": "suite_stop",
                        "reason": "stop_on_first_problem",
                        "prompt_index": idx,
                        "problem": ("timed_out" if run.timed_out else "fallback" if run.fallback else "error"),
                    },
                )
                break
    _log_line(
        log_path,
        {
            "event": "suite_complete",
            "executed_prompts": len(runs),
            "requested_prompts": len(prompts),
            "elapsed_s": round(time.time() - suite_started, 3),
        },
    )
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeatable planning benchmark suite.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8443")
    parser.add_argument("--repo", default="prscope")
    parser.add_argument("--config-root", default=".")
    parser.add_argument("--prompts-file", default=None, help="JSON file with prompt array")
    parser.add_argument(
        "--create-timeout-seconds",
        type=int,
        default=15,
        help="HTTP timeout for session creation request.",
    )
    parser.add_argument("--poll-timeout-seconds", type=int, default=60)
    parser.add_argument(
        "--poll-request-timeout-seconds",
        type=int,
        default=3,
        help="HTTP timeout for each polling request.",
    )
    parser.add_argument(
        "--max-consecutive-poll-timeouts",
        type=int,
        default=3,
        help="Mark prompt unhealthy after this many consecutive poll timeouts.",
    )
    parser.add_argument(
        "--poll-unhealthy-window-seconds",
        type=int,
        default=20,
        help="Require this sustained timeout window before marking poll unhealthy.",
    )
    parser.add_argument(
        "--max-consecutive-create-failures",
        type=int,
        default=1,
        help="Stop suite after this many consecutive session-create failures.",
    )
    parser.add_argument(
        "--stop-on-first-problem",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop benchmark at first prompt problem (default: enabled).",
    )
    parser.add_argument(
        "--health-check-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run cheap API health checks only (no requirements draft prompts).",
    )
    parser.add_argument(
        "--max-suite-seconds",
        type=int,
        default=900,
        help="Stop benchmark early when total runtime exceeds this limit (0 disables).",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_dir = output_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    best_file = output_dir / "best_performance.json"

    prompts = _load_prompts(Path(args.prompts_file).expanduser().resolve() if args.prompts_file else None)
    _wait_server(args.base_url)

    started_at = datetime.now(timezone.utc).isoformat()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_file = history_dir / f"run-{stamp}.log"
    _log_line(
        log_file,
        {
            "event": "suite_start",
            "started_at": started_at,
            "base_url": args.base_url,
            "repo": args.repo,
            "prompt_count": len(prompts),
            "create_timeout_seconds": args.create_timeout_seconds,
            "poll_timeout_seconds": args.poll_timeout_seconds,
            "poll_request_timeout_seconds": args.poll_request_timeout_seconds,
            "max_consecutive_poll_timeouts": args.max_consecutive_poll_timeouts,
            "poll_unhealthy_window_seconds": args.poll_unhealthy_window_seconds,
            "max_consecutive_create_failures": args.max_consecutive_create_failures,
            "stop_on_first_problem": args.stop_on_first_problem,
            "health_check_only": args.health_check_only,
            "max_suite_seconds": args.max_suite_seconds,
        },
    )
    if args.health_check_only:
        health = run_health_check(
            base_url=args.base_url,
            repo=args.repo,
            create_timeout_seconds=args.create_timeout_seconds,
            log_path=log_file,
        )
        payload = {
            "started_at": started_at,
            "base_url": args.base_url,
            "repo": args.repo,
            "mode": "health_check_only",
            "config": _load_config_metadata(Path(args.config_root).expanduser().resolve()),
            "health": health,
            "results": [],
            "summary": {
                "runs": 0,
                "non_fallback_runs": 0,
                "fallback_runs": 0,
                "timed_out_runs": 0,
                "error_runs": 0,
                "avg_create_elapsed_s": None,
                "avg_time_to_plan_s": None,
                "avg_server_initial_draft_elapsed_s": None,
                "avg_client_detect_gap_s": None,
                "max_time_to_plan_s": None,
                "avg_quality_score": None,
                "total_poll_timeouts": 0,
                "avg_poll_latency_ms": None,
                "slowest_prompt_index": None,
                "slowest_prompt_time_to_plan_s": None,
            },
            "log_file": _redact_home_path(log_file),
        }
        run_file = history_dir / f"run-{stamp}.json"
        run_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "run_file": _redact_home_path(run_file),
                    "log_file": _redact_home_path(log_file),
                    "best_file": _redact_home_path(best_file),
                    "best_updated": False,
                },
                indent=2,
            )
        )
        print(json.dumps(payload["summary"], indent=2))
        print(json.dumps({"health": health}, indent=2))
        return

    runs = run_benchmark(
        base_url=args.base_url,
        repo=args.repo,
        prompts=prompts,
        create_timeout_seconds=args.create_timeout_seconds,
        poll_timeout_seconds=args.poll_timeout_seconds,
        poll_request_timeout_seconds=args.poll_request_timeout_seconds,
        max_consecutive_poll_timeouts=args.max_consecutive_poll_timeouts,
        poll_unhealthy_window_seconds=args.poll_unhealthy_window_seconds,
        max_consecutive_create_failures=args.max_consecutive_create_failures,
        stop_on_first_problem=args.stop_on_first_problem,
        max_suite_seconds=(None if args.max_suite_seconds <= 0 else args.max_suite_seconds),
        log_path=log_file,
    )
    payload = {
        "started_at": started_at,
        "base_url": args.base_url,
        "repo": args.repo,
        "config": _load_config_metadata(Path(args.config_root).expanduser().resolve()),
        "prompt_count": len(prompts),
        "prompts": prompts,
        "results": [asdict(run) for run in runs],
        "summary": _summary(runs),
        "log_file": _redact_home_path(log_file),
    }

    run_file = history_dir / f"run-{stamp}.json"
    run_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    best_updated = False
    if best_file.exists():
        try:
            best_payload = json.loads(best_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            best_payload = {}
        if _is_better(payload, best_payload):
            best_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            best_updated = True
    else:
        best_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        best_updated = True

    print(
        json.dumps(
            {
                "run_file": _redact_home_path(run_file),
                "log_file": _redact_home_path(log_file),
                "best_file": _redact_home_path(best_file),
                "best_updated": best_updated,
            },
            indent=2,
        )
    )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
