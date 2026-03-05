# Agent Harness Guide

This guide documents the runtime harness that executes planning sessions end-to-end (API + orchestration + SSE + UI projection + benchmark loop).

Audience:

- use this doc for operational behavior and integration expectations
- use `docs/planning-state-machine.md` for strict invariants and state rules

## What "Harness" Means in Prscope

In `prscope`, the harness is the operational envelope around planning agents and session state:

- **Transport**: FastAPI HTTP commands plus SSE event stream
- **Runtime orchestration**: deterministic session lifecycle and round execution
- **State kernel**: explicit state machine in `PlanningCore.transition_and_snapshot()`
- **Persistence**: SQLite session row as canonical UI state, turns as audit log
- **Tool/runtime telemetry**: tool call/result events and token/cost metrics
- **Benchmark harness**: repeatable startup + quality checks against the same API surface

## Architecture and Ownership

Core components:

- `prscope/web/api.py`
  - Canonical command endpoint: `POST /api/sessions/{id}/command`
  - Wrapper endpoints (`/message`, `/round`, `/approve`, `/export`, `/stop`) route through the same executor
  - Command gate order: replay -> allowed revalidation -> processing lock -> reserve row
  - SSE endpoint with **snapshot-on-connect** (`session_state` emitted first)
- `prscope/web/events.py`
  - Session-scoped, **multi-subscriber** emitter (`set` of queues per session)
- `prscope/planning/core.py`
  - Canonical state machine (`ALLOWED_TRANSITIONS`, `VALID_COMMANDS`)
  - Only protected-state write path: `transition_and_snapshot()`
  - Invariants for draft coherence, processing derivation, and round monotonicity
- `prscope/planning/executor.py`
  - Two-phase command execution (`reserve` -> `execute` -> `finalize`)
  - Centralized replay, locking, lease heartbeat, and snapshot persistence
  - Command handlers return `HandlerResult` only (no direct snapshot/lock logic)
- `prscope/planning/runtime/orchestration.py`
  - Integrates discovery/author/critic loops with core transitions
  - Enforces persist-then-emit sequencing for runtime events
  - Persists bounded active tool calls (`MAX_ACTIVE_TOOL_CALLS = 50`)
- `prscope/planning/runtime/discovery.py`
  - Generalized discovery engine (feature intent extraction + evidence-first questioning)
  - Registry-scored framework detection and signal indexing
  - Session-scoped bootstrap insights (`existing_feature`, `feature_label`, evidence paths)
- `prscope/store.py`
  - Session schema fields for canonical UI state (`status`, `phase_message`, `pending_questions_json`, etc.)
  - Runtime guard preventing direct writes to protected state fields
- `prscope/benchmark.py`
  - Harness validation loop for latency and quality

## Canonical Session State Model

The `planning_sessions` row is authoritative for UI state. Core fields:

- `status`
- `current_round`
- `phase_message`
- `is_processing`
- `pending_questions_json`
- `active_tool_calls_json`
- `processing_started_at`
- `current_command_id`

Turns (`planning_turns`) remain useful for traceability and debugging, but are not authoritative for live UI state.

For the full state contract and transition matrix, see `docs/planning-state-machine.md`.

## Session Lifecycle (Requirements Mode)

Typical flow:

1. `POST /api/sessions` with `mode=requirements`
2. Session is created and runtime schedules initial draft work
3. UI reads `GET /api/sessions/{id}` and then opens SSE stream
4. SSE emits `session_state` snapshot immediately on connect
5. Runtime emits progress/tool events while persisting state transitions
6. Plan versions are saved by round and exposed via session + export endpoints

Key design guarantees:

- `GET /api/sessions/{id}` is sufficient to render the correct screen
- SSE accelerates UI updates, but does not define canonical state
- Reloading any tab should preserve the exact state projection

Practical implication: if SSE is interrupted, the UI should still recover from the next GET + snapshot event without ambiguous intermediate logic.

## Discovery Behavior Contract

Discovery operates in this order:

1. extract feature intent from the latest user request
2. bootstrap-scan repository evidence (tools + grep/read snapshots)
3. infer framework/signals from shared scan results
4. decide whether feature likely exists
5. ask only unresolved decision questions

Expected behavior:

- If evidence indicates a feature already exists, discovery should avoid "create new X" planning.
- If framework evidence is present, discovery should avoid asking "which backend/framework?".
- Clarifying questions should be batched and non-duplicative in UI rendering.

## Command Model

Primary command endpoint:

- `POST /api/sessions/{session_id}/command`

Command safety:

- All mutating commands carry `command_id` (UUID) for idempotent replay
- Replay source of truth is durable `planning_commands.result_snapshot_json`
- Concurrent commands while processing return `409` with `reason=processing_lock`
- Invalid command/state combinations return `409` with `reason=invalid_status`

This behavior is intentional: commands are rejected deterministically rather than queued implicitly.

Executor details:

- Reserve phase writes `planning_commands` row as `running` with lease
- Execute phase runs handler/pipeline and renews lease heartbeat
- Finalize phase writes canonical snapshot, marks command `completed`, clears `current_command_id`

## SSE Event Model

Primary events:

- `session_state` (versioned: `v: 1`, full canonical snapshot)
- `tool_call`
- `tool_result`
- `plan_ready`
- `thinking`
- `warning`
- `error`
- `complete`
- `token_usage`
- `clarification_needed`
- `state_snapshot` events emitted by command finalize

Event semantics:

- **Snapshot first**: every new SSE connection gets a fresh `session_state` event before any subsequent stream events
- **Persist then emit**: runtime persists state first, then emits events
- **Multi-tab safe**: all subscribers on the same session receive the same stream

Frontend integration expectation:

- treat `session_state` as total replacement for session UI state, not a partial merge

## Runtime Invariants Enforced by Core

High-value invariants:

- Only `transition_and_snapshot()` mutates protected session state
- `is_processing` is derived (`status in WORK_STATES` and `phase_message` is set)
- `current_round` may change only during `refining -> refining`
- In `draft`, questions and processing message cannot coexist
- Non-draft states clear pending questions

## Tool Call Persistence Rules

- Active tool calls are persisted on the session row (`active_tool_calls_json`)
- Updates are done under explicit transactional boundaries in orchestration
- Entries are sorted by `created_at` and truncated to most recent 50
- Terminal `complete`/closure transitions clear or shrink volatile in-flight state

## Crash Recovery and Startup Reconciliation

On app startup, before serving requests:

- expired `planning_commands` rows are marked failed (`timeout`) idempotently
- `PlanningCore.reconcile_stuck_sessions()` scans sessions with `is_processing = true`
- Uses `processing_started_at` timeout to identify stale work
- Transitions stale sessions to `error` via `transition_and_snapshot()`

Tradeoff: in-flight generation is not resumed; coherence is prioritized.

Operator note: after a restart, stale sessions may surface as `error` and require an explicit user retry.

## API Surface Used by Harness

- `GET /api/sessions`
- `POST /api/sessions`
- `GET /api/sessions/{session_id}`
- `DELETE /api/sessions/{session_id}`
- `POST /api/sessions/{session_id}/command`
- `POST /api/sessions/{session_id}/message` (wrapper)
- `POST /api/sessions/{session_id}/round` (wrapper)
- `POST /api/sessions/{session_id}/clarify`
- `POST /api/sessions/{session_id}/approve`
- `POST /api/sessions/{session_id}/export`
- `GET /api/sessions/{session_id}/download/{kind}`
- `GET /api/sessions/{session_id}/diff`
- `GET /api/sessions/{session_id}/events`

## Benchmark Harness

`prscope-benchmark` (or `python -m prscope.benchmark`) validates startup and quality against the same API:

- Creates session
- Polls lightweight session state
- Tracks first-plan timing and quality score

Artifacts:

- `benchmarks/results/history/run-<timestamp>.json`
- `benchmarks/results/history/run-<timestamp>.log`
- `benchmarks/results/best_performance.json`

## Local Operations Runbook

Start API + static UI:

```bash
python3 -m uvicorn prscope.web.server:create_server_app --factory --host 127.0.0.1 --port 8420
```

Quick one-prompt benchmark smoke:

```bash
python3 -m prscope.benchmark \
  --base-url http://127.0.0.1:8420 \
  --repo prscope \
  --config-root /path/to/config/root \
  --prompts-file /path/to/prompts-1.json
```

## Troubleshooting

### UI looks inconsistent after reconnect

- Confirm the first SSE event received is `session_state`.
- Compare UI against `GET /api/sessions/{id}` payload.
- Verify frontend is using snapshot replacement, not incremental merge.

### Commands unexpectedly rejected

- Inspect returned `409` payload (`reason`, `status`, `phase_message`, `allowed_commands`).
- Confirm the submitted `command_id` is unique for new attempts.

### Session appears stuck in processing

- Check `status`, `is_processing`, and `processing_started_at`.
- Restart server and verify startup reconciliation logs for stale-session recovery.

## Change-Safety Checklist

When editing harness code, keep these fixed:

- preserve persist-then-emit ordering
- keep snapshot-first behavior on SSE connect
- avoid adding client-side state reconstruction from turns/events
- keep command rejection payloads structured and deterministic

