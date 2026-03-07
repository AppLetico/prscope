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

- `src/prscope/web/api.py`
  - Canonical command endpoint: `POST /api/sessions/{id}/command`
  - Wrapper endpoints (`/message`, `/round`, `/approve`, `/export`, `/stop`) route through the same executor
  - Command gate order: replay -> allowed revalidation -> processing lock -> reserve row
  - SSE endpoint with **snapshot-on-connect** (`session_state` emitted first)
- `src/prscope/web/events.py`
  - Session-scoped, **multi-subscriber** emitter (`set` of queues per session)
- `src/prscope/planning/core.py`
  - Canonical state machine (`ALLOWED_TRANSITIONS`, `VALID_COMMANDS`)
  - Only protected-state write path: `transition_and_snapshot()`
  - Invariants for draft coherence, processing derivation, and round monotonicity
- `src/prscope/planning/executor.py`
  - Two-phase command execution (`reserve` -> `execute` -> `finalize`)
  - Centralized replay, locking, lease heartbeat, and snapshot persistence
  - Command handlers return `HandlerResult` only (no direct snapshot/lock logic)
- `src/prscope/planning/runtime/orchestration.py`
  - Acts as session coordinator (locks, lifecycle, core transitions, command flow)
  - Delegates orchestration concerns to `src/prscope/planning/runtime/orchestration_support/*`
    (`event_router`, `state_snapshots`, `initial_draft`, `session_starts`, `chat_flow`, `round_entry`)
  - Delegates initial draft planning prompt construction/execution to `AuthorAgent.run_initial_draft()`
  - Enforces persist-then-emit sequencing for runtime events
  - Emits unified `tool_update` events (replacing separate `tool_call`/`tool_result`)
  - Stamps every SSE event with a monotonic `session_version` for ordering guarantees
  - Persists completed tool groups with `sequence` and `created_at` before emitting snapshots
  - Persists bounded active tool calls (`MAX_ACTIVE_TOOL_CALLS = 50`)
- `src/prscope/planning/runtime/pipeline/*`
  - `adversarial_loop.py`: runs staged refinement rounds
  - `stages.py`: stage implementations (`design_review` -> `repair` -> `revise` -> `validation` -> `convergence`)
  - `round_context.py`: round context assembly
  - Stage dependencies are injected explicitly (author, critic, manifesto checker, event + memory adapters), avoiding full runtime coupling
- `src/prscope/planning/runtime/discovery.py`
  - Discovery turn orchestrator and compatibility façade
  - Delegates helper logic to `src/prscope/planning/runtime/discovery_support/*`
    (`models`, `signals`, `existing_feature`, `bootstrap`, `llm`)
  - Delegates semantic routing policy to `src/prscope/planning/runtime/reasoning/discovery_reasoner.py`
  - Keeps session-scoped bootstrap insights (`existing_feature`, `feature_label`, evidence paths)
- `src/prscope/planning/runtime/reasoning/*`
  - Shared Layer 3 policy package: `base`, `models`, `discovery_reasoner`, `refinement_reasoner`, `review_reasoner`, `convergence_reasoner`
  - Consumes `ReasoningContext` and returns provenance-carrying decisions (`confidence`, `evidence`, `decision_source`, `reasoner_version`)
- `src/prscope/planning/runtime/followups/*`
  - Decision graph extraction, merge, and follow-up generation
  - Persisted plan artifacts include `decision_graph_json` and `followups_json`
- `src/prscope/planning/runtime/authoring/*`
  - Author subsystem: `models`, `discovery`, `validation`, `repair`, `pipeline`
- `src/prscope/store.py`
  - Session schema fields for canonical UI state (`status`, `phase_message`, `pending_questions_json`, etc.)
  - Runtime guard preventing direct writes to protected state fields
- `src/prscope/benchmark.py`
  - Harness validation loop for latency and quality

## Canonical Session State Model

The `planning_sessions` row is authoritative for UI state. Core fields:

- `status`
- `current_round`
- `phase_message`
- `is_processing`
- `pending_questions_json`
- `active_tool_calls_json`
- `completed_tool_call_groups_json` (structured: `{sequence, created_at, tools}[]`)
- `event_seq` (monotonic counter for deterministic ordering of turns and tool groups)
- `processing_started_at`
- `current_command_id`

Turns (`planning_turns`) carry a `sequence` field for deterministic timeline ordering. Turns remain useful for traceability and debugging, but are not authoritative for live UI state.

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
4. build signal payloads (`FrameworkSignals`, `ExistingFeatureSignals`, follow-up choice signals)
5. call `DiscoveryReasoner`
6. ask only unresolved decision questions or execute the chosen discovery mode

Expected behavior:

- If evidence indicates a feature already exists, discovery should avoid "create new X" planning.
- If framework evidence is present, discovery should avoid asking "which backend/framework?".
- Clarifying questions should be batched and non-duplicative in UI rendering.
- `discovery_support/*` may collect and score evidence, but should not directly choose discovery routes.

## Refinement Behavior Contract

Refinement now follows the same layered pattern:

1. extract `RefinementMessageSignals` from the latest message + recent context
2. optionally classify ambiguous routing with the author model
3. call `RefinementReasoner`
4. execute the chosen path (`author_chat`, `lightweight_refine`, `full_refine`, or follow-up/issue resolution)

Expected behavior:

- `chat_flow.py` should remain execution-oriented: invoke the reasoner, run the selected path, persist, and emit SSE events.
- lightweight issue resolution and open-question handling should be explainable through reasoner provenance, not hidden heuristics.
- routing telemetry should carry provenance fields so ambiguous paths can be debugged without re-reading orchestration code.

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

- `session_state` (versioned: `v: 1`, full canonical snapshot including `completed_tool_call_groups` and `active_tool_calls`)
- `tool_update` (unified event with `call_id`, `name`, `status: running|done`, `durationMs`; replaces legacy `tool_call`/`tool_result`)
- `plan_ready`
- `thinking`
- `warning`
- `error`
- `complete`
- `token_usage`
- `clarification_needed`
- `state_snapshot` events emitted by command finalize

Every SSE event carries a `session_version` field — a monotonic integer incremented per emission. The frontend uses this to discard stale or reordered events (any event with `session_version ≤ lastVersionSeen` is dropped).

Event semantics:

- **Snapshot first**: every new SSE connection gets a fresh `session_state` event before any subsequent stream events
- **Persist then emit**: runtime persists state first, then emits events. Completed tool groups are persisted *before* the `session_state` snapshot is built and emitted.
- **Multi-tab safe**: all subscribers on the same session receive the same stream
- **Unified tool events**: tool execution is streamed as `tool_update` with upsert semantics (keyed by `call_id`), not separate start/result event pairs

Frontend integration expectation:

- treat `session_state` as total replacement for session UI state, not a partial merge
- treat `tool_update` as a hint for transient active tool state; the next `session_state` snapshot is authoritative

Issue snapshot expectation:

- session snapshots include a backward-compatible flat issue view (`open_issues`) and an additive graph payload (`issue_graph`)
- `issue_graph` includes deterministic replay fields: `nodes`, `edges`, `duplicate_alias`, and `summary`
- adjacency indexes are runtime-derived and should not be persisted
- plan version payloads also carry `decision_graph` and `followups`; the decision graph is the primary planning-state artifact and markdown extraction is compatibility/backfill only

## Runtime Invariants Enforced by Core

High-value invariants:

- Only `transition_and_snapshot()` mutates protected session state
- `is_processing` is derived (`status in WORK_STATES` and `phase_message` is set)
- `current_round` may change only during `refining -> refining`
- In `draft`, questions and processing message cannot coexist
- Non-draft states clear pending questions

## Tool Call Persistence Rules

- Active tool calls are persisted on the session row (`active_tool_calls_json`)
- Completed tool groups are persisted as `completed_tool_call_groups_json` with structured entries: `{sequence, created_at, tools[]}`
- `sequence` is a monotonic integer from the session's `event_seq` counter, shared with turns for deterministic cross-type ordering
- Completed groups are persisted *before* emitting the `session_state` snapshot (fixes race condition where snapshot could contain stale groups)
- Updates are done under explicit transactional boundaries in orchestration
- Active entries are sorted by `created_at` and truncated to most recent 50
- Completed groups are truncated to most recent 50
- Terminal `complete`/closure transitions clear or shrink volatile in-flight state

## Issue Graph Runtime Rules

- Public issue operations canonicalize IDs first (alias-safe traversal and mutation)
- Duplicate detection maps aliases to canonical IDs; duplicate nodes are not created
- Auto-resolution propagates only along `causes` edges (never `depends_on`)
- Root-open computation uses incoming `causes` edges only
- Dependency-chain checks count open nodes with unresolved `depends_on` targets

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

- preserve persist-then-emit ordering (especially: persist completed tool groups before snapshot)
- keep snapshot-first behavior on SSE connect
- avoid adding client-side state reconstruction from turns/events
- keep command rejection payloads structured and deterministic
- stamp all SSE events with monotonic `session_version`
- use `tool_update` (not separate `tool_call`/`tool_result`) for tool event emission
- assign `sequence` from `event_seq` to new turns and tool groups for deterministic ordering

