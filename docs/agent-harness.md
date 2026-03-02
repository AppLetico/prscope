# Agent Harness Guide

This document describes the agent harness used by `prscope` to run planning sessions end-to-end (API + runtime + UI + benchmark loop).

## What "Agent Harness" Means Here

In `prscope`, the harness is the full execution envelope around the planning agents:

- **Transport layer**: FastAPI endpoints and SSE event streaming
- **Runtime orchestration**: session lifecycle, locks, round execution, convergence
- **Agent roles**: discovery, author, and critic loops
- **Sandboxed tools**: read/search/list operations exposed to the author loop
- **Persistence + telemetry**: SQLite-backed session/turn/version state, timing, token/cost events
- **Benchmark harness**: repeatable prompt runner for startup latency + output quality

## High-Level Architecture

Core components and responsibilities:

- `prscope/web/server.py`
  - Boots FastAPI app
  - Serves built frontend assets from `prscope/web/static`
  - Loads `.env` files for direct `uvicorn` runs
- `prscope/web/api.py`
  - HTTP API for session creation, messaging, rounds, approve/export/diff
  - Background kickoff for initial requirements drafting
  - SSE endpoint for live UI events
- `prscope/web/events.py`
  - In-memory, session-scoped SSE queue with bounded capacity and drop-oldest backpressure
- `prscope/planning/runtime/orchestration.py`
  - Main runtime coordinator (`PlanningRuntime`)
  - Owns author/critic/discovery agents, locks, clarification gates, and round transitions
- `prscope/planning/runtime/author.py`
  - Author loop, tool-call enforcement, fallback behavior, token usage emission
- `prscope/planning/runtime/critic.py`
  - Structured critique and constraints validation
- `prscope/planning/runtime/discovery.py`
  - Chat-first discovery turns and question handling
- `prscope/planning/runtime/tools.py`
  - Sandboxed file tools (`list_files`, `read_file`, `grep_code`, etc.) with repo-boundary safety checks
- `prscope/benchmark.py`
  - Repeatable benchmark runner against the API surface

## Session Lifecycle (Requirements Mode)

Typical path when creating a session from requirements:

1. `POST /api/sessions` with `mode=requirements`
2. Runtime creates session record (status starts in drafting flow)
3. API launches initial draft work as a background task
4. UI opens session and subscribes to SSE events
5. Author/critic/discovery events stream while drafting/refining progresses
6. Plan versions are persisted; UI displays `current_plan` when available

Important implementation details:

- Runtime instances are cached per repo in `RuntimeRegistry` so per-session locks survive across requests.
- Initial drafting timing is captured and exposed as `draft_timing` in session payloads.
- SSE subscriptions are session-scoped; cleanup aborts pending clarification waits on disconnect.

## API Surface Used by the Harness

Primary endpoints:

- `GET /api/sessions`
- `POST /api/sessions`
- `GET /api/sessions/{session_id}`
- `DELETE /api/sessions/{session_id}`
- `POST /api/sessions/{session_id}/message` (discovery turn)
- `POST /api/sessions/{session_id}/round` (adversarial round)
- `POST /api/sessions/{session_id}/clarify`
- `POST /api/sessions/{session_id}/approve`
- `POST /api/sessions/{session_id}/export`
- `GET /api/sessions/{session_id}/download/{kind}`
- `GET /api/sessions/{session_id}/diff`
- `GET /api/sessions/{session_id}/events` (SSE)

Repo routing behavior:

- API can resolve repo context from explicit `repo` parameter/body, or from stored session metadata.
- Frontend persists active repo context in local storage and appends `repo` query/body where needed.

## Event Model (SSE)

The frontend subscribes to `EventSource(/api/sessions/{id}/events)` and normalizes these event types:

- `thinking`
- `tool_call`
- `complete`
- `error`
- `warning`
- `token_usage`
- `clarification_needed`

Operational notes:

- Event queue is bounded (default `maxsize=1000`) with drop-oldest behavior when full.
- Events are only enqueued while at least one subscriber is connected.
- Current design is intentionally single-consumer oriented per session (one active UI tab is the intended model).

## Agent Roles and Orchestration

### Discovery

- Collects requirements interactively from user chat turns.
- Can ask targeted clarification questions.

### Author

- Generates and refines plan drafts.
- Uses codebase tools through a sandboxed executor.
- Enforces per-call timeout and model fallback behavior.
- Emits token and cost telemetry per call.

### Critic

- Validates plans against constraints, evidence, and quality expectations.
- Returns structured issue counts/violations used in convergence decisions.

### Round Control

- `run_adversarial_round` executes critic + author refinement cycles.
- Convergence state and score are persisted and reflected in UI status badges/actions.

## Tooling and Safety Boundaries

`ToolExecutor` safety model:

- Restricts file access to repo root; blocks path escape attempts.
- Blocks sensitive file name patterns (e.g., `.env`, credentials-like names).
- Skips binary/non-text extensions for read/search operations.
- Tracks accessed/read files for grounding and audit-style checks.

Available tool functions:

- `list_files`
- `read_file`
- `grep_code`
- `ask_clarification`
- `get_memory_block`

## Configuration Inputs

Primary config source:

- `prscope.yml` loaded from `PRSCOPE_CONFIG_ROOT` when set, otherwise repo root auto-detection.

Planning-specific knobs include:

- `planning.author_model`
- `planning.critic_model`
- `planning.discovery_tool_rounds`
- `planning.author_tool_rounds`
- `planning.max_adversarial_rounds`
- `planning.convergence_threshold`
- `planning.scanner`
- `planning.require_verified_file_references`
- `planning.clarification_timeout_seconds`

Repo mapping gotcha:

- If `local_repo` points to a nonexistent or wrong path, session creation/drafting can fail for valid repo names.

## Benchmark Harness (Load/Latency/Quality Loop)

The benchmark harness (`python -m prscope.benchmark` or `prscope-benchmark`) validates startup behavior through the same API used by UI:

- Creates sessions via `POST /api/sessions`
- Polls session state via `GET /api/sessions/{id}?lightweight=true`
- Records first-plan timing, fallback/timed-out outcomes, and quality heuristic score

Artifacts:

- `benchmarks/results/history/run-<timestamp>.json`
- `benchmarks/results/history/run-<timestamp>.log`
- `benchmarks/results/best_performance.json`

Key timing metrics:

- `time_to_plan_s` (client-observed)
- `server_initial_draft_elapsed_s` (server-reported)
- `client_detect_gap_s` (difference between the two)

## Failure Modes and Recovery Behavior

Common failure classes:

- Repo resolution/config mapping errors (400/invalid repo)
- Model timeout or provider errors during author/critic calls
- Long-poll latency spikes while session remains healthy
- Clarification gate timeouts/aborts

Recovery patterns in harness:

- Author fallback model attempts on timeout/non-chat incompatibility paths
- Poll retry with expanded timeout before counting hard poll timeout
- Session-level cleanup of runtime state and tool artifacts on delete

UI-facing severity policy:

- Recoverable author fallback is emitted as a **warning** (not fatal error) so transient recovery does not appear as hard failure.

## Local Operations Runbook

### Start API + UI

```bash
python3 -m uvicorn prscope.web.server:create_server_app --factory --host 127.0.0.1 --port 8420
```

### Run one-prompt benchmark smoke

```bash
python3 -m prscope.benchmark \
  --base-url http://127.0.0.1:8420 \
  --repo prscope \
  --config-root /path/to/config/root \
  --prompts-file /path/to/prompts-1.json
```

### Verify live event stream quickly

- Open session page in UI and confirm:
  - tool call chips appear
  - warning/error banners appear when relevant
  - plan panel transitions from in-progress state to rendered markdown

## Troubleshooting

### "No plan generated yet" during active drafting

- Check session `status` and recent SSE events.
- If status is in progress but no plan yet, this is usually transient generation latency.
- Confirm static assets are current (rebuild frontend if source changed and backend serves stale bundle).

### Stale frontend after code changes

- Rebuild:

```bash
cd prscope/web/frontend
npm run build
```

- Reload page (or restart server if process pinned old assets in environment-specific workflows).

### Repo not found / wrong repo path

- Validate `prscope.yml` `local_repo` and/or multi-repo mapping.
- Ensure requested `repo` matches resolved repo name/path.

### Benchmark appears slower than server draft time

- Compare `time_to_plan_s` vs `server_initial_draft_elapsed_s`.
- Large `client_detect_gap_s` usually indicates polling cadence/network latency, not purely model generation delay.

## Recommended Team Usage

- Use a cheap-model config for iterative harness diagnostics.
- Keep a one-prompt smoke benchmark in your normal dev loop.
- Track `best_performance.json` deltas for regressions before merging runtime changes.
- Prefer warning severity for recoverable runtime paths; reserve error severity for non-recoverable failures.

