# Planning State Machine

This document defines the runtime contract for planning sessions.

It is written for both:

- operators/product engineers who need predictable behavior under retries/reloads
- contributors who need clear invariants before changing orchestration code

If you remember one thing: the database session row is canonical state, and everything else is a projection or audit trail.

## Source of Truth

`planning_sessions` is the canonical state row for UI and command handling. The frontend should render from:

- `GET /api/sessions/{id}` (authoritative baseline)
- `session_state` SSE snapshots (real-time projection updates)

Turns and tool traces are audit artifacts, not control state.

Why this matters:

- reloads are deterministic
- retries are safe
- UI behavior is not dependent on parsing assistant text

## Protected Session Fields

These fields are state-machine controlled:

- `status`
- `phase_message`
- `pending_questions_json`
- `is_processing`
- `processing_started_at`
- `active_tool_calls_json`
- `completed_tool_call_groups_json`
- `event_seq`

They must be written through `PlanningCore.transition_and_snapshot()` only.

`Store.update_planning_session()` enforces this via a runtime guard unless `_bypass_protection=True` is explicitly used by trusted core paths.

## States

Canonical statuses:

- `draft`
- `refining`
- `converged`
- `approved`
- `error`

Stable user-visible checkpoints are typically:

- `draft` (requirements capture and initial draft generation)
- `refining` (manual critique rounds available)
- `converged` (ready to approve or run another round)
- `approved`

## Allowed Transitions

Defined in `PlanningCore.ALLOWED_TRANSITIONS`:

- `draft -> draft | refining | error`
- `refining -> refining | converged | error`
- `converged -> refining | approved | error`
- `approved -> approved | error`
- `error -> draft`

## Valid Commands by State

Executor command matrix (server-derived):

- `draft`: `message`, `reset`
- `refining`: `run_round`, `message`, `reset`
- `converged`: `run_round`, `message`, `approve`, `reset`
- `approved`: `export`, `reset`
- while processing with live lease: `cancel` only

## Core Transition Function

`transition_and_snapshot()` is the kernel operation:

1. Begins `BEGIN IMMEDIATE` transaction
2. Reads current session
3. Validates transition and invariants
4. Derives state fields (`is_processing`, `processing_started_at`, etc.)
5. Persists full protected state atomically
6. Returns a versioned snapshot payload

Snapshot shape:

- `type: "session_state"`
- `v: 1`
- `status`
- `phase_message`
- `is_processing`
- `current_round`
- `pending_questions`
- `active_tool_calls`
- `completed_tool_call_groups` (structured: `{sequence, created_at, tools[]}[]`)
- `session_version` (monotonic ordering counter)

Runtime file snapshots under `.prscope/sessions/{session_id}.json` are also versioned with
`schema_version` (current: `1`). Restore logic defaults missing versions to legacy (`0`) and
applies backward-compatible fallback parsing.

Design intent: transition logic is "closed world." Illegal states should be unrepresentable.

## Invariants

Hard invariants enforced in core:

- **Single write path**: no direct writes to protected fields outside transition kernel
- **Draft coherence**:
  - if `status == draft` and `pending_questions_json != null`, then `phase_message == null`
  - if `status == draft` and `phase_message != null`, then `pending_questions_json` is cleared
- **Non-draft question clearing**: pending questions are cleared for all non-`draft` states
- **Processing derivation**: `is_processing = status in WORK_STATES and phase_message is not null`
- **Round monotonicity**:
  - `current_round` can only change on `refining -> refining`
  - that transition requires increasing round, unless explicit stability mode is requested for internal state-only updates

## Command Idempotency and Rejection Order

`POST /api/sessions/{id}/command` is canonical (wrapper endpoints route here) and uses executor reserve/finalize semantics:

1. Idempotency replay check (`planning_commands` by `(session_id, command_id)`)
2. Allowed command revalidation (transactional, current DB status)
3. Processing lock check (derived from running/finalizing command + lease)
4. Reserve command row (`running`, `started_at`, `lease_expires_at`)
5. Execute handler outside transaction with lease heartbeat
6. Finalize transaction (`finalizing -> completed`) with persisted `result_snapshot_json`

Rejections return `409` with structured payload including:

- `status`
- `phase_message`
- `allowed_commands`
- `reason` (`processing_lock`, `invalid_status`, `unknown_command`, `timeout`, `cancelled`)

This keeps client behavior simple: the server explains what actions are valid next.

## Event Sequencing Contract

For state-relevant runtime activity:

1. Persist session mutation in DB (including `completed_tool_call_groups_json`)
2. Commit transaction
3. Emit `session_state` snapshot (now reflects persisted groups)
4. Emit secondary events (`tool_update`, `plan_ready`, etc.)

Critical ordering rule: completed tool groups must be persisted *before* the `session_state` snapshot is built. This ensures the snapshot always contains the latest groups and the frontend never needs client-side reconciliation.

Every emitted SSE event carries a monotonic `session_version` integer. The backend increments this counter before each emission. The frontend discards any event with `session_version ≤ lastVersionSeen`, which prevents stale or reordered events from causing UI inconsistencies.

No event should be emitted that implies a state the DB does not yet hold.

This is the key crash-safety rule for avoiding split-brain between live UI and persisted state.

## Command Execution and Timeout Recovery

Commands are now executor-backed and tracked in durable `planning_commands` command-log rows:

- reserve phase inserts `running` command row and sets `current_command_id`
- execute phase runs handler/pipeline outside DB transaction and renews lease
- finalize phase persists snapshot and marks `completed`

Timeout monitor behavior:

1. scans expired `running` rows (`lease_expires_at < now`)
2. marks them `failed` with `timeout` reason
3. clears `current_command_id` only if pointing to timed-out row

`finalizing` rows are never timed out by monitor.

## SSE Contract

`GET /api/sessions/{id}/events` must emit:

1. Fresh `session_state` snapshot immediately on connection
2. Then subsequent live events

This supports reconnect correctness and multi-tab consistency.

## Tool Call State

### Active Tool Calls

`active_tool_calls_json` is bounded, volatile state:

- Persisted on session row for restart/reload continuity
- Updated with deterministic ordering (`created_at`)
- Truncated to most recent 50 entries
- Cleared/shrunk as calls complete
- Streamed to the frontend via unified `tool_update` SSE events (keyed by `call_id`, status `running` or `done`)

### Completed Tool Groups

`completed_tool_call_groups_json` stores finalized tool execution batches:

- Each group is a structured object: `{sequence, created_at, tools[]}`
- `sequence` is a monotonic integer from the session's `event_seq` counter, shared with `planning_turns.sequence` for cross-type timeline ordering
- Groups are persisted before the `session_state` snapshot is emitted (no race condition)
- Truncated to most recent 50 groups
- The frontend merges turns and groups by `sequence` into a single chronological timeline

## Issue Graph and Dedupe Configuration

Issue tracking in refinement rounds is graph-backed. Runtime behavior is configurable via
`planning.issue_dedupe` and `planning.issue_graph`.

Issue dedupe configuration (`planning.issue_dedupe`):

- `embeddings_enabled`: `auto | true | false`
- `embedding_model`: LiteLLM embedding model string
- `similarity_threshold`: semantic duplicate threshold
- `fallback_mode`: `lexical | none`

Issue graph configuration (`planning.issue_graph`):

- `max_nodes`: hard cap for tracked issue nodes
- `max_edges`: hard cap for tracked issue edges
- `causality_extraction_enabled`: enable/disable prose-based causal inference
- `causality_max_edges_per_review`: cap inferred edges per review pass
- `causality_min_text_len`: minimum phrase length for inferred issue text
- `causality_patterns`: lexical causal markers used by extractor
- `causality_negative_patterns`: vague/noisy phrases rejected by extractor

Design intent:

- semantic dedupe can be enabled independently of author/critic model providers
- when `embeddings_enabled=auto`, runtime attempts semantic matching first and can fall back to lexical matching based on `fallback_mode`
- duplicate findings are canonicalized via alias mapping (no duplicate issue nodes are created)
- issue graph persistence is deterministic (`nodes`, `edges`, `duplicate_alias`) and replayable on restart
- convergence uses root-open + unresolved dependency-chain checks, in addition to existing stability gates

## Crash Recovery

At startup (before serving requests):

- `reconcile_stuck_sessions()` scans sessions with:
  - `is_processing = true`
  - `status` in work states
  - stale `processing_started_at`
- Stale sessions transition to `error` via `transition_and_snapshot()`

This favors coherent state over resuming unknown in-flight work.

## Correctness Goals

The runtime model is designed so these remain true:

- A full page reload preserves the exact UI state.
- `GET /session` alone can render the correct screen.
- Missing SSE events recover on reconnect via snapshot-first behavior.
- Deleting turn history does not break core session lifecycle behavior.

## Contributor Checklist

Before merging runtime/state changes, verify:

- no protected fields are mutated outside `transition_and_snapshot()`
- command endpoints still use gate order (idempotency -> processing -> allowed)
- `session_state` remains versioned and snapshot-complete (includes `completed_tool_call_groups`)
- completed tool groups are persisted before snapshot emission
- all SSE events carry monotonic `session_version`
- tool events use unified `tool_update` format (not separate `tool_call`/`tool_result`)
- new turns and tool groups receive `sequence` from `event_seq` for deterministic ordering
- reconnect still emits snapshot first
- UI state still renders correctly from `GET /api/sessions/{id}` alone
- issue graph snapshot payload remains deterministic (`nodes`, `edges`, `duplicate_alias`, `summary`)

## Runtime Module Map (Ownership)

Use this as a quick ownership guide when making state-machine-adjacent changes:

- `prscope/planning/runtime/orchestration.py`
  - Session coordinator only: locking, lifecycle entrypoints, core transition calls, and event dispatch wiring.
  - Should not re-absorb stage implementation logic.
- `prscope/planning/runtime/pipeline/adversarial_loop.py`
  - Executes staged refinement round flow and sequencing.
- `prscope/planning/runtime/pipeline/stages.py`
  - Stage implementations (`design_review`, `repair_plan`, `revise_plan`, `validation_review`, `convergence_check`).
  - Uses explicit dependency injection (author, critic, manifesto checker, emit/memory adapters), not full runtime access.
- `prscope/planning/runtime/author.py`
  - Author-side planning behavior, including initial draft flow (`run_initial_draft`) and author loop/tool policy.
- `prscope/planning/runtime/context/*`
  - Context assembly, token budgeting, critique compression, and clarification gate logic.
- `prscope/planning/runtime/review/*`
  - Critique support services: manifesto validation and issue similarity/dedupe.
- `prscope/planning/runtime/events/*`
  - Event shaping, token accounting, and tool event state persistence.

Design intent:

- keep `PlanningCore.transition_and_snapshot()` as the only protected-state write path
- keep orchestration entrypoints thin and explicit
- keep stage dependencies explicit to avoid hidden coupling
