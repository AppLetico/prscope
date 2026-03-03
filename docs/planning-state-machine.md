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

They must be written through `PlanningCore.transition_and_snapshot()` only.

`Store.update_planning_session()` enforces this via a runtime guard unless `_bypass_protection=True` is explicitly used by trusted core paths.

## States

Canonical statuses:

- `created`
- `preparing`
- `discovering`
- `drafting`
- `refining`
- `converged`
- `approved`
- `exported`
- `error`

Stable user-visible checkpoints are typically:

- `discovering` (Q&A)
- `refining` (manual critique rounds available)
- `converged` (ready to approve or run another round)
- `approved` / `exported`

## Allowed Transitions

Defined in `PlanningCore.ALLOWED_TRANSITIONS`:

- `created -> discovering | error`
- `preparing -> discovering | error`
- `discovering -> drafting | error`
- `drafting -> refining | error`
- `refining -> refining | converged | error`
- `converged -> refining | approved | error`
- `approved -> exported | error`
- `exported -> (terminal)`
- `error -> (terminal)`

## Valid Commands by State

Defined in `PlanningCore.VALID_COMMANDS`:

- `discovering`: `message`
- `refining`: `message`, `round`
- `converged`: `round`, `approve`
- `approved`: `export`
- all other states: none

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

Design intent: transition logic is "closed world." Illegal states should be unrepresentable.

## Invariants

Hard invariants enforced in core:

- **Single write path**: no direct writes to protected fields outside transition kernel
- **Discovery coherence**:
  - if `status == discovering` and `pending_questions_json != null`, then `phase_message == null`
  - if `status == discovering` and `phase_message != null`, then `pending_questions_json` is cleared
- **Non-discovery question clearing**: pending questions are cleared for all non-`discovering` states
- **Processing derivation**: `is_processing = status in WORK_STATES and phase_message is not null`
- **Round monotonicity**:
  - `current_round` can only change on `refining -> refining`
  - that transition requires increasing round, unless explicit stability mode is requested for internal state-only updates

## Command Idempotency and Rejection Order

Command endpoints (`message`, `round`, `approve`) require `command_id` and use this order:

1. Idempotency replay check (`last_commands_json[command_type]`)
2. Processing lock check (`is_processing`)
3. Allowed command check (`VALID_COMMANDS`)

Rejections return `409` with structured payload including:

- `status`
- `phase_message`
- `allowed_commands`

This keeps client behavior simple: the server explains what actions are valid next.

## Event Sequencing Contract

For state-relevant runtime activity:

1. Persist session mutation in DB
2. Commit transaction
3. Emit `session_state`
4. Emit secondary events (`tool_call`, `tool_result`, `plan_ready`, etc.)

No event should be emitted that implies a state the DB does not yet hold.

This is the key crash-safety rule for avoiding split-brain between live UI and persisted state.

## SSE Contract

`GET /api/sessions/{id}/events` must emit:

1. Fresh `session_state` snapshot immediately on connection
2. Then subsequent live events

This supports reconnect correctness and multi-tab consistency.

## Active Tool Calls

`active_tool_calls_json` is bounded, volatile state:

- Persisted on session row for restart/reload continuity
- Updated with deterministic ordering (`created_at`)
- Truncated to most recent 50 entries
- Cleared/shrunk as calls complete

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
- `session_state` remains versioned and snapshot-complete
- reconnect still emits snapshot first
- UI state still renders correctly from `GET /api/sessions/{id}` alone
