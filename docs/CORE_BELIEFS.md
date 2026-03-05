# Core Beliefs

Golden principles for writing code in this repository. These apply to both human and agent contributors.

When a principle here conflicts with expediency, the principle wins.

## 1. The Database is the Source of Truth

Session state lives in `store.py`. The frontend, SSE events, and in-memory caches are projections of database state, never the reverse. If it's not persisted, it didn't happen.

Corollary: never reconstruct canonical state from event streams, turn history, or client-side logic.

## 2. Persist Then Emit

Every state mutation follows the same sequence: write to database, commit transaction, then emit events. No event should imply a state the database does not yet hold. This is the crash-safety rule.

## 3. Validate at Boundaries

Parse and validate data at ingestion points — API request handlers, config loaders, file readers, LLM response parsers. Interior code operates on validated types, not raw strings or untyped dicts.

Corollary: wrap `json.loads()` and YAML parsing in try/except. Malformed external data must not crash the process.

## 4. Fail Loudly, Recover Gracefully

Raise typed exceptions for programming errors and invalid state transitions. For operational failures (LLM timeouts, file not found, network errors), degrade to a safe fallback state and surface the issue to the user.

The manifesto constraint system is an example: hard constraint violations raise; soft constraint issues warn.

## 5. Layers Import Downward

The architecture has four layers: Foundation → Storage → Intelligence → Interface. Each layer may import from itself or below. Never import upward. This is enforced by `tests/test_architecture.py`.

Adding a new module means deciding which layer it belongs to first.

## 6. One Write Path for Protected State

Protected session fields (`status`, `phase_message`, `is_processing`, `pending_questions_json`, `active_tool_calls_json`) are modified only through `PlanningCore.transition_and_snapshot()`. No shortcut writes. No exceptions.

## 7. Idempotent Commands

Every mutating command carries a `command_id` for replay. The server returns the cached result on replay, rejects concurrent commands with `409`, and never silently drops or double-executes.

## 8. Prefer Shared Utilities Over Hand-Rolled Helpers

If a pattern appears in more than one module, extract it. Token budgeting lives in `budget.py`. Cost estimation lives in `pricing.py`. Telemetry lives in `telemetry.py`. Don't reinvent these per-call-site.

Corollary: when adding a new utility, check if an existing module already covers the concern.

## 9. Keep Leaf Modules Leaf

Runtime leaf modules (`tools.py`, `context/budget.py`, `telemetry.py`, `events/analytics_emitter.py`, `context/clarification.py`, `context/compression.py`) have minimal or zero internal imports. They do not import from `planning.core` or `planning.executor`. This keeps the dependency graph shallow and testable.

`orchestration.py` may import broadly as session coordinator, but stage implementations should receive explicit dependencies instead of importing runtime internals ad hoc.

## 10. Context is Finite — Budget It

LLM context windows are a hard constraint. The planning runtime uses `TokenBudgetManager` to allocate context across requirements, manifesto, memory, and critique. Adding content to a prompt means something else gets displaced.

Corollary: keep manifesto constraints concise. Keep memory blocks within configured char caps. Use on-demand `get_memory_block()` instead of front-loading.

## 11. Tests Prove Behavior, Not Coverage

Write tests for invariants, boundary conditions, and regression cases. Don't write tests that merely exercise happy-path code to inflate coverage numbers. Every test should answer: "what breaks if this test is deleted?"

The architecture tests (`test_architecture.py`) are an example: they enforce structural rules that prevent silent architectural drift.

## 12. Documentation Earns Its Keep

Every doc in `docs/` must be referenced from at least one other document or from `AGENTS.md`. Orphaned docs rot. If a document isn't worth linking to, it isn't worth having.

When behavior changes, update the relevant doc in the same PR. Stale docs are worse than no docs.

## Applying These Principles

- Before writing code, identify which principle(s) apply.
- If a change violates a principle, call it out in the PR description and explain the tradeoff.
- If a principle is wrong, update this document — don't silently ignore it.
