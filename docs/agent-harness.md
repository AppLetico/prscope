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
  - SSE endpoint with **snapshot-on-connect** (`session_state` emitted first); optional query `repo=` uses the same repo-scoped `Store` as other session routes for that initial snapshot
  - `POST /api/sessions/{id}/clarify` **does not** go through the command executor (see [Clarification vs commands](#clarification-vs-commands))
- `src/prscope/web/server.py`
  - `run_server` / `ensure_server_running` refuse binding to `0.0.0.0` or `::` unless `PRSCOPE_ALLOW_PUBLIC_BIND=1` (no API auth by default)
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
7. **Export (PRD)**: `export_plan_documents` writes `PRD.md` (and a stub `conversation.md`) under `{repo.output_dir or ./plans}/{sanitized_session_title}/`. The PRD template includes an **After approval (handoff)** section: Prscope does not edit the repo; teams implement and run CI outside the harness. **Suggested verification** lists example commands; when `plan_json` on the current version includes `acceptance_criteria`, those lines are inlined for traceability.

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

### Codebase tools: `grep_code` and `glob_files`

- **`grep_code`** searches under the repo sandbox. When `ripgrep` (`rg`) is on `PATH` and `planning.tools.grep_backend` is `auto` (default) or `ripgrep`, searches use **ripgrep** for speed and `.gitignore`-aware exclusions (configurable via `planning.tools.ripgrep_respect_ignore_files`). If `rg` is missing or ripgrep errors, execution falls back to the built-in Python line scan.
- Use **`output_mode: content`** (default) when you need matching lines for evidence (discovery and feature verification). Use **`output_mode: files_with_matches`** to list only file paths and save tokens when you only need to know where a pattern appears before opening files with `read_file`.
- Optional **`glob`**, **`type`**, and **`case_insensitive`** apply to the ripgrep path; the Python fallback ignores `glob`/`type` (a note is included in the tool result).
- Optional **`context`** (integer) adds lines before/after each match via ripgrep `-C`; results include `line_kind` of `match` or `context` in content mode. The Python fallback **ignores** `context` (a note is included).
- Optional **`offset`** skips the first N result rows before applying `head_limit` / `max_results`. For **content** mode this is match rows (and context rows when `context` is set); for **`files_with_matches`** it is the path list. The Python fallback applies offset after collecting matches (see tool notes).
- **`glob_files`** finds paths by wildcard under a directory (for example `**/*.py`). It uses **Python** `glob` semantics: **bash-style brace expansion is not supported** (patterns such as `**/*.{py,go}` typically match nothing). Use one extension per pattern or issue separate `glob_files` calls. Prefer **`list_files`** for a single directory listing; use **`glob_files`** for recursive filename patterns.

### `read_file`: line windows vs full-file reads

- Default behavior reads from the beginning of the file for up to `max_lines` (tool default 200). For large files, discovery and authoring should prefer **windowed** reads: `around_line` + `radius` after `grep_code` reports line numbers, or `start_line` + `max_lines` for sequential chunks. This avoids loading only the file header when the relevant code (routes, `create_app`, middleware) lives thousands of lines in.

### `read_file`: fast path vs streaming (large files on disk)

- Files whose **on-disk size** is at most `planning.tools.read_file_fast_path_max_bytes` (default **10 MiB**, similar to Claude Code’s fast path) are read with `read_text` + `splitlines` in memory.
- **Larger files** use a **streaming** line reader: only the requested line window is held in memory, so peak RAM does not scale with file size. Accurate `line_count` still requires scanning the entire file once (I/O cost, not a giant in-memory buffer).
- Reads are **refused** when `stat().st_size` exceeds `planning.tools.read_file_max_file_bytes` (default **1 GiB**). Use `grep_code`, smaller windows, or raise the cap in `prscope.yml` if you intentionally work with very large text artifacts.

### `tool_result_max_chars` and artifact offload

- Each tool result is JSON-encoded; if its size exceeds `max(1024, planning.tools.tool_result_max_chars)` (default **8000**), the runtime writes the full payload under `.prscope/tool-results/...` and returns a short summary plus `stored_at`. Models are instructed to call `read_file` on that path when they need the full payload. If discovery often hits artifact offload and loses inline context, **raise** `planning.tools.tool_result_max_chars` in `prscope.yml` (trades larger prompts for fewer artifacts).
- **Per-tool overrides:** `planning.tools.tool_result_max_chars_by_tool` is an optional map of tool name → max chars (each value is still clamped to at least **1024**). For example, allow a larger inline `grep_code` payload without raising the global cap for every tool.

### `read_file_max_output_chars` (post-window)

- After line-windowing, the returned `content` string is capped at `planning.tools.read_file_max_output_chars` (default **256000**). If truncated, the payload includes `content_truncated: true` and a `note`. This guards against huge single lines (minified JSON, etc.) that would otherwise dominate the prompt.

### Planner draft diagnostics: `author_self_review_used`

- **`author_self_review_used` is not an optional “quality second pass.”** It becomes true only when the **initial draft planner** had a **failed validation** on an attempt and `self_review_draft` produced **revision hints** for a subsequent redraft. When the first planner draft passes validation, this flag stays **false** and log lines such as `self_review=False` are **expected** — they mean no failure-path hint pass ran, not that a review was skipped.

### Prompt context budgeting

- **Initial draft** (`AuthorAgent.run_initial_draft`) sizes the **user-visible** planning blob with `TokenBudgetManager`. The effective context window comes from `MODEL_CONTEXT_WINDOWS` for the draft model (`initial_draft_model` or `author_model` override), falling back to a conservative default when the model id is unknown.
- **Synthetic exploration tool updates**: `planning.synthetic_initial_draft_tool_updates` (default on) makes the planner pipeline emit bounded `tool_update` events after deterministic `explore_repo` so the UI can mirror agent-style activity. Each payload includes `synthetic: true` on the tool object so it is distinguishable from real LLM tool calls. Set to `false` in `prscope.yml` to disable.
- **Reserved space**: `planning.context_tool_overhead_tokens` is subtracted from the prompt budget to leave room for tool definitions, system wrappers, and completion (`planning.author_prompt_completion_reserve_tokens`). The budget targets what we assemble in the user message, not the full wire prompt to the provider.
- **Token estimates**: `planning.token_budget_estimator` is `heuristic` (default, char-based) or `tiktoken` when the optional `tiktoken` package is installed (better for OpenAI-style tokenization).
- **Discovery tool loops**: `planning.discovery_conversation_max_chars` caps the transcript before each LLM call (drops oldest `tool` messages first). `planning.tools.tool_result_max_chars` caps a single tool payload before artifact offload; discovery may further wrap oversized JSON with a truncated preview. See **`tool_result_max_chars` and artifact offload** above for tuning when tool results are often stored as artifacts.

## Refinement Behavior Contract

Refinement now follows the same layered pattern:

1. extract `RefinementMessageSignals` from the latest message + recent context
2. optionally classify ambiguous routing with the author model
3. call `RefinementReasoner`
4. inside refinement reasoning, optionally run a bounded evidence gate for pressured decisions
5. execute the chosen path (`author_chat`, `lightweight_refine`, `full_refine`, or follow-up/issue resolution)

Expected behavior:

- `chat_flow.py` should remain execution-oriented: invoke the reasoner, run the selected path, persist, and emit SSE events.
- lightweight issue resolution and open-question handling should be explainable through reasoner provenance, not hidden heuristics.
- routing telemetry should carry provenance fields so ambiguous paths can be debugged without re-reading orchestration code.
- refinement evidence refresh stays bounded (`<=3` search queries, `<=8` files, `<=5s`) and should remain selective rather than becoming the default path.
- options/tradeoff memos remain deferred until bounded evidence refresh proves value; the runtime currently persists decisions, not internal exploratory artifacts.
- broader `PLAN_PATCH` runtime simplification remains a post-benchmark evaluation item rather than part of the initial evidence-refresh rollout.

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

## Clarification vs commands

`POST /api/sessions/{id}/clarify` calls `PlanningRuntime.provide_clarification` directly and **does not** create a `planning_commands` row or take the processing lock.

Rationale:

- Clarification answers unblock an in-memory `ClarificationGate` and patch `clarifications_log_json` on the session row; the operation is synchronous and short-lived.
- Routing it through `execute_command` would either block concurrent `run_round` / other commands or require a new “non-locking command” category.

Revisit this if you need **durable audit rows for every user action** or unified lease semantics for clarification; until then, treat clarify as a coordination primitive, not a planning command.

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
- `GET /api/sessions/{id}` also emits an additive derived `impact_view` computed from `current_plan.decision_graph` plus snapshot `issue_graph`
- `impact_view` is not persisted; it is a deterministic read model that can include per-decision pressure, root-cause clusters, and reconsideration candidates
- persisted JSON artifacts are authoritative; runtime graph objects and read models must stay deterministic functions of those explicit persisted inputs

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

## Context compaction strategy catalog

Each path below is **independent**; see `PlanningConfig` in `src/prscope/config.py` for YAML keys.

### Discovery transcript trim

- **Where:** [`DiscoveryLLMClient.llm_call_with_tools`](src/prscope/planning/runtime/discovery_support/llm.py) calls `trim_discovery_messages_for_budget` before **each** LiteLLM completion inside the discovery tool loop.
- **Config:** `planning.discovery_conversation_max_chars` (runtime enforces a minimum of 8192 characters). Tool results are also capped per round (`per_tool_cap = max(4096, max_conv // 8)`).
- **Optional token cap:** `planning.discovery_conversation_max_input_tokens` (omit or `0` = disabled). When set, the serialized discovery message list is estimated with `planning.token_budget_estimator` (`heuristic` = `len/3.5`, or `tiktoken` when installed for OpenAI-style models). Trimming runs until **both** the character cap and the token cap (when enabled) are satisfied—characters remain a hard backstop.
- **Compaction circuit breaker:** `planning.discovery_compaction_failure_max` (default `0` = unlimited). When `> 0`, each `context_compaction` emission for `discovery_transcript_trim` increments a counter; if it exceeds this limit, discovery aborts with a user-visible message and SSE `discovery_circuit_breaker` (`reason: compaction_emit_limit`).
- **What survives:** Recent messages and tool-output text within the budget; older content is dropped by the trim helper.
- **SSE:** `context_compaction` with `reason: "discovery_transcript_trim"` and `session_stage: "discovery"`.

### Prior-critique compaction (adversarial refinement)

- **When:** [`PlanningRuntime._should_compact_context`](src/prscope/planning/runtime/orchestration.py) returns true if **any** of: adversarial `round_number >= 2`, at least **four** critic turns in the conversation, **recent peak** `max_prompt_tokens` above **65%** of the min author/critic context window, or current plan text length **≥ 14_000** characters.
- **Entry:** [`PlanningRuntime._prepare_adversarial_compaction_context`](src/prscope/planning/runtime/orchestration.py) runs at the start of each adversarial round ([`adversarial_loop.py`](src/prscope/planning/runtime/pipeline/adversarial_loop.py)); it sets `state.working_summary` when compaction is needed.
- **Heuristic summary:** [`CritiqueCompressor`](src/prscope/planning/runtime/context/compression.py) — knobs `planning.critique_compress_max_recent_chars` and `planning.critique_compress_max_summary_chars`. Produces a short “older rounds” digest plus the latest critique excerpt.
- **Optional LLM summary:** If `planning.critique_llm_summarize_enabled` is true **and** `max_prompt_tokens >= planning.critique_llm_summarize_prompt_tokens_threshold`, prior critiques are summarized with the author LLM (`planning.critique_llm_summarize_model` or `critic_model`).
- **SSE:** `context_compaction` with `reason: "critique_heuristic_summary"` or `"critique_llm_summary"` and `session_stage: "refinement"`.

### Token budgeting elsewhere

Initial draft and refinement prompts use `TokenBudgetManager` and memory block caps (`planning.memory_block_max_chars`, repo overrides). That is **budgeting**, not the same as “compaction” above; see [`docs/memory-context-manifesto.md`](memory-context-manifesto.md).

## Prompt construction and provider cache boundaries (audit)

- **Relatively stable (good cache prefix candidates):** fixed system prompts (e.g. reviewer system prompt in `src/prscope/planning/runtime/critic.py`), tool schemas in `src/prscope/planning/runtime/tools.py`, manifesto/skills text when unchanged per `instruction_context_refresh`.
- **Volatile (per turn):** user requirements, current plan markdown, discovery transcript, tool results, memory blocks pulled on demand, critic/author turn history.
- **Author path:** [`AuthorLLMClient`](src/prscope/planning/runtime/transport/llm_client.py) uses LiteLLM `completion` (or OpenAI Responses API for `gpt-5-*`). Message order is **system + user/tool messages** as assembled by callers.
- **Critic path:** [`CriticAgent._llm_call`](src/prscope/planning/runtime/critic.py) builds `system` + a large `user` blob (requirements, plan, constraints, etc.).
- **Provider cache hints:** LiteLLM and each provider differ (e.g. Anthropic prompt caching, OpenAI prompt caching). Prscope does **not** attach `cache_control` or other provider-specific fields by default; enable only after measuring cost/latency on your target model. Prefer **deduplicating** identical large strings in prompts before adding provider extras.

## Long-phase SSE pings

When discovery or design review runs longer than **`planning.long_phase_ping_first_after_seconds`** (default **45**), the runtime emits additional **`thinking`** events every **`planning.long_phase_ping_interval_seconds`** (default **35**). Set **`long_phase_ping_first_after_seconds`** to **0** to disable.

**Limitation:** This only helps while the **async** phase is awaiting work. A blocking call inside `asyncio.to_thread` (e.g. a long synchronous LiteLLM completion) cannot be interrupted mid-flight; the UI may still look quiet until the call returns or times out.

## Planning harness invariants (critic + convergence)

Refinement convergence is gated by **feasibility**, not only “review complete”:

- **`plan_rubric`**: five axes (`specificity`, `testability`, `coherence`, `evidence_alignment`, `intent_alignment`) in [0,10]. Convergence requires `min(axis scores) >= planning.plan_rubric_floor` (default **7.0**, configurable in `prscope.yml`). Missing axes default to `MISSING_RUBRIC_SCORE` (0) and set **`rubric_incomplete`**, which **blocks** convergence.
- **`blocking_categories`**: closed vocabulary `testability` | `evidence` | `vagueness` | `scope`. Unknown tokens **fail closed** (convergence blocked). Empty `[]` means no category blockers. **`_apply_scope_discipline` does not strip issues** when any category blocker is present or the list is invalid.
- **`acceptance_criteria`**: falsifiable bullets; a structural check rejects vague wording. Satisfaction is computed from **plan markdown evidence** (shallow text overlap), not a critic self-report flag.
- **`stalled_refinement`** shortcut was **removed**; convergence must satisfy the full legacy stability checks **and** the harness gates above. A **postcondition** assert runs after each convergence decision (see `acceptance_contract.verify_convergence_postcondition`).
- **Debug**: `PlanningStages` logs `convergence_gate` at DEBUG with min rubric, floor, gate booleans, rationale, **`failed_gate`**, **`failure_delta`** (`rubric_floor` gap below floor when applicable, `acceptance_missing_count`), **`acceptance_missing`**, and **`blocking_categories`**.

### Failure distribution (tuning workflow)

Use DEBUG logs (e.g. `~/.prscope/server.log` when running the web server) to see **why** refinement did not converge, then aggregate across runs:

1. Run **N** comparable sessions (same rough requirements class, same models).
2. Extract lines containing `convergence_gate` (JSON payload after the message).
3. Histogram **`failed_gate`** counts: `rubric_incomplete`, `rubric_floor`, `blockers`, `acceptance_structural`, `acceptance_evidence`, `legacy`.
4. Cross-check **`failure_delta.rubric_floor`** (how far below floor) vs **`acceptance_missing_count`** to separate near-miss rubric from brittle acceptance evidence.

Interpretation hints:

- Many **`rubric_floor`** → critic rubric calibration / few-shots (avoid lowering the floor first).
- Many **`acceptance_evidence`** with stable **`acceptance_missing`** → matcher or criterion wording.
- Many **`blockers`** with unstable **`blocking_categories`** across runs → critic inconsistency.
- **`legacy`** with harness gates passing → convergence reasoner / non-harness signals.

Helper: `python scripts/summarize_convergence_logs.py ~/.prscope/server.log` (or stdin).

## Ablations (methodology)

When changing models or prompts, treat harness components as **hypotheses**: remove or relax one gate at a time, re-run `pytest` and the HTTP **benchmark** (`prscope-benchmark`), and record which pieces are load-bearing. Prefer documenting outcomes in this file or `docs/QUALITY_SCORE.md` rather than leaving behavior implicit in code.

## Future: context reset experiments

If long sessions show quality cliffs or premature convergence, consider **round-boundary context resets** with a handoff payload of current plan + decision graph + open issues (not full chat). This is **not** implemented by default; validate with benchmarks before enabling.

## Change-Safety Checklist

When editing harness code, keep these fixed:

- preserve persist-then-emit ordering (especially: persist completed tool groups before snapshot)
- keep snapshot-first behavior on SSE connect
- avoid adding client-side state reconstruction from turns/events
- keep command rejection payloads structured and deterministic
- stamp all SSE events with monotonic `session_version`
- use `tool_update` (not separate `tool_call`/`tool_result`) for tool event emission
- assign `sequence` from `event_seq` to new turns and tool groups for deterministic ordering

