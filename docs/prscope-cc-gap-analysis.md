# Prscope vs Claude Code (CC) — comparison and gap list

**Purpose:** First-pass comparison against CC-style harness practices (see [harness-comparison-checklist.md](./harness-comparison-checklist.md), P0–P3). **CC** here means patterns described in public writeups and mirrored source snapshots—not a live product endorsement.

**Prscope’s scope:** Planning engine with Author/Critic, repo sandbox tools, SQLite sessions, web + API. It is **not** a full terminal coding agent (no arbitrary shell/edit loop in core).

---

## Summary table

| Area | CC pattern (reference) | Prscope today | Gap / candidate work | Priority |
| --- | --- | --- | --- | --- |
| Code search & discovery | Ripgrep, glob, ignore-aware, path-only mode | Implemented (`grep_code`, `glob_files`, config, discovery caps) | Keep aligned with checklist; tune defaults per stage | Done / maintain |
| Context budgeting | Large-window aware, reserved overhead, truncation | `TokenBudgetManager`, model windows, discovery conversation trim | Extend if new stages need budgets | Done / maintain |
| Instruction reload | Project instructions **re-read every turn** | Skills + manifesto **cached per session** in `PlanningState` after first load | `planning.instruction_context_refresh`: **`off`** (default), **`on_change`** (manifesto + `skills/*.md` mtime/size fingerprint), **`each_turn`** (reload every `_state` access). | P1 / done |
| Parallel read-only tools | Run Glob/Grep reads concurrently | `StageRunner.execute_tool_calls` runs **`list_files`/`read_file`/`grep_code`/`glob_files` in parallel** when a turn emits **two or more** of only those tools (`asyncio.gather` + `ToolExecutor._access_lock` on path tracking) | — | Done / maintain |
| Hooks / extension API | 20+ lifecycle events; inject context on `UserPromptSubmit` | No hook system; events are fixed SSE types | Add **optional** hook points (e.g. subprocess or HTTP) only if product needs integrations—high maintenance | P2 |
| MCP | External tools via MCP, deferred load | **Not present** | MCP server bridge only if Prscope becomes a general agent shell | P3 / out of scope |
| Permission rule engine | Multi-stage allow/deny + patterns + classifier | Repo **sandbox** + `ToolSafetyError`; **`planning.tools.path_allowlist`** shipped (repo-relative prefixes per `read_file` / `list_files` / `grep_code` / `glob_files`). | CC-style multi-stage hooks / classifier pipeline—**not** a Prscope goal unless product asks | P2 / done (declarative prefixes) |
| Compaction strategies | Micro → snip → auto → collapse (ordered, cheap first) | Heuristic `CritiqueCompressor`; optional LLM summarize; adversarial rounds call **`_prepare_adversarial_compaction_context`** (uses `_should_compact_context` + `summarize_critiques_for_compaction` with `event_callback`) and thread **`state.working_summary`** into critic `prior_critique` and author **`revise_plan`**. **`context_compaction`** SSE when compaction runs with a callback. | Done / maintain |
| Post-edit verification | Typecheck/lint gates (often user-driven in CLAUDE.md) | **No code write tool** in planning harness | N/A for plan-only; if execution tools appear later, add verify hooks | N/A now |
| Async streaming loop | Single async generator; cancel-friendly | SSE + async planning; LLM calls via `asyncio.to_thread` | **Speculative tool start** (run tools before assistant message finishes) needs streaming parser—large change | P3 |
| Retry / backoff | Retry-After, 529 fallback, long-run heartbeat | LiteLLM + configurable **`author_completion_fallback_model`** / **`discovery_completion_fallback_model`**; **`planning.llm_retry`** (exponential backoff, Retry-After, jitter) on author + discovery completions | Stream stall detection, long-run heartbeat—**incremental** | P2 / partial |
| Subagents / parallel workers | Isolated contexts, fork/worktree | **None** | Multi-plan or parallel evidence gathering is **product decision**; not required for core planning | P3 |
| Semantic / AST | LSP or index for rename safety | **Grep + read** only | Document in manifesto/skills; optional **scanner** upgrade (existing repomap path) for memory—not for tool rename | P2 docs / P3 product |
| Prompt cache boundary | Static prefix vs per-turn tail for API cache | Not explicitly structured for provider **prompt cache** | Audit largest stable prefixes (manifesto excerpt, tool schemas) for LiteLLM/OpenAI cache hints—**optimization** | P2 |

---

## Already strong relative to “chat wrapper”

- **Grounded planning:** Author/Critic, manifesto constraints, file reference validation paths in pipeline.
- **State + persistence:** `PlanningCore`, SQLite sessions, replay-oriented design ([planning-state-machine.md](./planning-state-machine.md)).
- **Evidence tools:** Sandbox, ripgrep-backed search, glob, caps and artifacts for large tool payloads.
- **Telemetry:** Token usage, warnings near context limits, SSE event stream.

---

## Recommended next implementations (ordered)

1. ~~**P1 — Emit `context_compaction` from runtime**~~ **Done:** discovery trim + adversarial `summarize_critiques_for_compaction(..., event_callback=...)`.
2. ~~**P1 — Optional skills/manifesto refresh**~~ **Done:** `planning.instruction_context_refresh` (`off` \| `on_change` \| `each_turn`).
3. ~~**P1 — Parallelize read-only tool batch**~~ **Done:** `StageRunner` parallel batch for `list_files`/`read_file`/`grep_code`/`glob_files` only; `ask_clarification` / `get_memory_block` stay sequential.
4. ~~**P1 — Wire adversarial refinement**~~ **Done:** `_prepare_adversarial_compaction_context` at start of adversarial round; `PlanningStages._compose_prior_critique`; `prior_rounds_compact` on `revise_plan`.
5. ~~**P2 — Retry policy**~~ **Done:** `planning.runtime.llm_retry` (transient detection, Retry-After + message parse, backoff+jitter); wired into `AuthorLLMClient` and discovery `safe_completion_call`; **`planning.llm_retry`** YAML + **`author_completion_fallback_model`** / **`discovery_completion_fallback_model`** in config.
6. ~~**P2 — `prscope.yml` tool permissions**~~ **Done:** **`planning.tools.path_allowlist`** — optional repo-relative prefix lists per tool; empty list = tool denied; omitted tool = no extra restriction beyond sandbox.

**Status:** The ordered backlog **1–6** above is **fully implemented** in tree. Remaining rows in the **summary table** are **ongoing posture** (maintain / incremental / out of scope), not a second backlog—see each row’s “Gap” column.

---

## Explicit non-goals (unless roadmap changes)

- Full **MCP** ecosystem in core.
- **LSP** tool parity.
- **Unbounded subagents** with separate worktrees.
- Replacing planning with a **single-turn chat** product.

---

## How to extend this doc

After each release, add a row or bump priorities. Link new design decisions to [design-docs/index.md](./design-docs/index.md) when you implement a gap.
