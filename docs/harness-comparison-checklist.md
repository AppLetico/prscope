# Harness comparison checklist

Use this when comparing **Prscope** to other agent harnesses (e.g. Claude Code, Cursor-style agents). It is a **working list**: for each row, note how Prscope behaves today, how the reference behaves, and gaps or non-goals.

**Related docs:** [agent-harness.md](./agent-harness.md), [memory-context-manifesto.md](./memory-context-manifesto.md), [planning-state-machine.md](./planning-state-machine.md).

---

## Prioritized comparison order (Prscope-first)

Prscope optimizes for **grounded plans**, **adversarial refinement**, and **bounded tools**—not full parity with a terminal coding OS. Use this order so comparison work hits **highest leverage** first; skip lower tiers until a product decision pulls them in.

| Tier | Sections (in suggested order) | Why this priority |
| --- | --- | --- |
| **P0** | **1** → **2** (main table) → **2d** → **2c** → **8** | **Evidence and trust**: search/read quality, **context budgets and compaction**, **instruction layers** (manifesto/skills parity), **silent limits and verification** (wrong answers vs wrong bytes). |
| **P1** | **5** → **6** → **9a**, **9b**, **9f**, **9d** (as needed) | **Session UX and resilience**: persistence/resume, streaming/cancel/cost, **core loop shape**, **tool ordering**, **retries**, **prompt/cache boundaries** when tuning API or runtime. |
| **P2** | **2b** → **3** (+ **3b**) → **4** → **9c**, **9e**, **9g** | **Advanced**: index-first memory (only if evolving memory architecture), **MCP/hooks**, **permission rule engines**, deeper **compaction** and **extensibility** blueprints. |
| **P3** | **7** → **9h** + [Non-goals](#non-goals-typical) | **Meta**: benchmarks; **re-scope** before scope creep. |

**First-pass minimum (if time-boxed):** complete **P0** only, then stop and write findings. **P1** when improving orchestration, SSE, or reliability. **P2–P3** for spikes or enterprise-style features.

---

## 1. Codebase access and search

| Topic | What to compare |
| --- | --- |
| Content search | Backend (ripgrep vs in-process), regex dialect, `.gitignore` / ignore files, timeouts, output modes (lines vs paths-only), caps |
| File discovery | Glob/recursive patterns vs single-directory listing; performance on large trees |
| Read scope | Max lines, binary handling, path sandbox rules |
| Structural context | Tree scanners, repomap/repomix vs LSP/symbols |

---

## 2. Context and memory

| Topic | What to compare |
| --- | --- |
| Prompt budgeting | Heuristic vs tokenizer; reserved space for tools/system; per-model windows |
| Transcript growth | Compaction, summarization, what is dropped vs preserved |
| Stable instructions | Layered project files vs manifesto + skills + memory blocks; precedence when they conflict |
| Long-session memory | Session recall, embeddings, session search vs Prscope recall |
| On-demand pulls | Tools or APIs to fetch more context without front-loading everything |

### 2b. Structured / long-lived memory (index-first harnesses)

Some agents treat **memory as an index, not a dump**: a small always-on layer points to detail elsewhere; heavy or volatile material is fetched or searched only when needed. Use this block when comparing to systems that emphasize **constrained, self-healing** memory (e.g. pointer files, topic shards, grep-only transcripts).

| Topic | What to compare |
| --- | --- |
| Index vs storage | Always-loaded layer is **pointers/summaries** (short lines) vs full knowledge; where “real” content lives (topic files, repo, tools) |
| Bandwidth layers | What is **always** in context vs **on-demand** vs **never fully read** (e.g. transcripts only **grep’d**, not loaded wholesale) |
| Write discipline | Order of operations (e.g. **write detail → then update index**); rules **against** dumping large content into the index (entropy / context pollution) |
| Background rewriting | Jobs that **merge, dedupe, resolve contradictions**, tighten vague → concrete, **prune** aggressively; memory as **edited** state, not only append-only logs |
| Staleness & truth | Policy when **memory ≠ reality** (e.g. code/repo wins); avoiding persistence of **code-derived facts** as if they were durable; **truncating** or invalidating the index when wrong |
| Isolation of consolidation | Consolidation runs in **separate process/agent** with **narrow tool** surface so rewrite passes don’t corrupt main session context |
| Skeptical retrieval | Memory is a **hint**; model or runtime **verifies** (e.g. read/grep) before treating as fact |
| What not to store | Explicit **negative list**: e.g. debug logs, full code structure dumps, PR history—if **re-derivable** from repo/tools, don’t persist as memory |

### 2c. Instruction files (every query vs once per session)

Some harnesses **reload** project/user instructions on **every** user turn (not only at session start), so edits to those files take effect immediately. Compare:

| Topic | What to compare |
| --- | --- |
| Reload semantics | Every turn vs session start vs manual refresh |
| Hierarchy & precedence | e.g. global → project → modular rules → private/local; merge order when files conflict |
| Size budget | Max characters per layer or total (e.g. tens of k); empty vs overstuffed files |
| Content guidance | Architecture, conventions, tests, “never do X”—what the harness expects vs freeform |

### 2d. Transcript compaction (strategy catalog)

Reference harnesses often implement **multiple** compaction paths (time-based tool trim, span summary, session file extract, full history summary, oldest-message drop). Compare **which** strategies exist and **when** each runs (user-invoked vs automatic).

| Topic | What to compare |
| --- | --- |
| User-initiated compact | Explicit “compact” command vs only auto under pressure |
| Tool-result trimming | Age-based or size-based clearing of old tool outputs in the transcript |
| Span / full history summary | What gets summarized and what identifiers survive |
| Session memory file | Extracting durable notes to disk vs keeping everything in chat |
| Prefix / message-group truncation | Dropping oldest groups vs summarizing |
| Large tool payloads | Spill to disk with **small preview** in context vs inline only |
| Long context modes | Optional larger context window (e.g. model suffix or flag) for big refactors |

---

## 3. Tools and extensibility

| Topic | What to compare |
| --- | --- |
| Fixed tool set | Names, schemas, sandbox boundaries |
| MCP / plugins | Optional servers, auth, resource limits; **deferred loading** so unused servers don’t tax every request |
| MCP tool discovery | Lazy registration vs upfront enumeration; “search for tool by need” patterns |
| Bash / execution | Allowed commands, cwd, timeouts, streaming, dangerous-op policies |
| Edit model | Patch vs replace, conflicts, rollback |
| Subagents / delegation | Isolated workers, context boundaries, parent/child correlation |
| Parallelism vs ordering | **Concurrent** read-only ops (read/search/glob) vs **serial** mutating ops (edit/write/bash) to avoid races |
| Subagent execution models | e.g. fork (shared prompt prefix / cache-friendly), separate pane/mailbox, isolated **git worktree** per agent |
| Prompt-cache / shared prefix | Whether parallel workers reuse identical system+instruction prefix (provider cache behavior) |

### 3b. Hooks and lifecycle extension points

Some harnesses expose **many** lifecycle events (before/after tool, prompt submit, session start/end) and hook types (command, prompt-inject, sub-agent, HTTP, in-process function). Compare:

| Topic | What to compare |
| --- | --- |
| Event surface | Which stages are hookable; can hooks inject **additional context** into the next model call |
| Hook types | Shell vs LLM vs agent vs webhook vs embedded code |
| Use cases | Lint-before-write, test-after-edit, auto-attach diffs or test output to every prompt |

---

## 4. Permissions and safety

| Topic | What to compare |
| --- | --- |
| Filesystem | Repo root enforcement, sensitive path rules |
| Network / secrets | Exfiltration boundaries, secret handling in prompts and logs |
| User approval | Per-action prompts vs auto-run modes |
| Policy cascade | Order of resolution: e.g. org/policy → CLI flag → local → project → user defaults |
| Allow / deny patterns | Declarative rules (globs, tool names) vs ad hoc clicks each time |
| Permission modes | e.g. unrestricted vs edits-only vs classifier-mediated “auto” allow/deny |
| Audit | Immutable logs, approval history (if applicable) |

---

## 5. Session lifecycle and reliability

| Topic | What to compare |
| --- | --- |
| State machine | Phases, transitions, idempotency (see planning-state-machine) |
| Persistence | SQLite vs other stores; what survives restart |
| Resume / replay | Crash recovery, deterministic replay for debugging |
| Session artifacts | Transcript format on disk (e.g. append-only JSONL); **continue** vs **resume id** vs **fork from past session** |
| Session memory across compactions | Structured carryover: task spec, file lists, errors, learnings vs raw chat only |
| Retries | Model fallback, rate limits, backoff |
| Retry & transport detail | Backoff+jitter, OAuth refresh on auth errors, stream stall watchdog, fallback from streaming to non-streaming |

---

## 6. Cost, telemetry, and UX

| Topic | What to compare |
| --- | --- |
| Token and cost | Per-call and session totals; caps and warnings |
| UI surfacing | Context usage, cost, phase, errors (e.g. SSE events) |
| Plan vs act | Approval before execution; how “planning only” is enforced |
| Clarification | Batching, timeouts, UX for blocking questions |
| Streaming & cancellation | Whether users can **abort** in-flight generation cheaply; how much prior context is kept after cancel |

---

## 7. Testing and quality

| Topic | What to compare |
| --- | --- |
| Harness tests | Replay fixtures, golden traces, eval suites |
| Benchmarks | Prscope `benchmarks/` policy; what is measured and how often |

---

## 8. Verification, silent limits, and system defaults (reference patterns)

Third-party writeups and leaked snapshots go stale fast. Treat the rows below as **questions to answer from source + your own traces**, not as facts about any vendor.

### 8a. What counts as “success” after an edit

| Topic | What to compare |
| --- | --- |
| Write success vs quality | Whether the runtime treats “bytes written” as success vs requiring typecheck/tests/lint |
| Built-in post-edit verification | Whether the harness runs checks automatically, or only under certain builds/env flags |
| User / project mitigation | Teams often encode **mandatory verify** steps in instruction files (e.g. `tsc`, `eslint`) because the base loop may not |

### 8b. Aggressive compaction and “context amputation”

| Topic | What to compare |
| --- | --- |
| Auto-compact threshold | Approximate token pressure where automatic compaction runs (varies by model and version) |
| What survives | Caps on retained file reads, whether prior tool I/O and reasoning are summarized vs dropped |
| Interaction with codebase noise | Dead imports, unused exports, and large irrelevant text **burn context** and can push compaction earlier—compare to **clean-first** workflows |
| Mid-task safety | Whether long refactors should be **phased** (few files per phase) so compaction does not fire mid-task |

### 8c. System defaults vs user intent (“brevity mandate”)

| Topic | What to compare |
| --- | --- |
| Default style directives | Instructions such as “smallest change,” “simplest approach,” “avoid extra refactor” appear in system prompts |
| Conflict with user ask | User may request architectural fixes while defaults bias toward minimal diffs—**instruction files** or explicit reframing may be needed |
| Prscope angle | Manifesto/skills/critic loops can **pull toward** rigor; compare how each product resolves this tension |

### 8d. Silent truncation: reads and tool results

| Topic | What to compare |
| --- | --- |
| Per-read line/token caps | Hard limits on file reads; whether the **model is warned** when content is truncated |
| Chunked reads | Offset/limit or paging so large files are read in multiple calls |
| Large tool outputs | When results exceed a size, **persist to disk** and inject only a **preview** into context; whether the model is told preview is partial |
| Honest reporting | Risk: model summarizes “3 matches” when more existed outside the preview—compare **narrower queries** and **re-read before edit** mitigations |

### 8e. Parallel workers vs one long thread

| Topic | What to compare |
| --- | --- |
| Isolated budgets | Subagents with separate compaction/token budgets vs one shared transcript |
| When to parallelize | Independent file batches vs tasks that need shared reasoning |
| Prscope | Planning is often **session-scoped**; compare to harnesses built for **many concurrent code workers** |

### 8f. Grep and text search are not an AST

| Topic | What to compare |
| --- | --- |
| Rename / signature change | Need **multiple** search strategies: direct refs, types, strings, dynamic `import()`, re-exports, barrels, tests/mocks |
| False confidence | Single `grep` missing callers does not mean the rename is safe |

### 8g. Optional: project-level mitigations (any harness)

Teams sometimes mirror “employee-grade” discipline in **project instructions** (not Prscope defaults): phased file limits, forced verify commands, re-read after compaction, chunked reads for large files, explicit truncation suspicion, multi-pattern search for refactors. Map these to **manifesto**, **skills**, or **review gates** in Prscope rather than copying vendor-specific filenames.

---

## 9. Production harness blueprint (architectural patterns)

Use this section when designing or comparing **orchestration shape**, not only feature checklists. Counts (lines, stages, strategies) differ by product version—**verify in source** before treating any reference as exact.

### 9a. Core loop: streaming and control

| Topic | What to compare |
| --- | --- |
| Event model | Single **async** pipeline vs request/response batches; whether **model tokens, tool calls, and errors** stream as first-class events |
| UI coupling | Character/token streaming vs waiting for full completion |
| Cancellation | Cheap **abort** of in-flight generation; what state survives (see also section 6) |
| Nesting | Subagents or child runs embedded in the same event architecture (see **8e**) |

### 9b. Tool execution: latency and ordering

| Topic | What to compare |
| --- | --- |
| Speculative / early execution | Whether **read-only tools** can start when **arguments are known** without waiting for the full assistant message to finish (latency win) |
| Parallel vs serial | **Concurrent** reads/searches vs **serial** mutating steps (edits, shell) to avoid races—speed vs safety |
| Tool descriptions | **Static** tool schemas vs descriptions **patched from live environment** (paths, cwd, capabilities) |

### 9c. Compaction: ordered strategies (cheapest first)

Reference harnesses often stack **multiple** compaction passes and run **lower-cost** fixes before expensive summarization. Compare names and triggers; see **section 2d** for the full catalog. Typical pattern dimensions:

| Topic | What to compare |
| --- | --- |
| Per-turn cheap wins | e.g. deduping or caching **unchanged** tool output before touching history |
| Light trim | Trimming **old** turns while preserving a **recency window** |
| Heavier summarize | When token pressure crosses thresholds; what is summarized vs dropped |
| Deepest compression | Long-session or staged collapse when lighter steps are insufficient |

### 9d. Prompt structure and cache boundaries

| Topic | What to compare |
| --- | --- |
| Static vs dynamic split | Prefix stable across many calls (provider **prompt cache**) vs per-session or per-turn tail |
| Instruction hierarchy | Multiple layers (e.g. org, user, project, local) and merge order |
| Boundary discipline | What must sit **before** vs **after** the cacheable/static boundary so caching stays effective |

### 9e. Permission pipeline (rule engine, not a single toggle)

| Topic | What to compare |
| --- | --- |
| Stages | Ordered checks: validation → deny → allow → tool-specific policy → hooks → classifier → user prompt |
| Pattern language | Glob-like or structured rules per tool family (`Bash(git *)`, `Edit(src/**)`, …) |
| Roles | Enterprise vs project vs user override surfaces |

### 9f. Error recovery as first-class infrastructure

| Topic | What to compare |
| --- | --- |
| Rate limits | Respect **Retry-After**, tiered backoff (short vs long cooldowns) |
| Model / transport fallback | e.g. repeated overload errors → alternate model or non-streaming path |
| Context errors | **Inline** budget adjustment and retry vs hard crash |
| Long-running / CI | Extended retry policy, **heartbeat** or keepalive for idle streams |
| Codebase size | Dedicated retry module vs scattered try/except |

### 9g. Extensibility without recompiling core

| Topic | What to compare |
| --- | --- |
| Drop-in instructions | Markdown or config that **injects** prompts and narrows tool surface (“skills”) |
| Drop-in automation | Hooks (shell or scripts) on lifecycle events |
| External tools | MCP or similar; **transports** and deferred loading |
| Bundles | Plugins that package skills + hooks + MCP metadata in one installable unit |

### 9h. Mapping to Prscope (intentional scope)

Prscope is **planning-first** (Author/Critic, state machine, grounded tools)—not a 1:1 clone of a terminal coding OS. Use section 9 to ask: which patterns belong in **transport** (SSE, streaming), **runtime** (retries, budgets), **policy** (manifesto, permissions), vs **out of scope** for this product.

---

## How to use this doc

1. Start from **[Prioritized comparison order](#prioritized-comparison-order-prscope-first)** (P0 → P1 → …) unless you have a narrow question (e.g. only permissions → jump to section 4).
2. Pick a **reference harness** and a **Prscope release or branch**.
3. For each section, add a short **Prscope** bullet and **reference** bullet (or “N/A / out of scope”).
4. Use **section 2b** for **index-first / self-healing memory**; **2c** for **instruction-file reload and hierarchy**; **2d** for **compaction strategy catalogs** and large-tool spill behavior.
5. Use **section 3b** for **hooks and lifecycle extension points** (inject context every turn, lint gates, webhooks).
6. Use **section 8** for **verification vs “write succeeded”**, **compaction side effects**, **system default frugality**, **silent read/tool truncation**, and **grep-vs-rename** safety—especially when debugging “it said done but the repo is broken.”
7. Use **section 9** for **production harness architecture**: async streaming loop, early tool execution, **compaction ordering**, **prompt-cache boundaries**, **multi-stage permissions**, **deep retry/heartbeat** design, and **drop-in extensibility**—when building or evaluating orchestration, not just features.
8. File gaps as issues or design docs when the difference affects product goals—not every reference-harness feature belongs in Prscope (see **Non-goals** below).

---

## Non-goals (typical)

Full parity with a general coding agent (infinite chat, arbitrary shell, LSP in-process) is usually **not** the goal unless the product direction changes. Prscope optimizes for **planning**, **adversarial refinement**, and **grounded exports**; use this list to borrow **patterns**, not to duplicate every surface area.
