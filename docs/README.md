# prscope Docs

Technical documentation for architecture, design, runtime behavior, and quality tracking.

## Index

### Architecture and Design

- [Core Beliefs](./CORE_BELIEFS.md) — golden principles for writing code in this repo
- [Design Philosophy](./DESIGN.md) — product decisions and their reasoning
- [Frontend Architecture](./FRONTEND.md) — React/Vite frontend conventions and component map
- [Quality Scores](./QUALITY_SCORE.md) — per-domain grades and known gaps
- [Design Docs](./design-docs/index.md) — catalog of architectural decision records

### Runtime and Operations

- [Agent Harness Guide](./agent-harness.md) — API, runtime, SSE, benchmark harness
- [Harness comparison checklist](./harness-comparison-checklist.md) — dimensions for comparing Prscope to other agent harnesses (**P0–P3 prioritized order**, verification footguns, **production blueprint** in section 9)
- [Prscope vs Claude Code gap analysis](./prscope-cc-gap-analysis.md) — first-pass comparison and **recommended add/change** list (living doc)
- [Production readiness](./PRODUCTION_READINESS.md) — bind safety, auth expectations, deployment modes, go/no-go checklist
- [Decision and Issue Graphs](./decision-and-issue-graphs.md) — graph artifacts, persistence, and UI/runtime roles
- [Planning State Machine](./planning-state-machine.md) — state contract, transitions, invariants
- [Memory, Context, and Manifesto](./memory-context-manifesto.md) — memory layers, context budgets
- [Skills and Session Recall](./skills-and-recall.md) — skills and episodic recall

### Limitations (what to expect)

- **Convergence and acceptance:** Refinement uses structured critic output, rubric floors, and **shallow** plan-text checks for acceptance criteria—not semantic proof. See [Agent Harness Guide](./agent-harness.md) (harness invariants, failure tuning).
- **Scanners:** Default `grep` backend has the widest test coverage; `repomap` / `repomix` need optional dependencies (`aider-chat`, `repomix`). Misconfigured backends fall back to `grep` with a log warning.
- **When sessions won’t converge:** Aggregate `convergence_gate` logs or run `scripts/summarize_convergence_logs.py` on a server log file (see [Agent Harness — Failure distribution](./agent-harness.md#failure-distribution-tuning-workflow)).

### Current Runtime Shape

- Signal extraction and evidence helpers live under `planning/runtime/discovery_support/*`.
- Semantic policy lives under `planning/runtime/reasoning/*`.
- Persisted plan artifacts now include `plan_json`, `decision_graph_json`, and `followups_json`.
- The rendered markdown plan is a projection over persisted planning state, not the only durable representation.

## Notes

- Architecture and layer rules live in the root `ARCHITECTURE.md`.
- Agent orientation lives in the root `AGENTS.md`.
- Keep docs in this folder focused on internals, operations, and decision records.
- Keep high-level user onboarding in the repository `README.md`.
