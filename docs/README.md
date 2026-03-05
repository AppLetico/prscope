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
- [Planning State Machine](./planning-state-machine.md) — state contract, transitions, invariants
- [Memory, Context, and Manifesto](./memory-context-manifesto.md) — memory layers, context budgets
- [Skills and Session Recall](./skills-and-recall.md) — skills and episodic recall

## Notes

- Architecture and layer rules live in the root `ARCHITECTURE.md`.
- Agent orientation lives in the root `AGENTS.md`.
- Keep docs in this folder focused on internals, operations, and decision records.
- Keep high-level user onboarding in the repository `README.md`.
