# AGENTS.md — Prscope

## What This Is

Local-first planning engine. Requirements (or upstream PR context) → adversarial Author↔Critic refinement → grounded PRD/RFC outputs. Python 3.9+, FastAPI backend, React/Vite frontend.

## Repo Map

```
AGENTS.md           ← you are here
ARCHITECTURE.md     ← domain layers, dependency rules, package map
CONTRIBUTING.md     ← PR workflow, benchmark policy, acceptance gates
docs/
├── README.md              ← docs index
├── CORE_BELIEFS.md        ← golden principles for writing code in this repo
├── DESIGN.md              ← design philosophy and product decisions
├── FRONTEND.md            ← frontend architecture, conventions, component map
├── QUALITY_SCORE.md       ← per-domain quality grades and known gaps
├── design-docs/index.md   ← design doc catalog
├── agent-harness.md       ← API, runtime, SSE, benchmark harness
├── planning-state-machine.md ← state contract, transitions, invariants
├── memory-context-manifesto.md ← memory, context budgets, manifesto
└── skills-and-recall.md   ← skills and session recall
plans/                     ← versioned execution plans (PRD, RFC, conversation logs)
.prscope/
├── manifesto.md           ← hard/soft constraints enforced by Critic
└── skills/*.md            ← stable team planning patterns
prscope/                   ← Python package (see ARCHITECTURE.md)
prscope/web/frontend/      ← React/Vite frontend (see docs/FRONTEND.md)
tests/                     ← pytest suite, co-located with package
benchmarks/                ← prompt suites, configs, historical results
```

## How to Work Here

1. Read `ARCHITECTURE.md` for layer rules before changing imports or adding modules.
2. Read `docs/CORE_BELIEFS.md` for coding principles.
3. State machine changes require reading `docs/planning-state-machine.md` first.
4. Runtime/orchestration changes require reading `docs/agent-harness.md` first.
5. Frontend changes require reading `docs/FRONTEND.md` first.

## Key Conventions

- **State writes**: only through `PlanningCore.transition_and_snapshot()`. Never mutate protected session fields directly.
- **Persist then emit**: DB commit before SSE event. No split-brain.
- **Manifesto**: `.prscope/manifesto.md` defines hard/soft constraints for the Critic. Keep constraints concise and actionable.
- **Memory layers** (injection order): Manifesto → Skills → Recall → Memory blocks. Historical precedent never overrides policy.
- **Import boundaries**: see `ARCHITECTURE.md` — layers import downward only.

## Running

```bash
make dev            # install with dev deps
make check          # lint + format check + tests
make ci             # full CI parity (includes frontend)
make web-backend    # uvicorn on :8420
make web-frontend   # vite on :5173
```

## Testing

```bash
pytest -v                  # all tests
pytest tests/test_store.py # single file
make ci                    # full CI (python + frontend)
```

Performance-sensitive changes require a benchmark run — see `CONTRIBUTING.md`.

## Before You Change Anything

- Read the relevant doc from the map above.
- Run `make check` before committing.
- Keep PRs focused on one logical change.
- Do not hard-code secrets, tokens, or credentials anywhere (HARD_CONSTRAINT_001).
- Do not propose irreversible data ops without rollback steps (HARD_CONSTRAINT_002).

## Config

- `prscope.yml` — main config (repos, planning settings, scanner backend)
- `prscope.features.yml` — feature definitions for PR scoring
- `.env` — API keys (never committed)
