# Architecture

This document defines prscope's domain decomposition, layer ordering, and dependency rules.

Read this before adding modules, changing imports, or restructuring packages.

## Domain Map

Prscope has four domains. Each domain owns a slice of functionality and has clear boundaries.

| Domain | Responsibility | Key Modules |
|---|---|---|
| **Foundation** | Config, pricing, models, profile | `config.py`, `pricing.py`, `model_catalog.py`, `profile.py`, `semantic.py` |
| **Storage** | Persistence, session state, PR data | `store.py` |
| **Intelligence** | Planning runtime, author/critic, memory, scoring | `planning/`, `memory.py`, `scoring.py`, `llm.py`, `planner.py` |
| **Interface** | CLI, web API, frontend, benchmark | `cli.py`, `web/`, `benchmark.py` |

## Layer Ordering

Layers are ordered top-to-bottom. A module may import from its own layer or any layer below it. **Never import upward.**

```
┌─────────────────────────────────────────────────────┐
│  Interface                                          │
│  cli.py, web/api.py, web/server.py, benchmark.py   │
├─────────────────────────────────────────────────────┤
│  Intelligence                                       │
│  planning/core.py, planning/executor.py,            │
│  planning/runtime/*, planning/render.py,            │
│  planning/scanners/*, memory.py, scoring.py,        │
│  llm.py, planner.py, github.py                     │
├─────────────────────────────────────────────────────┤
│  Storage                                            │
│  store.py                                           │
├─────────────────────────────────────────────────────┤
│  Foundation                                         │
│  config.py, pricing.py, model_catalog.py,           │
│  profile.py, semantic.py                            │
└─────────────────────────────────────────────────────┘
```

### Dependency Direction Rules

1. **Foundation** imports nothing from prscope (stdlib + third-party only).
2. **Storage** imports from Foundation only (`config`).
3. **Intelligence** imports from Storage and Foundation. Never from Interface.
4. **Interface** imports from any layer below it.

### Within Intelligence: Planning Sublayers

The `planning/` package has its own internal layering:

```
planning/runtime/orchestration.py   ← top: wires everything together
    ↓
planning/runtime/{discovery,author,critic,round_controller,state}.py
    ↓
planning/runtime/authoring/{models,discovery,validation,repair}.py
    ↓
planning/runtime/transport/{llm_client}.py
    ↓
planning/runtime/{tools,budget,telemetry}.py  ← leaf utilities
    ↓
planning/runtime/{context/*,events/*,pipeline/*,review/*}
    ↓
planning/{core,executor,render}.py  ← planning kernel
    ↓
planning/scanners/*                 ← codebase scanning backends
```

- `orchestration.py` is the only module that imports broadly across runtime.
- `tools.py`, `budget.py`, `telemetry.py`, and `transport/llm_client.py` are leaf modules with no imports from orchestration.
- `core.py` and `executor.py` import from Storage and Foundation only — never from runtime.
- Runtime agents (`author.py`, `critic.py`, `discovery.py`) import from Foundation + their own leaf utilities. They do **not** import from `core.py` or `executor.py` directly.

### Web Sublayers

```
web/server.py      ← static file serving, app factory wrapper
    ↓
web/api.py         ← FastAPI routes, command handling, SSE
    ↓
web/events.py      ← pure async event emitter (no prscope imports)
```

Frontend (`web/frontend/`) is a separate Vite/React app that communicates with the backend exclusively via HTTP + SSE. No shared Python imports.

## Package Map

```
prscope/
├── __init__.py
├── config.py                  [Foundation] config loading, repo profiles
├── pricing.py                 [Foundation] model pricing tables
├── model_catalog.py           [Foundation] model registry and availability
├── profile.py                 [Foundation] codebase profiling
├── semantic.py                [Foundation] code search utilities
├── store.py                   [Storage]    SQLite persistence, session/turn/PR schema
├── memory.py                  [Intelligence] memory blocks, manifesto, skills
├── llm.py                     [Intelligence] LLM call abstraction
├── scoring.py                 [Intelligence] PR scoring rules
├── github.py                  [Intelligence] GitHub API client, PR sync
├── planner.py                 [Intelligence] high-level planner facade
├── planning/
│   ├── core.py                [Intelligence] state machine, transitions, convergence
│   ├── executor.py            [Intelligence] command reserve/execute/finalize
│   ├── render.py              [Intelligence] plan export rendering
│   ├── scanners/
│   │   ├── base.py            [Intelligence] scanner interface
│   │   ├── grep.py            [Intelligence] grep-backed scanner
│   │   ├── repomap.py         [Intelligence] repomap scanner
│   │   └── repomix.py         [Intelligence] repomix scanner
│   └── runtime/
│       ├── orchestration.py   [Intelligence] top-level runtime wiring
│       ├── discovery.py       [Intelligence] generalized intent/evidence discovery engine
│       ├── author.py          [Intelligence] plan authoring agent
│       ├── authoring/         [Intelligence] author subsystem package
│       │   ├── models.py      [Intelligence] author dataclasses + markdown helpers
│       │   ├── discovery.py   [Intelligence] deterministic author-side repo discovery
│       │   ├── validation.py  [Intelligence] author draft validation gates
│       │   ├── repair.py      [Intelligence] author plan repair/revision helpers
│       │   └── pipeline.py    [Intelligence] planner pipeline coordinator
│       ├── critic.py          [Intelligence] adversarial critic agent
│       ├── tools.py           [Intelligence] tool definitions, file ops
│       ├── budget.py          [Intelligence] token budget management
│       ├── telemetry.py       [Intelligence] completion telemetry
│       ├── transport/         [Intelligence] LLM transport adapters
│       │   └── llm_client.py  [Intelligence] responses/chat compatibility layer
│       ├── context/           [Intelligence] context assembly, budgeting, clarification, compression
│       ├── events/            [Intelligence] runtime analytics/tool/token event state
│       ├── pipeline/          [Intelligence] adversarial stage loop and stage orchestration
│       ├── review/            [Intelligence] issue graph, similarity, causality, manifesto checks
│       ├── round_controller.py[Intelligence] round progression logic
│       └── state.py           [Intelligence] runtime planning state model
├── cli.py                     [Interface] Click CLI
├── benchmark.py               [Interface] HTTP-based benchmark harness
└── web/
    ├── server.py              [Interface] uvicorn + static serving
    ├── api.py                 [Interface] FastAPI routes
    └── events.py              [Interface] async SSE emitter
```

## Structural Invariants

These are enforced by `tests/test_architecture.py`:

1. **Foundation modules** have zero intra-package imports (only stdlib + third-party).
2. **Storage** imports only from Foundation.
3. **Intelligence** never imports from Interface.
4. **Interface** never imports from other Interface modules except `web/` internal layering (`server→api→events`).
5. **Runtime leaf modules** (`tools`, `budget`, `telemetry`, `transport/llm_client`) do not import from `planning.core` or `planning.executor`.

Violations are caught at CI time. See `tests/test_architecture.py`.

## Discovery Engine Notes

`planning/runtime/discovery.py` is intentionally data-driven and feature-agnostic:

- feature detection is derived from user intent + code evidence, not hardcoded feature names
- framework detection is registry-based and scored from observed route signals
- directory/file prioritization is heuristic-scored (`_select_scan_directories`, `_route_file_score`)
- evidence is confidence-ranked and deduplicated before planner decisions
- bootstrap insights are session-scoped and keyed by generic fields (`existing_feature`, `feature_label`)

Guardrail: avoid introducing feature-specific conditionals in engine logic. Add new feature/framework support by extending registries and patterns, not by adding special-case branches.

## Adding a New Module

1. Decide which layer it belongs to (Foundation / Storage / Intelligence / Interface).
2. Only import from that layer or below.
3. Add the module to the package map in this document.
4. If it's a new planning runtime module, keep it as a leaf unless it genuinely needs to import broadly.
5. Run `make check` to verify no import boundary violations.
