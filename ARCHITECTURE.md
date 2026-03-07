# Architecture

This document defines prscope's domain decomposition, layer ordering, and dependency rules.

Read this before adding modules, changing imports, or restructuring packages.

## Domain Map

Prscope has four domains. Each domain owns a slice of functionality and has clear boundaries.

| Domain | Responsibility | Key Modules |
|---|---|---|
| **Foundation** | Config, pricing, models, profile | `config.py`, `pricing.py`, `model_catalog.py`, `profile.py`, `semantic.py` |
| **Storage** | Persistence, session state, PR data | `store/` |
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
│  store/*                                            │
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

The `planning/` package has its own internal layering and several conceptual runtime subdomains:

```
planning/runtime/orchestration.py   ← top: wires everything together
    ↓
planning/runtime/{discovery,author,critic,round_controller,state}.py
    ↓
planning/runtime/orchestration_support/{event_router,state_snapshots,initial_draft,session_starts,chat_flow,round_entry}.py
    ↓
planning/runtime/discovery_support/{models,signals,existing_feature,bootstrap,llm}.py
    ↓
planning/runtime/reasoning/{base,models,discovery_reasoner,refinement_reasoner,review_reasoner,convergence_reasoner}.py
    ↓
planning/runtime/authoring/{models,discovery,validation,repair,pipeline}.py
    ↓
planning/runtime/transport/{llm_client}.py
    ↓
planning/runtime/{tools,telemetry}.py  ← leaf utilities
    ↓
planning/runtime/{context/*,events/*,pipeline/*,review/*}  ← context/ includes budget
    ↓
planning/{core,executor,render}.py  ← planning kernel
    ↓
planning/scanners/*                 ← codebase scanning backends
```

- `orchestration.py` remains the public coordinator/facade; orchestration internals live under `orchestration_support/*`.
- `discovery.py`, `author.py`, and `critic.py` are runtime entrypoints coordinated by orchestration, not independent policy owners.
- `discovery_support/*` plus `planning/scanners/*` form the evidence layer: signal extraction, heuristics, and repository facts.
- `reasoning/*` is the policy layer: it interprets structured evidence and must not depend on review state, `issue_graph`, or other persisted critique artifacts.
- `authoring/*` plus `pipeline/*` form the drafting/refinement engine.
- `review/*` is the critique subsystem; it may consume reasoning outputs when needed, but it does not own orchestration or decision persistence.
- `followups/*` plus `decision_catalog.py` hold persisted decision/follow-up artifacts.
- `tools.py`, `telemetry.py`, `transport/llm_client.py`, `context/{budget,clarification,compression}.py`, and `events/analytics_emitter.py` are runtime leaf utilities. They may import Foundation plus explicitly approved leaf helpers, but not orchestration, reasoning, pipeline, review, followups, `planning.core`, or `planning.executor`.
- `context/` (including budget) and `events/` remain infrastructure-style helpers for context assembly, clarification, compression, and event bookkeeping; they do not own control flow.
- `core.py` and `executor.py` import from Storage and Foundation only — never from runtime.
- Runtime agents (`author.py`, `critic.py`, `discovery.py`) import from Foundation + their own leaf utilities. They do **not** import from `core.py` or `executor.py` directly.

Minimal dependency direction within runtime:

- orchestration may coordinate discovery, authoring/pipeline, review, followups, context/events, and reasoning
- pipeline may depend on authoring and review outputs
- review may consume reasoning outputs, but not pipeline orchestration
- reasoning remains policy-oriented and does not depend on review state or persisted critique artifacts
- leaf/context/event utilities remain reusable helpers rather than mini-controllers

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
src/prscope/
├── __init__.py
├── config.py                  [Foundation] config loading, repo profiles
├── pricing.py                 [Foundation] model pricing tables
├── model_catalog.py           [Foundation] model registry and availability
├── profile.py                 [Foundation] codebase profiling
├── semantic.py                [Foundation] code search utilities
├── store/                     [Storage]    SQLite persistence package
│   ├── __init__.py            [Storage]    stable storage exports and compatibility surface
│   ├── service.py             [Storage]    composed Store facade
│   ├── connection.py          [Storage]    sqlite connection + timestamp helpers
│   ├── schema.py              [Storage]    schema DDL + migrations
│   ├── models.py              [Storage]    storage dataclasses
│   ├── repo_data.py           [Storage]    repo/PR/evaluation/artifact persistence
│   ├── planning_commands.py   [Storage]    command queue/lease lifecycle
│   ├── planning_sessions.py   [Storage]    planning session CRUD/search/update guards
│   ├── planning_history.py    [Storage]    turns/plan versions/round metrics persistence
│   └── file_stats.py          [Storage]    filesystem-backed stats and rounds logs
├── memory.py                  [Intelligence] memory blocks, manifesto, skills
├── llm.py                     [Intelligence] LLM call abstraction
├── scoring.py                 [Intelligence] PR scoring rules
├── github.py                  [Intelligence] GitHub API client, PR sync
├── planner.py                 [Intelligence] high-level planner facade
├── planning/
│   ├── core.py                [Intelligence] state machine, transitions, convergence
│   ├── executor.py            [Intelligence] command reserve/execute/finalize
│   ├── render.py              [Intelligence] plan export rendering
│   ├── decision_catalog.py    [Intelligence] stable decision identity/catalog entries
│   ├── scanners/
│   │   ├── base.py            [Intelligence] scanner interface
│   │   ├── grep.py            [Intelligence] grep-backed scanner
│   │   ├── repomap.py         [Intelligence] repomap scanner
│   │   └── repomix.py         [Intelligence] repomix scanner
│   └── runtime/
│       ├── orchestration.py   [Intelligence] top-level runtime wiring
│       ├── orchestration_support/ [Intelligence] orchestration helper package
│       │   ├── event_router.py [Intelligence] SSE versioning + tool event normalization/persistence
│       │   ├── state_snapshots.py [Intelligence] runtime snapshot persistence/list/read helpers
│       │   ├── initial_draft.py [Intelligence] memory-prep + initial-draft bootstrap flow
│       │   ├── session_starts.py [Intelligence] requirements/chat/PR start-mode adapters
│       │   ├── chat_flow.py    [Intelligence] discovery turn + author chat + clarification flows
│       │   └── round_entry.py  [Intelligence] adversarial round entry/context setup
│       ├── discovery.py       [Intelligence] discovery orchestrator / compatibility façade
│       ├── discovery_support/ [Intelligence] discovery helper package
│       │   ├── models.py      [Intelligence] discovery dataclasses + parsing helpers
│       │   ├── signals.py     [Intelligence] raw signal extraction + framework evidence helpers
│       │   ├── existing_feature.py [Intelligence] existing-feature evidence/summarization helpers
│       │   ├── bootstrap.py   [Intelligence] first-turn bootstrap scanning + evidence ingest
│       │   └── llm.py         [Intelligence] discovery LLM/tool-call loop wrapper
│       ├── reasoning/         [Intelligence] shared reasoning contract + policy modules
│       │   ├── base.py        [Intelligence] `Reasoner` interface
│       │   ├── models.py      [Intelligence] `ReasoningContext`, decisions, signal payloads
│       │   ├── discovery_reasoner.py [Intelligence] discovery routing policy
│       │   ├── refinement_reasoner.py [Intelligence] refinement routing + open-question policy
│       │   ├── review_reasoner.py [Intelligence] issue-to-decision interpretation
│       │   └── convergence_reasoner.py [Intelligence] convergence policy
│       ├── author.py          [Intelligence] plan authoring agent
│       ├── authoring/         [Intelligence] author subsystem package
│       │   ├── models.py      [Intelligence] author dataclasses + markdown helpers
│       │   ├── discovery.py   [Intelligence] deterministic author-side repo discovery
│       │   ├── validation.py  [Intelligence] author draft validation gates
│       │   ├── repair.py      [Intelligence] author plan repair/revision helpers
│       │   └── pipeline.py    [Intelligence] planner pipeline coordinator
│       ├── critic.py          [Intelligence] adversarial critic agent
│       ├── tools.py           [Intelligence] tool definitions, file ops
│       ├── telemetry.py       [Intelligence] completion telemetry
│       ├── transport/         [Intelligence] LLM transport adapters
│       │   └── llm_client.py  [Intelligence] responses/chat compatibility layer
│       ├── context/           [Intelligence] budget, context assembly, clarification, compression
│       ├── events/            [Intelligence] runtime analytics/tool/token event state
│       ├── followups/         [Intelligence] decision graph + post-plan follow-up artifacts
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
5. **Runtime leaf modules** may import only Foundation, approved leaf helpers, and their own local leaf types — not broader runtime subsystems, `planning.core`, or `planning.executor`.
6. **Reasoning modules** do not import runtime review state.

Violations are caught at CI time. See `tests/test_architecture.py`.

## Discovery Engine Notes

`planning/runtime/discovery.py` is intentionally orchestration-focused and feature-agnostic:

- feature detection is derived from user intent + code evidence, not hardcoded feature names
- framework detection is registry-based and scored from observed route signals
- directory/file prioritization is heuristic-scored (`_select_scan_directories`, `_route_file_score`) but remains evidence, not routing policy
- evidence is confidence-ranked and deduplicated before reasoner decisions
- bootstrap insights are session-scoped and keyed by generic fields (`existing_feature`, `feature_label`)
- semantic routing belongs in `planning/runtime/reasoning/*`, not in discovery helpers or orchestration wrappers

Guardrail: avoid introducing feature-specific conditionals in engine logic. Add new feature/framework support by extending registries and patterns, not by adding special-case branches.

## Evidence → Reasoning → Decision

Planning decisions should follow a causal flow:

`Evidence -> Reasoning -> Decision`

Role boundaries:

- **Evidence** modules extract structured observations and repository facts. Examples: `planning/runtime/discovery_support/signals.py`, `planning/runtime/discovery_support/existing_feature.py`, `planning/runtime/discovery_support/bootstrap.py`, `planning/scanners/*`, `semantic.py`.
- **Reasoning** modules interpret evidence into policy outcomes. Examples: `planning/runtime/reasoning/*`.
- **Decision** modules persist stable architectural state and follow-up artifacts. Examples: `planning/runtime/followups/decision_graph.py`, `planning/decision_catalog.py`.
- **Orchestration** coordinates the flow around these layers, but it does not interpret raw evidence into architectural decisions.

Guardrails:

- evidence modules produce observations, not architecture choices or routing policy
- reasoners consume structured evidence; they do not scan repos, execute tools, or persist decisions directly
- orchestration may coordinate evidence collection and invoke reasoners, but it must not derive architectural decisions directly from raw evidence
- authoring, review, and discovery helpers must not bypass reasoning when turning evidence into architectural conclusions

This separation keeps decisions traceable from persisted decision state back to reasoner outputs and underlying repository evidence.

## Decision Graph Notes

Plan versions now persist a first-class decision graph plus follow-up artifacts.

- `planning/runtime/followups/decision_graph.py` owns graph extraction, merge, and JSON serialization
- `planning/decision_catalog.py` provides stable catalog-backed node identity for common architecture decisions
- `planning/runtime/orchestration.py` treats persisted graph state as primary and uses markdown extraction as compatibility/backfill
- `planning/runtime/review/issue_graph.py` remains separate from the decision graph; links are additive via `related_decision_ids`
- `issue_graph` may reference decision nodes, but `decision_graph` does not import or depend on review state
- decision graph state remains deterministic and plan-version scoped
- decision graph nodes represent architectural state, not critic-derived evaluation state

## State And Persistence Notes

Protected planning-session lifecycle state must be written through `PlanningCore.transition_and_snapshot()`. See `docs/planning-state-machine.md` and `docs/agent-harness.md`.

- planning lifecycle state includes stage, round progression, convergence status, and plan-version transitions
- runtime modules may still persist auxiliary artifacts through the storage layer, such as telemetry, event state, issue graphs, decision graphs, and related metadata

This invariant is about protected session state, not about banning all runtime persistence.

## Testing And Validation Notes

Use `tests/` intentionally rather than as a dumping ground:

- architecture boundary tests protect import rules and sublayer invariants
- runtime/unit tests cover deterministic helpers and reasoners
- integration/web/API tests cover interface and persistence flows
- regression tests cover planning behavior and previously fixed edge cases

When reviewing runtime architecture changes, inspect newly introduced `runtime/<subdomain> -> runtime/<different_subdomain>` imports. Coordinator edges from `runtime.orchestration` and `runtime.orchestration_support` are expected; other new cross-subdomain edges deserve scrutiny even if they do not violate a hard CI boundary.

## Adding a New Module

1. Decide which layer it belongs to (Foundation / Storage / Intelligence / Interface).
2. Only import from that layer or below.
3. Add the module to the package map in this document.
4. If it's a new planning runtime module, keep it as a leaf unless it genuinely needs to import broadly.
5. Run `make check` to verify no import boundary violations.
