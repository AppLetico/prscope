# Prscope

<p align="center">
  <img src="prscope-banner.svg" alt="Prscope" width="800" />
</p>

<p align="center">
  <b>PLAN. REFINE. SHIP.</b>
  <br />
  <i>AI-assisted architecture planning for real codebases.</i>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/version-0.1.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/status-Alpha-yellow.svg" alt="Status">
</p>

---

Prscope is a **local-first AI-assisted architecture planning system**.

Instead of jumping directly from requirements or upstream code changes to generated code, Prscope creates a **grounded implementation plan first**, using your real repository structure, codebase signals, and engineering constraints.

It combines:

- **Codebase-aware planning** — scans your repo into structured memory blocks
- **Author ↔ Critic refinement loops** — adversarial review before code is written
- **Architecture constraints** — enforce design rules through a manifesto
- **Grounded plan exports** — outputs reference real file paths, modules, and constraints

The result is a **reviewable implementation plan (`PRD.md`) in your configured output directory** that fits your codebase before code generation begins.

---

## Why Prscope Exists

Modern AI coding tools can generate code quickly, but they rarely reason about **architecture**.

That often leads to:

- inconsistent designs
- hidden dependency issues
- large refactors later

Prscope moves that reasoning earlier in the workflow:

```text
requirements or upstream changes
-> architecture plan (Prscope)
-> implementation
```

By forcing architectural critique and constraint validation first, teams can catch design problems **before writing large amounts of code**.

---

## Table of Contents

- [Why Prscope Exists](#why-prscope-exists)
- [Install](#install)
- [Quickstart](#quickstart)
- [Example](#example)
- [Three Ways to Start a Plan](#three-ways-to-start-a-plan)
- [Web UI Workflow](#web-ui-workflow)
- [Planning Features](#planning-features)
- [CLI Reference](#cli-reference)
- [Development](#development)
- [Technical Docs](#technical-docs)

---

## What Prscope Does

Prscope gives you a CLI + web UI workflow for turning requirements into grounded plan/spec outputs for real codebases:

1. Scans your codebase into structured memory blocks
2. Starts a planning session in the web UI where an **Author LLM** drafts the plan
3. Runs up to 10 adversarial **Author ↔ Critic** refinement rounds
4. Exports a final plan (`PRD.md`) grounded in your actual file paths and constraints

Upstream PR tracking (`prscope upstream sync` / `prscope upstream evaluate`) feeds directly into this planning flow — instead of generating standalone plans, relevant PRs become high-signal inputs to planning sessions.

---

## Install

```bash
git clone https://github.com/your-org/prscope.git
cd prscope
pip install -e .
```

Or install from PyPI when published:

```bash
pip install prscope
```

---

## Quickstart

### 1. Set up environment

```bash
cp env.example .env
```

Edit `.env` with your keys:

```bash
GITHUB_TOKEN=github_pat_your_token_here
OPENAI_API_KEY=sk-your-openai-key
# ANTHROPIC_API_KEY=sk-ant-your-anthropic-key  # optional — only if using a Claude model
```

### 2. Initialize in your repo

```bash
cd /path/to/your/repo
prscope init
```

This creates `prscope.yml`, `prscope.features.yml`, and `.prscope/` in your repo.

### 3. Configure `prscope.yml`

`prscope init` creates a `prscope.yml` — edit the planning section at minimum:

```yaml
local_repo: .   # path to your repo (default: current directory)

planning:
  author_model: gpt-4o        # compatibility default for authoring stages
  critic_model: gpt-4o-mini   # compatibility default for critic stages
  memory_model: gpt-4o-mini   # optional: codebase memory build (defaults to author_model)
  initial_draft_model: gemini-2.5-flash          # optional stage override
  author_refine_model: gpt-4o-mini               # optional stage override
  critic_review_model: gpt-4o-mini               # optional stage override
  structured_output_fallback_model: gpt-4o-mini  # fallback for JSON-heavy stages
  output_dir: ./plans

# Optional: seed plans from upstream GitHub PRs
upstream:
  - repo: owner/upstream-repo
```

> **All model fields can be any LiteLLM-supported string.** `author_model` / `critic_model` remain compatibility defaults, while `initial_draft_model`, `author_refine_model`, and `critic_review_model` let you split roles by stage. `structured_output_fallback_model` is used only when JSON-heavy stages hit repeated contract failures. `memory_model` is used only for the "Preparing codebase memory" step; if omitted it defaults to `author_model`.

### 4. (Optional) Define your architecture constraints

`prscope init` also creates `.prscope/manifesto.md` — a plain markdown file where you document principles and machine-readable constraints the Critic will enforce every round:

```yaml
# inside .prscope/manifesto.md
constraints:
  - id: C-001
    text: "No synchronous I/O on the main thread"
    severity: hard    # blocks approval and CI
  - id: C-002
    text: "Prefer stdlib over new dependencies"
    severity: soft    # advisory only
```

Skip this entirely if you don't need governance — it's optional. Edit it later with `prscope plan manifesto --edit`.

### 5. Build codebase memory and start planning

```bash
# Scan your codebase into structured memory blocks
prscope profile

# Start an interactive planning session (opens browser UI)
prscope plan chat
```

The web UI opens. The Author LLM asks clarifying questions, drafts a plan, and supports iterative Critic refinement rounds.

> `prscope plan ...` auto-starts the local web server when needed. If you prefer managing it explicitly, run `prscope web` first.

---

## Example

```bash
prscope plan start "Add rate limiting middleware for authentication endpoints"
```

Prscope will:

1. Inspect the repository for framework and architecture context
2. Draft a grounded implementation plan
3. Run adversarial **Author ↔ Critic** review rounds
4. Export a final spec (`PRD.md`) in your configured output directory

The resulting plan references **real modules, files, and constraints in your codebase**, making it ready for implementation or code generation.

---

## Three Ways to Start a Plan

### From requirements text

```bash
prscope plan start "Add multi-agent task routing with priority queues"
```

### From a GitHub PR (seeded with upstream analysis)

```bash
# First pull in upstream PR data
prscope upstream sync
prscope upstream evaluate

# Then start a plan seeded from a specific PR
prscope plan start --from-pr owner/upstream-repo 42
```

### Chat-first discovery

```bash
prscope plan chat
```

The Author LLM interviews you to define scope before drafting anything.

Discovery now uses a generalized, repository-agnostic evidence flow:

- extracts a feature intent from the request (for example: "rate limiting middleware")
- scans code once, then reuses a shared signal index for framework + evidence inference
- detects existing implementations before proposing creation plans
- avoids asking framework-identification questions when framework evidence is already present

In short: discovery should ask for decisions, not facts already visible in code.

---

## Web UI Workflow

```bash
# Start API + web UI
prscope web

# Start a planning session from requirements
prscope plan start "Add rate limiting for auth endpoints"

# Resume an existing session in browser UI
prscope plan resume <session-id>
```

You can also pass `--no-open` to `plan start` / `plan chat` if you do not want the browser opened automatically.
For background operation, run `prscope web --background`.

### Model Selection (UI + API)

- The New Session screen includes `Author model` and `Critic model` selectors.
- The Planning view header keeps these selectors available per interaction, so you can switch model pairs before sending a message or running a critique round.
- Model availability is key-aware (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`) and exposed from `GET /api/models`.
- The backend validates every selected model at request time; if a model becomes unavailable, the request fails with a clear validation error so the UI can prompt reselection.
- Last selected model pair is persisted per repo in browser local storage for convenience.

---

## Runtime State Model

Prscope now uses a server-authoritative runtime model for planning sessions:

- Session row is the canonical state machine (`status`, `phase_message`, `pending_questions`, `is_processing`, `active_tool_calls`)
- `session_state` SSE events carry full snapshots and are emitted snapshot-first on connect
- `POST /round` executes through the command executor with lease-backed lock/idempotency semantics
- Commands are idempotent via `command_id` and validated against explicit state/command rules
- Frontend renders from canonical session state rather than reconstructing from turn text

Why this matters:

- page reloads are predictable
- duplicate submits/retries are safely handled
- multi-tab behavior stays consistent

Further reading:

- `docs/planning-state-machine.md` (formal contract and invariants)
- `docs/agent-harness.md` (end-to-end API/runtime/SSE behavior)

---

## Planning Features


| Feature                        | Description                                                                                     |
| ------------------------------ | ----------------------------------------------------------------------------------------------- |
| **Structured codebase memory** | Auto-generates `architecture.md`, `modules.md`, `patterns.md`, `entrypoints.md` per repo        |
| **Manifesto constraints**      | Hard/soft constraints in `.prscope/manifesto.md` enforced by the Critic                         |
| **Skills memory**              | Always-on team patterns loaded from `.prscope/skills/*.md` with deterministic ordering          |
| **Session recall**             | `prscope recall` searches prior sessions; optional auto-injection during planning                |
| **Convergence detection**      | Multi-signal: hash equality, diff ratio, structural regression, critic `major_issues_remaining` |
| **Tool-use enforcement**       | Author must verify file paths in the repo before finalizing; accessed paths tracked             |
| **Diff view**                  | See exactly what the Critic forced to change each round                                         |
| **CI validation**              | `prscope plan validate <session-id>` exits non-zero on constraint violations                    |
| **Drift detection**            | `prscope plan status` compares planned vs merged files post-implementation                      |
| **Multi-repo support**         | Named repo profiles each with isolated memory, manifesto, and output dir                        |


---

## Memory Stack: Skills + Recall

Prscope uses a layered memory architecture for durable planning quality:

1. **Manifesto** (philosophy + hard constraints)
2. **Skills** (persistent team planning defaults)
3. **Recall** (episodic memory from past planning sessions)
4. **Memory blocks** (current codebase context)

The runtime injects these in the fixed order above so historical precedent cannot override policy constraints.

### Skills

Add markdown files under `.prscope/skills/`:

```bash
mkdir -p .prscope/skills
$EDITOR .prscope/skills/api-guidelines.md
```

Skills are loaded deterministically and truncated at file boundaries to fit budget.

### Recall

Search prior sessions:

```bash
prscope recall "auth token rotation endpoint compatibility"
prscope recall "auth token rotation endpoint compatibility" --all-repos
prscope recall "auth token rotation endpoint compatibility" --full
```

Start clean-slate sessions without recall anchoring:

```bash
prscope plan start --no-recall "Design a new auth subsystem"
prscope plan chat --no-recall
```

For runtime details, see [`docs/skills-and-recall.md`](docs/skills-and-recall.md).

---

## Full Configuration Reference

```yaml
# prscope.yml

# Single-repo (backward-compatible)
local_repo: ~/workspace/my-repo

# Multi-repo profiles (optional — enables `--repo <name>` on all commands)
repos:
  my-repo:
    path: ~/workspace/my-repo
    upstream:
      - repo: owner/upstream-repo
    # manifesto_file: optional override
    # output_dir: optional override

# Upstream PR ingestion (feeds planning seeds)
upstream:
  - repo: owner/upstream-repo
    filters:
      # state: merged   # optional per-repo override

sync:
  state: merged
  max_prs: 100
  fetch_files: true
  since: 90d          # 90d, 6m, 1y, or ISO date
  incremental: true
  eval_batch_size: 25

llm:
  enabled: true
  model: gpt-4o
  temperature: 0.2
  max_tokens: 3000

# Planning mode
planning:
  author_model: gpt-4o                        # compatibility default for authoring stages
  critic_model: claude-3-5-sonnet-20241022    # compatibility default for critic stages
  memory_model: gpt-4o-mini                   # optional: codebase memory build (defaults to author_model)
  initial_draft_model: gemini-2.5-flash       # optional stage override
  author_refine_model: gpt-4o-mini            # optional stage override
  critic_review_model: gpt-4o-mini            # optional stage override
  structured_output_fallback_model: gpt-4o-mini
  issue_dedupe:
    embeddings_enabled: auto                   # auto | true | false
    embedding_model: text-embedding-3-small    # any LiteLLM embedding model
    similarity_threshold: 0.82                 # semantic duplicate cutoff
    fallback_mode: lexical                     # lexical | none
  issue_graph:
    max_nodes: 50                              # hard cap for issue nodes
    max_edges: 100                             # hard cap for issue edges
    causality_extraction_enabled: false        # infer causes edges from critic prose
    causality_max_edges_per_review: 8          # cap inferred edges each review pass
    causality_min_text_len: 12                 # reject short/vague inferred issue text
  skills_max_chars: 3000                      # budget for .prscope/skills injection
  recall_prior_sessions: false                # enable session recall injection
  recall_top_k: 2                             # max recalled sessions considered
  recall_max_chars: 1500                      # budget for injected recall context
  max_adversarial_rounds: 10
  convergence_threshold: 0.05
  output_dir: ./plans
  memory_concurrency: 2                       # parallel LLM calls during memory build
  discovery_max_turns: 5                      # max Q&A turns before forcing draft
  seed_token_budget: 4000                     # max tokens from PR seed context
  require_verified_file_references: false     # set true for strict mode
  validate_temperature: 0.0                   # deterministic CI validation
  validate_audit_log: true                    # save critic JSON to ~/.prscope/.../audit/
```

Notes:
- `author_model` / `critic_model` are still accepted everywhere for backward compatibility, but stage-specific overrides now take precedence when present.
- `initial_draft_model` is a good place to try faster/cheaper experimental models while keeping `critic_review_model` and `author_refine_model` on a JSON-stable baseline.
- `structured_output_fallback_model` is used only for structured JSON stages after repeated contract failures; it does not change the initial draft path.
- `memory_model` is used only for the "Preparing codebase memory" step. Omit it to use `author_model` (backward compatible). Set it to a cheaper model (e.g. `gpt-4o-mini`) to reduce cost for memory build while keeping a stronger author/critic.
- `issue_dedupe.embedding_model` is independent from `author_model` / `critic_model` / `memory_model`, so you can run Claude or Gemini for planning while using a different embedding provider.
- `embeddings_enabled: auto` attempts semantic dedupe first and falls back to lexical dedupe if embeddings are unavailable.
- With `fallback_mode: none`, dedupe will not match when embeddings fail.
- Issue tracking is graph-backed with canonical IDs; duplicates are tracked via alias mapping (not duplicate nodes).
- Convergence gating uses root-open + unresolved dependency-chain checks, then existing stability/implementability checks.
- `GET /api/sessions/{id}/diagnostics` now includes selected stage models/providers plus JSON retry/fallback counters for `critic_review` and `author_refine`.

---

## Manifesto Constraints

`.prscope/manifesto.md` is created by `prscope init` and is optional. Add a `constraints:` YAML block to define architecture rules the Critic enforces every round. See the [Quickstart](#4-optional-define-your-architecture-constraints) for the format.

Severity levels:

- `hard` — blocks `prscope plan approve` and fails `prscope plan validate` (CI)
- `soft` — Critic flags but does not block
- `optional: true` — Critic notes in passing, never a violation

Edit at any time with `prscope plan manifesto --edit`.

---

## CLI Reference

### Core

```
prscope init                              Initialize Prscope in current repo
prscope profile                           Scan and profile local codebase
prscope web [--background]                Run web UI server
prscope repos list                        Show configured repo profiles and memory age
prscope scanners list                     Show scanner backends and status
prscope analytics [--repo <name>]         Show planning analytics and quality trends
```

### Planning

```
prscope plan chat                                  Start chat-first discovery
prscope plan chat --no-recall                      Start discovery without prior-session recall
prscope plan start "requirements"                  Start from text requirements
prscope plan start --no-recall "requirements"      Start requirements mode without recall
prscope plan start --from-pr owner/repo 42         Seed from upstream PR
prscope plan resume <session-id>                   Resume in browser UI
prscope plan list [--repo <name>]                  List sessions
prscope plan diff <session-id> [--round N]         Unified diff to stdout
prscope plan export <session-id>                   Write plan (PRD.md) and conversation
prscope plan validate <session-id>                 Headless CI check (exits 0/1/2)
prscope plan status <session-id> --pr-number N     Post-merge drift detection
prscope plan memory [--rebuild]                    Build/show memory blocks
prscope plan manifesto [--edit]                    Create/open manifesto
```

### Upstream (PR intelligence input)

```
prscope upstream sync                               Fetch upstream PRs
prscope upstream evaluate [--batch N]               Score PRs for planning relevance
prscope upstream digest                             Show top relevant PRs
prscope upstream history [--decision ...]           View evaluation history
```

> Legacy top-level commands (`prscope sync`, `prscope evaluate`, etc.) still work but will print a deprecation notice pointing to `prscope upstream ...`.

### Recall (episodic planning memory)

```
prscope recall "query terms with enough signal"         Search current repo by default
prscope recall "query terms with enough signal" --repo my-repo
prscope recall "query terms with enough signal" --all-repos
prscope recall "query terms with enough signal" --full
```

---

## Environment Variables


| Variable            | Required                                | Purpose                      |
| ------------------- | --------------------------------------- | ---------------------------- |
| `GITHUB_TOKEN`      | For upstream sync                       | GitHub personal access token |
| `OPENAI_API_KEY`    | If using OpenAI models                  | `gpt-4o`, `o1`, etc.         |
| `ANTHROPIC_API_KEY` | Optional — only if using a Claude model | `claude-3-5-sonnet`, etc.    |
| `GEMINI_API_KEY`    | If using Google Gemini models          | `gemini-2.5-flash`, etc.     |


Prscope uses [LiteLLM](https://docs.litellm.ai/docs/providers) — any provider it supports works as `author_model`, `critic_model`, `memory_model`, or `issue_dedupe.embedding_model`.
Ollama local models need no API key.

---

## Project Structure

```
src/prscope/
├── cli.py                      # Click CLI (plan, upstream, recall, repos groups)
├── config.py                   # Config + RepoProfile dataclasses
├── store.py                    # SQLite storage (sessions, turns, versions, session recall BM25)
├── memory.py                   # Codebase memory builder, manifesto parser, load_skills()
├── profile.py                  # Local repo file tree profiling
├── scoring.py                  # Upstream PR relevance scoring
├── github.py                   # GitHub REST client
├── llm.py                      # LLM call abstraction (LiteLLM)
├── model_catalog.py            # Model registry and API-key-aware availability
├── pricing.py                  # Model pricing tables
├── planner.py                  # PlanningEngine shim
├── benchmark.py                # HTTP-based benchmark harness
├── semantic.py                 # Code search and similarity utilities
├── web/
│   ├── api.py                  # FastAPI app factory + planning session endpoints
│   ├── server.py               # Web server bootstrap + lifecycle
│   ├── events.py               # Session-scoped SSE emitter
│   ├── frontend/               # Vite/React UI source
│   └── static/                 # Built frontend assets served by backend
├── planning/
│   ├── core.py                 # Pure state machine + convergence logic
│   ├── executor.py             # Command reserve/execute/finalize, replay, locking
│   ├── render.py               # plan Jinja2 rendering
│   ├── scanners/               # Codebase scanning backends (grep, repomap, repomix)
│   └── runtime/
│       ├── orchestration.py    # Session coordinator (locks, lifecycle, command flow)
│       ├── orchestration_support/  # event_router, state_snapshots, initial_draft, session_starts, chat_flow, round_entry
│       ├── discovery.py        # Discovery orchestrator (intent → evidence → insight)
│       ├── discovery_support/  # models, signals, existing_feature, bootstrap, llm
│       ├── author.py           # Author agent + initial draft + tool enforcement
│       ├── authoring/          # models, discovery, validation, repair, pipeline
│       ├── critic.py           # Critic LLM + JSON contract validation
│       ├── tools.py            # Sandboxed read_file/search_codebase/list_dir
│       ├── pipeline/           # Adversarial loop, stages, round context
│       ├── context/            # Budgeting, context assembly, compression, clarification
│       ├── review/             # Issue graph, similarity, causality, manifesto checker
│       ├── events/             # Analytics emitter, tool event state, token accounting
│       ├── transport/          # LLM client (responses/chat compatibility)
│       ├── round_controller.py # Round progression logic
│       ├── state.py            # Runtime planning state model
│       └── telemetry.py        # Completion telemetry
├── plan_templates/
│   └── plan.md.j2              # plan Jinja2 template
tests/
├── test_config.py             # Config + multi-repo parsing
├── test_store.py              # DB including planning tables + search_sessions
├── test_memory_skills.py      # load_skills() boundary and truncation
├── test_planning_core.py      # Convergence logic + constraint parsing
├── test_cli.py                # CLI command surface
├── test_scoring.py            # PR relevance scoring
├── test_profile.py            # Codebase profiling
├── test_github.py             # GitHub client
├── test_semantic.py           # Semantic similarity
└── test_architecture.py       # Import boundary and layer rules
```

For layer ordering, dependency rules, and the full package map, see **`ARCHITECTURE.md`**.

---

## Benchmark Harness

Prscope includes a repeatable benchmark harness for planning speed + quality:

```bash
prscope-benchmark --base-url http://127.0.0.1:8443 --repo my-repo --config-root /path/to/repo
```

Artifacts:

- Run history JSON: `benchmarks/results/history/run-<timestamp>.json`
- Diagnostics log: `benchmarks/results/history/run-<timestamp>.log`
- Best-known baseline: `benchmarks/results/best_performance.json`

See:

- `benchmarks/README.md` for benchmark flags, quality heuristic, and debugging long runs
- `CONTRIBUTING.md` for mandatory performance benchmark policy on performance-sensitive PRs

Recommended default for expensive model stacks:

- Use cheap models in a dedicated config for benchmark test loops
- Run with `--stop-on-first-problem` during diagnostics
- Use `--health-check-only` before larger suites

---

## Development

```bash
# Install with dev dependencies
make dev

# Run tests
make test

# Lint + format check + tests
make check

# Profile → upstream sync → evaluate
make run

# Start discovery-mode planning
make plan-chat

# Run the web app (two terminals)
make web              # show instructions
make web-backend      # API on :8420 (loads .env)
make web-frontend     # Vite on :5173
```

For performance runs, use the [Benchmark Harness](#benchmark-harness) section above.

---

## Technical Docs

For runtime internals, architecture notes, and operational details, start with `docs/README.md`.

---

## Contributing

See `CONTRIBUTING.md` for development workflow, performance benchmark requirements, and PR checklist.

---

## License

MIT