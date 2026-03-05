# Prscope

<p align="center">
  <img src="prscope-banner.svg" alt="Prscope" width="800" />
</p>

<p align="center">
  <b>PLAN. REFINE. SHIP.</b>
  <br />
  <i>A planning engine that turns requirements (or upstream PR context) into grounded plan/spec outputs for real codebases.</i>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/version-0.1.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/status-Alpha-yellow.svg" alt="Status">
</p>

---

## Contributing

See `CONTRIBUTING.md` for development workflow, performance benchmark requirements, and PR checklist.

---

## Table of Contents

- [What Prscope Does](#what-prscope-does)
- [Install](#install)
- [Quickstart](#quickstart)
- [Three Ways to Start a Plan](#three-ways-to-start-a-plan)
- [Web UI Workflow](#web-ui-workflow)
- [Runtime State Model](#runtime-state-model)
- [Planning Features](#planning-features)
- [Memory Stack: Skills + Recall](#memory-stack-skills--recall)
- [Full Configuration Reference](#full-configuration-reference)
- [Manifesto Constraints](#manifesto-constraints)
- [CLI Reference](#cli-reference)
- [Environment Variables](#environment-variables)
- [Project Structure](#project-structure)
- [Benchmark Harness](#benchmark-harness)
- [Development](#development)

---

## What Prscope Does

Prscope is a local-first planning system with a CLI + web UI workflow. You give it requirements (or seed it from an upstream GitHub PR), and it:

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
  author_model: gpt-4o        # drafts and refines the plan
  critic_model: gpt-4o-mini   # stress-tests it (same provider = one key)
  output_dir: ./plans

# Optional: seed plans from upstream GitHub PRs
upstream:
  - repo: owner/upstream-repo
```

> **Both models can be any LiteLLM-supported string.** Using two OpenAI models means you only need `OPENAI_API_KEY`. Swap in `claude-3-5-sonnet-20241022` for the critic if you want cross-provider adversarial tension.

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
- Model availability is key-aware (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`/`GEMINI_API_KEY`) and exposed from `GET /api/models`.
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
  author_model: gpt-4o                        # any LiteLLM string
  critic_model: claude-3-5-sonnet-20241022    # any LiteLLM string
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
- `issue_dedupe.embedding_model` is independent from `author_model` / `critic_model`, so you can run Claude or Gemini for planning while using a different embedding provider.
- `embeddings_enabled: auto` attempts semantic dedupe first and falls back to lexical dedupe if embeddings are unavailable.
- With `fallback_mode: none`, dedupe will not match when embeddings fail.
- Issue tracking is graph-backed with canonical IDs; duplicates are tracked via alias mapping (not duplicate nodes).
- Convergence gating uses root-open + unresolved dependency-chain checks, then existing stability/implementability checks.

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
| `GOOGLE_API_KEY`    | If using Google models                  | `gemini-pro`, etc.           |


Prscope uses [LiteLLM](https://docs.litellm.ai/docs/providers) — any provider it supports works as `author_model`, `critic_model`, or `issue_dedupe.embedding_model`.
Ollama local models need no API key.

---

## Project Structure

```
prscope/
├── cli.py                      # Click CLI (plan, upstream, recall, repos groups)
├── config.py                   # Config + RepoProfile dataclasses
├── store.py                    # SQLite storage (sessions, turns, versions, session recall BM25)
├── memory.py                   # Codebase memory builder, manifesto parser, load_skills()
├── profile.py                  # Local repo file tree profiling
├── scoring.py                  # Upstream PR relevance scoring
├── github.py                   # GitHub REST client
├── planner.py                  # PlanningEngine shim
├── web/
│   ├── api.py                  # FastAPI endpoints for planning sessions
│   ├── server.py               # Web server bootstrap + lifecycle
│   ├── frontend/               # Vite/React UI source
│   └── static/                 # Built frontend assets served by backend
├── planning/
│   ├── core.py                 # Pure state machine + convergence logic
│   ├── render.py               # plan Jinja2 rendering
│   └── runtime/
│       ├── orchestration.py    # Session coordinator (locks, lifecycle, command flow)
│       ├── author.py           # Author agent + initial draft pipeline + tool enforcement
│       ├── critic.py           # Critic LLM + JSON contract validation
│       ├── discovery.py        # Generalized discovery engine (intent → evidence → insight)
│       ├── tools.py            # Sandboxed read_file/search_codebase/list_dir
│       ├── pipeline/           # Adversarial loop + stage implementations + round context
│       ├── context/            # Budgeting, context assembly, compression, clarification gate
│       ├── review/             # Issue similarity and manifesto validation helpers
│       └── events/             # Analytics emitter + token/tool event persistence helpers
├── plan_templates/
│   └── plan.md.j2               # plan Jinja2 template
tests/
├── test_config.py              # Config + multi-repo parsing
├── test_store.py               # DB including planning tables + search_sessions
├── test_memory_skills.py        # load_skills() boundary and truncation
├── test_planning_core.py       # Convergence logic + constraint parsing
├── test_cli.py                 # CLI command surface
├── test_scoring.py             # PR relevance scoring
├── test_profile.py             # Codebase profiling
├── test_github.py              # GitHub client
└── test_semantic.py            # Semantic similarity
```

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

## License

MIT