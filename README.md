# Prscope

<p align="center">
  <img src="prscope-banner.svg" alt="Prscope" width="800" />
</p>

<p align="center">
  <b>PLAN. REFINE. SHIP.</b>
  <br />
  <i>An adversarial Author/Critic planning engine that turns requirements — or an upstream PR — into high-quality PRD and RFC documents, grounded in your actual codebase.</i>
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

## What Prscope Does

Prscope is a local-first CLI planning tool. You give it requirements (or seed it from an upstream GitHub PR), and it:

1. Scans your codebase into structured memory blocks
2. Opens a terminal TUI where an **Author LLM** drafts the plan
3. Runs up to 10 adversarial **Author ↔ Critic** refinement rounds
4. Exports a final `PRD.md` and `RFC.md` grounded in your actual file paths and constraints

Upstream PR tracking (`sync` / `evaluate`) feeds directly into this planning flow — instead of generating standalone PRDs, relevant PRs become high-signal inputs to planning sessions.

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
cp env.sample .env
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

# Start an interactive planning session
prscope plan chat
```

The TUI opens. The Author LLM asks you clarifying questions, drafts a plan, and you can trigger Critic rounds (`Ctrl+K`) to refine it.

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

---

## TUI Keybindings

| Key | Action |
|-----|--------|
| `Ctrl+K` | Trigger adversarial Critic round |
| `Ctrl+D` | Toggle diff view (current vs previous round) |
| `Ctrl+A` | Approve plan |
| `Ctrl+E` | Export PRD + RFC |
| `Ctrl+Q` | Quit (session is saved) |

Status bar format: `[repo: my-repo]  Round: 2/10  REFINING  Δ12%  [Seeded: PR #42]`

---

## Planning Features

| Feature | Description |
|---------|-------------|
| **Structured codebase memory** | Auto-generates `architecture.md`, `modules.md`, `patterns.md`, `entrypoints.md` per repo |
| **Manifesto constraints** | Hard/soft constraints in `.prscope/manifesto.md` enforced by the Critic |
| **Convergence detection** | Multi-signal: hash equality, diff ratio, structural regression, critic `major_issues_remaining` |
| **Tool-use enforcement** | Author must verify file paths in the repo before finalizing; accessed paths tracked |
| **Diff view** | See exactly what the Critic forced to change each round |
| **CI validation** | `prscope plan validate <session-id>` exits non-zero on constraint violations |
| **Drift detection** | `prscope plan status` compares planned vs merged files post-implementation |
| **Multi-repo support** | Named repo profiles each with isolated memory, manifesto, and output dir |

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
prscope init            Initialize Prscope in current repo
prscope profile         Scan and profile local codebase
```

### Planning

```
prscope plan chat                                  Start chat-first discovery
prscope plan start "requirements"                  Start from text requirements
prscope plan start --from-pr owner/repo 42         Seed from upstream PR
prscope plan resume <session-id>                   Resume in TUI
prscope plan list [--repo <name>]                  List sessions
prscope plan diff <session-id> [--round N]         Unified diff to stdout
prscope plan export <session-id>                   Write PRD.md + RFC.md
prscope plan validate <session-id>                 Headless CI check (exits 0/1/2)
prscope plan status <session-id> --pr-number N     Post-merge drift detection
prscope plan memory [--rebuild]                    Build/show memory blocks
prscope plan manifesto [--edit]                    Create/open manifesto
```

### Repos

```
prscope repos list      Show configured repo profiles and memory age
```

### Upstream (PR intelligence input)

```
prscope upstream sync                      Fetch upstream PRs
prscope upstream evaluate [--batch N]      Score PRs for planning relevance
prscope upstream digest                    Show top relevant PRs
prscope upstream history [--decision ...]  View evaluation history
```

> Legacy top-level commands (`prscope sync`, `prscope evaluate`, etc.) still work but will print a deprecation notice pointing to `prscope upstream ...`.

---

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `GITHUB_TOKEN` | For upstream sync | GitHub personal access token |
| `OPENAI_API_KEY` | If using OpenAI models | `gpt-4o`, `o1`, etc. |
| `ANTHROPIC_API_KEY` | Optional — only if using a Claude model | `claude-3-5-sonnet`, etc. |
| `GOOGLE_API_KEY` | If using Google models | `gemini-pro`, etc. |

Prscope uses [LiteLLM](https://docs.litellm.ai/docs/providers) — any provider it supports works as `author_model` or `critic_model`. Ollama local models need no API key.

---

## Project Structure

```
prscope/
├── cli.py                      # Click CLI (plan, upstream, repos groups)
├── config.py                   # Config + RepoProfile dataclasses
├── store.py                    # SQLite storage (sessions, turns, versions)
├── memory.py                   # Codebase memory builder + manifesto parser
├── profile.py                  # Local repo file tree profiling
├── scoring.py                  # Upstream PR relevance scoring
├── github.py                   # GitHub REST client
├── tui.py                      # Textual TUI (PlanningTUI)
├── planner.py                  # PlanningEngine shim
├── planning/
│   ├── core.py                 # Pure state machine + convergence logic
│   ├── render.py               # PRD/RFC Jinja2 rendering
│   └── runtime/
│       ├── orchestration.py    # PlanningRuntime entry points
│       ├── author.py           # Author LLM loop + tool enforcement
│       ├── critic.py           # Critic LLM + JSON contract validation
│       ├── discovery.py        # Chat-first discovery flow
│       └── tools.py            # Sandboxed read_file/search_codebase/list_dir
├── plan_templates/
│   ├── prd.md.j2               # PRD Jinja2 template
│   └── rfc.md.j2               # RFC Jinja2 template (file path validation)
└── templates/
    └── prd.md.j2               # Legacy upstream PRD template

tests/
├── test_config.py              # Config + multi-repo parsing
├── test_store.py               # DB including planning tables
├── test_planning_core.py       # Convergence logic + constraint parsing
├── test_cli.py                 # CLI command surface
├── test_scoring.py             # PR relevance scoring
├── test_profile.py             # Codebase profiling
├── test_github.py              # GitHub client
└── test_semantic.py            # Semantic similarity
```

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

# Run repeatable planning benchmark suite
prscope-benchmark --base-url http://127.0.0.1:8443 --repo my-repo --config-root /path/to/repo
```

---

## License

MIT
