# Memory, Context, and Manifesto

This document explains how `prscope` builds planning memory, manages prompt context budgets, and enforces manifesto constraints during discovery, authoring, and refinement.

## Why This Exists

The planning runtime needs three things to produce reliable plans:

- **Repository memory**: concise, reusable summaries of architecture and codebase structure.
- **Context discipline**: budgeted prompt construction so important information is always present.
- **Manifesto constraints**: explicit policy and non-negotiable guardrails injected into planning loops.

The system is designed so these three concerns work together instead of competing for tokens.

## Memory System Overview

`MemoryStore` in `prscope/memory.py` is responsible for building and loading memory artifacts.

### Memory Blocks

Default block set:

- `architecture`
- `modules`
- `patterns`
- `entrypoints`

For non-grep scanners (`repomap`, `repomix`), a single rich `context.md` block is used and stub block files are created for compatibility.

### Where Memory Lives

Memory is stored per repo under:

- `~/.prscope/repos/<repo-name>/memory`

Metadata is tracked in:

- `~/.prscope/repos/<repo-name>/memory/_meta.json`

The meta file includes `git_sha`, build timestamp, and repo path so memory can be rebuilt only when needed.

### Rebuild Logic

Memory rebuild is triggered when:

- explicit `rebuild` is requested, or
- tracked `git_sha` changes, or
- expected block files are missing.

This keeps startup fast for normal runs while still invalidating stale memory.

### Scanner + Summarization Path

There are two high-level build modes:

- **Grep-backed mode**
  - Scanner builds rich repository context text.
  - LLM summarizes this into structured block markdown.
  - Block builds run concurrently (`planning.memory_concurrency`).
- **Repomap/repomix mode**
  - Scanner provides rich context directly.
  - `context.md` is written directly.
  - stub block files are generated for the classic block names.

If LLM summarization fails, memory falls back to deterministic summaries so planning can proceed.

## Manifesto Integration

The manifesto is loaded from the repo profile path:

- default: `<repo>/.prscope/manifesto.md`
- override: repo-level `manifesto_file`

If it does not exist, `MemoryStore.ensure_manifesto()` creates a starter manifesto template.

### Machine-Readable Constraints

`MemoryStore.load_constraints()` parses a YAML-like machine-readable block from markdown:

- supports:
  - `extends: ...` (currently logged as unresolved V2 parent, not merged)
  - `constraints:` list with:
    - `id`
    - `text`
    - `severity` (`hard` or `soft`, defaults to `hard`)
    - `optional`

Malformed YAML or invalid shapes fail safely (empty constraint list rather than crash).

### Constraint Lifecycle

Constraints are injected into runtime loops via `PlanningRuntime._constraints()` and consumed by critic/author orchestration logic. They are part of the contract used to detect hard-constraint violations during refinement.

## Context Management in Runtime

`PlanningRuntime` controls context assembly and token budgeting for initial draft and refinement rounds.

### Budget Strategy

The runtime uses a token budget manager to ensure required content always fits:

- requirements text is mandatory
- manifesto excerpt is mandatory
- context index is included within remaining budget
- additional memory blocks are injected based on available budget and ratio enforcement

Manifesto and memory sections are truncated by configurable char caps before budgeting:

- `planning.memory_block_max_chars`
  - defaults:
    - `architecture: 3000`
    - `modules: 2000`
    - `manifesto: 1500`
- repo-level overrides merge on top of planning defaults

### Context Index Pattern

The runtime builds a short context index that tells the model what memory is available and how to fetch more through tools. This keeps initial prompts compact while preserving discoverability.

### On-Demand Memory Pull

The tool surface includes `get_memory_block(key)` so the model can fetch targeted memory only when needed, rather than front-loading all memory into every call.

Supported keys are bounded and validated (`architecture`, `modules`, `patterns`, `entrypoints`, `context`).

## Discovery + Author + Critic Interplay

Memory and manifesto are used in multiple phases:

- **Discovery**
  - uses memory + manifesto to shape clarifying questions and early framing.
- **Initial draft (author planner phase)**
  - starts with requirements + manifesto excerpt + context index.
  - can pull additional memory via tools.
- **Adversarial refinement**
  - carries manifesto/constraints and prior critique context into critic/author iterations.
  - uses token budget controls to keep prompt size stable.

This staged usage minimizes token bloat while preserving policy fidelity.

## Safety and Governance Properties

The current design provides:

- **Determinism guardrails**
  - structured constraints and repeatable memory blocks.
- **Policy visibility**
  - manifesto path and excerpt explicitly included in planning prompts.
- **Graceful degradation**
  - fallback summaries and safe parse failures avoid full pipeline collapse.
- **Bounded context growth**
  - char caps + budgeted allocation + on-demand memory pull.

## Configuration Surface

Relevant knobs:

- `planning.memory_concurrency`
- `planning.memory_block_max_chars`
- `planning.scanner` (`grep`, `repomap`, `repomix`)
- repo-level `memory_block_max_chars` overrides
- repo-level `manifesto_file` override

Related runtime/state behavior is documented in `docs/agent-harness.md` and `docs/planning-state-machine.md`.

## Practical Tips

- Keep manifesto constraints concise and actionable; vague constraints produce noisy critiques.
- Prefer hard constraints only for true non-negotiables; overusing hard severity can stall refinement.
- Tune memory block caps before increasing raw model context; smaller, cleaner blocks often work better.
- Use scanner backends intentionally:
  - `grep` for low dependency/default reliability
  - `repomap`/`repomix` when richer structural context is worth the setup.
