# Design Philosophy

This document captures the foundational design decisions behind prscope and the reasoning that produced them.

## Product Intent

Prscope exists to reduce the gap between "I know roughly what I want" and "here is a grounded, reviewable plan that references real code." It is a planning tool, not a code generator — its output is plan documents, not pull requests.

## Design Decisions

### Local-First

All state lives on disk (SQLite + flat files). No cloud dependency for core planning. API keys are the only external requirement, and only for LLM calls.

**Why:** Planning tools that require infrastructure setup create friction before the first useful output. Local-first means `pip install` → useful work in minutes.

### Server-Authoritative State

The database session row is the single source of truth for UI state. SSE events accelerate updates but never define canonical state. A full page reload always recovers the correct view.

**Why:** Split-brain between server and client is the #1 source of subtle planning UI bugs. Server-authoritative eliminates the class entirely.

### Adversarial Refinement

Plans are refined through an Author ↔ Critic loop (up to 10 rounds). The Critic is structurally adversarial — it is prompted to find flaws, not confirm quality.

**Why:** Single-pass LLM output has systematic blind spots. Adversarial refinement catches missing edge cases, ungrounded claims, and constraint violations that a single author pass misses.

### Manifesto as Policy

Hard constraints (secrets, destructive ops) are encoded in `.prscope/manifesto.md` and enforced mechanically by the Critic. They are injected into every planning loop, not left to model discretion.

**Why:** Governance rules that depend on model memory are unreliable. Explicit injection ensures policy is always present in context.

### Memory as Progressive Disclosure

The planning runtime does not front-load all codebase context. It provides a compact index and lets the model pull additional memory blocks on demand via tools.

**Why:** Token budgets are finite. Front-loading wastes context on irrelevant details. On-demand pull keeps the important content (requirements, manifesto, critique) in the high-attention window.

### Deterministic State Machine

Session lifecycle follows an explicit state machine (`draft → refining → converged → approved`) with transitions enforced in a single function (`transition_and_snapshot`). Invalid states are unrepresentable by construction.

**Why:** Planning sessions have long lifecycles (minutes to hours). Implicit state management leads to inconsistencies on crash recovery, reconnect, and concurrent access.

### Benchmark-Driven Quality

Planning output quality is measured by a repeatable benchmark harness with historical baselines. Performance-sensitive changes require before/after comparison.

**Why:** LLM-based systems are easy to regress silently. Quantitative baselines make regressions visible before merge.

## Non-Goals

- **Code generation**: Prscope produces plans, not code. Execution is out of scope.
- **Multi-tenant hosting**: The server is designed for local or single-team use, not SaaS.
- **Model training**: Prscope consumes LLM APIs; it does not fine-tune or train models.
- **Real-time collaboration**: One user per session. Multi-tab is supported; multi-user is not.

## Related Documents

- [Architecture](../ARCHITECTURE.md) — layer rules and package map
- [Planning State Machine](./planning-state-machine.md) — transition contract
- [Agent Harness](./agent-harness.md) — runtime operational behavior
- [Core Beliefs](./CORE_BELIEFS.md) — coding principles for contributors
