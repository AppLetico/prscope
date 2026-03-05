# Quality Scores

Per-domain quality grades for the prscope codebase. Updated as gaps are identified and resolved.

Grading: **A** = solid, well-tested, documented | **B** = functional, minor gaps | **C** = works but needs attention | **D** = known significant issues

## Domain Grades

| Domain | Grade | Notes |
|---|---|---|
| **Foundation** (`config`, `pricing`, `model_catalog`, `profile`, `semantic`) | B | Well-tested. `config.py` is stable. `semantic.py` could use more edge-case coverage. |
| **Storage** (`store.py`) | A | Thorough test coverage. Protected field guards. Schema is stable. |
| **Planning Core** (`planning/core.py`, `planning/executor.py`) | A | State machine is well-specified with invariant enforcement. Executor has durable command log. |
| **Planning Runtime** (`planning/runtime/*`) | A- | Runtime is now split into `pipeline/`, `context/`, `review/`, and `events/` subpackages with explicit stage dependency injection and author-owned initial draft flow. `orchestration.py` remains large but risk is reduced. |
| **Planning Scanners** (`planning/scanners/*`) | B | grep backend is reliable. repomap/repomix backends are less exercised. |
| **Memory** (`memory.py`) | B | Rebuild logic and skill loading are tested. Manifesto parsing is solid. Memory block summarization depends on LLM availability. |
| **Scoring** (`scoring.py`) | B | Rule-based scoring with good unit tests. Feature config is stable. |
| **GitHub Integration** (`github.py`) | B | PR sync tested. Rate limiting and pagination could be more robust. |
| **Web API** (`web/api.py`) | B | Command model is well-tested. SSE contract is documented. Some wrapper endpoints have lighter coverage. |
| **Web Frontend** (`web/frontend/`) | B | 2 test files with 12 tests covering timeline reducer, buildTimeline, upsertToolCall, and hasRunningToolCalls. No integration or e2e tests. Timeline architecture is well-structured with reducer-based state management. |
| **Benchmark** (`benchmark.py`) | B | HTTP-based, repeatable. Historical tracking works. No automated regression gate in CI yet. |
| **Documentation** | B | Runtime docs are strong (`agent-harness.md`, `planning-state-machine.md`). Architecture and design docs are new. |
| **CI / Linting** | B | Standard ruff + eslint. Structural import lints are new (`test_architecture.py`). No custom lint rules with agent-friendly remediation messages yet. |

## Known Gaps

### High Priority

- [ ] Frontend test coverage: `PlanningView.test.ts` (10 tests: reducer, buildTimeline, upsertToolCall) and `ChatPanel.test.ts` (2 tests: hasRunningToolCalls) exist. No component unit tests for `ActionBar`, `PlanPanel`, `ToolCallStream`.
- [ ] No e2e test harness for the web UI (no Playwright/Cypress).
- [ ] `orchestration.py` is still the largest runtime module. Continue migrating formatting/prompt helper logic into specialized modules to reduce coordinator size further.

### Medium Priority

- [ ] Benchmark results are not gated in CI — regressions are caught manually via `CONTRIBUTING.md` policy.
- [ ] Scanner backends (`repomap`, `repomix`) have limited test coverage vs. `grep`.
- [ ] No structured logging enforcement (log format varies between `loguru` and `logging`).

### Low Priority

- [ ] `semantic.py` edge cases (empty repos, binary files).
- [ ] Memory block summarization fallback paths are tested but the happy path depends on live LLM calls.
- [ ] Stale doc detection is manual — no automated doc-gardening process yet.

## Updating This Document

When you fix a gap, update the grade and move the item to a "resolved" section or delete it. When you discover a new gap, add it to the appropriate priority bucket.

This document is intended to be maintained continuously, not written once and forgotten.
