# Quality Scores

Per-domain quality grades for the prscope codebase. Updated as gaps are identified and resolved.

Grading: **A** = solid, well-tested, documented | **B** = functional, minor gaps | **C** = works but needs attention | **D** = known significant issues

## Domain Grades

| Domain | Grade | Notes |
|---|---|---|
| **Foundation** (`config`, `pricing`, `model_catalog`, `profile`, `semantic`) | B | Well-tested. `config.py` is stable. `semantic.py` could use more edge-case coverage. |
| **Storage** (`store.py`) | A | Thorough test coverage. Protected field guards. Schema is stable. |
| **Planning Core** (`planning/core.py`, `planning/executor.py`) | A | State machine is well-specified with invariant enforcement. Executor has durable command log. |
| **Planning Runtime** (`planning/runtime/*`) | B | Orchestration is complex but well-documented. Discovery, author, critic agents have good test coverage. `orchestration.py` is the largest file and carries the most risk. |
| **Planning Scanners** (`planning/scanners/*`) | B | grep backend is reliable. repomap/repomix backends are less exercised. |
| **Memory** (`memory.py`) | B | Rebuild logic and skill loading are tested. Manifesto parsing is solid. Memory block summarization depends on LLM availability. |
| **Scoring** (`scoring.py`) | B | Rule-based scoring with good unit tests. Feature config is stable. |
| **GitHub Integration** (`github.py`) | B | PR sync tested. Rate limiting and pagination could be more robust. |
| **Web API** (`web/api.py`) | B | Command model is well-tested. SSE contract is documented. Some wrapper endpoints have lighter coverage. |
| **Web Frontend** (`web/frontend/`) | C | Functional but has limited test coverage (2 test files). No integration or e2e tests. |
| **Benchmark** (`benchmark.py`) | B | HTTP-based, repeatable. Historical tracking works. No automated regression gate in CI yet. |
| **Documentation** | B | Runtime docs are strong (`agent-harness.md`, `planning-state-machine.md`). Architecture and design docs are new. |
| **CI / Linting** | B | Standard ruff + eslint. Structural import lints are new (`test_architecture.py`). No custom lint rules with agent-friendly remediation messages yet. |

## Known Gaps

### High Priority

- [ ] Frontend test coverage: only `PlanningView.test.ts` and `ChatPanel.test.ts` exist. No component unit tests for `ActionBar`, `PlanPanel`, `ToolCallStream`.
- [ ] No e2e test harness for the web UI (no Playwright/Cypress).
- [ ] `orchestration.py` is ~large and carries high coupling risk. Consider extracting phase handlers.

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
