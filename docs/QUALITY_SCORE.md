# Quality Scores

Per-domain quality grades for the prscope codebase. Updated as gaps are identified and resolved.

Grading: **A** = solid, well-tested, documented | **B** = functional, minor gaps | **C** = works but needs attention | **D** = known significant issues

## Domain Grades

| Domain | Grade | Notes |
|---|---|---|
| **Foundation** (`config`, `pricing`, `model_catalog`, `profile`, `semantic`) | B | Well-tested. `config.py` is stable. `semantic.py` could use more edge-case coverage. |
| **Storage** (`store.py`) | A | Thorough test coverage. Protected field guards. Schema is stable. |
| **Planning Core** (`planning/core.py`, `planning/executor.py`) | A | State machine is well-specified with invariant enforcement. Executor has durable command log. |
| **Planning Runtime** (`planning/runtime/*`) | A | Runtime has `signals -> reasoners -> orchestration`, graph-backed follow-up state, and **Tier-A harness gates**: `plan_rubric` min-floor, closed `blocking_categories`, `acceptance_criteria` + plan-evidence checks, no `stalled_refinement` escape, convergence postcondition (`acceptance_contract.py`). Ablate gates deliberately when tuning models (see `docs/agent-harness.md`). Some orchestration logic lives under `orchestration_support/*` (e.g. adversarial compaction); `orchestration.py` remains large. |
| **Planning Scanners** (`planning/scanners/*`) | B | `grep` backend is reliable and heavily tested. `repomap` / `repomix` have registry and fallback tests (`tests/test_scanner_registry.py`); deep behavior vs. fixtures is still thinner than `grep`. |
| **Memory** (`memory.py`) | B | Rebuild logic and skill loading are tested. Manifesto parsing is solid. Memory block summarization depends on LLM availability. |
| **Scoring** (`scoring.py`) | B | Rule-based scoring with good unit tests. Feature config is stable. |
| **GitHub Integration** (`github.py`) | B | PR sync tested. Rate limiting and pagination could be more robust. |
| **Web API** (`web/api.py`) | B | Command model is well-tested. SSE contract is documented. Module is large (~1.9k lines); some wrapper endpoints have lighter coverage. |
| **Web Frontend** (`web/frontend/`) | B | Vitest: `PlanningView`, `ChatPanel`, `ActionBar`, `PlanPanel`, `impactView`, `decisionGraphRender`. **Tier 1 Playwright** smoke (health, `/api/sessions`, `/` and `/new`) in CI. No LLM-backed e2e in default CI. Timeline architecture is reducer-based. |
| **Benchmark** (`benchmark.py`) | B | HTTP-based, repeatable. Historical tracking works. **Default PR CI does not** run full prompt suites or LLM spend. **Manual** GitHub workflow `.github/workflows/benchmark.yml` (`workflow_dispatch`) runs health-style checks with a started API; pre-release harness tuning still relies on `CONTRIBUTING.md` + local `make benchmark-smoke` / `benchmark-health`. |
| **Documentation** | A | Runtime docs cover reasoning, decision-graph artifacts, and frontend graph rendering. **Internal** relative links in `docs/*.md` and key root markdown are validated by `scripts/check_doc_links.py` (`make check` / CI). No automated external link crawl or semantic doc freshness checks. |
| **CI / Linting** | B | Ruff + pytest + doc link script + eslint + frontend build; Playwright Tier 1 in `make ci`. Structural import rules (`test_architecture.py`). No custom lint rules with agent-friendly remediation messages yet. |

## Known Gaps

### High Priority

- [x] Component unit tests for `ActionBar` and `PlanPanel` (`src/prscope/web/frontend/src/components/*.test.ts`).
- [x] `ToolCallStream` — Vitest + Testing Library (`src/prscope/web/frontend/src/components/ToolCallStream.test.tsx`); jsdom test env in `vite.config.ts`.
- [x] Tier 1 Playwright smoke in CI (`src/prscope/web/frontend/e2e/`).
- [ ] **Full-stack integration** (create session → draft → refinement round) with real API keys — optional / manual; main blind spot for UI ↔ API ↔ provider regressions.
- [ ] `orchestration.py` is still the largest runtime module (~1.2k+ lines). Continue incremental splits into `orchestration_support/*` as seams clear. **Same for** `web/api.py` (thin route modules per area when touching files).

### Medium Priority

- [ ] **Benchmark regression:** not gated on every PR (by design, to avoid flaky LLM cost). Mitigations: manual workflow, local policy in `CONTRIBUTING.md`, record runs when changing harness defaults or critic prompts.
- [ ] Scanner backends (`repomap`, `repomix`): registry and fallback coverage exists; end-to-end scanning on small fixtures is still lighter than `grep`.
- [ ] Logging style varies (`loguru` vs stdlib `logging`); `CONTRIBUTING.md` asks stdlib for new/edited Python under `src/prscope/` — no mass migration.

### Low Priority

- [ ] `semantic.py` edge cases (empty repos, binary files).
- [ ] Memory block summarization fallback paths are tested but the happy path depends on live LLM calls.
- [ ] Frontend decision-graph UX is projection-only; there is no direct decision-node editing surface yet.

## Release / integration checklist

Use before a release or when merging large harness/UI changes:

- [ ] Run `make check` (local parity with Python CI: lint, tests, doc links).
- [ ] Run `make ci` if you changed frontend or want Playwright Tier 1 smoke.
- [ ] If you changed compaction, critic prompts, rubric floors, or convergence behavior: run benchmark smoke / health per `CONTRIBUTING.md` and note results (or link a run).
- [ ] If behavior or user-facing expectations shifted: update this file (`QUALITY_SCORE.md`) in the same PR.
- [ ] Full-stack path with keys remains optional: run once manually if the change touches session creation, SSE, or provider routing.

## Updating This Document

When you fix a gap, update the grade and move the item to a "resolved" section or delete it. When you discover a new gap, add it to the appropriate priority bucket.

**Release / merge reminder:** If a PR changes planning behavior, web API contracts, or frontend architecture in a way that would move a grade or resolve a “Known gap” bullet, update this file in the same PR.

This document is intended to be maintained continuously, not written once and forgotten.
