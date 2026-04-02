# Contributing

Thanks for contributing to Prscope.

## Pull Request Workflow

1. Create a focused branch from `main`.
2. Keep PRs small and scoped to one logical change when possible.
3. Open a PR early if you want design feedback before polishing.
4. Ensure all required CI checks pass before requesting final review.
5. Do not merge with unresolved review comments or failing checks.

## Development Setup

```bash
pip install -e ".[dev]"
# or
make dev
```

Run checks before opening a PR:

```bash
ruff check .
pytest
# or
make check
```

Frontend changes should also pass. From repo root:

```bash
cd src/prscope/web/frontend
npm ci
npm run lint
npm run build
```

Or run full CI parity (Python lint, format check, tests, plus frontend lint and build):

```bash
make ci
```

## CI Expectations

The repository uses GitHub Actions CI on pushes and PRs to `main`.

- **Python Checks**: dependency install, `ruff check .`, `pytest`, `scripts/check_doc_links.py`
- **Frontend Checks**: `npm ci`, `npm run lint`, `npm run build`

If your PR touches runtime, API, store, or frontend code, CI must be green before merge.

**Manual benchmark workflow:** [`.github/workflows/benchmark.yml`](.github/workflows/benchmark.yml) is **workflow_dispatch only**. It installs the package, starts the API, and runs `python -m prscope.benchmark --health-check-only`. Use it to validate the benchmark harness in a clean environment; optional API keys can be set as repository secrets if session create/delete steps need them.

## Performance Benchmark Policy (Required)

If your PR can affect planning behavior, latency, or output quality, benchmarking is required.

This includes changes in (or adjacent to):

- `src/prscope/planning/runtime/`
- `src/prscope/store.py`
- `src/prscope/memory.py`
- `src/prscope/scoring.py`
- model/tool orchestration and prompt behavior

**Harness tuning:** If you change compaction defaults (`planning.discovery_conversation_max_chars`, `critique_compress_*`, `critique_llm_summarize_*`), critic system prompts or rubric handling, or `planning.plan_rubric_floor`, treat that as performance-sensitive: run the benchmark suite and attach artifacts like any other planning change.

Run benchmark suite:

```bash
prscope-benchmark --base-url http://127.0.0.1:8443 --repo <repo-name> --config-root <path-to-repo>
```

Quick checks from the repo root (API must be up; use `make web-backend` on port **8420** if you match the URL below):

```bash
make benchmark-health   # API health only, no planning prompts
make benchmark-smoke    # full `benchmarks/prompts.json` suite against http://127.0.0.1:8420
```

Benchmark artifacts:

- Current run: `benchmarks/results/history/run-<timestamp>.json`
- Best known baseline: `benchmarks/results/best_performance.json`
- Prompt suite: `benchmarks/prompts.json`

Use the same prompt suite when comparing to baseline. If you change prompts, explain why in the PR and provide before/after for both old and new suite.

**When convergence “feels stuck”:** See [Agent Harness — Failure distribution](docs/agent-harness.md#failure-distribution-tuning-workflow). Aggregate DEBUG `convergence_gate` lines or run `python scripts/summarize_convergence_logs.py` on `~/.prscope/server.log` (or your log path).

## Logging convention

For **new or heavily edited** Python modules under `src/prscope/`, prefer **`logging`**: `logger = logging.getLogger(__name__)` and structured messages. Existing `loguru` usage in the web layer is fine to leave as-is; do not mass-convert logging in unrelated PRs.

## PR Requirements For Performance-Sensitive Changes

Include in your PR description:

- Path to the new benchmark artifact
- Comparison vs `benchmarks/results/best_performance.json`
- Explicit statement for:
  - fallback regression (must not increase)
  - `avg_time_to_plan_s` delta
  - `avg_quality_score` delta

## Recommended Acceptance Gates

Unless explicitly approved otherwise by maintainers:

- `fallback_runs` must not regress
- `avg_time_to_plan_s` should not regress by more than 10%
- `avg_quality_score` should not regress by more than 0.03

If a change intentionally trades speed for quality (or vice versa), call it out clearly and get maintainer approval in the PR.

## PR Checklist

- [ ] Code is tested and linted
- [ ] CI is green for Python + frontend checks (includes `make docs-linkcheck` parity via `scripts/check_doc_links.py`)
- [ ] Benchmark run executed (if performance-sensitive)
- [ ] Benchmark artifact linked in PR
- [ ] Baseline comparison included
- [ ] Any benchmark prompt changes justified
- [ ] [`docs/QUALITY_SCORE.md`](docs/QUALITY_SCORE.md) updated if a domain grade or known gap changed materially
