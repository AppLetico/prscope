# Contributing

Thanks for contributing to Prscope.

## Development Setup

```bash
pip install -e ".[dev]"
```

Run checks before opening a PR:

```bash
ruff check .
pytest
```

## Performance Benchmark Policy (Required)

If your PR can affect planning behavior, latency, or output quality, benchmarking is required.

This includes changes in (or adjacent to):

- `prscope/planning/runtime/`
- `prscope/store.py`
- `prscope/memory.py`
- `prscope/scoring.py`
- model/tool orchestration and prompt behavior

Run benchmark suite:

```bash
prscope-benchmark --base-url http://127.0.0.1:8443 --repo <repo-name> --config-root <path-to-repo>
```

Benchmark artifacts:

- Current run: `benchmarks/results/history/run-<timestamp>.json`
- Best known baseline: `benchmarks/results/best_performance.json`
- Prompt suite: `benchmarks/prompts.json`

Use the same prompt suite when comparing to baseline. If you change prompts, explain why in the PR and provide before/after for both old and new suite.

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
- [ ] Benchmark run executed (if performance-sensitive)
- [ ] Benchmark artifact linked in PR
- [ ] Baseline comparison included
- [ ] Any benchmark prompt changes justified
