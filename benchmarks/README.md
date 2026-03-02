# Prompt Benchmarking

Repeatable startup benchmark for planning speed + output quality.

## Run

```bash
prscope-benchmark --base-url http://127.0.0.1:8443 --repo clasper-core --config-root ~/workspace/clasper-core
```

Equivalent module invocation:

```bash
python -m prscope.benchmark --base-url http://127.0.0.1:8443 --repo clasper-core --config-root ~/workspace/clasper-core
```

## Inputs

- Prompt suite: `benchmarks/prompts.json` (JSON array of strings)
- Optional override: `--prompts-file /path/to/prompts.json`

## Outputs

- Per-run history: `benchmarks/results/history/run-<timestamp>.json`
- Per-run diagnostics log: `benchmarks/results/history/run-<timestamp>.log`
- Best-known record: `benchmarks/results/best_performance.json`

The best-known file is automatically replaced only when a run is better under this ranking:

1. Fewer fallback runs
2. Faster average time to first plan
3. Higher average quality score

## Quality Heuristic

Quality is a normalized [0, 1] score from:

- Required section coverage
- Grounding signal (file-path references)
- Generic placeholder penalties
- Fallback penalty

## Debugging Long Runs

Use these controls to avoid runaway suites and improve diagnosis:

- `--create-timeout-seconds <n>` (default `15`)
- `--poll-timeout-seconds <n>` (default `60`)
- `--poll-request-timeout-seconds <n>` (default `5`)
- `--max-consecutive-poll-timeouts <n>` (default `3`)
- `--max-consecutive-create-failures <n>` (default `1`)
- `--stop-on-first-problem` / `--no-stop-on-first-problem` (default stop early)
- `--health-check-only` (cheap API readiness check, no requirements drafting)
- `--max-suite-seconds <n>` (default `900`, set `0` to disable)

The diagnostics log includes per-prompt lifecycle events, poll latencies, poll timeouts, and timeout outcomes.
