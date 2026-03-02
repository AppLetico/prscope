# Prompt Benchmarking

Repeatable startup benchmark for planning speed + output quality.

## Run

```bash
prscope-benchmark --base-url http://127.0.0.1:8443 --repo prscope --config-root /Users/jasongelinas/workspace/prscope/benchmarks/configs/prscope-self
```

Equivalent module invocation:

```bash
python -m prscope.benchmark --base-url http://127.0.0.1:8443 --repo prscope --config-root /Users/jasongelinas/workspace/prscope/benchmarks/configs/prscope-self
```

## Inputs

- Prompt suite: `benchmarks/prompts.json` (JSON array of strings)
- Optional override: `--prompts-file /path/to/prompts.json`

## Outputs

- Per-run history: `benchmarks/results/history/run-<timestamp>.json`
- Per-run diagnostics log: `benchmarks/results/history/run-<timestamp>.log`
- Best-known record: `benchmarks/results/best_performance.json`

Timing fields now include both perspectives:

- `time_to_plan_s`: client-detected time until plan is first visible via API polling
- `server_initial_draft_elapsed_s`: server-reported initial draft runtime (source-of-truth on generation time)
- `client_detect_gap_s`: `time_to_plan_s - server_initial_draft_elapsed_s` when both are present

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
- `--poll-unhealthy-window-seconds <n>` (default `20`)
- `--max-consecutive-create-failures <n>` (default `1`)
- `--stop-on-first-problem` / `--no-stop-on-first-problem` (default stop early)
- `--health-check-only` (cheap API readiness check, no requirements drafting)
- `--max-suite-seconds <n>` (default `900`, set `0` to disable)

The diagnostics log includes per-prompt lifecycle events, poll latencies, poll timeouts, and timeout outcomes.

## E2E Model Selection Check

For a quick web E2E validation after benchmark changes:

1. Start backend + frontend.
2. In New Session, choose explicit `Author model` and `Critic model`.
3. Run at least one discovery/critique interaction.
4. Confirm `token_usage` events show the selected `model` value for both author and critic calls.
