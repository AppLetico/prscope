# CI check

Run lint and type checks (and optionally full CI) on the repo; fix any reported issues.

## Quick (lint + typecheck only, no tests)

1. Run: `make lint-typecheck`
   - Backend: ruff check, ruff format --check
   - Frontend: npm run lint (eslint), tsc --noEmit (TypeScript)
2. If anything fails, fix the reported errors (format, lint rules, or type errors) and re-run until the command succeeds.

## Full CI parity (satisfies all CI checks)

To match what CI runs and ensure the PR will pass:

1. Run: `make ci`
   - Backend: ruff check, ruff format --check, pytest
   - Frontend: npm run lint, npm run build (eslint + tsc + vite build)
2. Fix any failures and re-run until `make ci` succeeds.
