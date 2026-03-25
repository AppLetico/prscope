# Production readiness checklist

Prscope is built as a **local-first** developer tool. This checklist is a go/no-go aid if you run it beyond a single machine or share access.

## Bind and network exposure

- [ ] API listens on **127.0.0.1** (default) unless you have a deliberate reason not to.
- [ ] Binding to **0.0.0.0** or **::** requires `PRSCOPE_ALLOW_PUBLIC_BIND=1` when using `prscope.web.server.run_server` or CLI-spawned uvicorn; direct `uvicorn` invocations do not enforce this — set the env var yourself if needed.
- [ ] If the API is reachable from other hosts, assume **full session control** (no authentication today).

## Authentication and authorization

- [ ] There is **no API auth**; session IDs are opaque capabilities.
- [ ] Do not expose the raw FastAPI app to the public internet without a reverse proxy, TLS, and your own auth layer.

## Secrets

- [ ] API keys live in **`.env`** (never committed). See `AGENTS.md` / project README for provider variables.
- [ ] CI and shared hosts must inject secrets via the platform’s secret store, not files in the repo.

## Data and SQLite

- [ ] Default DB path is under the prscope user directory (`get_prscope_dir()` / `prscope.db`). Plan **backup and restore** if sessions matter.
- [ ] Single global DB today; repo-scoped routes use the same file with a `repo_name` column.

## Testing and regression

- [ ] **CI:** Python lint/tests + frontend lint/build; **Tier 1 Playwright** smoke (health, SPA shell, `/api/sessions`) when the e2e job is enabled.
- [ ] **Full LLM e2e** (create session, draft, round) is optional/manual — requires keys and is not required on every PR.

## Supported deployment shapes

| Mode | Notes |
|------|--------|
| Local dev | Vite `:5173` + API `:8420` with CORS for localhost. |
| Static on API | Built frontend in `web/static`; SPA fallback from `create_server_app`. |
| Split / tunnel | Treat tunneled URL as **untrusted** unless you add auth in front. |

## Harness tuning (operational)

- [ ] For critic/matcher calibration, collect **failure distribution** from DEBUG `convergence_gate` logs (`failed_gate`, `failure_delta`) — see `docs/agent-harness.md` and `scripts/summarize_convergence_logs.py`.

## Documentation map

- Runtime and API behavior: `docs/agent-harness.md`
- State machine: `docs/planning-state-machine.md`
- Repo conventions: `AGENTS.md`, `CONTRIBUTING.md`
