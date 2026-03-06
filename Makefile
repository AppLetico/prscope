# Prscope Makefile
# Common development tasks

# Use python3 on macOS, python elsewhere
PYTHON := $(shell command -v python3 2>/dev/null || echo python)

.PHONY: install dev test lint format check lint-typecheck ci clean help web web-backend web-frontend web-kill sessions-delete reset

# Default target
help:
	@echo "Prscope Development Commands"
	@echo ""
	@echo "  make install      Install package (editable)"
	@echo "  make dev          Install with dev dependencies"
	@echo "  make test         Run unit tests"
	@echo "  make lint         Run linter (ruff check)"
	@echo "  make format       Format code (ruff format)"
	@echo "  make check        Run lint + format check + tests"
	@echo "  make lint-typecheck  Run lint + format check + typecheck (backend + frontend, no tests)"
	@echo "  make ci            Same as CI: lint + format check + tests + frontend lint + frontend build"
	@echo "  make clean         Remove build artifacts"
	@echo ""
	@echo "Prscope Workflow Commands"
	@echo ""
	@echo "  make run          Incremental: profile -> sync -> evaluate (planner seed prep)"
	@echo "  make run-full     Full: profile -> sync-full -> evaluate-all (planner seed prep)"
	@echo "  make sync         Sync PRs (incremental, only new since last sync)"
	@echo "  make sync-full    Sync PRs (full, ignores watermark)"
	@echo "  make evaluate     Evaluate PRs (one batch)"
	@echo "  make evaluate-all Evaluate all pending PRs (loops until done)"
	@echo "  make plan-chat    Start interactive discovery mode"
	@echo "  make digest       Show PR digest"
	@echo ""
	@echo "Web app (run in two terminals)"
	@echo ""
	@echo "  make web-backend   Start API server (port 8420, loads .env)"
	@echo "  make web-frontend  Start Vite dev server (port 5173)"
	@echo "  make web-kill      Kill processes on 8420 and 5173 (free the ports)"
	@echo "  make sessions-delete  Delete all planning sessions and session cache (prompts for confirmation)"
	@echo "  make reset            Blank slate: delete sessions + clear all planning cache (prompts; use -y to skip)"
	@echo "  make web           Show instructions to run backend + frontend"
	@echo ""

# Install package in editable mode
install:
	$(PYTHON) -m pip install -e .

# Install with dev dependencies
dev:
	$(PYTHON) -m pip install -e ".[dev]"

# Run tests
test:
	pytest -v

# Run tests with coverage
test-cov:
	pytest --cov=prscope --cov-report=term-missing

# Lint code
lint:
	ruff check .

# Format code
format:
	ruff format .

# Check all (lint, format check, tests)
check: lint
	ruff format --check .
	pytest -q

# Lint and typecheck only (no tests): backend ruff + frontend eslint + tsc
lint-typecheck: lint
	ruff format --check .
	@cd src/prscope/web/frontend && npm run lint && npx tsc --noEmit

# Full CI parity: what .github/workflows/ci.yml runs (lint, format check, tests, frontend lint + build)
ci: lint
	ruff format --check .
	pytest -q
	@cd src/prscope/web/frontend && npm run lint && npm run build

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Initialize prscope in current directory
init:
	$(PYTHON) -m prscope.cli init

# Profile local codebase
profile:
	$(PYTHON) -m prscope.cli profile

# Sync upstream PRs (incremental - only new since last sync)
sync:
	$(PYTHON) -m prscope.cli sync

# Sync all PRs (full - ignores watermark, uses since window)
sync-full:
	$(PYTHON) -m prscope.cli sync --full

# Evaluate PRs (one batch)
evaluate:
	$(PYTHON) -m prscope.cli evaluate

# Evaluate all pending PRs (loops until done)
evaluate-all:
	@while $(PYTHON) -m prscope.cli evaluate 2>&1 | grep -q "Remaining"; do \
		echo "Processing next batch..."; \
	done
	@echo "All PRs evaluated"

# Start planning discovery
plan-chat:
	$(PYTHON) -m prscope.cli plan chat

# Show digest
digest:
	$(PYTHON) -m prscope.cli digest

# Incremental workflow: profile -> sync (incremental) -> evaluate (one batch)
run: profile sync evaluate
	@echo "Prscope workflow complete"

# Full workflow: profile -> sync (full) -> evaluate (all)
run-full: profile sync-full evaluate-all
	@echo "Prscope full workflow complete"

# --- Web app ---

# Show how to run the web app
web:
	@echo "Run the app in two terminals:"
	@echo "  Terminal 1: make web-backend"
	@echo "  Terminal 2: make web-frontend"
	@echo "Then open: http://localhost:5173/new?repo=prscope"
	@echo ""

# Start API server (loads .env from repo root). Use --reload for dev (auto-restart on file changes).
web-backend:
	@bash -c 'set -a; [ -f .env ] && . ./.env; set +a; exec uvicorn prscope.web.api:create_app --factory --host 127.0.0.1 --port 8420 --reload'

# Start Vite dev server (frontend)
web-frontend:
	cd src/prscope/web/frontend && npm run dev

# Kill processes on backend (8420) and frontend (5173) ports
web-kill:
	@-lsof -ti :8420 | xargs kill 2>/dev/null || true
	@-lsof -ti :5173 | xargs kill 2>/dev/null || true
	@echo "Killed processes on 8420 and 5173 (if any)."

# Delete all planning sessions (uses prscope plan delete --all; also clears session cache)
sessions-delete:
	$(PYTHON) -m prscope.cli plan delete --all

# Blank slate for testing: delete all sessions and clear planning cache (memory, rounds, audit, tool-results)
reset:
	$(PYTHON) -m prscope.cli plan reset --yes
