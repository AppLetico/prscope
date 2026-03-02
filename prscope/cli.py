"""
Prscope CLI - Planning-first workflow with upstream PR intelligence.

Commands:
    init      - Initialize Prscope in current repository
    profile   - Scan and profile local codebase
    upstream  - Upstream sync/evaluate/history utilities
    plan      - Interactive planning (PRD + RFC generation)
    repos     - Repo profile management
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

import click
from dotenv import load_dotenv

# Load .env file from current directory or repo root
load_dotenv()  # Loads from current directory
load_dotenv(Path.cwd() / ".env")  # Explicit current dir
# Also try repo root
from .config import get_repo_root
try:
    load_dotenv(get_repo_root() / ".env")
except Exception:
    pass

from . import __version__
from .config import (
    PrscopeConfig,
    get_repo_root,
    ensure_prscope_dir,
)
from .store import Store
from .profile import build_profile, hash_profile
from .github import GitHubClient, sync_repo_prs, GitHubAPIError
from .scoring import evaluate_pr
from .planning.runtime import PlanningRuntime
from .web.server import DEFAULT_HOST, DEFAULT_PORT, ensure_server_running


# Sample configuration files
SAMPLE_CONFIG = """\
# Prscope Configuration
# See: https://github.com/prscope/prscope

# Local repository to profile and compare against upstream PRs
# Can be absolute path or relative to this config file
local_repo: .  # Current directory (default)
# local_repo: ~/workspace/my-project
# local_repo: /absolute/path/to/repo

# Upstream repositories to monitor
upstream:
  - repo: openclaw/openclaw
    filters:
      # Override sync defaults per-repo (optional)
      # state: merged  # merged (default), open, closed, all
      # labels: [security, agents]  # Optional label filter

# Sync settings (how to fetch PRs from upstream)
sync:
  state: merged       # merged (default), open, closed, all
  max_prs: 100        # Maximum PRs to fetch per repo
  fetch_files: true   # Fetch file changes (needed for path matching)
  since: 90d          # Initial date window (90d = 90 days, 6m = 6 months, or ISO date)
  incremental: true   # Only fetch new PRs after initial sync
  eval_batch_size: 25 # Max PRs to evaluate per run (controls LLM costs)

# Scoring thresholds
scoring:
  min_rule_score: 0.3    # Minimum score to consider (0-1)
  min_final_score: 0.5   # Threshold for "relevant" decision (0-1)
  keyword_weight: 0.4    # Weight for keyword matching
  path_weight: 0.6       # Weight for path matching

# Upstream PR semantic evaluation (RECOMMENDED for high-quality, noise-free results)
# Without this, only rule-based keyword/path matching is used (more noise)
# API keys are read from environment (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
# See: https://docs.litellm.ai/docs/providers
upstream_eval:
  enabled: true          # Enable for production use
  model: gpt-4o          # LiteLLM model string
  # model: claude-3-opus # Anthropic
  # model: gemini-pro    # Google
  # model: ollama/llama2 # Local Ollama
  temperature: 0.2       # Lower = more consistent decisions
  max_tokens: 3000

# Planning mode configuration (PRD + RFC generation)
planning:
  author_model: gpt-4o
  critic_model: claude-3-5-sonnet-20241022
  max_adversarial_rounds: 10
  convergence_threshold: 0.05
  output_dir: ./plans
  require_verified_file_references: false

# Optional multi-repo profiles
# repos:
#   my-repo:
#     path: ~/workspace/my-repo
#     upstream:
#       - repo: owner/upstream-repo
"""

SAMPLE_FEATURES = """\
# Prscope Feature Definitions
# Define features to match against upstream PRs

features:
  # Security-related changes
  security:
    keywords:
      - security
      - auth
      - authentication
      - authorization
      - jwt
      - tls
      - injection
      - vulnerability
      - cve
    paths:
      - "**/security/**"
      - "**/auth/**"
      - "**/*auth*.py"
      - "**/*auth*.ts"

  # Streaming/real-time features
  streaming:
    keywords:
      - stream
      - streaming
      - sse
      - websocket
      - real-time
      - realtime
      - chunk
      - flush
    paths:
      - "**/streaming*"
      - "**/stream*"
      - "**/*sse*"

  # Tool/function calling
  tools:
    keywords:
      - tool
      - function
      - function_call
      - tool_call
      - proxy
    paths:
      - "**/tools/**"
      - "**/*tool*"
      - "**/*proxy*"

  # API changes
  api:
    keywords:
      - api
      - endpoint
      - route
      - handler
      - rest
      - graphql
    paths:
      - "**/api/**"
      - "**/routes/**"
      - "**/handlers/**"
      - "**/routers/**"
"""


@click.group()
@click.version_option(version=__version__)
def main():
    """Prscope - Planning-first tool with upstream PR intelligence."""
    pass


def _warn_legacy_command(old: str, new: str) -> None:
    click.echo(
        f"⚠️  Deprecated command `{old}`. Please use `{new}` instead.",
        err=True,
    )


@main.group(name="upstream")
def upstream_group() -> None:
    """Upstream PR ingestion and evaluation commands."""


@main.command()
@click.option("--force", is_flag=True, help="Overwrite existing configuration")
def init(force: bool):
    """Initialize Prscope in the current repository."""
    repo_root = get_repo_root()
    click.echo(f"Initializing Prscope in: {repo_root}")
    
    # Create .prscope directory
    prscope_dir = ensure_prscope_dir(repo_root)
    click.echo(f"  Created: {prscope_dir}")
    
    # Initialize database
    store = Store()
    click.echo(f"  Database: {store.db_path}")
    
    # Create sample config files
    config_path = repo_root / "prscope.yml"
    if not config_path.exists() or force:
        config_path.write_text(SAMPLE_CONFIG)
        click.echo(f"  Created: {config_path}")
    else:
        click.echo(f"  Skipped: {config_path} (already exists)")
    
    features_path = repo_root / "prscope.features.yml"
    if not features_path.exists() or force:
        features_path.write_text(SAMPLE_FEATURES)
        click.echo(f"  Created: {features_path}")
    else:
        click.echo(f"  Skipped: {features_path} (already exists)")
    
    # Add to .gitignore
    gitignore_path = repo_root / ".gitignore"
    gitignore_entry = "\n# Prscope\n.prscope/\n.env\n"
    if gitignore_path.exists():
        content = gitignore_path.read_text()
        if ".prscope" not in content:
            with open(gitignore_path, "a") as f:
                f.write(gitignore_entry)
            click.echo(f"  Updated: {gitignore_path}")
    else:
        gitignore_path.write_text(gitignore_entry)
        click.echo(f"  Created: {gitignore_path}")
    
    # Create env templates if not exists
    env_example_path = repo_root / "env.example"
    env_sample_path = repo_root / "env.sample"
    if not env_example_path.exists() or force:
        # Copy from package
        import importlib.resources
        try:
            # Try to read from installed package
            pkg_env_example = Path(__file__).parent.parent / "env.example"
            if pkg_env_example.exists():
                env_example_path.write_text(pkg_env_example.read_text())
                click.echo(f"  Created: {env_example_path}")
        except Exception:
            pass
    if not env_sample_path.exists() or force:
        pkg_env_sample = Path(__file__).parent.parent / "env.sample"
        if pkg_env_sample.exists():
            env_sample_path.write_text(pkg_env_sample.read_text())
            click.echo(f"  Created: {env_sample_path}")
    
    click.echo("\nPrscope initialized! Next steps:")
    click.echo("  1. Edit prscope.yml to add upstream repositories")
    click.echo("  2. Edit prscope.features.yml to define features")
    click.echo("  3. Set GITHUB_TOKEN environment variable")
    click.echo("  4. Run: prscope profile")
    click.echo("  5. Run: prscope sync")
    click.echo("  6. Run: prscope evaluate")
    click.echo("  7. Start planning: prscope plan chat")


@main.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def profile(as_json: bool):
    """Scan and profile the local codebase."""
    config = PrscopeConfig.load(get_repo_root())
    repo_root = config.get_local_repo_path()
    
    if not as_json:
        click.echo(f"Profiling repository: {repo_root}")
    
    # Build profile
    profile_data = build_profile(repo_root)
    profile_sha = hash_profile(profile_data)
    
    # Save to store
    store = Store()
    store.save_profile(
        repo_root=str(repo_root),
        profile_sha=profile_sha,
        profile_json=json.dumps(profile_data),
    )
    
    if as_json:
        click.echo(json.dumps({
            "profile_sha": profile_sha,
            "git_sha": profile_data.get("git_sha"),
            "total_files": profile_data["file_tree"]["total_files"],
            "extensions": profile_data["file_tree"]["extensions"],
            "import_stats": profile_data["import_stats"],
        }, indent=2))
    else:
        click.echo(f"\nProfile saved:")
        click.echo(f"  SHA: {profile_sha}")
        click.echo(f"  Git HEAD: {profile_data.get('git_sha', 'unknown')}")
        click.echo(f"  Total files: {profile_data['file_tree']['total_files']}")
        
        ext_counts = profile_data["file_tree"]["extensions"]
        top_exts = sorted(ext_counts.items(), key=lambda x: -x[1])[:5]
        if top_exts:
            click.echo(f"  Top extensions: {', '.join(f'{e}({c})' for e, c in top_exts)}")
        
        stats = profile_data["import_stats"]
        click.echo(f"  Files analyzed: {stats['files_analyzed']}")
        click.echo(f"  Python imports: {stats['python_imports']}")
        click.echo(f"  JS imports: {stats['js_imports']}")


@main.command(hidden=True)
@click.option("--repo", help="Sync specific repository (owner/repo)")
@click.option("--state", default=None, help="PR state filter (merged, open, closed, all)")
@click.option("--max-prs", default=None, type=int, help="Maximum PRs to fetch")
@click.option("--since", default=None, help="Only fetch PRs after date (ISO date or 90d/6m/1y)")
@click.option("--full", is_flag=True, help="Full sync (ignore incremental watermark)")
@click.option("--no-files", is_flag=True, help="Skip fetching file lists")
def sync(
    repo: str | None,
    state: str | None,
    max_prs: int | None,
    since: str | None,
    full: bool,
    no_files: bool,
    warn_legacy: bool = True,
):
    """Fetch PRs from upstream repositories.
    
    By default, uses incremental sync (only PRs newer than last sync).
    First sync uses --since window (default: 90 days).
    
    Examples:
    
        prscope sync                    # Incremental (new PRs only)
        prscope sync --since 30d        # Last 30 days
        prscope sync --since 2024-01-01 # Since specific date
        prscope sync --full             # Ignore watermark, use --since window
    """
    if warn_legacy:
        _warn_legacy_command("prscope sync", "prscope upstream sync")

    repo_root = get_repo_root()
    config = PrscopeConfig.load(repo_root)
    
    if not config.upstream:
        click.echo("No upstream repositories configured.")
        click.echo("Add repositories to prscope.yml")
        sys.exit(1)
    
    # Filter repos if specified
    repos_to_sync = config.upstream
    if repo:
        repos_to_sync = [r for r in repos_to_sync if r.full_name == repo]
        if not repos_to_sync:
            click.echo(f"Repository not found in config: {repo}")
            sys.exit(1)
    
    # Initialize GitHub client
    client = GitHubClient()
    store = Store()
    
    # Use config defaults, allow CLI overrides
    default_state = state or config.sync.state
    default_max_prs = max_prs or config.sync.max_prs
    default_since = since or config.sync.since
    default_incremental = config.sync.incremental and not full
    default_fetch_files = config.sync.fetch_files and not no_files
    
    click.echo(f"Sync settings: state={default_state}, max_prs={default_max_prs}, since={default_since}")
    if default_incremental:
        click.echo("  Mode: incremental (only new PRs since last sync)")
    else:
        click.echo("  Mode: full (using date window)")
    
    total_new = 0
    total_updated = 0
    total_skipped = 0
    
    for upstream in repos_to_sync:
        click.echo(f"\n{'─' * 60}")
        click.echo(f"📦 Syncing: {upstream.full_name}")
        click.echo(f"{'─' * 60}")
        
        # Per-repo filters can override defaults
        pr_state = upstream.filters.get("state", default_state)
        
        # Progress callback for real-time updates
        last_stage = [None]
        def progress(stage: str, current: int, total: int, message: str):
            if stage == "fetch" and current == 0:
                click.echo(f"  ⏳ {message}")
            elif stage == "fetch" and current > 0:
                click.echo(f"  ✓ {message}")
            elif stage == "process":
                # Show progress every 5 PRs or for small batches
                if total <= 10 or current % 5 == 0 or current == total:
                    pct = int(current / total * 100) if total > 0 else 0
                    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                    click.echo(f"\r  [{bar}] {current}/{total} PRs processed", nl=False)
                    if current == total:
                        click.echo()  # Newline at end
            elif stage == "done":
                pass  # Handled below
        
        try:
            new_count, updated_count, skipped_count = sync_repo_prs(
                client=client,
                store=store,
                repo_name=upstream.full_name,
                state=pr_state,
                max_prs=default_max_prs,
                fetch_files=default_fetch_files,
                since=default_since,
                incremental=default_incremental,
                progress_callback=progress,
            )
            
            click.echo(f"\n  ✅ Done: {new_count} new, {updated_count} updated, {skipped_count} unchanged")
            
            total_new += new_count
            total_updated += updated_count
            total_skipped += skipped_count
            
        except GitHubAPIError as e:
            click.echo(f"\n  ❌ Error: {e}", err=True)
            continue
    
    click.echo(f"\n{'═' * 60}")
    click.echo(f"📊 Sync Summary: {total_new} new, {total_updated} updated, {total_skipped} unchanged")
    click.echo(f"{'═' * 60}")


@main.command(hidden=True)
@click.option("--repo", help="Evaluate PRs from specific repository")
@click.option("--pr", "pr_number", type=int, help="Evaluate specific PR number")
@click.option("--batch", default=None, type=int, help="Max PRs to evaluate (limits LLM calls)")
@click.option("--force", is_flag=True, help="Re-evaluate even if already done")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def evaluate(
    repo: str | None,
    pr_number: int | None,
    batch: int | None,
    force: bool,
    as_json: bool,
    warn_legacy: bool = True,
):
    """Evaluate PRs for relevance using multi-stage analysis.
    
    Uses 3-stage pipeline:
    1. Rule-based filtering (keywords + paths)
    2. Semantic similarity (detect already-implemented)
    3. LLM analysis (final decision with reasoning)
    
    Examples:
    
        prscope evaluate               # Evaluate all unevaluated PRs
        prscope evaluate --batch 10    # Limit to 10 PRs (control LLM costs)
        prscope evaluate --pr 123      # Evaluate specific PR
        prscope evaluate --force       # Re-evaluate all
    """
    if warn_legacy:
        _warn_legacy_command("prscope evaluate", "prscope upstream evaluate")

    config = PrscopeConfig.load(get_repo_root())
    local_repo_path = config.get_local_repo_path()
    store = Store()
    
    if not config.features:
        click.echo("No features defined in prscope.features.yml")
        sys.exit(1)
    
    # Get current profile
    profile = store.get_latest_profile(str(local_repo_path))
    if not profile:
        click.echo("No profile found. Run: prscope profile")
        sys.exit(1)
    
    local_profile_sha = profile.profile_sha
    
    # Determine batch size
    batch_size = batch or config.sync.eval_batch_size
    
    if not as_json:
        click.echo(f"\n{'═' * 60}")
        click.echo(f"🔍 Evaluating PRs")
        click.echo(f"{'═' * 60}")
        click.echo(f"  Profile: {local_profile_sha[:8]}")
        if config.upstream_eval.enabled:
            click.echo(f"  Upstream eval LLM: {config.upstream_eval.model}")
            click.echo(f"  Mode: semantic + AI analysis (3-stage pipeline)")
            click.echo(f"  Batch limit: {batch_size} PRs")
        else:
            click.echo("  Mode: rule-based only (enable LLM for better accuracy)")
    
    # Get PRs to evaluate
    prs = store.list_pull_requests()
    if repo:
        upstream = store.get_upstream_repo(repo)
        if upstream:
            prs = [pr for pr in prs if pr.repo_id == upstream.id]
    
    if pr_number:
        prs = [pr for pr in prs if pr.number == pr_number]
    
    # Count already-evaluated PRs upfront
    pending_prs = []
    already_evaluated = 0
    for pr in prs:
        if not force and pr.head_sha:
            if store.evaluation_exists(pr.id, local_profile_sha, pr.head_sha):
                already_evaluated += 1
                continue
        pending_prs.append(pr)
    
    total_pending = len(pending_prs)
    to_process = min(total_pending, batch_size)
    
    if not as_json:
        click.echo(f"{'─' * 60}")
        click.echo(f"  Total PRs in database: {len(prs)}")
        click.echo(f"  Already evaluated: {already_evaluated}")
        click.echo(f"  Pending evaluation: {total_pending}")
        click.echo(f"  Will process: {to_process} (batch limit: {batch_size})")
        click.echo(f"{'─' * 60}")
    
    evaluated = []
    batch_limited = 0
    
    # Load full profile data for LLM
    profile_data = profile.profile_data if config.upstream_eval.enabled else None
    
    for i, pr in enumerate(pending_prs[:batch_size], 1):
        if not as_json:
            pct = int(i / to_process * 100) if to_process > 0 else 0
            click.echo(f"\n  [{i}/{to_process}] PR #{pr.number}: {pr.title[:45]}...")
            click.echo(f"      ⏳ Analyzing...", nl=False)
        
        files = store.get_pr_files(pr.id)
        result = evaluate_pr(
            pr=pr,
            files=files,
            config=config,
            local_profile_sha=local_profile_sha,
            store=store,
            local_profile=profile_data,
            local_repo_path=local_repo_path,
        )
        
        if result:
            evaluated.append({
                "pr_id": pr.id,
                "number": pr.number,
                "title": pr.title,
                "rule_score": result.rule_score,
                "final_score": result.final_score,
                "final_decision": result.final_decision,
                "llm_decision": result.llm_decision,
                "llm_confidence": result.llm_confidence,
                "llm_reasoning": result.llm_reasoning,
                "matched_features": result.matched_features,
                "has_existing_impl": result.has_existing_implementation,
                "should_seed_plan": result.should_seed_plan(),
            })
            
            if not as_json:
                icon = {"relevant": "✅", "maybe": "⚠️", "skip": "❌"}.get(result.final_decision, "  ")
                conf_pct = int(result.llm_confidence * 100)
                click.echo(f"\r      {icon} {result.final_decision.upper()} ({conf_pct}% confidence)")
                if result.llm_reasoning:
                    click.echo(f"      💬 {result.llm_reasoning[:70]}")
    
    # Calculate remaining
    batch_limited = total_pending - len(evaluated)
    
    if as_json:
        click.echo(json.dumps({
            "evaluated": evaluated,
            "skipped": already_evaluated,
            "batch_limited": batch_limited,
            "profile_sha": local_profile_sha,
        }, indent=2))
    else:
        # Summary by decision
        implement = [r for r in evaluated if r["llm_decision"] == "implement"]
        partial = [r for r in evaluated if r["llm_decision"] == "partial"]
        skip_prs = [r for r in evaluated if r["llm_decision"] == "skip"]
        plan_seed_ready = [r for r in evaluated if r["should_seed_plan"]]
        
        click.echo(f"\n{'═' * 60}")
        click.echo(f"📊 Evaluation Summary")
        click.echo(f"{'═' * 60}")
        click.echo(f"  Evaluated this run: {len(evaluated)}")
        click.echo(f"  Previously evaluated: {already_evaluated}")
        if batch_limited > 0:
            click.echo(f"  Remaining (batch limit): {batch_limited}")
        
        click.echo(f"\n  📈 Results:")
        click.echo(f"      ✅ Implement: {len(implement)}")
        click.echo(f"      ⚠️  Partial:   {len(partial)}")
        click.echo(f"      ❌ Skip:      {len(skip_prs)}")
        click.echo(f"      🗺️  Plan-seed ready: {len(plan_seed_ready)}")
        
        if implement:
            click.echo(f"\n{'─' * 60}")
            click.echo(f"✅ RECOMMENDED TO IMPLEMENT ({len(implement)})")
            click.echo(f"{'─' * 60}")
            for r in sorted(implement, key=lambda x: -x["llm_confidence"]):
                conf_pct = int(r['llm_confidence'] * 100)
                click.echo(f"  PR #{r['number']}: {r['title'][:50]}")
                click.echo(f"    Confidence: {conf_pct}%")
                if r['llm_reasoning']:
                    click.echo(f"    Reason: {r['llm_reasoning'][:65]}")
                click.echo()
        
        if partial:
            click.echo(f"\n{'─' * 60}")
            click.echo(f"⚠️  PARTIAL RELEVANCE ({len(partial)})")
            click.echo(f"{'─' * 60}")
            for r in partial[:5]:  # Limit to 5
                conf_pct = int(r['llm_confidence'] * 100)
                click.echo(f"  PR #{r['number']}: {r['title'][:50]} ({conf_pct}%)")
        
        if batch_limited > 0:
            click.echo(f"\n💡 Tip: Run `prscope evaluate` again to process {batch_limited} more PRs")
        
        if plan_seed_ready:
            click.echo(
                "\n💡 Tip: seed planning directly from upstream context with:\n"
                "   `prscope plan start --from-pr <owner/repo> <pr-number>`"
            )


@main.command(hidden=True)
@click.option("--limit", default=10, help="Number of PRs to show")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def digest(limit: int, as_json: bool, warn_legacy: bool = True):
    """Show summary of relevant PRs."""
    if warn_legacy:
        _warn_legacy_command("prscope digest", "prscope upstream digest")
    store = Store()
    
    # Get recent relevant evaluations
    evaluations = store.list_evaluations(decision="relevant", limit=limit)
    
    digest_data = []
    for eval in evaluations:
        pr = store.get_pull_request_by_id(eval.pr_id)
        if not pr:
            continue
        
        # Get repo name
        repos = store.list_upstream_repos()
        repo_name = None
        for r in repos:
            if r.id == pr.repo_id:
                repo_name = r.full_name
                break
        
        digest_data.append({
            "repo": repo_name,
            "number": pr.number,
            "title": pr.title,
            "author": pr.author,
            "score": eval.final_score,
            "matched_features": eval.matched_features,
            "url": pr.html_url,
            "evaluated_at": eval.created_at,
        })
    
    if as_json:
        click.echo(json.dumps(digest_data, indent=2))
    else:
        if not digest_data:
            click.echo("No relevant PRs found.")
            click.echo("Run: prscope sync && prscope evaluate")
            return
        
        click.echo(f"Top {len(digest_data)} Relevant PRs:\n")
        for item in digest_data:
            click.echo(f"[{item['repo']}] PR #{item['number']}: {item['title']}")
            click.echo(f"  Score: {item['score']:.2f} | Features: {', '.join(item['matched_features'])}")
            click.echo(f"  Author: {item['author']} | {item['url']}")
            click.echo()


@main.command(hidden=True)
@click.option("--limit", default=20, help="Number of evaluations to show")
@click.option("--decision", type=click.Choice(["relevant", "maybe", "skip"]), help="Filter by decision")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def history(limit: int, decision: str | None, as_json: bool, warn_legacy: bool = True):
    """View evaluation history."""
    if warn_legacy:
        _warn_legacy_command("prscope history", "prscope upstream history")
    store = Store()
    
    evaluations = store.list_evaluations(decision=decision, limit=limit)
    
    history_data = []
    for eval in evaluations:
        pr = store.get_pull_request_by_id(eval.pr_id)
        if not pr:
            continue
        
        history_data.append({
            "id": eval.id,
            "pr_number": pr.number,
            "pr_title": pr.title,
            "rule_score": eval.rule_score,
            "final_score": eval.final_score,
            "decision": eval.decision,
            "matched_features": eval.matched_features,
            "profile_sha": eval.local_profile_sha[:8],
            "pr_sha": eval.pr_head_sha[:8] if eval.pr_head_sha else "unknown",
            "created_at": eval.created_at,
        })
    
    if as_json:
        click.echo(json.dumps(history_data, indent=2))
    else:
        if not history_data:
            click.echo("No evaluation history found.")
            return
        
        click.echo(f"Evaluation History ({len(history_data)} entries):\n")
        for item in history_data:
            decision_icon = {"relevant": "✓", "maybe": "?", "skip": "✗"}.get(item["decision"], " ")
            click.echo(f"[{decision_icon}] #{item['id']} - PR #{item['pr_number']}: {item['pr_title'][:40]}")
            click.echo(f"    Score: {item['final_score']:.2f} | Profile: {item['profile_sha']} | PR: {item['pr_sha']}")
            click.echo(f"    Features: {', '.join(item['matched_features']) or 'none'}")
            click.echo(f"    Date: {item['created_at']}")
            click.echo()


@upstream_group.command("sync")
@click.option("--repo", help="Sync specific repository (owner/repo)")
@click.option("--state", default=None, help="PR state filter (merged, open, closed, all)")
@click.option("--max-prs", default=None, type=int, help="Maximum PRs to fetch")
@click.option("--since", default=None, help="Only fetch PRs after date (ISO date or 90d/6m/1y)")
@click.option("--full", is_flag=True, help="Full sync (ignore incremental watermark)")
@click.option("--no-files", is_flag=True, help="Skip fetching file lists")
def upstream_sync(
    repo: str | None,
    state: str | None,
    max_prs: int | None,
    since: str | None,
    full: bool,
    no_files: bool,
) -> None:
    """Fetch upstream PR metadata for planning seeds."""
    sync(
        repo=repo,
        state=state,
        max_prs=max_prs,
        since=since,
        full=full,
        no_files=no_files,
        warn_legacy=False,
    )


@upstream_group.command("evaluate")
@click.option("--repo", help="Evaluate PRs from specific repository")
@click.option("--pr", "pr_number", type=int, help="Evaluate specific PR number")
@click.option("--batch", default=None, type=int, help="Max PRs to evaluate (limits LLM calls)")
@click.option("--force", is_flag=True, help="Re-evaluate even if already done")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def upstream_evaluate(
    repo: str | None,
    pr_number: int | None,
    batch: int | None,
    force: bool,
    as_json: bool,
) -> None:
    """Score upstream PRs for planning relevance."""
    evaluate(
        repo=repo,
        pr_number=pr_number,
        batch=batch,
        force=force,
        as_json=as_json,
        warn_legacy=False,
    )


@upstream_group.command("digest")
@click.option("--limit", default=10, help="Number of PRs to show")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def upstream_digest(limit: int, as_json: bool) -> None:
    """Summarize high-signal upstream PR candidates."""
    digest(limit=limit, as_json=as_json, warn_legacy=False)


@upstream_group.command("history")
@click.option("--limit", default=20, help="Number of evaluations to show")
@click.option("--decision", type=click.Choice(["relevant", "maybe", "skip"]), help="Filter by decision")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def upstream_history(limit: int, decision: str | None, as_json: bool) -> None:
    """Show stored upstream evaluation history."""
    history(limit=limit, decision=decision, as_json=as_json, warn_legacy=False)


def _load_planning_runtime(repo_name: str | None) -> tuple[PrscopeConfig, Store, PlanningRuntime]:
    repo_root = get_repo_root()
    config = PrscopeConfig.load(repo_root)
    repo_profile = config.resolve_repo(repo_name, cwd=Path.cwd())
    store = Store()
    runtime = PlanningRuntime(store=store, config=config, repo=repo_profile)
    return config, store, runtime


def _format_age(path: Path) -> str:
    if not path.exists():
        return "never"
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    delta = datetime.now(tz=timezone.utc) - mtime
    minutes = int(delta.total_seconds() // 60)
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    return f"{days}d ago"


def _open_web_session(session_id: str, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    _, base_url = ensure_server_running(host=host, port=port, open_browser=False)
    webbrowser.open(f"{base_url}/sessions/{session_id}")


@main.group(name="repos")
def repos_group() -> None:
    """Repository profile utilities."""


@repos_group.command("list")
def repos_list() -> None:
    """List configured repo profiles."""
    config = PrscopeConfig.load(get_repo_root())
    repos = config.list_repos()
    if not repos:
        click.echo("No repositories configured.")
        return
    click.echo("Name\tPath\tUpstreams\tMemory")
    for repo in repos:
        meta_path = repo.memory_dir / "_meta.json"
        click.echo(
            f"{repo.name}\t{repo.resolved_path}\t{len(repo.upstream)}\t{_format_age(meta_path)}"
        )


@main.group(name="plan")
def plan_group() -> None:
    """Interactive planning commands."""


@plan_group.command("start")
@click.argument("requirements", required=False)
@click.option("--repo", "repo_name", help="Repo profile name")
@click.option("--from-pr", "from_pr", nargs=2, type=str, help="Seed from upstream repo + PR number")
@click.option("--no-open", is_flag=True, help="Do not open browser after creating session")
@click.option("--rebuild-memory", is_flag=True, help="Force memory rebuild")
def plan_start(
    requirements: str | None,
    repo_name: str | None,
    from_pr: tuple[str, str] | None,
    no_open: bool,
    rebuild_memory: bool,
) -> None:
    """Start a planning session from requirements, PR seed, or discovery chat."""
    _, _, runtime = _load_planning_runtime(repo_name)

    if from_pr:
        upstream_repo, pr_num_raw = from_pr
        session = asyncio.run(
            runtime.start_from_pr(
                upstream_repo=upstream_repo,
                pr_number=int(pr_num_raw),
                rebuild_memory=rebuild_memory,
            )
        )
    elif requirements:
        session = asyncio.run(
            runtime.start_from_requirements(
                requirements=requirements,
                rebuild_memory=rebuild_memory,
            )
        )
    else:
        session, opening = asyncio.run(runtime.start_from_chat(rebuild_memory=rebuild_memory))
        click.echo(opening)

    click.echo(f"Created planning session: {session.id}")
    if not no_open:
        _open_web_session(session.id)


@plan_group.command("chat")
@click.option("--repo", "repo_name", help="Repo profile name")
@click.option("--no-open", is_flag=True, help="Do not open browser after creating session")
@click.option("--rebuild-memory", is_flag=True, help="Force memory rebuild")
def plan_chat(repo_name: str | None, no_open: bool, rebuild_memory: bool) -> None:
    """Start chat-first discovery mode."""
    _, _, runtime = _load_planning_runtime(repo_name)
    session, opening = asyncio.run(runtime.start_from_chat(rebuild_memory=rebuild_memory))
    click.echo(opening)
    click.echo(f"Created planning session: {session.id}")
    if not no_open:
        _open_web_session(session.id)


@plan_group.command("resume")
@click.argument("session_id")
@click.option("--repo", "repo_name", help="Optional override for repo profile")
def plan_resume(session_id: str, repo_name: str | None) -> None:
    """Resume an existing planning session in the browser UI."""
    store = Store()
    session = store.get_planning_session(session_id)
    if session is None:
        raise click.ClickException(f"Session not found: {session_id}")
    _open_web_session(session_id)


@plan_group.command("round")
@click.argument("session_id")
@click.option("--repo", "repo_name", help="Repo profile name")
@click.option("--input", "user_input", default=None, help="Optional user input for this round")
def plan_round(session_id: str, repo_name: str | None, user_input: str | None) -> None:
    """Run one adversarial refinement round in the CLI."""
    _, store, runtime = _load_planning_runtime(repo_name)
    session = store.get_planning_session(session_id)
    if session is None:
        raise click.ClickException(f"Session not found: {session_id}")

    async def _run() -> None:
        running_cost = float(session.session_total_cost_usd or 0.0)
        rendered_progress = False
        round_label = f"[Round {session.current_round + 1}/{runtime.planning_config.max_adversarial_rounds}]"

        async def on_event(event: dict[str, object]) -> None:
            nonlocal running_cost, rendered_progress
            event_type = str(event.get("type", ""))
            if event_type == "token_usage":
                running_cost = float(event.get("session_total_usd", running_cost) or running_cost)
                click.echo(f"\r{round_label} Refining plan... ${running_cost:.4f}", nl=False)
                rendered_progress = True
                return
            if event_type == "clarification_needed":
                question = str(event.get("question", "")).strip() or "Clarification needed"
                click.echo()
                click.echo(f"[?] {question}")
                answer = click.prompt("Your answer", type=str)
                runtime.provide_clarification(session_id, [answer])
                return
            if event_type == "warning":
                click.echo()
                click.echo(f"[warning] {event.get('message', '')}")

        critic, author, convergence = await runtime.run_adversarial_round(
            session_id=session_id,
            user_input=user_input,
            event_callback=on_event,
        )
        if rendered_progress:
            click.echo()
        click.echo(
            "Round complete: "
            f"{critic.major_issues_remaining} major / {critic.minor_issues_remaining} minor remaining, "
            f"cost ${running_cost:.4f}, converged={convergence.converged}"
        )
        click.echo(f"Updated plan length: {len(author.plan)} chars")

    asyncio.run(_run())


@main.command("web")
@click.option("--dev", is_flag=True, help="Run frontend Vite dev server with backend")
@click.option("--host", default=DEFAULT_HOST, show_default=True)
@click.option("--port", default=DEFAULT_PORT, show_default=True, type=int)
@click.option("--resume", "resume_session_id", default=None, help="Open specific session ID")
@click.option("--repo", "repo_name", default=None, help="Repo profile name for web UI context")
@click.option("--repo-root", "repo_root", default=None, help="Directory containing prscope.yml")
@click.option("--background", "-b", is_flag=True, help="Detach server to background")
def run_web(
    dev: bool,
    host: str,
    port: int,
    resume_session_id: str | None,
    repo_name: str | None,
    repo_root: str | None,
    background: bool,
) -> None:
    """Run prscope web UI server (foreground by default, -b for background)."""
    from .web.server import run_server, is_port_open

    if repo_root:
        os.environ["PRSCOPE_CONFIG_ROOT"] = str(Path(repo_root).expanduser().resolve())

    base_url = f"http://{host}:{port}"
    repo_query = f"?repo={repo_name}" if repo_name else ""

    if dev:
        _, base_url = ensure_server_running(host=host, port=port, open_browser=False)
        frontend_dir = Path(__file__).parent / "web" / "frontend"
        if not frontend_dir.exists():
            raise click.ClickException(f"Frontend directory not found: {frontend_dir}")
        click.echo(f"Backend running at {base_url}")
        click.echo("Starting Vite dev server...")
        subprocess.run(["npm", "run", "dev"], cwd=frontend_dir, check=True)
        return

    if background:
        _, base_url = ensure_server_running(host=host, port=port, open_browser=False)
        if resume_session_id:
            webbrowser.open(f"{base_url}/sessions/{resume_session_id}{repo_query}")
        else:
            webbrowser.open(f"{base_url}/{repo_query}")
        click.echo(f"Web UI available at {base_url}")
        click.echo("Server logs: ~/.prscope/server.log")
        return

    if is_port_open(host, port):
        click.echo(f"Server already running at {base_url}")
        if resume_session_id:
            webbrowser.open(f"{base_url}/sessions/{resume_session_id}{repo_query}")
        else:
            webbrowser.open(f"{base_url}/{repo_query}")
        return

    import threading
    def _open_browser() -> None:
        import time as _time
        _time.sleep(1.5)
        target = (
            f"{base_url}/sessions/{resume_session_id}{repo_query}"
            if resume_session_id
            else f"{base_url}/{repo_query}"
        )
        webbrowser.open(target)

    threading.Thread(target=_open_browser, daemon=True).start()
    click.echo(f"Starting prscope server at {base_url} (Ctrl+C to stop)")
    run_server(host=host, port=port)


@plan_group.command("list")
@click.option("--repo", "repo_name", help="Filter sessions by repo profile")
def plan_list(repo_name: str | None) -> None:
    """List planning sessions."""
    store = Store()
    sessions = store.list_planning_sessions(repo_name=repo_name, limit=200)
    if not sessions:
        click.echo("No planning sessions found.")
        return
    click.echo("Session ID\tRepo\tStatus\tRound\tTitle")
    for session in sessions:
        click.echo(
            f"{session.id}\t{session.repo_name}\t{session.status}\t"
            f"{session.current_round}\t{session.title}"
        )


@plan_group.command("export")
@click.argument("session_id")
@click.option("--repo", "repo_name", help="Repo profile name")
def plan_export(session_id: str, repo_name: str | None) -> None:
    """Export PRD and RFC markdown files."""
    _, store, runtime = _load_planning_runtime(repo_name)
    if store.get_planning_session(session_id) is None:
        raise click.ClickException(f"Session not found: {session_id}")
    paths = runtime.export(session_id)
    click.echo(f"Exported:\n- {paths['prd']}\n- {paths['rfc']}\n- {paths['conversation']}")


@plan_group.command("diff")
@click.argument("session_id")
@click.option("--repo", "repo_name", help="Repo profile name")
@click.option("--round", "round_number", type=int, default=None, help="Diff specific round against previous")
def plan_diff(session_id: str, repo_name: str | None, round_number: int | None) -> None:
    """Show plan diff as unified text."""
    _, store, runtime = _load_planning_runtime(repo_name)
    if store.get_planning_session(session_id) is None:
        raise click.ClickException(f"Session not found: {session_id}")
    diff_text = runtime.plan_diff(session_id, round_number=round_number)
    click.echo(diff_text or "No diff available.")


@plan_group.command("memory")
@click.option("--repo", "repo_name", help="Repo profile name")
@click.option("--rebuild", is_flag=True, help="Force rebuild memory blocks")
def plan_memory(repo_name: str | None, rebuild: bool) -> None:
    """Build/show memory blocks for active repo."""
    _, _, runtime = _load_planning_runtime(repo_name)
    profile = build_profile(runtime.repo.resolved_path)
    asyncio.run(runtime.memory.ensure_memory(profile, rebuild=rebuild))
    click.echo(f"Memory dir: {runtime.repo.memory_dir}")
    for block in ("architecture", "modules", "patterns", "entrypoints"):
        path = runtime.repo.memory_dir / f"{block}.md"
        click.echo(f"- {block}: {path} ({_format_age(path)})")


@plan_group.command("manifesto")
@click.option("--repo", "repo_name", help="Repo profile name")
@click.option("--edit", "open_editor", is_flag=True, help="Open manifesto in editor")
def plan_manifesto(repo_name: str | None, open_editor: bool) -> None:
    """Create or open repo manifesto file."""
    _, _, runtime = _load_planning_runtime(repo_name)
    path = runtime.memory.ensure_manifesto()
    if open_editor:
        click.edit(filename=str(path))
    else:
        click.echo(path)


@plan_group.command("validate")
@click.argument("session_id")
@click.option("--repo", "repo_name", help="Repo profile name")
def plan_validate(session_id: str, repo_name: str | None) -> None:
    """Headless validation for CI."""
    _, store, runtime = _load_planning_runtime(repo_name)
    session = store.get_planning_session(session_id)
    if session is None:
        raise click.ClickException(f"Session not found: {session_id}")
    try:
        result = asyncio.run(runtime.validate_session(session_id))
    except Exception as exc:
        click.echo(f"FAILED: strict validation error: {exc}")
        sys.exit(2)
    hard_count = len(result.hard_constraint_violations)
    major = result.major_issues_remaining
    if hard_count > 0 and major == 0:
        click.echo(
            "FAILED: hard constraint violations only "
            f"({', '.join(result.hard_constraint_violations)})"
        )
        sys.exit(2)
    if major > 0 or hard_count > 0:
        click.echo(
            f"FAILED: {major} major issues, hard violations: "
            f"{', '.join(result.hard_constraint_violations) or 'none'}"
        )
        sys.exit(1)
    click.echo("Plan validated - 0 major issues, 0 hard constraint violations")


@plan_group.command("status")
@click.argument("session_id")
@click.option("--repo", "repo_name", help="Repo profile name")
@click.option("--pr-number", type=int, default=None, help="Merged PR number for drift detection")
def plan_status(session_id: str, repo_name: str | None, pr_number: int | None) -> None:
    """Compare planned file refs with merged PR files."""
    _, store, runtime = _load_planning_runtime(repo_name)
    session = store.get_planning_session(session_id)
    if session is None:
        raise click.ClickException(f"Session not found: {session_id}")

    chosen_pr = pr_number
    upstream_repo = None
    if chosen_pr is None and session.seed_ref and "#" in session.seed_ref:
        upstream_repo, pr_raw = session.seed_ref.rsplit("#", 1)
        if pr_raw.isdigit():
            chosen_pr = int(pr_raw)

    if chosen_pr is None:
        raise click.ClickException("Provide --pr-number or use a session seeded from an upstream PR.")
    if upstream_repo is None:
        raise click.ClickException("Session does not include upstream repo name in seed_ref.")

    upstream = store.get_upstream_repo(upstream_repo)
    if upstream is None:
        raise click.ClickException(f"Upstream repo not found in DB: {upstream_repo}")
    pr = store.get_pull_request(upstream.id, chosen_pr)
    if pr is None:
        raise click.ClickException(f"PR not found in DB: {upstream_repo}#{chosen_pr}")
    merged_files = {f.path for f in store.get_pr_files(pr.id)}
    drift = runtime.status(session_id, merged_pr_files=merged_files)

    click.echo(f"Plan Status: {upstream_repo}#{chosen_pr}")
    click.echo(f"Implemented as planned: {drift['implemented_count']} files")
    click.echo(f"Planned but not touched: {drift['missing_count']} files")
    for item in drift["missing"][:20]:
        click.echo(f"  - {item}")
    click.echo(f"Unplanned changes: {drift['unplanned_count']} files")
    for item in drift["unplanned"][:20]:
        click.echo(f"  - {item}")
    if drift.get("session_cost_usd") is not None:
        click.echo(f"Session cost: ${float(drift['session_cost_usd']):.4f}")
    if drift.get("max_prompt_tokens") is not None:
        click.echo(f"Max prompt tokens: {int(drift['max_prompt_tokens'])}")
    if drift.get("confidence_trend") is not None:
        click.echo(f"Confidence trend: {float(drift['confidence_trend']):+.3f}")
    click.echo(f"Converged early: {'yes' if drift.get('converged_early') else 'no'}")
    if drift["missing_count"] > 0:
        sys.exit(1)


@plan_group.command("abort")
@click.argument("session_id")
@click.option("--repo", "repo_name", help="Repo profile name")
def plan_abort(session_id: str, repo_name: str | None) -> None:
    """Abort an active clarification wait for a session."""
    _, store, runtime = _load_planning_runtime(repo_name)
    session = store.get_planning_session(session_id)
    if session is None:
        raise click.ClickException(f"Session not found: {session_id}")
    runtime.abort_clarification(session_id)
    click.echo(f"Aborted clarification wait for session {session_id[:8]}.")


@plan_group.command("clarify")
@click.argument("session_id")
@click.option("--repo", "repo_name", help="Repo profile name")
@click.option("--answer", "answers", multiple=True, help="Clarification answer (repeatable)")
def plan_clarify(session_id: str, repo_name: str | None, answers: tuple[str, ...]) -> None:
    """Provide clarification answers for a paused session."""
    _, store, runtime = _load_planning_runtime(repo_name)
    session = store.get_planning_session(session_id)
    if session is None:
        raise click.ClickException(f"Session not found: {session_id}")
    if answers:
        payload = list(answers)
    else:
        payload = [click.prompt("Your answer", type=str)]
    runtime.provide_clarification(session_id, payload)
    click.echo(f"Submitted {len(payload)} clarification answer(s) for {session_id[:8]}.")


@plan_group.command("delete")
@click.argument("session_id", required=False)
@click.option("--all", "delete_all", is_flag=True, help="Delete all sessions")
@click.option("--repo", "repo_name", help="Limit --all to a specific repo profile")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def plan_delete(
    session_id: str | None,
    delete_all: bool,
    repo_name: str | None,
    yes: bool,
) -> None:
    """Delete one or all planning sessions (including turns and plan versions)."""
    store = Store()

    if delete_all:
        sessions = store.list_planning_sessions(repo_name=repo_name)
        if not sessions:
            click.echo("No sessions to delete.")
            return
        scope = f"repo '{repo_name}'" if repo_name else "ALL repos"
        click.echo(f"This will delete {len(sessions)} session(s) from {scope}:")
        for s in sessions[:10]:
            click.echo(f"  {s.id[:8]}  [{s.status}]  {s.title}")
        if len(sessions) > 10:
            click.echo(f"  ... and {len(sessions) - 10} more")
        if not yes:
            click.confirm("Delete all?", abort=True)
        count = store.delete_all_planning_sessions(repo_name=repo_name)
        click.echo(f"Deleted {count} session(s).")
        return

    if not session_id:
        raise click.UsageError("Provide SESSION_ID or use --all.")

    session = store.get_planning_session(session_id)
    if session is None:
        raise click.ClickException(f"Session not found: {session_id}")

    if not yes:
        click.confirm(
            f"Delete session '{session.title}' ({session_id[:8]})?", abort=True
        )
    store.delete_planning_session(session_id)
    click.echo(f"Deleted session {session_id[:8]}.")


@main.command("scanners")
def list_scanners_cmd() -> None:
    """Show available codebase scanner backends and their status."""
    from .planning.scanners import list_scanners

    backends = list_scanners()
    click.echo("\nCODEBASE SCANNER BACKENDS\n")
    click.echo(f"  {'NAME':<12} {'STATUS':<14} DESCRIPTION")
    click.echo("  " + "-" * 60)

    descriptions = {
        "grep":    "Default — no extra deps, file-tree + README",
        "repomap": "tree-sitter symbol map (pip install aider-chat)",
        "repomix": "full repo pack (npm install -g repomix)",
    }
    for b in backends:
        name = str(b["name"])
        status = click.style("available", fg="green") if b["available"] else click.style("not installed", fg="yellow")
        desc = descriptions.get(name, "")
        click.echo(f"  {name:<12} {status:<14}  {desc}")

    click.echo()
    click.echo("Set in prscope.yml:  planning:\n                       scanner: repomap")
    click.echo()


@main.command("analytics")
@click.option("--repo", "repo_name", default=None, help="Repo profile name")
def analytics(repo_name: str | None) -> None:
    """Show planning analytics (cost, confidence, convergence, constraints)."""
    store = Store()
    sessions = store.list_planning_sessions(repo_name=repo_name, limit=500)
    if not sessions:
        click.echo("No planning sessions found.")
        return

    total_cost = 0.0
    costs: list[float] = []
    confidence_values: list[float] = []
    confidence_trends: list[float] = []
    rounds_to_convergence: list[int] = []
    constraint_counts: dict[str, int] = {}
    round_total_tokens: list[int] = []
    clarification_pauses = 0
    total_rounds = 0
    model_costs: dict[str, float] = {}

    for session in sessions:
        cost = float(session.session_total_cost_usd or 0.0)
        total_cost += cost
        costs.append(cost)
        if session.confidence_trend is not None:
            confidence_trends.append(float(session.confidence_trend))

        metrics = store.get_round_metrics(session.id)
        if session.status in {"converged", "approved", "exported"}:
            rounds_to_convergence.append(session.current_round)
        for metric in metrics:
            total_rounds += 1
            if metric.critic_confidence is not None:
                confidence_values.append(float(metric.critic_confidence))
            round_total_tokens.append(
                int(metric.author_prompt_tokens or 0)
                + int(metric.author_completion_tokens or 0)
                + int(metric.critic_prompt_tokens or 0)
                + int(metric.critic_completion_tokens or 0)
            )
            if int(metric.clarifications_this_round or 0) > 0:
                clarification_pauses += 1
            for cid in metric.constraint_violations:
                constraint_counts[cid] = constraint_counts.get(cid, 0) + 1

    if repo_name:
        persisted_stats = store.get_constraint_stats(repo_name)
        for cid, payload in persisted_stats.items():
            if isinstance(payload, dict):
                constraint_counts[cid] = max(
                    constraint_counts.get(cid, 0), int(payload.get("violations", 0))
                )

    avg_cost = total_cost / len(sessions) if sessions else 0.0
    sorted_costs = sorted(costs)
    p75_cost = sorted_costs[int(0.75 * (len(sorted_costs) - 1))] if sorted_costs else 0.0
    avg_conf = (sum(confidence_values) / len(confidence_values)) if confidence_values else 0.0
    avg_rounds = (
        sum(rounds_to_convergence) / len(rounds_to_convergence)
        if rounds_to_convergence
        else 0.0
    )
    avg_conf_trend = (
        sum(confidence_trends) / len(confidence_trends) if confidence_trends else 0.0
    )
    sorted_round_tokens = sorted(round_total_tokens)
    p75_round_tokens = (
        sorted_round_tokens[int(0.75 * (len(sorted_round_tokens) - 1))]
        if sorted_round_tokens
        else 0
    )
    clarification_rate = (
        (clarification_pauses / total_rounds) * 100.0 if total_rounds else 0.0
    )
    top_expensive_sessions = sorted(
        sessions,
        key=lambda item: float(item.session_total_cost_usd or 0.0),
        reverse=True,
    )[:5]

    repo_names = sorted({session.repo_name for session in sessions})
    for candidate_repo in repo_names:
        rounds_log = Path.home() / ".prscope" / "repos" / candidate_repo / "rounds.jsonl"
        if not rounds_log.exists():
            continue
        try:
            for line in rounds_log.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                per_model = payload.get("model_costs", {})
                if not isinstance(per_model, dict):
                    continue
                for model, value in per_model.items():
                    try:
                        model_costs[str(model)] = model_costs.get(str(model), 0.0) + float(value)
                    except (TypeError, ValueError):
                        continue
        except (OSError, json.JSONDecodeError):
            continue

    click.echo("Planning Analytics")
    click.echo(f"Sessions: {len(sessions)}")
    click.echo(f"Total spend: ${total_cost:.4f}")
    click.echo(f"Average session cost: ${avg_cost:.4f}")
    click.echo(f"p75 session cost: ${p75_cost:.4f}")
    click.echo(f"Average critic confidence: {avg_conf:.3f}")
    click.echo(f"Average confidence trend/session: {avg_conf_trend:+.3f}")
    click.echo(f"Average rounds-to-convergence: {avg_rounds:.2f}")
    click.echo(f"p75 tokens/round: {p75_round_tokens}")
    click.echo(f"Clarification pause frequency: {clarification_rate:.1f}% of rounds")
    click.echo("Most expensive sessions:")
    for item in top_expensive_sessions:
        click.echo(f"  - {item.id[:8]} [{item.repo_name}] ${float(item.session_total_cost_usd or 0.0):.4f}")
    click.echo("Cost by model:")
    if not model_costs:
        click.echo("  - unavailable (no rounds.jsonl model_costs data yet)")
    else:
        for model, cost in sorted(model_costs.items(), key=lambda entry: entry[1], reverse=True):
            click.echo(f"  - {model}: ${cost:.4f}")
    click.echo("Top hard-constraint violations:")
    if not constraint_counts:
        click.echo("  - none")
    else:
        for cid, count in sorted(constraint_counts.items(), key=lambda item: item[1], reverse=True)[:10]:
            click.echo(f"  - {cid}: {count}")


if __name__ == "__main__":
    main()
