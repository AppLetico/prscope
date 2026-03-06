from __future__ import annotations

import json
from typing import Any

from .models import Artifact, Evaluation, PRFile, PullRequest, RepoProfile, UpstreamRepo


class StoreRepoDataMixin:
    # =========================================================================
    # Repo Profiles
    # =========================================================================

    def save_profile(self, repo_root: str, profile_sha: str, profile_json: str) -> RepoProfile:
        """Save or update a repo profile."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO repo_profiles (repo_root, profile_sha, profile_json, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(repo_root, profile_sha) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    created_at = excluded.created_at
                """,
                (repo_root, profile_sha, profile_json, self._now()),
            )
            row = conn.execute(
                "SELECT * FROM repo_profiles WHERE repo_root = ? AND profile_sha = ?",
                (repo_root, profile_sha),
            ).fetchone()
            return RepoProfile(**dict(row))

    def get_profile(self, repo_root: str, profile_sha: str) -> RepoProfile | None:
        """Get a specific profile by SHA."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM repo_profiles WHERE repo_root = ? AND profile_sha = ?",
                (repo_root, profile_sha),
            ).fetchone()
            return RepoProfile(**dict(row)) if row else None

    def get_latest_profile(self, repo_root: str) -> RepoProfile | None:
        """Get the most recent profile for a repo."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM repo_profiles WHERE repo_root = ? ORDER BY created_at DESC LIMIT 1",
                (repo_root,),
            ).fetchone()
            return RepoProfile(**dict(row)) if row else None

    # =========================================================================
    # Upstream Repos
    # =========================================================================

    def upsert_upstream_repo(self, full_name: str) -> UpstreamRepo:
        """Insert or get an upstream repo."""
        with self._connect() as conn:
            conn.execute("INSERT OR IGNORE INTO upstream_repos (full_name) VALUES (?)", (full_name,))
            row = conn.execute("SELECT * FROM upstream_repos WHERE full_name = ?", (full_name,)).fetchone()
            return UpstreamRepo(**dict(row))

    def update_repo_sync_time(self, repo_id: int, last_seen_updated_at: str | None = None) -> None:
        """Update last sync time for a repo."""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE upstream_repos
                SET last_synced_at = ?, last_seen_updated_at = COALESCE(?, last_seen_updated_at)
                WHERE id = ?
                """,
                (self._now(), last_seen_updated_at, repo_id),
            )

    def get_upstream_repo(self, full_name: str) -> UpstreamRepo | None:
        """Get upstream repo by full name."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM upstream_repos WHERE full_name = ?", (full_name,)).fetchone()
            return UpstreamRepo(**dict(row)) if row else None

    def list_upstream_repos(self) -> list[UpstreamRepo]:
        """List all upstream repos."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM upstream_repos ORDER BY full_name").fetchall()
            return [UpstreamRepo(**dict(row)) for row in rows]

    # =========================================================================
    # Pull Requests
    # =========================================================================

    def upsert_pull_request(
        self,
        repo_id: int,
        number: int,
        state: str,
        title: str,
        body: str | None = None,
        author: str | None = None,
        labels: list[str] | None = None,
        updated_at: str | None = None,
        merged_at: str | None = None,
        head_sha: str | None = None,
        html_url: str | None = None,
    ) -> PullRequest:
        """Insert or update a pull request."""
        labels_json = json.dumps(labels) if labels else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO pull_requests
                    (repo_id, number, state, title, body, author, labels_json,
                     updated_at, merged_at, head_sha, html_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(repo_id, number) DO UPDATE SET
                    state = excluded.state,
                    title = excluded.title,
                    body = excluded.body,
                    author = excluded.author,
                    labels_json = excluded.labels_json,
                    updated_at = excluded.updated_at,
                    merged_at = excluded.merged_at,
                    head_sha = excluded.head_sha,
                    html_url = excluded.html_url
                """,
                (
                    repo_id,
                    number,
                    state,
                    title,
                    body,
                    author,
                    labels_json,
                    updated_at,
                    merged_at,
                    head_sha,
                    html_url,
                ),
            )
            row = conn.execute(
                "SELECT * FROM pull_requests WHERE repo_id = ? AND number = ?", (repo_id, number)
            ).fetchone()
            return PullRequest(**dict(row))

    def get_pull_request(self, repo_id: int, number: int) -> PullRequest | None:
        """Get a specific PR."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM pull_requests WHERE repo_id = ? AND number = ?", (repo_id, number)
            ).fetchone()
            return PullRequest(**dict(row)) if row else None

    def get_pull_request_by_id(self, pr_id: int) -> PullRequest | None:
        """Get a PR by its database ID."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM pull_requests WHERE id = ?", (pr_id,)).fetchone()
            return PullRequest(**dict(row)) if row else None

    def list_pull_requests(
        self,
        repo_id: int | None = None,
        state: str | None = None,
        limit: int = 100,
    ) -> list[PullRequest]:
        """List pull requests with optional filters."""
        with self._connect() as conn:
            query = "SELECT * FROM pull_requests WHERE 1=1"
            params: list[Any] = []

            if repo_id is not None:
                query += " AND repo_id = ?"
                params.append(repo_id)
            if state is not None:
                query += " AND state = ?"
                params.append(state)

            query += " ORDER BY updated_at DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            return [PullRequest(**dict(row)) for row in rows]

    # =========================================================================
    # PR Files
    # =========================================================================

    def save_pr_files(self, pr_id: int, files: list[dict[str, Any]]) -> None:
        """Save files for a PR (replaces existing)."""
        with self._connect() as conn:
            # Delete existing files
            conn.execute("DELETE FROM pr_files WHERE pr_id = ?", (pr_id,))
            # Insert new files
            for f in files:
                conn.execute(
                    "INSERT INTO pr_files (pr_id, path, additions, deletions) VALUES (?, ?, ?, ?)",
                    (pr_id, f.get("path", ""), f.get("additions", 0), f.get("deletions", 0)),
                )

    def get_pr_files(self, pr_id: int) -> list[PRFile]:
        """Get files for a PR."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM pr_files WHERE pr_id = ?", (pr_id,)).fetchall()
            return [PRFile(**dict(row)) for row in rows]

    # =========================================================================
    # Evaluations
    # =========================================================================

    def evaluation_exists(self, pr_id: int, local_profile_sha: str, pr_head_sha: str) -> bool:
        """Check if an evaluation already exists (deduplication)."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM evaluations
                WHERE pr_id = ? AND local_profile_sha = ? AND pr_head_sha = ?
                """,
                (pr_id, local_profile_sha, pr_head_sha),
            ).fetchone()
            return row is not None

    def save_evaluation(
        self,
        pr_id: int,
        local_profile_sha: str,
        pr_head_sha: str,
        rule_score: float,
        final_score: float,
        matched_features: list[str],
        signals: dict[str, Any],
        llm_result: dict[str, Any] | None,
        decision: str,
    ) -> Evaluation:
        """Save an evaluation result."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO evaluations
                    (pr_id, local_profile_sha, pr_head_sha, rule_score, final_score,
                     matched_features_json, signals_json, llm_json, decision, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pr_id, local_profile_sha, pr_head_sha) DO UPDATE SET
                    rule_score = excluded.rule_score,
                    final_score = excluded.final_score,
                    matched_features_json = excluded.matched_features_json,
                    signals_json = excluded.signals_json,
                    llm_json = excluded.llm_json,
                    decision = excluded.decision,
                    created_at = excluded.created_at
                """,
                (
                    pr_id,
                    local_profile_sha,
                    pr_head_sha,
                    rule_score,
                    final_score,
                    json.dumps(matched_features),
                    json.dumps(signals),
                    json.dumps(llm_result) if llm_result else None,
                    decision,
                    self._now(),
                ),
            )
            row = conn.execute(
                """
                SELECT * FROM evaluations
                WHERE pr_id = ? AND local_profile_sha = ? AND pr_head_sha = ?
                """,
                (pr_id, local_profile_sha, pr_head_sha),
            ).fetchone()
            return Evaluation(**dict(row))

    def get_evaluation(self, pr_id: int, local_profile_sha: str, pr_head_sha: str) -> Evaluation | None:
        """Get a specific evaluation."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM evaluations
                WHERE pr_id = ? AND local_profile_sha = ? AND pr_head_sha = ?
                """,
                (pr_id, local_profile_sha, pr_head_sha),
            ).fetchone()
            return Evaluation(**dict(row)) if row else None

    def get_evaluation_by_id(self, evaluation_id: int) -> Evaluation | None:
        """Get evaluation by ID."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM evaluations WHERE id = ?", (evaluation_id,)).fetchone()
            return Evaluation(**dict(row)) if row else None

    def list_evaluations(
        self,
        decision: str | None = None,
        limit: int = 100,
    ) -> list[Evaluation]:
        """List evaluations with optional filters."""
        with self._connect() as conn:
            query = "SELECT * FROM evaluations WHERE 1=1"
            params: list[Any] = []

            if decision is not None:
                query += " AND decision = ?"
                params.append(decision)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            return [Evaluation(**dict(row)) for row in rows]

    # =========================================================================
    # Artifacts
    # =========================================================================

    def save_artifact(self, evaluation_id: int, artifact_type: str, path: str) -> Artifact:
        """Save an artifact record."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO artifacts (evaluation_id, type, path, created_at) VALUES (?, ?, ?, ?)",
                (evaluation_id, artifact_type, path, self._now()),
            )
            row = conn.execute(
                "SELECT * FROM artifacts WHERE evaluation_id = ? AND type = ? AND path = ?",
                (evaluation_id, artifact_type, path),
            ).fetchone()
            return Artifact(**dict(row))

    def get_artifacts(self, evaluation_id: int) -> list[Artifact]:
        """Get artifacts for an evaluation."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM artifacts WHERE evaluation_id = ?", (evaluation_id,)).fetchall()
            return [Artifact(**dict(row)) for row in rows]
