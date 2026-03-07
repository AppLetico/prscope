from __future__ import annotations

import sqlite3

CURRENT_SCHEMA_VERSION = 19

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

-- Repo profile snapshots keyed by git HEAD SHA
CREATE TABLE IF NOT EXISTS repo_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_root TEXT NOT NULL,
    profile_sha TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(repo_root, profile_sha)
);

-- Tracked upstream repositories
CREATE TABLE IF NOT EXISTS upstream_repos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT UNIQUE NOT NULL,
    last_synced_at TEXT,
    last_seen_updated_at TEXT
);

-- PR metadata
CREATE TABLE IF NOT EXISTS pull_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_id INTEGER NOT NULL,
    number INTEGER NOT NULL,
    state TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT,
    author TEXT,
    labels_json TEXT,
    updated_at TEXT,
    merged_at TEXT,
    head_sha TEXT,
    html_url TEXT,
    FOREIGN KEY (repo_id) REFERENCES upstream_repos(id),
    UNIQUE(repo_id, number)
);

-- Files changed per PR
CREATE TABLE IF NOT EXISTS pr_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pr_id INTEGER NOT NULL,
    path TEXT NOT NULL,
    additions INTEGER DEFAULT 0,
    deletions INTEGER DEFAULT 0,
    FOREIGN KEY (pr_id) REFERENCES pull_requests(id)
);

-- Evaluation results with deduplication key
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pr_id INTEGER NOT NULL,
    local_profile_sha TEXT NOT NULL,
    pr_head_sha TEXT NOT NULL,
    rule_score REAL,
    final_score REAL,
    matched_features_json TEXT,
    signals_json TEXT,
    llm_json TEXT,
    decision TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (pr_id) REFERENCES pull_requests(id),
    UNIQUE(pr_id, local_profile_sha, pr_head_sha)
);

-- Generated artifacts (plans, specs, etc.)
CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id INTEGER NOT NULL,
    type TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (evaluation_id) REFERENCES evaluations(id)
);

-- Planning sessions (interactive plan lifecycle)
CREATE TABLE IF NOT EXISTS planning_sessions (
    id TEXT PRIMARY KEY,
    repo_name TEXT NOT NULL,
    title TEXT NOT NULL,
    requirements TEXT NOT NULL,
    author_model TEXT,
    critic_model TEXT,
    status TEXT NOT NULL,
    seed_type TEXT NOT NULL,
    seed_ref TEXT,
    current_round INTEGER NOT NULL DEFAULT 0,
    no_recall INTEGER NOT NULL DEFAULT 0,
    pending_questions_json TEXT,
    phase_message TEXT,
    is_processing INTEGER NOT NULL DEFAULT 0,
    processing_started_at TEXT,
    last_commands_json TEXT,
    active_tool_calls_json TEXT,
    completed_tool_call_groups_json TEXT,
    current_command_id TEXT,
    session_total_cost_usd REAL,
    max_prompt_tokens INTEGER,
    confidence_trend REAL,
    converged_early INTEGER,
    clarifications_log_json TEXT,
    diagnostics_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Durable command log rows for executor semantics
CREATE TABLE IF NOT EXISTS planning_commands (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    command TEXT,
    command_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    payload_json TEXT NOT NULL,
    result_snapshot_json TEXT,
    started_at TEXT,
    completed_at TEXT,
    last_error TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    worker_id TEXT,
    lease_expires_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES planning_sessions(id),
    UNIQUE(session_id, command_id)
);

-- Planning conversation turns
CREATE TABLE IF NOT EXISTS planning_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    round INTEGER NOT NULL,
    major_issues_remaining INTEGER,
    minor_issues_remaining INTEGER,
    hard_constraint_violations TEXT,
    parse_error TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES planning_sessions(id)
);

-- Plan snapshots by round
CREATE TABLE IF NOT EXISTS plan_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    round INTEGER NOT NULL,
    plan_content TEXT NOT NULL,
    plan_json TEXT,
    decision_graph_json TEXT,
    followups_json TEXT,
    plan_sha TEXT NOT NULL,
    changed_sections TEXT,
    diff_from_previous TEXT,
    convergence_score REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES planning_sessions(id)
);

-- Round-level planning metrics
CREATE TABLE IF NOT EXISTS planning_round_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    round INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    author_prompt_tokens INTEGER,
    author_completion_tokens INTEGER,
    critic_prompt_tokens INTEGER,
    critic_completion_tokens INTEGER,
    max_prompt_tokens INTEGER,
    major_issues INTEGER,
    minor_issues INTEGER,
    critic_confidence REAL,
    vagueness_score REAL,
    citation_count INTEGER,
    constraint_violations_json TEXT,
    resolved_since_last_round_json TEXT,
    clarifications_this_round INTEGER,
    call_cost_usd REAL,
    issues_resolved INTEGER,
    issues_introduced INTEGER,
    net_improvement INTEGER,
    time_to_first_tool_call INTEGER,
    grounding_ratio REAL,
    static_injection_tokens_pct REAL,
    rejected_for_no_discovery INTEGER,
    rejected_for_grounding INTEGER,
    rejected_for_budget INTEGER,
    average_read_depth_per_round REAL,
    time_between_tool_calls REAL,
    rejection_reasons_json TEXT,
    plan_quality_score REAL,
    unsupported_claims_count INTEGER,
    missing_evidence_count INTEGER,
    FOREIGN KEY (session_id) REFERENCES planning_sessions(id)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_pr_repo ON pull_requests(repo_id);
CREATE INDEX IF NOT EXISTS idx_pr_state ON pull_requests(state);
CREATE INDEX IF NOT EXISTS idx_pr_updated ON pull_requests(updated_at);
CREATE INDEX IF NOT EXISTS idx_eval_pr ON evaluations(pr_id);
CREATE INDEX IF NOT EXISTS idx_eval_decision ON evaluations(decision);
CREATE INDEX IF NOT EXISTS idx_turns_session_round ON planning_turns(session_id, round);
CREATE INDEX IF NOT EXISTS idx_versions_session_round ON plan_versions(session_id, round);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON planning_sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_repo ON planning_sessions(repo_name);
CREATE INDEX IF NOT EXISTS idx_round_metrics_session_round ON planning_round_metrics(session_id, round);
CREATE INDEX IF NOT EXISTS idx_planning_commands_status_created ON planning_commands(status, created_at);
CREATE INDEX IF NOT EXISTS idx_planning_commands_session_status ON planning_commands(session_id, status);
"""


class StoreSchemaMixin:
    def _ensure_schema(self) -> None:
        """Ensure database schema exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            self._run_migrations(conn)

    def _get_schema_version(self, conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1").fetchone()
        if row is None:
            return 0
        value = row[0]
        return int(value) if value is not None else 0

    def _set_schema_version(self, conn: sqlite3.Connection, version: int) -> None:
        conn.execute("DELETE FROM schema_version")
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))

    def _column_exists(self, conn: sqlite3.Connection, table: str, column: str) -> bool:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return any(str(row[1]) == column for row in rows)

    def _run_migrations(self, conn: sqlite3.Connection) -> None:
        current = self._get_schema_version(conn)
        if current >= CURRENT_SCHEMA_VERSION:
            return

        # v1 -> v2: persist diff/convergence metadata on plan versions
        if current < 2:
            if not self._column_exists(conn, "plan_versions", "diff_from_previous"):
                conn.execute("ALTER TABLE plan_versions ADD COLUMN diff_from_previous TEXT")
            if not self._column_exists(conn, "plan_versions", "convergence_score"):
                conn.execute("ALTER TABLE plan_versions ADD COLUMN convergence_score REAL")
            current = 2

        # v2 -> v3: session telemetry + round metrics
        if current < 3:
            if not self._column_exists(conn, "planning_sessions", "session_total_cost_usd"):
                conn.execute("ALTER TABLE planning_sessions ADD COLUMN session_total_cost_usd REAL")
            if not self._column_exists(conn, "planning_sessions", "max_prompt_tokens"):
                conn.execute("ALTER TABLE planning_sessions ADD COLUMN max_prompt_tokens INTEGER")
            if not self._column_exists(conn, "planning_sessions", "clarifications_log_json"):
                conn.execute("ALTER TABLE planning_sessions ADD COLUMN clarifications_log_json TEXT")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS planning_round_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    round INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    author_prompt_tokens INTEGER,
                    author_completion_tokens INTEGER,
                    critic_prompt_tokens INTEGER,
                    critic_completion_tokens INTEGER,
                    max_prompt_tokens INTEGER,
                    major_issues INTEGER,
                    minor_issues INTEGER,
                    critic_confidence REAL,
                    vagueness_score REAL,
                    citation_count INTEGER,
                    constraint_violations_json TEXT,
                    resolved_since_last_round_json TEXT,
                    clarifications_this_round INTEGER,
                    call_cost_usd REAL,
                    issues_resolved INTEGER,
                    issues_introduced INTEGER,
                    net_improvement INTEGER,
                    FOREIGN KEY (session_id) REFERENCES planning_sessions(id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_round_metrics_session_round "
                "ON planning_round_metrics(session_id, round)"
            )
            current = 3

        # v3 -> v4: confidence/convergence metadata on sessions
        if current < 4:
            if not self._column_exists(conn, "planning_sessions", "confidence_trend"):
                conn.execute("ALTER TABLE planning_sessions ADD COLUMN confidence_trend REAL")
            if not self._column_exists(conn, "planning_sessions", "converged_early"):
                conn.execute("ALTER TABLE planning_sessions ADD COLUMN converged_early INTEGER")
            current = 4

        # v4 -> v5: additional harness metrics on round metrics
        if current < 5:
            additions = (
                ("planning_round_metrics", "time_to_first_tool_call", "INTEGER"),
                ("planning_round_metrics", "grounding_ratio", "REAL"),
                ("planning_round_metrics", "static_injection_tokens_pct", "REAL"),
                ("planning_round_metrics", "rejected_for_no_discovery", "INTEGER"),
                ("planning_round_metrics", "rejected_for_grounding", "INTEGER"),
                ("planning_round_metrics", "rejected_for_budget", "INTEGER"),
            )
            for table, column, sql_type in additions:
                if not self._column_exists(conn, table, column):
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
            current = 5

        # v5 -> v6: deeper exploration and rejection telemetry
        if current < 6:
            additions = (
                ("planning_round_metrics", "average_read_depth_per_round", "REAL"),
                ("planning_round_metrics", "time_between_tool_calls", "REAL"),
                ("planning_round_metrics", "rejection_reasons_json", "TEXT"),
            )
            for table, column, sql_type in additions:
                if not self._column_exists(conn, table, column):
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
            current = 6

        # v6 -> v7: plan quality and evidence-gap metrics
        if current < 7:
            additions = (
                ("planning_round_metrics", "plan_quality_score", "REAL"),
                ("planning_round_metrics", "unsupported_claims_count", "INTEGER"),
                ("planning_round_metrics", "missing_evidence_count", "INTEGER"),
            )
            for table, column, sql_type in additions:
                if not self._column_exists(conn, table, column):
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
            current = 7

        # v7 -> v8: persist per-session model defaults
        if current < 8:
            additions = (
                ("planning_sessions", "author_model", "TEXT"),
                ("planning_sessions", "critic_model", "TEXT"),
            )
            for table, column, sql_type in additions:
                if not self._column_exists(conn, table, column):
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
            current = 8

        # v8 -> v9: persist per-session no_recall mode
        if current < 9:
            if not self._column_exists(conn, "planning_sessions", "no_recall"):
                conn.execute("ALTER TABLE planning_sessions ADD COLUMN no_recall INTEGER NOT NULL DEFAULT 0")
            current = 9

        # v9 -> v10: canonical session workflow state columns
        if current < 10:
            additions = (
                ("planning_sessions", "pending_questions_json", "TEXT"),
                ("planning_sessions", "phase_message", "TEXT"),
                ("planning_sessions", "is_processing", "INTEGER NOT NULL DEFAULT 0"),
                ("planning_sessions", "processing_started_at", "TEXT"),
                ("planning_sessions", "last_commands_json", "TEXT"),
                ("planning_sessions", "active_tool_calls_json", "TEXT"),
                ("planning_sessions", "current_command_id", "TEXT"),
            )
            for table, column, sql_type in additions:
                if not self._column_exists(conn, table, column):
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
            conn.execute("UPDATE planning_sessions SET status = 'draft' WHERE status = 'discovery'")
            current = 10

        # v10 -> v11: durable async command log rows
        if current < 11:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS planning_commands (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    command TEXT,
                    command_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    payload_json TEXT NOT NULL,
                    result_snapshot_json TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    last_error TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    worker_id TEXT,
                    lease_expires_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES planning_sessions(id),
                    UNIQUE(session_id, command_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_planning_commands_status_created "
                "ON planning_commands(status, created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_planning_commands_session_status "
                "ON planning_commands(session_id, status)"
            )
            current = 11

        # v11 -> v12: command executor fields + lock/query index
        if current < 12:
            additions = (
                ("planning_sessions", "current_command_id", "TEXT"),
                ("planning_commands", "command", "TEXT"),
                ("planning_commands", "result_snapshot_json", "TEXT"),
                ("planning_commands", "started_at", "TEXT"),
                ("planning_commands", "completed_at", "TEXT"),
                ("planning_commands", "last_error", "TEXT"),
            )
            for table, column, sql_type in additions:
                if not self._column_exists(conn, table, column):
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_planning_commands_session_status "
                "ON planning_commands(session_id, status)"
            )
            current = 12

        # v12 -> v13: naming harmonization planning_jobs -> planning_commands
        if current < 13:
            jobs_exists = (
                conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'planning_jobs'").fetchone()
                is not None
            )
            commands_exists = (
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'planning_commands'"
                ).fetchone()
                is not None
            )
            if jobs_exists and not commands_exists:
                conn.execute("ALTER TABLE planning_jobs RENAME TO planning_commands")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_planning_commands_status_created "
                "ON planning_commands(status, created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_planning_commands_session_status "
                "ON planning_commands(session_id, status)"
            )
            current = 13

        # v13 -> v14: normalize legacy planning status names to canonical artifact states
        if current < 14:
            conn.execute(
                """
                UPDATE planning_sessions
                SET status = 'draft'
                WHERE status IN ('created', 'preparing', 'discovery', 'discovering', 'drafting')
                """
            )
            conn.execute(
                """
                UPDATE planning_sessions
                SET status = 'approved'
                WHERE status = 'exported'
                """
            )
            current = 14

        # v14 -> v15: persist completed tool call groups for refresh-safe UI history
        if current < 15:
            if not self._column_exists(conn, "planning_sessions", "completed_tool_call_groups_json"):
                conn.execute("ALTER TABLE planning_sessions ADD COLUMN completed_tool_call_groups_json TEXT")
            conn.execute(
                """
                UPDATE planning_sessions
                SET completed_tool_call_groups_json = COALESCE(completed_tool_call_groups_json, '[]')
                """
            )
            current = 15

        if current < 16:
            if not self._column_exists(conn, "planning_sessions", "event_seq"):
                conn.execute("ALTER TABLE planning_sessions ADD COLUMN event_seq INTEGER NOT NULL DEFAULT 0")
            if not self._column_exists(conn, "planning_turns", "sequence"):
                conn.execute("ALTER TABLE planning_turns ADD COLUMN sequence INTEGER")
            current = 16

        # v16 -> v17: structured plan source-of-truth columns
        if current < 17:
            if not self._column_exists(conn, "plan_versions", "plan_json"):
                conn.execute("ALTER TABLE plan_versions ADD COLUMN plan_json TEXT")
            if not self._column_exists(conn, "plan_versions", "changed_sections"):
                conn.execute("ALTER TABLE plan_versions ADD COLUMN changed_sections TEXT")
            current = 17

        # v17 -> v18: persist compact per-session diagnostics payload
        if current < 18:
            if not self._column_exists(conn, "planning_sessions", "diagnostics_json"):
                conn.execute("ALTER TABLE planning_sessions ADD COLUMN diagnostics_json TEXT")
            conn.execute(
                """
                UPDATE planning_sessions
                SET diagnostics_json = COALESCE(diagnostics_json, '{}')
                """
            )
            current = 18

        # v18 -> v19: versioned decision graph and follow-up artifacts on plan versions
        if current < 19:
            if not self._column_exists(conn, "plan_versions", "decision_graph_json"):
                conn.execute("ALTER TABLE plan_versions ADD COLUMN decision_graph_json TEXT")
            if not self._column_exists(conn, "plan_versions", "followups_json"):
                conn.execute("ALTER TABLE plan_versions ADD COLUMN followups_json TEXT")
            current = 19

        self._set_schema_version(conn, current)
