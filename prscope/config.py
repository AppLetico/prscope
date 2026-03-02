"""
Configuration management for Prscope.

Loads and validates:
- prscope.yml: Main configuration (upstream repos, thresholds, planning config)
- prscope.features.yml: Feature definitions for matching
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class UpstreamRepo:
    """Configuration for an upstream repository to monitor."""

    repo: str  # full_name like "owner/repo"
    filters: dict[str, Any] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        return self.repo


@dataclass
class ScoringConfig:
    """Scoring thresholds and weights."""

    min_rule_score: float = 0.3
    min_final_score: float = 0.5
    keyword_weight: float = 0.4
    path_weight: float = 0.6


@dataclass
class SyncConfig:
    """Default sync settings."""

    state: str = "merged"  # merged, open, closed, all
    max_prs: int = 100
    fetch_files: bool = True
    since: str = "90d"  # Initial window (ISO date or relative)
    incremental: bool = True
    eval_batch_size: int = 25


@dataclass
class UpstreamEvalConfig:
    """LLM configuration for upstream PR semantic scoring."""

    enabled: bool = False
    model: str = "gpt-4o"
    temperature: float = 0.3
    max_tokens: int = 2000


# Backwards-compatible alias
LLMConfig = UpstreamEvalConfig


@dataclass
class PlanningConfig:
    """Planning mode configuration."""

    author_model: str = "gpt-4o"
    critic_model: str = "claude-3-5-sonnet-20241022"
    max_adversarial_rounds: int = 10
    convergence_threshold: float = 0.05
    output_dir: str = "./plans"
    scan_depth: int = 3
    memory_concurrency: int = 2
    discovery_max_turns: int = 5
    discovery_tool_rounds: int = 25  # max codebase tool calls per discovery turn (like Cursor plan mode)
    author_tool_rounds: int = 25     # max codebase tool calls per author draft/refinement round
    seed_token_budget: int = 4000
    require_verified_file_references: bool = False
    validate_temperature: float = 0.0
    validate_audit_log: bool = True
    # Codebase scanner backend: "grep" (default, no deps) | "repomap" (aider tree-sitter) | "repomix" (repomix CLI)
    scanner: str = "grep"
    memory_block_max_chars: dict[str, int] = field(
        default_factory=lambda: {
            "architecture": 3000,
            "modules": 2000,
            "manifesto": 1500,
        }
    )
    clarification_timeout_seconds: int = 600


@dataclass
class Feature:
    """A feature definition for matching PRs."""

    name: str
    keywords: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class ProjectConfig:
    """Project description for LLM context."""

    name: str = ""
    description: str = ""


@dataclass
class RepoProfile:
    """Per-repository planning profile."""

    name: str
    path: str
    upstream: list[UpstreamRepo] = field(default_factory=list)
    manifesto_file: str | None = None
    output_dir: str | None = None
    memory_block_max_chars: dict[str, int] | None = None

    @property
    def resolved_path(self) -> Path:
        return Path(self.path).expanduser().resolve()

    @property
    def resolved_manifesto(self) -> Path:
        if self.manifesto_file:
            return Path(self.manifesto_file).expanduser().resolve()
        return self.resolved_path / ".prscope" / "manifesto.md"

    @property
    def memory_dir(self) -> Path:
        return Path.home() / ".prscope" / "repos" / self.name / "memory"

    @property
    def audit_dir(self) -> Path:
        return Path.home() / ".prscope" / "repos" / self.name / "audit"


@dataclass
class PrscopeConfig:
    """Complete Prscope configuration."""

    local_repo: str | None = None
    repos: dict[str, RepoProfile] = field(default_factory=dict)
    project: ProjectConfig = field(default_factory=ProjectConfig)
    upstream: list[UpstreamRepo] = field(default_factory=list)
    sync: SyncConfig = field(default_factory=SyncConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    upstream_eval: UpstreamEvalConfig = field(default_factory=UpstreamEvalConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    features: list[Feature] = field(default_factory=list)
    repo_root: Path | None = None

    def get_repo(self, name: str) -> RepoProfile:
        if name in self.repos:
            return self.repos[name]
        raise ValueError(f"Repo '{name}' not found. Run: prscope repos list")

    def list_repos(self) -> list[RepoProfile]:
        if self.repos:
            return list(self.repos.values())
        fallback = self._fallback_repo()
        return [fallback] if fallback else []

    def _fallback_repo(self) -> RepoProfile | None:
        if self.local_repo:
            path = Path(self.local_repo).expanduser()
            if not path.is_absolute():
                base = self.repo_root or get_repo_root()
                path = base / path
            resolved = path.resolve()
            return RepoProfile(
                name=resolved.name,
                path=str(resolved),
                upstream=self.upstream.copy(),
                output_dir=self.planning.output_dir,
            )
        return None

    def resolve_repo(self, name: str | None = None, cwd: Path | None = None) -> RepoProfile:
        """Resolve active repo from explicit name or cwd."""
        if name:
            if name in self.repos:
                return self.repos[name]
            fallback = self._fallback_repo()
            if fallback and name in {fallback.name, fallback.path, str(fallback.resolved_path)}:
                return fallback
            raise ValueError(f"Repo '{name}' not found. Run: prscope repos list")

        if cwd is None:
            cwd = get_repo_root()
        cwd = cwd.resolve()

        for repo in self.repos.values():
            try:
                cwd.relative_to(repo.resolved_path)
                return repo
            except ValueError:
                continue

        fallback = self._fallback_repo()
        if fallback:
            return fallback

        raise ValueError("No repo specified and cannot auto-detect. Use --repo <name>.")

    def get_local_repo_path(self) -> Path:
        """Backward-compatible single-repo accessor."""
        try:
            return self.resolve_repo().resolved_path
        except ValueError:
            return get_repo_root()

    @classmethod
    def load(cls, repo_root: Path) -> "PrscopeConfig":
        """Load configuration from repo root directory."""
        config = cls(repo_root=repo_root.resolve())

        main_config_path = repo_root / "prscope.yml"
        if main_config_path.exists():
            with open(main_config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            config = cls._parse_main_config(data, repo_root=repo_root.resolve())

        features_path = repo_root / "prscope.features.yml"
        if features_path.exists():
            with open(features_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            config.features = cls._parse_features(data)

        return config

    @staticmethod
    def _parse_upstream_repos(raw: list[Any]) -> list[UpstreamRepo]:
        upstream: list[UpstreamRepo] = []
        for repo_data in raw:
            if isinstance(repo_data, str):
                upstream.append(UpstreamRepo(repo=repo_data))
            elif isinstance(repo_data, dict):
                upstream.append(
                    UpstreamRepo(
                        repo=repo_data.get("repo", ""),
                        filters=repo_data.get("filters", {}),
                    )
                )
        return upstream

    @classmethod
    def _parse_main_config(cls, data: dict[str, Any], repo_root: Path) -> "PrscopeConfig":
        config = cls(repo_root=repo_root)
        config.local_repo = data.get("local_repo")

        project_data = data.get("project", {})
        config.project = ProjectConfig(
            name=project_data.get("name", ""),
            description=project_data.get("description", ""),
        )

        config.upstream = cls._parse_upstream_repos(data.get("upstream", []))

        scoring_data = data.get("scoring", {})
        config.scoring = ScoringConfig(
            min_rule_score=scoring_data.get("min_rule_score", 0.3),
            min_final_score=scoring_data.get("min_final_score", 0.5),
            keyword_weight=scoring_data.get("keyword_weight", 0.4),
            path_weight=scoring_data.get("path_weight", 0.6),
        )

        sync_data = data.get("sync", {})
        config.sync = SyncConfig(
            state=sync_data.get("state", "merged"),
            max_prs=sync_data.get("max_prs", 100),
            fetch_files=sync_data.get("fetch_files", True),
            since=sync_data.get("since", "90d"),
            incremental=sync_data.get("incremental", True),
            eval_batch_size=sync_data.get("eval_batch_size", 25),
        )

        # Accept both "upstream_eval:" (new) and "llm:" (backwards compat)
        upstream_eval_data = data.get("upstream_eval") or data.get("llm") or {}
        config.upstream_eval = UpstreamEvalConfig(
            enabled=upstream_eval_data.get("enabled", False),
            model=upstream_eval_data.get("model", "gpt-4o"),
            temperature=upstream_eval_data.get("temperature", 0.3),
            max_tokens=upstream_eval_data.get("max_tokens", 2000),
        )

        planning_data = data.get("planning", {})
        memory_caps_raw = planning_data.get("memory_block_max_chars", {})
        memory_caps = {
            "architecture": int(memory_caps_raw.get("architecture", 3000)),
            "modules": int(memory_caps_raw.get("modules", 2000)),
            "manifesto": int(memory_caps_raw.get("manifesto", 1500)),
        }
        config.planning = PlanningConfig(
            author_model=planning_data.get("author_model", "gpt-4o"),
            critic_model=planning_data.get("critic_model", "claude-3-5-sonnet-20241022"),
            max_adversarial_rounds=planning_data.get("max_adversarial_rounds", 10),
            convergence_threshold=planning_data.get("convergence_threshold", 0.05),
            output_dir=planning_data.get("output_dir", "./plans"),
            scan_depth=planning_data.get("scan_depth", 3),
            memory_concurrency=planning_data.get("memory_concurrency", 2),
            discovery_max_turns=planning_data.get("discovery_max_turns", 5),
            seed_token_budget=planning_data.get("seed_token_budget", 4000),
            require_verified_file_references=planning_data.get(
                "require_verified_file_references",
                False,
            ),
            validate_temperature=planning_data.get("validate_temperature", 0.0),
            validate_audit_log=planning_data.get("validate_audit_log", True),
            discovery_tool_rounds=planning_data.get("discovery_tool_rounds", 25),
            author_tool_rounds=planning_data.get("author_tool_rounds", 25),
            scanner=planning_data.get("scanner", "grep"),
            memory_block_max_chars=memory_caps,
            clarification_timeout_seconds=int(
                planning_data.get("clarification_timeout_seconds", 600)
            ),
        )

        repos_data = data.get("repos", {})
        if isinstance(repos_data, dict):
            for name, repo_data in repos_data.items():
                if not isinstance(repo_data, dict):
                    continue
                raw_path = repo_data.get("path", "")
                path = Path(raw_path).expanduser()
                if not path.is_absolute():
                    path = repo_root / path

                profile = RepoProfile(
                    name=name,
                    path=str(path.resolve()),
                    upstream=cls._parse_upstream_repos(repo_data.get("upstream", [])),
                    manifesto_file=repo_data.get("manifesto_file"),
                    output_dir=repo_data.get("output_dir"),
                    memory_block_max_chars={
                        "architecture": int(
                            repo_data.get("memory_block_max_chars", {}).get(
                                "architecture", memory_caps["architecture"]
                            )
                        ),
                        "modules": int(
                            repo_data.get("memory_block_max_chars", {}).get(
                                "modules", memory_caps["modules"]
                            )
                        ),
                        "manifesto": int(
                            repo_data.get("memory_block_max_chars", {}).get(
                                "manifesto", memory_caps["manifesto"]
                            )
                        ),
                    },
                )
                config.repos[name] = profile

        return config

    @classmethod
    def _parse_features(cls, data: dict[str, Any]) -> list[Feature]:
        features: list[Feature] = []
        features_data = data.get("features", {})

        for name, feature_data in features_data.items():
            if isinstance(feature_data, dict):
                features.append(
                    Feature(
                        name=name,
                        keywords=feature_data.get("keywords", []),
                        paths=feature_data.get("paths", []),
                        description=feature_data.get("description", ""),
                    )
                )

        return features


def get_repo_root() -> Path:
    """Find the repository root (directory containing .git)."""

    current = Path.cwd()
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return Path.cwd()


def get_prscope_dir(repo_root: Path | None = None) -> Path:
    """Get the .prscope directory path."""

    if repo_root is None:
        repo_root = get_repo_root()
    return repo_root / ".prscope"


def ensure_prscope_dir(repo_root: Path | None = None) -> Path:
    """Ensure .prscope directory exists and return its path."""

    prscope_dir = get_prscope_dir(repo_root)
    prscope_dir.mkdir(parents=True, exist_ok=True)
    return prscope_dir
