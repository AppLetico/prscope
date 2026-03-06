from __future__ import annotations

from unittest.mock import patch

from prscope.config import PrscopeConfig


def test_config_load_project_section(tmp_path):
    config_path = tmp_path / "prscope.yml"
    config_path.write_text(
        """
local_repo: .
project:
  name: My Project
  description: |
    This is a test project.
    RELEVANT: feature A
    NOT RELEVANT: feature B
        """.strip()
    )

    config = PrscopeConfig.load(tmp_path)

    assert config.project.name == "My Project"
    assert "test project" in config.project.description
    assert "RELEVANT" in config.project.description


def test_config_project_defaults_to_empty(tmp_path):
    config_path = tmp_path / "prscope.yml"
    config_path.write_text("local_repo: .")

    config = PrscopeConfig.load(tmp_path)

    assert config.project.name == ""
    assert config.project.description == ""


def test_config_load_local_repo_and_defaults(tmp_path):
    config_path = tmp_path / "prscope.yml"
    config_path.write_text(
        """
local_repo: ./local
sync:
  state: open
  max_prs: 5
  fetch_files: false
llm:
  enabled: true
  model: claude-3-opus
  temperature: 0.2
  max_tokens: 1500
        """.strip()
    )

    config = PrscopeConfig.load(tmp_path)

    assert config.local_repo == "./local"

    # Mock get_repo_root to return tmp_path for relative path resolution
    with patch("prscope.config.get_repo_root", return_value=tmp_path):
        assert config.get_local_repo_path() == (tmp_path / "local").resolve()

    assert config.sync.state == "open"
    assert config.sync.max_prs == 5
    assert config.sync.fetch_files is False
    assert config.upstream_eval.enabled is True
    assert config.upstream_eval.model == "claude-3-opus"
    assert config.upstream_eval.temperature == 0.2
    assert config.upstream_eval.max_tokens == 1500


def test_config_loads_repos_and_planning_settings(tmp_path):
    (tmp_path / "repo-a").mkdir()
    (tmp_path / "repo-b").mkdir()
    config_path = tmp_path / "prscope.yml"
    config_path.write_text(
        """
repos:
  alpha:
    path: ./repo-a
    upstream:
      - repo: owner/repo-a
  beta:
    path: ./repo-b
planning:
  author_model: gpt-4o
  critic_model: claude-3-5-sonnet-20241022
  max_adversarial_rounds: 7
  require_verified_file_references: true
        """.strip()
    )

    config = PrscopeConfig.load(tmp_path)

    assert set(config.repos.keys()) == {"alpha", "beta"}
    assert config.get_repo("alpha").resolved_path == (tmp_path / "repo-a").resolve()
    assert config.get_repo("alpha").upstream[0].repo == "owner/repo-a"
    assert config.planning.max_adversarial_rounds == 7
    assert config.planning.require_verified_file_references is True
    # memory_model defaults to author_model when omitted
    assert config.planning.memory_model == "gpt-4o"


def test_config_memory_model_explicit(tmp_path):
    """When memory_model is set in YAML it is used; otherwise it defaults to author_model."""
    (tmp_path / "repo-mem").mkdir()
    config_path = tmp_path / "prscope.yml"
    config_path.write_text(
        """
repos:
  memrepo:
    path: ./repo-mem
planning:
  author_model: claude-3-5-sonnet-20241022
  memory_model: gpt-4o-mini
        """.strip()
    )
    config = PrscopeConfig.load(tmp_path)
    assert config.planning.author_model == "claude-3-5-sonnet-20241022"
    assert config.planning.memory_model == "gpt-4o-mini"


def test_config_resolve_repo_from_cwd(tmp_path):
    (tmp_path / "repo-c").mkdir()
    (tmp_path / "repo-c" / "nested").mkdir()
    config_path = tmp_path / "prscope.yml"
    config_path.write_text(
        """
repos:
  gamma:
    path: ./repo-c
        """.strip()
    )
    config = PrscopeConfig.load(tmp_path)

    resolved = config.resolve_repo(None, cwd=(tmp_path / "repo-c" / "nested"))
    assert resolved.name == "gamma"
    assert resolved.resolved_path == (tmp_path / "repo-c").resolve()


def test_config_loads_skills_and_recall_settings(tmp_path):
    (tmp_path / "repo-d").mkdir()
    config_path = tmp_path / "prscope.yml"
    config_path.write_text(
        """
repos:
  delta:
    path: ./repo-d
planning:
  skills_max_chars: 4200
  recall_prior_sessions: true
  recall_top_k: 3
  recall_max_chars: 1800
        """.strip()
    )

    config = PrscopeConfig.load(tmp_path)
    repo = config.get_repo("delta")

    assert repo.skills_dir == (tmp_path / "repo-d" / ".prscope" / "skills").resolve()
    assert config.planning.skills_max_chars == 4200
    assert config.planning.recall_prior_sessions is True
    assert config.planning.recall_top_k == 3
    assert config.planning.recall_max_chars == 1800
