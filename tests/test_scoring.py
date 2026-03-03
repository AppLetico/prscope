from __future__ import annotations

from prscope.config import Feature, PrscopeConfig, ScoringConfig
from prscope.scoring import match_keyword, match_path_glob, score_pr, score_pr_rules, tokenize
from prscope.store import PRFile, PullRequest


def test_tokenize():
    tokens = tokenize("Add security checks for JWT auth")
    assert "add" in tokens
    assert "security" in tokens
    assert "jwt" in tokens
    assert "auth" in tokens


def test_match_keyword():
    tokens = tokenize("Add security checks for authentication")
    assert match_keyword("security", tokens) is True
    assert match_keyword("auth", tokens) is True  # substring match
    assert match_keyword("random", tokens) is False


def test_match_path_glob():
    assert match_path_glob("**/auth/**", "src/auth/handler.py") is True
    assert match_path_glob("**/auth/**", "src/api/users.py") is False
    assert match_path_glob("*.py", "handler.py") is True
    assert match_path_glob("**/*.ts", "src/components/Button.tsx") is False


def test_score_pr_rules():
    config = PrscopeConfig(
        scoring=ScoringConfig(min_rule_score=0.3, min_final_score=0.5),
        features=[
            Feature(name="security", keywords=["security", "auth"], paths=["**/auth/**"]),
        ],
    )

    pr = PullRequest(
        id=1,
        repo_id=1,
        number=42,
        state="closed",
        title="Fix auth bug",
        body="Security patch",
        author="alice",
        labels_json=None,
        updated_at=None,
        merged_at=None,
        head_sha="sha1",
        html_url="",
    )
    files = [PRFile(id=1, pr_id=1, path="src/auth/login.py", additions=5, deletions=2)]

    rule_score, matched, matches = score_pr_rules(pr, files, config)

    assert rule_score == 1.0
    assert "security" in matched
    assert matches[0].keyword_score == 1.0
    assert matches[0].path_score == 1.0


def test_score_pr_skip_low_relevance():
    config = PrscopeConfig(
        scoring=ScoringConfig(min_rule_score=0.3, min_final_score=0.5),
        features=[
            Feature(name="security", keywords=["security"], paths=["**/auth/**"]),
        ],
    )

    pr = PullRequest(
        id=1,
        repo_id=1,
        number=42,
        state="closed",
        title="Update README",
        body="Documentation changes",
        author="alice",
        labels_json=None,
        updated_at=None,
        merged_at=None,
        head_sha="sha1",
        html_url="",
    )
    files = [PRFile(id=1, pr_id=1, path="README.md", additions=10, deletions=0)]

    result = score_pr(pr, files, config, run_semantic=False, run_llm=False)

    assert result.rule_score == 0.0
    assert result.final_decision == "skip"


def test_score_pr_relevant_decision():
    config = PrscopeConfig(
        scoring=ScoringConfig(min_rule_score=0.3, min_final_score=0.5),
        features=[
            Feature(
                name="security",
                keywords=["security"],
                paths=["**/auth/**"],
            )
        ],
    )

    pr = PullRequest(
        id=1,
        repo_id=1,
        number=42,
        state="closed",
        title="Add security checks",
        body="",
        author="alice",
        labels_json=None,
        updated_at="2024-01-01T00:00:00Z",
        merged_at="2024-01-02T00:00:00Z",
        head_sha="sha123",
        html_url="https://example.com/pr/42",
    )
    files = [
        PRFile(
            id=1,
            pr_id=1,
            path="src/auth/guard.py",
            additions=10,
            deletions=0,
        )
    ]

    result = score_pr(pr, files, config, run_semantic=False, run_llm=False)

    assert result.rule_score == 1.0
    assert result.final_score == 1.0
    assert result.final_decision == "relevant"
    assert result.matched_features == ["security"]
