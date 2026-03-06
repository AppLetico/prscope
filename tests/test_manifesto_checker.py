from __future__ import annotations

from prscope.memory import ParsedConstraint
from prscope.planning.runtime.review.manifesto_checker import ManifestoChecker

_HC001 = ParsedConstraint(
    id="HARD_CONSTRAINT_001",
    text="Do not expose secrets.",
    severity="hard",
    evidence_keywords=["secret", "api_key", "token", "password"],
)
_HC002 = ParsedConstraint(
    id="HARD_CONSTRAINT_002",
    text="Do not perform destructive ops without rollback.",
    severity="hard",
    evidence_keywords=["drop table", "truncate", "rm -rf", "shutil.rmtree"],
)


def test_manifesto_checker_ignores_safe_negative_secret_guidance() -> None:
    checker = ManifestoChecker()
    result = checker.validate(
        plan_content=("Add observability without logging secrets. Do not print API keys or tokens in logs."),
        constraints=[_HC001],
    )
    assert result.violations == []
    assert result.warnings == []


def test_manifesto_checker_flags_actual_secret_handling_risk() -> None:
    checker = ManifestoChecker()
    result = checker.validate(
        plan_content="Log API keys in the health response for debugging.",
        constraints=[_HC001],
    )
    assert result.violations == ["HARD_CONSTRAINT_001"]


def test_manifesto_checker_requires_rollback_for_destructive_ops() -> None:
    checker = ManifestoChecker()
    result = checker.validate(
        plan_content="Drop table users during cleanup and recreate it later.",
        constraints=[_HC002],
    )
    assert result.violations == ["HARD_CONSTRAINT_002"]


def test_manifesto_checker_allows_destructive_ops_with_explicit_rollback() -> None:
    checker = ManifestoChecker()
    result = checker.validate(
        plan_content=(
            "If cleanup is required, truncate the cache table only after backup, "
            "with a rollback plan to restore from the snapshot."
        ),
        constraints=[_HC002],
    )
    assert result.violations == []
