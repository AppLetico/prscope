from __future__ import annotations

from pathlib import Path

import pytest

from prscope.config import PlanningConfig, PrscopeConfig, RepoProfile
from prscope.planning.runtime.discovery_support import RefinementEvidenceRefresh
from prscope.planning.runtime.orchestration import PlanningRuntime
from prscope.planning.runtime.tools import ToolExecutor
from prscope.store import Store


def test_refinement_evidence_refresh_reuses_known_anchors_and_reads_adjacent_tests(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    tests_dir = tmp_path / "tests"
    src_dir.mkdir()
    tests_dir.mkdir()
    (src_dir / "auth_service.py").write_text("def authenticate_user():\n    return True\n", encoding="utf-8")
    (tests_dir / "test_auth_service.py").write_text(
        "from src.auth_service import authenticate_user\n\n\ndef test_authenticate_user():\n    assert authenticate_user()\n",
        encoding="utf-8",
    )

    refresh = RefinementEvidenceRefresh(ToolExecutor(tmp_path))
    result = refresh.refresh(
        user_message="Revisit the auth service tradeoff and test coverage.",
        reason="architecture_tradeoff",
        known_anchor_paths=["src/auth_service.py"],
    )

    assert result.reused_known_anchors is True
    assert "src/auth_service.py" in result.read_paths
    assert any(path.startswith("tests/") for path in result.read_paths)
    assert result.elapsed_ms >= 0


@pytest.mark.asyncio
async def test_handle_refinement_message_passes_bounded_evidence_into_full_refine(tmp_path: Path) -> None:
    store = Store(tmp_path / "test.db")
    config = PrscopeConfig(
        local_repo=str(tmp_path),
        planning=PlanningConfig(author_model="gpt-4o-mini", critic_model="gpt-4o-mini"),
    )
    repo = RepoProfile(name="repo", path=str(tmp_path))
    runtime = PlanningRuntime(store=store, config=config, repo=repo)

    session = store.create_planning_session(
        repo_name=repo.name,
        title="refinement-investigation",
        requirements="Preserve current architecture.",
        seed_type="chat",
        status="refining",
    )
    runtime._core(session.id).save_plan_version("# Plan\n\n## Architecture\nKeep current auth owner.\n", round_number=1)

    captured: dict[str, object] = {}

    real_refresh = runtime.refresh_refinement_evidence

    def fake_refresh(**kwargs):  # type: ignore[no-untyped-def]
        captured["refresh_kwargs"] = kwargs
        return real_refresh(
            user_message="auth architecture tradeoff",
            reason="architecture_tradeoff",
            known_anchor_paths=[],
        )

    async def fake_round(*, refinement_evidence=None, **kwargs):  # type: ignore[no-untyped-def]
        captured["refinement_evidence"] = refinement_evidence
        captured["round_kwargs"] = kwargs
        return None

    runtime.refresh_refinement_evidence = fake_refresh  # type: ignore[method-assign]
    runtime.run_adversarial_round = fake_round  # type: ignore[method-assign]

    mode, reply = await runtime.handle_refinement_message(
        session_id=session.id,
        user_message="We should revisit the architecture tradeoff for auth ownership.",
    )

    assert mode == "refine_round"
    assert reply is None
    evidence = captured.get("refinement_evidence")
    assert isinstance(evidence, dict)
    assert evidence.get("reason") == "architecture_tradeoff"
