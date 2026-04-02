from __future__ import annotations

from pathlib import Path

import pytest

from prscope.config import PlanningConfig, PrscopeConfig, RepoProfile
from prscope.memory import instruction_context_fingerprint
from prscope.planning.runtime.orchestration import PlanningRuntime
from prscope.store import Store


def test_instruction_context_fingerprint_changes_with_manifesto_edit(tmp_path: Path) -> None:
    prscope_dir = tmp_path / ".prscope"
    prscope_dir.mkdir(parents=True)
    manifesto = prscope_dir / "manifesto.md"
    manifesto.write_text("v1", encoding="utf-8")
    skills = prscope_dir / "skills"
    skills.mkdir()
    fp1 = instruction_context_fingerprint(manifesto, skills)
    manifesto.write_text("v2", encoding="utf-8")
    fp2 = instruction_context_fingerprint(manifesto, skills)
    assert fp1 != fp2


def test_instruction_context_fingerprint_includes_skills_files(tmp_path: Path) -> None:
    prscope_dir = tmp_path / ".prscope"
    prscope_dir.mkdir(parents=True)
    manifesto = prscope_dir / "manifesto.md"
    manifesto.write_text("m", encoding="utf-8")
    skills = prscope_dir / "skills"
    skills.mkdir()
    (skills / "a.md").write_text("a", encoding="utf-8")
    fp1 = instruction_context_fingerprint(manifesto, skills)
    (skills / "b.md").write_text("b", encoding="utf-8")
    fp2 = instruction_context_fingerprint(manifesto, skills)
    assert fp1 != fp2


@pytest.mark.asyncio
async def test_on_change_reloads_manifesto_after_edit(tmp_path: Path) -> None:
    prscope_dir = tmp_path / ".prscope"
    prscope_dir.mkdir(parents=True)
    manifesto = prscope_dir / "manifesto.md"
    manifesto.write_text("first", encoding="utf-8")
    skills = prscope_dir / "skills"
    skills.mkdir()

    cfg = PrscopeConfig(
        local_repo=str(tmp_path),
        planning=PlanningConfig(instruction_context_refresh="on_change"),
    )
    repo = RepoProfile(name="r", path=str(tmp_path))
    runtime = PlanningRuntime(store=Store(tmp_path / "s.db"), config=cfg, repo=repo)
    session = runtime.store.create_planning_session(
        repo_name="r",
        title="t",
        requirements="req",
        seed_type="requirements",
        status="draft",
    )
    s1 = runtime._state(session.id)  # noqa: SLF001
    assert s1.manifesto == "first"

    manifesto.write_text("second", encoding="utf-8")
    s2 = runtime._state(session.id)  # noqa: SLF001
    assert s2.manifesto == "second"


@pytest.mark.asyncio
async def test_each_turn_reloads_manifesto_every_state_access(tmp_path: Path) -> None:
    prscope_dir = tmp_path / ".prscope"
    prscope_dir.mkdir(parents=True)
    manifesto = prscope_dir / "manifesto.md"
    manifesto.write_text("a", encoding="utf-8")

    cfg = PrscopeConfig(
        local_repo=str(tmp_path),
        planning=PlanningConfig(instruction_context_refresh="each_turn"),
    )
    repo = RepoProfile(name="r", path=str(tmp_path))
    runtime = PlanningRuntime(store=Store(tmp_path / "s.db"), config=cfg, repo=repo)
    session = runtime.store.create_planning_session(
        repo_name="r",
        title="t",
        requirements="req",
        seed_type="requirements",
        status="draft",
    )
    assert runtime._state(session.id).manifesto == "a"  # noqa: SLF001
    manifesto.write_text("b", encoding="utf-8")
    assert runtime._state(session.id).manifesto == "b"  # noqa: SLF001


@pytest.mark.asyncio
async def test_off_keeps_cached_manifesto_until_process(tmp_path: Path) -> None:
    prscope_dir = tmp_path / ".prscope"
    prscope_dir.mkdir(parents=True)
    manifesto = prscope_dir / "manifesto.md"
    manifesto.write_text("first", encoding="utf-8")

    cfg = PrscopeConfig(
        local_repo=str(tmp_path),
        planning=PlanningConfig(instruction_context_refresh="off"),
    )
    repo = RepoProfile(name="r", path=str(tmp_path))
    runtime = PlanningRuntime(store=Store(tmp_path / "s.db"), config=cfg, repo=repo)
    session = runtime.store.create_planning_session(
        repo_name="r",
        title="t",
        requirements="req",
        seed_type="requirements",
        status="draft",
    )
    runtime._state(session.id)  # noqa: SLF001
    manifesto.write_text("second", encoding="utf-8")
    s2 = runtime._state(session.id)  # noqa: SLF001
    assert s2.manifesto == "first"
