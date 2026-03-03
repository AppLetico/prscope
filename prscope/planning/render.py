"""
Plan export renderers for PRD/RFC markdown documents.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from ..config import RepoProfile
from ..store import PlanningSession, PlanVersion


def _template_env() -> Environment:
    template_dir = Path(__file__).resolve().parent.parent / "plan_templates"
    return Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def extract_file_refs(text: str) -> list[str]:
    refs = sorted(set(re.findall(r"`([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)`", text)))
    return refs


def _sanitize_title(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = cleaned.strip("-")
    return cleaned or "plan-session"


def render_prd(session: PlanningSession, plan: PlanVersion, repo: RepoProfile) -> str:
    env = _template_env()
    template = env.get_template("prd.md.j2")
    return template.render(
        session=session,
        plan=plan,
        repo=repo,
    )


def render_rfc(session: PlanningSession, plan: PlanVersion, repo: RepoProfile) -> str:
    env = _template_env()
    template = env.get_template("rfc.md.j2")
    refs = extract_file_refs(plan.plan_content)
    ref_rows: list[dict[str, Any]] = []
    for ref in refs:
        path = repo.resolved_path / ref
        ref_rows.append({"path": ref, "exists": path.exists()})
    return template.render(
        session=session,
        plan=plan,
        repo=repo,
        referenced_files=ref_rows,
    )


def export_plan_documents(
    repo: RepoProfile,
    session: PlanningSession,
    plan: PlanVersion,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    if output_dir is None:
        base = Path(repo.output_dir or "./plans").expanduser()
    else:
        base = output_dir.expanduser()
    if not base.is_absolute():
        base = repo.resolved_path / base

    target_dir = base / _sanitize_title(session.title)
    target_dir.mkdir(parents=True, exist_ok=True)

    prd_path = target_dir / "PRD.md"
    rfc_path = target_dir / "RFC.md"
    conversation_path = target_dir / "conversation.md"

    prd_path.write_text(render_prd(session, plan, repo), encoding="utf-8")
    rfc_path.write_text(render_rfc(session, plan, repo), encoding="utf-8")
    conversation_path.write_text(
        f"# Conversation\n\nSession: `{session.id}`\nRound: `{session.current_round}`\n",
        encoding="utf-8",
    )

    return {
        "prd": prd_path,
        "rfc": rfc_path,
        "conversation": conversation_path,
    }
