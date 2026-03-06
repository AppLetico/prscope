from __future__ import annotations

import json
import time
from typing import Any

from ....store import PullRequest


class RuntimeSessionStarts:
    def __init__(self, runtime: Any):
        self._runtime = runtime

    async def create_requirements_session(
        self,
        requirements: str,
        author_model: str | None = None,
        critic_model: str | None = None,
        title: str | None = None,
        no_recall: bool = False,
        rebuild_memory: bool = False,  # kept for API compatibility
    ) -> Any:
        del rebuild_memory
        session = self._runtime.store.create_planning_session(
            repo_name=self._runtime.repo.name,
            title=title or (requirements.splitlines()[0][:80] if requirements.strip() else "New Plan"),
            requirements=requirements,
            author_model=author_model or self._runtime.planning_config.author_model,
            critic_model=critic_model or self._runtime.planning_config.critic_model,
            seed_type="requirements",
            no_recall=no_recall,
            status="draft",
        )
        self._runtime._state(session.id, session).no_recall = no_recall
        return session

    async def continue_requirements_draft(
        self,
        session_id: str,
        requirements: str,
        author_model_override: str | None = None,
        event_callback: Any | None = None,
        rebuild_memory: bool = False,
    ) -> Any:
        return await self._runtime._continue_initial_draft(
            session_id=session_id,
            requirements=requirements,
            author_model_override=author_model_override,
            event_callback=event_callback,
            rebuild_memory=rebuild_memory,
        )

    async def start_from_requirements(
        self,
        requirements: str,
        author_model: str | None = None,
        critic_model: str | None = None,
        title: str | None = None,
        no_recall: bool = False,
        rebuild_memory: bool = False,
        event_callback: Any | None = None,
    ) -> Any:
        session = await self.create_requirements_session(
            requirements=requirements,
            author_model=author_model,
            critic_model=critic_model,
            title=title,
            no_recall=no_recall,
            rebuild_memory=rebuild_memory,
        )
        return await self._runtime._continue_initial_draft(
            session_id=session.id,
            requirements=requirements,
            author_model_override=author_model,
            event_callback=event_callback,
            rebuild_memory=rebuild_memory,
        )

    def build_pr_seed_context(self, pr: PullRequest, evaluation: Any, files: list[Any]) -> str:
        sections: list[str] = []
        sections.append(f"## PR #{pr.number}: {pr.title}\n{pr.body or ''}")
        file_list = files[:30]
        file_lines = "\n".join(f"- {f.path}" for f in file_list)
        if len(files) > 30:
            file_lines += f"\n- ... and {len(files) - 30} more files (omitted)"
        sections.append(f"## Changed Files\n{file_lines}")

        if evaluation is not None:
            llm_summary = ""
            if evaluation.llm_json:
                try:
                    llm = json.loads(evaluation.llm_json)
                    llm_summary = json.dumps(llm, indent=2)[:2000]
                except json.JSONDecodeError:
                    llm_summary = str(evaluation.llm_json)[:2000]
            sections.append(
                f"## Prior Analysis\n"
                f"Decision: {evaluation.decision}\n"
                f"Rule score: {evaluation.rule_score}\n"
                f"Final score: {evaluation.final_score}\n"
                f"LLM summary:\n{llm_summary}"
            )

        combined = "\n\n".join(sections)
        cap = max(self._runtime.planning_config.seed_token_budget, 500) * 4
        if len(combined) > cap:
            combined = combined[:cap] + "\n\n[Seed context truncated to fit token budget]"
        return combined

    async def start_from_pr(
        self,
        upstream_repo: str,
        pr_number: int,
        author_model: str | None = None,
        critic_model: str | None = None,
        no_recall: bool = False,
        rebuild_memory: bool = False,
        event_callback: Any | None = None,
    ) -> Any:
        upstream = self._runtime.store.get_upstream_repo(upstream_repo)
        if upstream is None:
            raise ValueError(f"Unknown upstream repo in store: {upstream_repo}")
        pr = self._runtime.store.get_pull_request(upstream.id, pr_number)
        if pr is None:
            raise ValueError(f"PR not found in store: {upstream_repo}#{pr_number}")
        evaluation = None
        if pr.head_sha:
            evaluations = self._runtime.store.list_evaluations(limit=200)
            for candidate in evaluations:
                if candidate.pr_id == pr.id:
                    evaluation = candidate
                    break
        files = self._runtime.store.get_pr_files(pr.id)
        requirements = self.build_pr_seed_context(pr, evaluation, files)
        session = self._runtime.store.create_planning_session(
            repo_name=self._runtime.repo.name,
            title=f"PR #{pr_number}: {pr.title}",
            requirements=requirements,
            author_model=author_model or self._runtime.planning_config.author_model,
            critic_model=critic_model or self._runtime.planning_config.critic_model,
            seed_type="upstream_pr",
            seed_ref=f"{upstream_repo}#{pr_number}",
            no_recall=no_recall,
            status="draft",
        )
        self._runtime._state(session.id, session).no_recall = no_recall
        core = self._runtime._core(session.id)
        self._runtime.tools.set_session(session.id)

        wrapped_event: Any = None
        if event_callback is not None:

            async def wrapped(event: dict[str, Any]) -> None:
                await self._runtime._emit_event(event_callback, event, session.id)

            wrapped_event = wrapped
            self._runtime.author.event_callback = wrapped_event

        await self._runtime._prepare_memory(rebuild_memory=rebuild_memory, event_callback=wrapped_event)

        author_result = await self._runtime._run_initial_draft(
            core,
            requirements=requirements,
            author_model_override=self._runtime._resolve_author_model(session, author_model),
        )
        self._runtime._record_session_reads(session.id, author_result.accessed_paths)
        self._runtime._state(session.id, session).design_record = self._runtime._design_record_from_payload(
            author_result.design_record
        )
        core.add_turn("author", self._runtime._author_chat_summary(author_result.plan, 0), round_number=0)
        core.save_plan_version(author_result.plan, round_number=0)
        core.transition("refining")
        return self._runtime.store.get_planning_session(session.id) or session

    async def start_from_chat(
        self,
        author_model: str | None = None,
        critic_model: str | None = None,
        no_recall: bool = False,
        rebuild_memory: bool = False,  # kept for API compatibility
    ) -> tuple[Any, str | None]:
        del rebuild_memory
        session = self._runtime.store.create_planning_session(
            repo_name=self._runtime.repo.name,
            title="New Plan (discovery)",
            requirements="",
            author_model=author_model or self._runtime.planning_config.author_model,
            critic_model=critic_model or self._runtime.planning_config.critic_model,
            seed_type="chat",
            no_recall=no_recall,
            status="draft",
        )
        self._runtime._state(session.id, session).no_recall = no_recall
        core = self._runtime._core(session.id)
        core.transition_and_snapshot("draft", phase_message="Preparing codebase memory...")
        updated = self._runtime.store.get_planning_session(session.id) or session
        return updated, None

    async def continue_chat_setup(
        self,
        session_id: str,
        rebuild_memory: bool = False,
        event_callback: Any | None = None,
    ) -> str:
        setup_started = time.perf_counter()

        async def wrapped_event(event: dict[str, Any]) -> None:
            await self._runtime._emit_event(event_callback, event, session_id)

        async def on_progress(step: str) -> None:
            await wrapped_event({"type": "setup_progress", "step": step})

        await wrapped_event({"type": "phase_timing", "session_stage": "chat_setup", "state": "start"})
        try:
            await self._runtime._prepare_memory(
                rebuild_memory=rebuild_memory,
                progress_callback=on_progress,
                event_callback=wrapped_event,
            )
            self._runtime.discovery.reset_session(session_id)
            opening = self._runtime.discovery.opening_prompt()
            core = self._runtime._core(session_id)
            core.add_turn("author", opening, round_number=0)
            snapshot = core.transition_and_snapshot("draft", phase_message=None)
            await wrapped_event(snapshot)
            await wrapped_event({"type": "discovery_ready", "opening": opening})
            await wrapped_event(
                {
                    "type": "phase_timing",
                    "session_stage": "chat_setup",
                    "state": "complete",
                    "elapsed_ms": round((time.perf_counter() - setup_started) * 1000.0, 2),
                }
            )
            return opening
        except Exception:
            await wrapped_event(
                {
                    "type": "phase_timing",
                    "session_stage": "chat_setup",
                    "state": "failed",
                    "elapsed_ms": round((time.perf_counter() - setup_started) * 1000.0, 2),
                }
            )
            raise
