"""
Planning runtime orchestration for start modes and adversarial rounds.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

from ...config import PlanningConfig, PrscopeConfig, RepoProfile
from ...memory import MemoryStore, ParsedConstraint
from ..scanners import get_scanner
from ...profile import build_profile
from ...store import PlanningSession, PullRequest, Store
from ..core import ConvergenceResult, PlanningCore
from ..render import export_plan_documents
from .analytics_emitter import AnalyticsEmitter
from .author import AuthorAgent, AuthorResult
from .budget import ContextWindowExceeded, TokenBudgetManager, estimate_tokens
from .clarification import ClarificationAborted, ClarificationGate
from .compression import CritiqueCompressor
from .critic import CriticAgent, CriticResult
from .discovery import DiscoveryManager, DiscoveryTurnResult
from .round_controller import compute_plan_delta
from .tools import ToolExecutor

MEMORY_BLOCK_KEYS = {"architecture", "modules", "patterns", "entrypoints", "context"}
# Keep startup responsive; deeper refinement can happen in subsequent rounds.
INITIAL_DRAFT_TIMEOUT_SECONDS = 90


class PlanningRuntime:
    def __init__(self, store: Store, config: PrscopeConfig, repo: RepoProfile):
        self.store = store
        self.config = config
        self.repo = repo
        self.planning_config: PlanningConfig = config.planning
        scanner = get_scanner(self.planning_config.scanner)
        self.memory = MemoryStore(repo, self.planning_config, scanner=scanner)
        self.tools = ToolExecutor(
            repo.resolved_path,
            memory_block_callback=self._memory_block_for_tool,
        )
        self.tools.maybe_cleanup_artifacts(max_age_days=7)
        self.author = AuthorAgent(self.planning_config, self.tools)
        self.critic = CriticAgent(self.planning_config, repo)
        self.discovery = DiscoveryManager(self.planning_config, self.tools, self.memory)
        self._memory_block_caps = dict(self.planning_config.memory_block_max_chars)
        if self.repo.memory_block_max_chars:
            self._memory_block_caps.update(self.repo.memory_block_max_chars)
        self._locks: dict[str, asyncio.Lock] = {}
        self._session_cost_usd: dict[str, float] = {}
        self._session_max_prompt_tokens: dict[str, int] = {}
        self._session_read_paths: dict[str, set[str]] = {}
        self._clarification_gates: dict[str, ClarificationGate] = {}
        self._clarification_logs: dict[str, list[dict[str, Any]]] = {}
        self._compressor = CritiqueCompressor()

    def _core(self, session_id: str) -> PlanningCore:
        return PlanningCore(self.store, session_id, self.planning_config)

    def _session_lock(self, session_id: str) -> asyncio.Lock:
        return self._locks.setdefault(session_id, asyncio.Lock())

    def cleanup_session_resources(self, session_id: str) -> None:
        self._locks.pop(session_id, None)
        self._session_cost_usd.pop(session_id, None)
        self._session_max_prompt_tokens.pop(session_id, None)
        self._session_read_paths.pop(session_id, None)
        self._clarification_gates.pop(session_id, None)
        self._clarification_logs.pop(session_id, None)
        self.tools.delete_session_artifacts(session_id)

    def _session_reads(self, session_id: str) -> set[str]:
        return self._session_read_paths.setdefault(session_id, set())

    def _record_session_reads(self, session_id: str, read_paths: set[str]) -> None:
        if not read_paths:
            return
        self._session_reads(session_id).update(read_paths)

    def _clarification_gate(self, session_id: str) -> ClarificationGate:
        gate = self._clarification_gates.get(session_id)
        if gate is None:
            gate = ClarificationGate(timeout_seconds=self.planning_config.clarification_timeout_seconds)
            self._clarification_gates[session_id] = gate
        return gate

    def provide_clarification(self, session_id: str, answers: list[str]) -> None:
        self._clarification_gate(session_id).provide_answer(answers)

    def abort_clarification(self, session_id: str) -> None:
        gate = self._clarification_gates.get(session_id)
        if gate is not None:
            gate.abort()

    def _truncate_memory_block(self, key: str, text: str) -> str:
        cap = self._memory_block_caps.get(key)
        if cap is None or cap <= 0 or len(text) <= cap:
            return text
        omitted = len(text) - cap
        return f"{text[:cap]}\n[Memory truncated - {omitted} chars omitted]"

    @staticmethod
    def _confidence_trend(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        return (values[-1] - values[0]) / float(len(values) - 1)

    @staticmethod
    def _plan_quality_score(
        *,
        critic: CriticResult,
        grounding_ratio: float | None,
    ) -> float:
        grounding = max(0.0, min(1.0, grounding_ratio if grounding_ratio is not None else 0.5))
        major_penalty = min(0.5, max(0.0, critic.major_issues_remaining * 0.12))
        minor_penalty = min(0.2, max(0.0, critic.minor_issues_remaining * 0.03))
        evidence_penalty = min(
            0.25,
            (len(critic.unsupported_claims) * 0.05) + (len(critic.missing_evidence) * 0.05),
        )
        readiness_bonus = 0.1 if critic.operational_readiness else 0.0
        confidence_component = max(0.0, min(1.0, critic.critic_confidence)) * 0.2
        score = (grounding * 0.55) + readiness_bonus + confidence_component
        score -= major_penalty + minor_penalty + evidence_penalty
        return round(max(0.0, min(1.0, score)), 4)

    def _extract_critic_turns(self, turns: list[Any]) -> list[str]:
        return [t.content for t in turns if t.role == "critic" and t.content]

    @staticmethod
    def _single_line(text: str, limit: int = 220) -> str:
        normalized = " ".join((text or "").split())
        if len(normalized) <= limit:
            return normalized
        return normalized[:limit].rstrip() + "..."

    @staticmethod
    def _cap_lines(items: list[str], limit: int) -> list[str]:
        if len(items) <= limit:
            return items
        return [*items[:limit], f"...and {len(items) - limit} more"]

    def _round_anchor_block(
        self,
        *,
        requirements: str,
        critic: CriticResult,
        session_id: str,
        round_number: int,
    ) -> str:
        objective = self._single_line(requirements or "Refine plan against latest critique.")
        major_issues = [
            f"major_issues_remaining={critic.major_issues_remaining}",
            *[self._single_line(item) for item in critic.hard_constraint_violations],
        ]
        major_issues = self._cap_lines([issue for issue in major_issues if issue.strip()], 5)
        clarifications = [
            self._single_line(item.get("answer", ""))
            for item in self._clarification_logs.get(session_id, [])
            if int(item.get("round", -1)) == round_number and str(item.get("answer", "")).strip()
        ]
        clarifications = self._cap_lines(clarifications, 5)
        inspected = sorted(self._session_reads(session_id))
        inspected = self._cap_lines(inspected, 10)
        return (
            "ROUND ANCHOR\n\n"
            f"Current Objective:\n{objective}\n\n"
            "Unresolved Major Issues:\n"
            + ("\n".join(f"- {item}" for item in major_issues) if major_issues else "- (none)")
            + "\n\nClarifications Received:\n"
            + (
                "\n".join(f"- {item}" for item in clarifications)
                if clarifications
                else "- (none)"
            )
            + "\n\nFiles Inspected So Far:\n"
            + ("\n".join(f"- {item}" for item in inspected) if inspected else "- (none)")
        )

    def _manifesto_relative_path(self) -> str:
        try:
            return str(self.repo.resolved_manifesto.relative_to(self.repo.resolved_path))
        except ValueError:
            return str(self.repo.resolved_manifesto)

    @staticmethod
    def _excerpt(text: str, max_lines: int = 40) -> str:
        lines = text.splitlines()
        return "\n".join(lines[:max_lines]).strip()

    def _build_context_index(self, blocks: dict[str, str]) -> str:
        descriptions = {
            "architecture": "high-level system structure and dependencies",
            "modules": "module responsibilities and file mapping",
            "patterns": "cross-cutting implementation patterns",
            "entrypoints": "runtime and CLI entrypoints",
            "context": "scanner-generated rich repository context",
        }
        lines: list[str] = []
        for key, value in blocks.items():
            if key == "manifesto":
                continue
            if not value:
                continue
            desc = descriptions.get(key, "repo memory block")
            lines.append(f"- {key} memory ({len(value)} chars): {desc}")
        lines.append("- Entire repo via grep_code, list_files, read_file")
        return "\n".join(lines)

    @staticmethod
    def _requirements_vagueness_score(text: str) -> float:
        tokens = [token for token in re.split(r"[^a-z0-9]+", text.lower()) if token]
        if not tokens:
            return 0.0
        vague_markers = {
            "maybe",
            "possibly",
            "some",
            "improve",
            "better",
            "stuff",
            "things",
            "etc",
            "roughly",
            "around",
        }
        marker_hits = sum(1 for token in tokens if token in vague_markers)
        return marker_hits / float(len(tokens))

    def _initial_draft_policy(self, requirements: str) -> tuple[int, int, float, int]:
        """Tune startup discovery budget from requirement complexity."""
        tokens = [token for token in re.split(r"[^a-z0-9]+", requirements.lower()) if token]
        uniq_tokens = {token for token in tokens if len(token) >= 4}
        path_mentions = re.findall(r"[A-Za-z0-9_./-]+\.[A-Za-z0-9]+", requirements)
        complexity_score = len(uniq_tokens) + (3 * len(path_mentions))
        if complexity_score >= 45:
            return 3, 4, 0.55, 1800
        if complexity_score >= 25:
            return 2, 3, 0.45, 1400
        return 2, 2, 0.35, 1200

    def _memory_block_for_tool(self, key: str) -> dict[str, Any]:
        if key not in MEMORY_BLOCK_KEYS:
            allowed = ", ".join(sorted(MEMORY_BLOCK_KEYS))
            raise ValueError(f"Unsupported memory block '{key}'. Allowed: {allowed}")
        raw = self.memory.load_block(key)
        truncated = self._truncate_memory_block(key, raw)
        return {
            "key": key,
            "truncated": truncated != raw,
            "content": truncated,
        }

    async def _emit_event(
        self, callback: Any | None, event: dict[str, Any], session_id: str
    ) -> None:
        # Aggregate cost/token telemetry centrally before forwarding downstream.
        if event.get("type") == "token_usage":
            call_cost = float(event.get("call_cost_usd", 0.0) or 0.0)
            prompt_tokens = int(event.get("prompt_tokens", 0) or 0)
            running_cost = self._session_cost_usd.get(session_id, 0.0) + call_cost
            max_prompt = max(self._session_max_prompt_tokens.get(session_id, 0), prompt_tokens)
            self._session_cost_usd[session_id] = running_cost
            self._session_max_prompt_tokens[session_id] = max_prompt
            self.store.update_planning_session(
                session_id,
                session_total_cost_usd=running_cost,
                max_prompt_tokens=max_prompt,
            )
            event = {
                **event,
                "session_total_usd": running_cost,
                "max_prompt_tokens": max_prompt,
            }
        emitter = AnalyticsEmitter(callback)
        await emitter.emit(event)

    async def _prepare_memory(self, rebuild_memory: bool = False) -> dict[str, str]:
        profile = build_profile(self.repo.resolved_path)
        await self.memory.ensure_memory(profile, rebuild=rebuild_memory)
        blocks = self.memory.load_all_blocks()
        blocks["manifesto"] = self.memory.load_manifesto()
        return blocks

    def _constraints(self) -> list[ParsedConstraint]:
        return self.memory.load_constraints(self.repo.resolved_manifesto)

    @staticmethod
    def _author_chat_summary(plan_content: str, round_number: int) -> str:
        referenced = sorted(re.findall(r"`([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)`", plan_content))
        file_count = len(set(referenced))
        return (
            f"Plan updated (round {round_number}). "
            f"Referenced files: {file_count}. "
            "Review full plan in the plan panel."
        )

    @staticmethod
    def _critic_chat_summary(critic: CriticResult) -> str:
        lines = [
            (
                f"Critic review: {critic.major_issues_remaining} major, "
                f"{critic.minor_issues_remaining} minor."
            ),
        ]
        if critic.hard_constraint_violations:
            lines.append(
                "Hard-constraint violations: "
                + ", ".join(critic.hard_constraint_violations[:5])
                + (
                    f" (+{len(critic.hard_constraint_violations) - 5} more)"
                    if len(critic.hard_constraint_violations) > 5
                    else ""
                )
            )
        if critic.parse_error:
            lines.append(f"Critic parse/runtime warning: {critic.parse_error}")
        if critic.prose:
            lines.append(critic.prose)
        return "\n\n".join(lines)

    def _initial_draft_fallback(self, requirements: str, reason: str) -> str:
        # Keep startup fail-fast by returning a deterministic fallback plan.
        base = self.author._fallback_plan(requirements)  # noqa: SLF001
        return (
            f"{base}\n\n"
            "## Runtime Notes\n"
            f"- Initial draft fallback was used due to runtime issue: {reason}\n"
            "- Run another round after reviewing clarifications to improve quality.\n"
        )

    async def _continue_initial_draft(
        self,
        session_id: str,
        requirements: str,
        event_callback: Any | None = None,
        timeout_seconds: int = INITIAL_DRAFT_TIMEOUT_SECONDS,
        rebuild_memory: bool = False,
    ) -> PlanningSession:
        async with self._session_lock(session_id):
            core = self._core(session_id)
            session = core.get_session()
            self.tools.set_session(session.id)

            async def wrapped_event(event: dict[str, Any]) -> None:
                await self._emit_event(event_callback, event, session_id)

            self.author.event_callback = wrapped_event
            await self._emit_event(
                event_callback,
                {"type": "thinking", "message": "Building initial plan draft..."},
                session_id,
            )
            memory_started = time.perf_counter()
            await self._prepare_memory(rebuild_memory=rebuild_memory)
            await self._emit_event(
                event_callback,
                {
                    "type": "thinking",
                    "message": (
                        "Memory ready for initial draft "
                        f"({time.perf_counter() - memory_started:.1f}s)."
                    ),
                },
                session_id,
            )
            try:
                author_result = await asyncio.wait_for(
                    self._run_initial_draft(core, requirements=requirements),
                    timeout=timeout_seconds,
                )
                plan_content = author_result.plan
                self._record_session_reads(session.id, author_result.accessed_paths)
                if "fallback plan because the authoring model was unavailable" in plan_content.lower():
                    fallback_details = next(
                        (
                            item.get("details", "")
                            for item in reversed(author_result.rejection_reasons)
                            if item.get("reason") == "AUTHOR_FALLBACK"
                        ),
                        "unknown author runtime error",
                    )
                    await self._emit_event(
                        event_callback,
                        {
                            "type": "warning",
                            "message": (
                                "Initial draft produced fallback content. "
                                f"Reason: {fallback_details}"
                            ),
                        },
                        session_id,
                    )
            except asyncio.TimeoutError:
                await self._emit_event(
                    event_callback,
                    {
                        "type": "warning",
                        "message": (
                            f"Initial draft timed out after {timeout_seconds}s. "
                            "Using fallback draft so session can continue."
                        ),
                    },
                    session_id,
                )
                plan_content = self._initial_draft_fallback(
                    requirements,
                    f"timeout after {timeout_seconds}s",
                )
            except Exception as exc:  # noqa: BLE001
                await self._emit_event(
                    event_callback,
                    {
                        "type": "error",
                        "message": f"Initial draft runtime failure: {exc}",
                    },
                    session_id,
                )
                plan_content = self._initial_draft_fallback(requirements, str(exc))
            core.add_turn("author", self._author_chat_summary(plan_content, 0), round_number=0)
            core.save_plan_version(plan_content, round_number=0)
            core.transition("refining")
            async def _background_refine_once() -> None:
                try:
                    await self._emit_event(
                        event_callback,
                        {"type": "thinking", "message": "Running automatic refinement pass..."},
                        session_id,
                    )
                    await self.run_adversarial_round(
                        session_id=session_id,
                        user_input=None,
                        event_callback=event_callback,
                    )
                except Exception as exc:  # noqa: BLE001
                    await self._emit_event(
                        event_callback,
                        {
                            "type": "warning",
                            "message": f"Automatic refinement pass skipped: {exc}",
                        },
                        session_id,
                    )
            background_task = asyncio.create_task(_background_refine_once())
            background_task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            await self._emit_event(
                event_callback,
                {"type": "complete", "message": "Initial draft complete"},
                session_id,
            )
            return self.store.get_planning_session(session_id) or session

    async def create_requirements_session(
        self,
        requirements: str,
        title: str | None = None,
        rebuild_memory: bool = False,
    ) -> PlanningSession:
        session = self.store.create_planning_session(
            repo_name=self.repo.name,
            title=title or (requirements.splitlines()[0][:80] if requirements.strip() else "New Plan"),
            requirements=requirements,
            seed_type="requirements",
            status="drafting",
        )
        return session

    async def continue_requirements_draft(
        self,
        session_id: str,
        requirements: str,
        event_callback: Any | None = None,
        rebuild_memory: bool = False,
    ) -> PlanningSession:
        return await self._continue_initial_draft(
            session_id=session_id,
            requirements=requirements,
            event_callback=event_callback,
            rebuild_memory=rebuild_memory,
        )

    async def start_from_requirements(
        self,
        requirements: str,
        title: str | None = None,
        rebuild_memory: bool = False,
        event_callback: Any | None = None,
    ) -> PlanningSession:
        session = await self.create_requirements_session(
            requirements=requirements,
            title=title,
            rebuild_memory=rebuild_memory,
        )
        return await self._continue_initial_draft(
            session_id=session.id,
            requirements=requirements,
            event_callback=event_callback,
            rebuild_memory=rebuild_memory,
        )

    def _build_pr_seed_context(self, pr: PullRequest, evaluation: Any, files: list[Any]) -> str:
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
        cap = max(self.planning_config.seed_token_budget, 500) * 4
        if len(combined) > cap:
            combined = combined[:cap] + "\n\n[Seed context truncated to fit token budget]"
        return combined

    async def start_from_pr(
        self,
        upstream_repo: str,
        pr_number: int,
        rebuild_memory: bool = False,
    ) -> PlanningSession:
        await self._prepare_memory(rebuild_memory=rebuild_memory)
        upstream = self.store.get_upstream_repo(upstream_repo)
        if upstream is None:
            raise ValueError(f"Unknown upstream repo in store: {upstream_repo}")
        pr = self.store.get_pull_request(upstream.id, pr_number)
        if pr is None:
            raise ValueError(f"PR not found in store: {upstream_repo}#{pr_number}")
        evaluation = None
        if pr.head_sha:
            evaluations = self.store.list_evaluations(limit=200)
            for candidate in evaluations:
                if candidate.pr_id == pr.id:
                    evaluation = candidate
                    break
        files = self.store.get_pr_files(pr.id)
        requirements = self._build_pr_seed_context(pr, evaluation, files)
        session = self.store.create_planning_session(
            repo_name=self.repo.name,
            title=f"PR #{pr_number}: {pr.title}",
            requirements=requirements,
            seed_type="upstream_pr",
            seed_ref=f"{upstream_repo}#{pr_number}",
            status="drafting",
        )
        core = self._core(session.id)
        self.tools.set_session(session.id)
        author_result = await self._run_initial_draft(core, requirements=requirements)
        self._record_session_reads(session.id, author_result.accessed_paths)
        core.add_turn("author", self._author_chat_summary(author_result.plan, 0), round_number=0)
        core.save_plan_version(author_result.plan, round_number=0)
        core.transition("refining")
        return self.store.get_planning_session(session.id) or session

    async def start_from_chat(self, rebuild_memory: bool = False) -> tuple[PlanningSession, str]:
        await self._prepare_memory(rebuild_memory=rebuild_memory)
        session = self.store.create_planning_session(
            repo_name=self.repo.name,
            title="New Plan (discovery)",
            requirements="",
            seed_type="chat",
            status="discovering",
        )
        opening = self.discovery.opening_prompt()
        self._core(session.id).add_turn("author", opening, round_number=0)
        return session, opening

    async def handle_discovery_turn(
        self,
        session_id: str,
        user_message: str,
        event_callback: Any | None = None,
    ) -> DiscoveryTurnResult:
        async with self._session_lock(session_id):
            core = self._core(session_id)
            self.tools.set_session(session_id)
            session = core.get_session()
            if session.status not in {"discovering", "discovery"}:
                raise ValueError("Session is not in discovery mode")

            previous_state = session.status
            previous_round = session.current_round
            async def wrapped_event(event: dict[str, Any]) -> None:
                await self._emit_event(event_callback, event, session_id)

            self.discovery.event_callback = wrapped_event
            self.author.event_callback = wrapped_event

            current_round = session.current_round
            core.add_turn("user", user_message, round_number=current_round)
            conversation = [
                {"role": turn.role, "content": turn.content}
                for turn in core.get_conversation()
            ]
            try:
                result = await self.discovery.handle_turn(conversation)
                core.add_turn("author", result.reply, round_number=current_round)

                if result.complete:
                    summary = result.summary or user_message
                    self.store.update_planning_session(session_id, requirements=summary)
                    core.transition("drafting")
                    author_result = await self._run_initial_draft(core, requirements=summary)
                    self._record_session_reads(session_id, author_result.accessed_paths)
                    core.add_turn(
                        "author",
                        self._author_chat_summary(author_result.plan, 0),
                        round_number=0,
                    )
                    core.save_plan_version(author_result.plan, round_number=0)
                    core.transition("refining")

                return result
            except Exception:
                self.store.update_planning_session(
                    session_id,
                    status=previous_state,
                    current_round=previous_round,
                )
                if event_callback:
                    maybe = event_callback({"type": "error", "message": "Discovery turn failed"})
                    if asyncio.iscoroutine(maybe):
                        await maybe
                raise

    async def _run_initial_draft(self, core: PlanningCore, requirements: str) -> AuthorResult:
        blocks = self.memory.load_all_blocks()
        manifesto = self._truncate_memory_block("manifesto", self.memory.load_manifesto())
        manifesto_excerpt = self._excerpt(manifesto, max_lines=40)
        context_index = self._build_context_index(blocks)
        manifesto_path = self._manifesto_relative_path()
        budget = TokenBudgetManager(context_window=128_000, max_completion_tokens=4000)
        budget.enforce_required([requirements, f"Manifesto:\n{manifesto_excerpt}"])
        remaining = budget.available_prompt_tokens - estimate_tokens(requirements)
        manifesto_block, used_manifesto = budget.allocate(
            f"PROJECT MANIFESTO (excerpt):\n{manifesto_excerpt}\n\n", remaining
        )
        remaining = max(0, remaining - used_manifesto)
        context_index_block, _ = budget.allocate(
            f"CONTEXT INDEX:\n{context_index}\n\n",
            remaining,
        )
        initial_ratio = budget.injection_ratio([requirements, manifesto_block, context_index_block])
        if initial_ratio > budget.enforce_ratio:
            target_tokens = int(budget.context_window * budget.enforce_ratio)
            remaining = max(0, target_tokens - estimate_tokens(requirements))
            manifesto_block, used_manifesto = budget.allocate(manifesto_block, remaining)
            remaining = max(0, remaining - used_manifesto)
            context_index_block, _ = budget.allocate(context_index_block, remaining)
        messages = [
            {
                "role": "user",
                "content": (
                    "You are planning changes to this repository.\n\n"
                    f"REQUIREMENTS:\n{requirements}\n\n"
                    f"{manifesto_block}"
                    f"Full manifesto available at: {manifesto_path}\n\n"
                    f"{context_index_block}"
                    "IMPORTANT:\n"
                    "- Do not assume file paths.\n"
                    "- Search before referencing files.\n"
                    "- Read files before proposing changes.\n"
                    "- Pull additional memory only when needed with get_memory_block(key).\n"
                ),
            }
        ]
        require_clarification_first = self._requirements_vagueness_score(requirements) > 0.25
        max_attempts, max_tool_calls, min_grounding_ratio, max_output_tokens = self._initial_draft_policy(requirements)
        result = await self.author.author_loop(
            messages,
            require_tool_calls=True,
            max_attempts=min(max_attempts, self.planning_config.author_tool_rounds),
            max_tool_calls=max_tool_calls,
            min_grounding_ratio=min_grounding_ratio,
            draft_phase="planner",
            require_clarification_first=require_clarification_first and self.author.clarification_handler is not None,
            max_output_tokens=max_output_tokens,
            quick_start_mode=True,
        )
        return result

    async def run_adversarial_round(
        self,
        session_id: str,
        user_input: str | None = None,
        event_callback: Any | None = None,
    ) -> tuple[CriticResult, AuthorResult, ConvergenceResult]:
        async with self._session_lock(session_id):
            core = self._core(session_id)
            self.tools.set_session(session_id)
            session = core.get_session()
            if session.status not in {"refining", "converged", "approved"}:
                raise ValueError(f"Session is not in refining state: {session.status}")

            previous_state = session.status
            previous_round = session.current_round
            if session.status == "converged":
                core.transition("refining")

            if event_callback:
                await self._emit_event(
                    event_callback,
                    {"type": "thinking", "message": "Running adversarial round..."},
                    session_id,
                )

            try:
                current = core.get_current_plan()
                if current is None:
                    raise ValueError("Cannot run adversarial round without initial plan")

                round_number = session.current_round + 1
                blocks = self.memory.load_all_blocks()
                manifesto = self._truncate_memory_block("manifesto", self.memory.load_manifesto())
                manifesto_excerpt = self._excerpt(manifesto, max_lines=40)
                context_index = self._build_context_index(blocks)
                manifesto_path = self._manifesto_relative_path()
                constraints = self._constraints()
                requirements = (session.requirements or "") + (
                    f"\n\nUser input:\n{user_input}" if user_input else ""
                )
                if user_input:
                    core.add_turn("user", user_input, round_number=round_number)
                prior_critic_turn = self.store.get_latest_critic_turn(session_id)
                prior_critique_context = ""
                critic_turns = self._extract_critic_turns(core.get_conversation())
                if critic_turns:
                    prior_critique_context = self._compressor.summarize(critic_turns)
                elif prior_critic_turn is not None:
                    prior_critique_context = prior_critic_turn.content[:5000]

                round_usage: dict[str, int] = {
                    "author_prompt": 0,
                    "author_completion": 0,
                    "critic_prompt": 0,
                    "critic_completion": 0,
                }
                author_tool_calls = 0
                time_to_first_tool_call: int | None = None
                round_model_costs: dict[str, float] = {}

                async def wrapped_event(event: dict[str, Any]) -> None:
                    nonlocal author_tool_calls, time_to_first_tool_call
                    if event.get("type") == "token_usage":
                        stage = str(event.get("session_stage", ""))
                        prompt = int(event.get("prompt_tokens", 0) or 0)
                        completion = int(event.get("completion_tokens", 0) or 0)
                        model = str(event.get("model", "unknown"))
                        call_cost = float(event.get("call_cost_usd", 0.0) or 0.0)
                        round_model_costs[model] = round_model_costs.get(model, 0.0) + call_cost
                        if stage == "author":
                            round_usage["author_prompt"] += prompt
                            round_usage["author_completion"] += completion
                        elif stage == "critic":
                            round_usage["critic_prompt"] += prompt
                            round_usage["critic_completion"] += completion
                    if event.get("type") == "tool_call" and str(event.get("session_stage", "")) == "author":
                        author_tool_calls += 1
                        if time_to_first_tool_call is None:
                            time_to_first_tool_call = author_tool_calls
                    await self._emit_event(event_callback, event, session_id)

                self.author.event_callback = wrapped_event
                self.critic.event_callback = wrapped_event
                gate = self._clarification_gate(session_id)

                async def handle_author_clarification(question: str, context: str) -> list[str]:
                    self._clarification_logs.setdefault(session_id, []).append(
                        {
                            "round": round_number,
                            "question": question,
                            "answer": "",
                            "source": "author",
                            "timed_out": False,
                        }
                    )
                    await self._emit_event(
                        event_callback,
                        {
                            "type": "clarification_needed",
                            "question": question,
                            "context": context,
                            "source": "author",
                        },
                        session_id,
                    )
                    try:
                        answers, timed_out = await gate.wait_for_answer()
                    except ClarificationAborted:
                        answers, timed_out = ([], False)
                    answer = answers[0] if answers else ""
                    for item in reversed(self._clarification_logs.get(session_id, [])):
                        if item["round"] == round_number and item["source"] == "author" and not item["answer"]:
                            item["answer"] = answer
                            item["timed_out"] = timed_out
                            break
                    self.store.update_planning_session(
                        session_id,
                        clarifications_log_json=json.dumps(self._clarification_logs.get(session_id, [])),
                    )
                    return answers

                self.author.clarification_handler = handle_author_clarification
                critic_result = await self.critic.run_critic(
                    requirements=requirements,
                    plan_content=current.plan_content,
                    manifesto=manifesto,
                    architecture=blocks.get("architecture", ""),
                    constraints=constraints,
                    prior_critique=prior_critique_context,
                )
                should_pause_for_clarification = (
                    bool(critic_result.clarification_questions)
                    and (
                        critic_result.critic_confidence < 0.6
                        or bool(critic_result.hard_constraint_violations)
                        or (
                            critic_result.vagueness_score > 0.25
                            and len(self._session_reads(session_id)) == 0
                        )
                    )
                )
                if should_pause_for_clarification:
                    for question in critic_result.clarification_questions:
                        log_item = {
                            "round": round_number,
                            "question": question,
                            "answer": "",
                            "source": "critic",
                            "timed_out": False,
                        }
                        self._clarification_logs.setdefault(session_id, []).append(log_item)
                        await self._emit_event(
                            event_callback,
                            {
                                "type": "clarification_needed",
                                "question": question,
                                "context": "critic requested clarification",
                                "source": "critic",
                            },
                            session_id,
                        )
                    try:
                        answers, timed_out = await gate.wait_for_answer()
                    except ClarificationAborted:
                        answers, timed_out = ([], False)
                    for idx, log in enumerate(self._clarification_logs.get(session_id, [])):
                        if log["round"] == round_number and log["source"] == "critic" and not log["answer"]:
                            if idx < len(answers):
                                log["answer"] = answers[idx]
                            log["timed_out"] = timed_out
                    self.store.update_planning_session(
                        session_id,
                        clarifications_log_json=json.dumps(self._clarification_logs.get(session_id, [])),
                    )
                critic_content = self._critic_chat_summary(critic_result)
                budget = TokenBudgetManager(context_window=128_000, max_completion_tokens=4000)
                budget.enforce_required([requirements, f"Manifesto:\n{manifesto_excerpt}"])
                remaining_prompt = budget.available_prompt_tokens - estimate_tokens(requirements)
                manifesto_block, used = budget.allocate(manifesto_excerpt, remaining_prompt)
                remaining_prompt = max(0, remaining_prompt - used)
                context_index_block, used = budget.allocate(context_index, remaining_prompt)
                remaining_prompt = max(0, remaining_prompt - used)
                current_plan_block, used = budget.allocate(current.plan_content, remaining_prompt)
                remaining_prompt = max(0, remaining_prompt - used)
                critique_block, _ = budget.allocate(critic_content, remaining_prompt)
                round_anchor = self._round_anchor_block(
                    requirements=requirements,
                    critic=critic_result,
                    session_id=session_id,
                    round_number=round_number,
                )
                injection_blocks = [
                    requirements,
                    round_anchor,
                    manifesto_block,
                    context_index_block,
                    current_plan_block,
                    critique_block,
                ]
                static_ratio = budget.injection_ratio(injection_blocks)
                if static_ratio > budget.warn_ratio:
                    await self._emit_event(
                        event_callback,
                        {
                            "type": "warning",
                            "message": (
                                f"Prompt injection at {static_ratio:.2%} of context window; "
                                "approaching attention degradation zone."
                            ),
                        },
                        session_id,
                    )
                if static_ratio > budget.enforce_ratio:
                    target_tokens = int(budget.context_window * budget.enforce_ratio)
                    required_tokens = estimate_tokens(requirements) + estimate_tokens(round_anchor)
                    remaining_prompt = max(0, target_tokens - required_tokens)
                    critique_block, used = budget.allocate(critique_block, remaining_prompt)
                    remaining_prompt = max(0, remaining_prompt - used)
                    current_plan_block, used = budget.allocate(current_plan_block, remaining_prompt)
                    remaining_prompt = max(0, remaining_prompt - used)
                    manifesto_block, used = budget.allocate(manifesto_block, remaining_prompt)
                    remaining_prompt = max(0, remaining_prompt - used)
                    context_index_block, _ = budget.allocate(context_index_block, remaining_prompt)
                core.add_turn(
                    "critic",
                    critic_content,
                    round_number=round_number,
                    major_issues_remaining=critic_result.major_issues_remaining,
                    minor_issues_remaining=critic_result.minor_issues_remaining,
                    hard_constraint_violations=critic_result.hard_constraint_violations,
                    parse_error=critic_result.parse_error,
                )

                author_messages = [
                    {
                        "role": "user",
                        "content": (
                            f"{round_anchor}\n\n"
                            f"PROJECT MANIFESTO (excerpt):\n{manifesto_block}\n\n"
                            f"Full manifesto available at: {manifesto_path}\n\n"
                            f"CONTEXT INDEX:\n{context_index_block}\n\n"
                            f"Requirements:\n{requirements}\n\n"
                            f"Current plan:\n{current_plan_block}\n\n"
                            f"Critique:\n{critique_block}\n\n"
                            "Refinement requirements:\n"
                            "1) Every implementation step must include exact file path(s), change type, and rationale.\n"
                            "2) Update or add a concrete Test Strategy section with specific validations.\n"
                            "3) Update or add a concrete Rollback Plan section with failure triggers and rollback actions.\n"
                            "4) Explicitly resolve major critique issues, including failure modes and tradeoff risks.\n"
                            "5) Keep Cursor-style plan shape with title, summary, changes, files changed, ordered to-dos, "
                            "example code snippets, and mermaid diagram (or explicit waiver).\n"
                        ),
                    }
                ]
                recent_critics = self.store.get_planning_turns(session_id)
                recent_major = [
                    int(turn.major_issues_remaining)
                    for turn in recent_critics
                    if turn.role == "critic" and turn.major_issues_remaining is not None
                ]
                oscillation_detected = (
                    len(recent_major) >= 2
                    and critic_result.major_issues_remaining > recent_major[-1] > recent_major[-2]
                )
                oscillation_reason: dict[str, str] | None = None
                if oscillation_detected:
                    oscillation_reason = {
                        "reason": "OSCILLATION",
                        "details": (
                            "major issues increased for two consecutive rounds: "
                            f"{recent_major[-2]} -> {recent_major[-1]} -> {critic_result.major_issues_remaining}"
                        ),
                    }
                    author_messages[0]["content"] += (
                        "\n\nRegression blocker:\n"
                        "Major issues have increased for two consecutive rounds. "
                        "Prioritize reducing major issues and do not introduce new ones."
                    )
                require_refiner_tool_calls = (
                    critic_result.major_issues_remaining > 0
                    or bool(critic_result.hard_constraint_violations)
                    or bool(critic_result.missing_evidence)
                    or bool(critic_result.unsupported_claims)
                )
                author_result = await self.author.author_loop(
                    author_messages,
                    require_tool_calls=require_refiner_tool_calls,
                    min_grounding_ratio=0.8,
                    grounding_paths=self._session_reads(session_id),
                    draft_phase="refiner",
                )
                self._record_session_reads(session_id, author_result.accessed_paths)
                core.add_turn(
                    "author",
                    self._author_chat_summary(author_result.plan, round_number),
                    round_number=round_number,
                )
                core.save_plan_version(author_result.plan, round_number=round_number)
                convergence = core.check_convergence()
                plan_quality_score = self._plan_quality_score(
                    critic=critic_result,
                    grounding_ratio=author_result.grounding_ratio,
                )
                previous_critic_turn = prior_critic_turn
                previous_critic: CriticResult | None = None
                if previous_critic_turn and previous_critic_turn.major_issues_remaining is not None:
                    previous_critic = CriticResult(
                        major_issues_remaining=int(previous_critic_turn.major_issues_remaining or 0),
                        minor_issues_remaining=int(previous_critic_turn.minor_issues_remaining or 0),
                        hard_constraint_violations=previous_critic_turn.hard_constraint_violations or [],
                        critique_complete=False,
                        failure_modes=[],
                        design_tradeoff_risks=[],
                        unsupported_claims=[],
                        missing_evidence=[],
                        critic_confidence=0.0,
                        operational_readiness=False,
                        vagueness_score=0.0,
                        citation_count=0,
                        clarification_questions=[],
                        prose=previous_critic_turn.content,
                        parse_error=previous_critic_turn.parse_error,
                    )
                delta = compute_plan_delta(previous_critic, critic_result)
                self.store.add_round_metrics(
                    session_id=session_id,
                    repo_name=self.repo.name,
                    round_number=round_number,
                    author_prompt_tokens=round_usage["author_prompt"],
                    author_completion_tokens=round_usage["author_completion"],
                    critic_prompt_tokens=round_usage["critic_prompt"],
                    critic_completion_tokens=round_usage["critic_completion"],
                    max_prompt_tokens=self._session_max_prompt_tokens.get(session_id),
                    major_issues=critic_result.major_issues_remaining,
                    minor_issues=critic_result.minor_issues_remaining,
                    critic_confidence=critic_result.critic_confidence,
                    vagueness_score=critic_result.vagueness_score,
                    citation_count=critic_result.citation_count,
                    constraint_violations=critic_result.hard_constraint_violations,
                    resolved_since_last_round=[],
                    clarifications_this_round=len(
                        [q for q in critic_result.clarification_questions if q.strip()]
                    ),
                    call_cost_usd=self._session_cost_usd.get(session_id, 0.0),
                    issues_resolved=delta.issues_resolved,
                    issues_introduced=delta.issues_introduced,
                    net_improvement=delta.net_improvement,
                    model_costs=round_model_costs,
                    time_to_first_tool_call=time_to_first_tool_call,
                    grounding_ratio=author_result.grounding_ratio,
                    static_injection_tokens_pct=static_ratio,
                    rejected_for_no_discovery=author_result.rejection_counts.get(
                        "rejected_for_no_discovery", 0
                    ),
                    rejected_for_grounding=author_result.rejection_counts.get(
                        "rejected_for_grounding", 0
                    ),
                    rejected_for_budget=author_result.rejection_counts.get(
                        "rejected_for_budget", 0
                    ),
                    average_read_depth_per_round=author_result.average_read_depth,
                    time_between_tool_calls=author_result.average_time_between_tool_calls,
                    rejection_reasons=(
                        [*author_result.rejection_reasons, oscillation_reason]
                        if oscillation_reason is not None
                        else author_result.rejection_reasons
                    ),
                    plan_quality_score=plan_quality_score,
                    unsupported_claims_count=len(critic_result.unsupported_claims),
                    missing_evidence_count=len(critic_result.missing_evidence),
                )
                metrics = self.store.get_round_metrics(session_id)
                confidence_points = [
                    float(metric.critic_confidence)
                    for metric in metrics
                    if metric.critic_confidence is not None
                ]
                confidence_trend = self._confidence_trend(confidence_points)
                converged_early = int(
                    any(point >= 0.85 for point in confidence_points)
                    and len(confidence_points) < self.planning_config.max_adversarial_rounds
                )
                self.store.update_planning_session(
                    session_id,
                    confidence_trend=confidence_trend,
                    converged_early=converged_early,
                )
                self.store.update_constraint_stats(
                    self.repo.name,
                    critic_result.hard_constraint_violations,
                    session_id=session_id,
                )
                await self._emit_event(
                    event_callback,
                    {
                        "type": "plan_delta",
                        "issues_resolved": delta.issues_resolved,
                        "issues_introduced": delta.issues_introduced,
                        "net_improvement": delta.net_improvement,
                    },
                    session_id,
                )
                if (
                    convergence.converged
                    and (
                        plan_quality_score < 0.7
                        or bool(critic_result.unsupported_claims)
                        or bool(critic_result.missing_evidence)
                        or not critic_result.operational_readiness
                    )
                ):
                    convergence = ConvergenceResult(
                        converged=False,
                        reason="quality_gate",
                        change_pct=convergence.change_pct,
                        regression=convergence.regression,
                        major_issues=convergence.major_issues,
                    )
                if convergence.converged:
                    core.transition("converged")
                else:
                    core.transition("refining")
                if event_callback:
                    await self._emit_event(
                        event_callback,
                        {"type": "complete", "message": "Adversarial round complete"},
                        session_id,
                    )
                return critic_result, author_result, convergence
            except ContextWindowExceeded as exc:
                await self._emit_event(
                    event_callback,
                    {"type": "error", "message": str(exc)},
                    session_id,
                )
                raise
            except Exception:
                self.store.update_planning_session(
                    session_id,
                    status=previous_state,
                    current_round=previous_round,
                )
                if event_callback:
                    await self._emit_event(
                        event_callback,
                        {"type": "error", "message": "Adversarial round failed"},
                        session_id,
                    )
                raise

    def approve(self, session_id: str, unverified_references: set[str] | None = None) -> None:
        core = self._core(session_id)
        core.approve(unverified_references=unverified_references)

    async def validate_session(self, session_id: str) -> CriticResult:
        core = self._core(session_id)
        session = core.get_session()
        plan = core.get_current_plan()
        if plan is None:
            raise ValueError("No plan version found for validation")
        constraints = self._constraints()
        blocks = self.memory.load_all_blocks()
        manifesto = self.memory.load_manifesto()

        return await self.critic.validate_headless(
            session_id=session.id,
            round_number=plan.round,
            plan_sha=plan.plan_sha,
            requirements=session.requirements,
            plan_content=plan.plan_content,
            manifesto=manifesto,
            architecture=blocks.get("architecture", ""),
            constraints=constraints,
        )

    def export(self, session_id: str, output_dir: Path | None = None) -> dict[str, Path]:
        core = self._core(session_id)
        session = core.get_session()
        plan = core.get_current_plan()
        if plan is None:
            raise ValueError("No plan to export")
        paths = export_plan_documents(
            repo=self.repo,
            session=session,
            plan=plan,
            output_dir=output_dir,
        )
        core.mark_exported()
        return paths

    def status(self, session_id: str, merged_pr_files: set[str]) -> dict[str, Any]:
        core = self._core(session_id)
        session = core.get_session()
        plan = core.get_current_plan()
        if plan is None:
            raise ValueError("No plan to compare")
        planned_files = set(re.findall(r"`([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)`", plan.plan_content))
        implemented = planned_files & merged_pr_files
        missing = planned_files - merged_pr_files
        unplanned = merged_pr_files - planned_files
        round_metrics = self.store.get_round_metrics(session_id)
        confidence_points = [
            float(metric.critic_confidence)
            for metric in round_metrics
            if metric.critic_confidence is not None
        ]
        return {
            "planned_total": len(planned_files),
            "implemented_count": len(implemented),
            "missing_count": len(missing),
            "unplanned_count": len(unplanned),
            "implemented": sorted(implemented),
            "missing": sorted(missing),
            "unplanned": sorted(unplanned),
            "session_cost_usd": session.session_total_cost_usd,
            "max_prompt_tokens": session.max_prompt_tokens,
            "confidence_trend": (
                session.confidence_trend
                if session.confidence_trend is not None
                else self._confidence_trend(confidence_points)
            ),
            "converged_early": bool(session.converged_early) if session.converged_early is not None else False,
        }

    def plan_diff(self, session_id: str, round_number: int | None = None) -> str:
        if round_number is not None:
            current = self.store.get_plan_version(session_id, round_number)
            if current is None:
                return ""
            return current.diff_from_previous or ""
        else:
            versions = self.store.get_plan_versions(session_id, limit=2)
            if not versions:
                return ""
            return versions[0].diff_from_previous or ""
