"""
Planning runtime orchestration for start modes and adversarial rounds.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from ...config import PlanningConfig, PrscopeConfig, RepoProfile
from ...memory import MemoryStore, ParsedConstraint
from ...pricing import MODEL_CONTEXT_WINDOWS
from ...store import PlanningSession, PullRequest, Store
from ..core import ConvergenceResult, PlanningCore
from ..render import export_plan_documents
from ..scanners import get_scanner
from .author import (
    AuthorAgent,
    AuthorResult,
    DesignRecord,
    PlanDocument,
    RepairPlan,
    RevisionResult,
)
from .context import ClarificationGate, ContextAssembler, CritiqueCompressor
from .critic import CriticAgent, CriticResult, ImplementabilityResult, ReviewResult
from .discovery import DiscoveryManager, DiscoveryTurnResult
from .events import ToolEventStateManager
from .followups import (
    FollowupEngine,
    decision_graph_from_json,
    decision_graph_from_plan,
    decision_graph_to_json,
    followups_to_json,
    merge_decision_graphs,
)
from .orchestration_support import (
    RuntimeChatFlow,
    RuntimeEventRouter,
    RuntimeInitialDraftFlow,
    RuntimeRoundEntry,
    RuntimeSessionStarts,
    RuntimeStateSnapshots,
)
from .pipeline import AdversarialPlanningLoop, PlanningRoundContext, PlanningStages
from .review import (
    IssueCausalityExtractor,
    IssueGraphTracker,
    IssueSimilarityService,
    ManifestoChecker,
)
from .state import PlanningState
from .tools import ToolExecutor

MEMORY_BLOCK_KEYS = {"architecture", "modules", "patterns", "entrypoints", "context", "mental_model"}
# Target: plans should arrive in < 30s. Hard timeout — errors propagate, no fallback.
MEMORY_PREP_TIMEOUT_SECONDS = 120
INITIAL_DRAFT_TIMEOUT_SECONDS = 45
REFINEMENT_AUTHOR_TIMEOUT_SECONDS = 420
REFINEMENT_MAX_TOOL_CALLS = 10
MAX_ACTIVE_TOOL_CALLS = 50
MAX_COMPLETED_TOOL_CALL_GROUPS = 50
CRITIC_MAX_CLARIFICATION_QUESTIONS_PER_ROUND = 1
MAX_STATE_CACHE = 200
logger = logging.getLogger(__name__)
STATE_SNAPSHOT_SCHEMA_VERSION = 1

PLAN_TITLE_PATTERN = re.compile(r"^#\s+(.+)$", flags=re.MULTILINE)


@lru_cache(maxsize=64)
def _section_heading_pattern(heading: str) -> re.Pattern[str]:
    return re.compile(
        rf"^##\s+{re.escape(heading)}\b(.*?)(?=^##\s+|\Z)",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )


class IssueTracker(IssueGraphTracker):
    """Compatibility alias preserving runtime isinstance checks."""


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
        self._states: dict[str, PlanningState] = {}
        self._session_version: dict[str, int] = {}
        self._clarification_gates: dict[str, ClarificationGate] = {}
        self._manifesto_checker = ManifestoChecker()
        self._compressor = CritiqueCompressor()
        self._issue_similarity = IssueSimilarityService(self.planning_config.issue_dedupe)
        self._causality_extractor = IssueCausalityExtractor(self.planning_config.issue_graph)
        self._tool_event_state = ToolEventStateManager(
            store=self.store,
            core_resolver=self._core,
            max_active_tool_calls=MAX_ACTIVE_TOOL_CALLS,
            max_completed_tool_call_groups=MAX_COMPLETED_TOOL_CALL_GROUPS,
        )
        self._context_assembler = ContextAssembler(
            repo=self.repo,
            planning_config=self.planning_config,
            memory=self.memory,
            store=self.store,
            state_getter=self._state,
            truncate_memory_block=self._truncate_memory_block,
            recall_disabled=self._recall_disabled,
        )
        self._stages = PlanningStages(
            emit_event=self._emit_event,
            attach_plan_artifacts=self._attach_plan_version_artifacts,
            repo_memory=self._repo_memory,
            critic=self.critic,
            author=self.author,
            design_record_payload=self._design_record_payload,
            design_record_from_payload=self._design_record_from_payload,
            review_chat_summary=self._review_chat_summary,
            manifesto_checker=self._manifesto_checker,
            causality_extractor=self._causality_extractor,
        )
        self._adversarial_loop = AdversarialPlanningLoop(self)
        self._event_router = RuntimeEventRouter(self)
        self._snapshot_io = RuntimeStateSnapshots(self, schema_version=STATE_SNAPSHOT_SCHEMA_VERSION)
        self._initial_draft = RuntimeInitialDraftFlow(
            self,
            memory_prep_timeout_seconds=MEMORY_PREP_TIMEOUT_SECONDS,
            initial_draft_timeout_seconds=INITIAL_DRAFT_TIMEOUT_SECONDS,
        )
        self._session_starts = RuntimeSessionStarts(self)
        self._chat_flow = RuntimeChatFlow(self)
        self._round_entry = RuntimeRoundEntry(self)
        self._followup_engine = FollowupEngine()

    def _resolve_author_model(
        self,
        session: PlanningSession,
        author_model_override: str | None,
    ) -> str:
        return author_model_override or session.author_model or self.planning_config.author_model

    def _resolve_critic_model(
        self,
        session: PlanningSession,
        critic_model_override: str | None,
    ) -> str:
        return critic_model_override or session.critic_model or self.planning_config.critic_model

    def _core(self, session_id: str) -> PlanningCore:
        return PlanningCore(self.store, session_id, self.planning_config)

    def _session_lock(self, session_id: str) -> asyncio.Lock:
        return self._locks.setdefault(session_id, asyncio.Lock())

    def cleanup_session_resources(self, session_id: str) -> None:
        self._locks.pop(session_id, None)
        self._clarification_gates.pop(session_id, None)
        self._states.pop(session_id, None)
        self.discovery.clear_session(session_id)
        self.tools.delete_session_artifacts(session_id)

    def _state(self, session_id: str, session: PlanningSession | None = None) -> PlanningState:
        state = self._states.get(session_id)
        if state is not None:
            # Move hot sessions to the end for LRU-like eviction.
            self._states.pop(session_id, None)
            self._states[session_id] = state
        if state is None:
            current = session or self.store.get_planning_session(session_id)
            requirements = str(getattr(current, "requirements", "") or "")
            no_recall = bool(getattr(current, "no_recall", 0)) if current is not None else False
            tracker = IssueTracker(
                self._issue_similarity,
                max_nodes=self.planning_config.issue_graph.max_nodes,
                max_edges=self.planning_config.issue_graph.max_edges,
            )
            snapshot = self.read_state_snapshot(session_id)
            if isinstance(snapshot, dict):
                snapshot_version = int(snapshot.get("schema_version", 0) or 0)
                issue_graph_payload = snapshot.get("issue_graph")
                if snapshot_version >= 1 and isinstance(issue_graph_payload, dict):
                    tracker.load_snapshot(issue_graph_payload)
                else:
                    # Backward compatibility for flat snapshots.
                    for raw in snapshot.get("open_issues", []):
                        if not isinstance(raw, dict):
                            continue
                        description = str(raw.get("description", "")).strip()
                        issue_id = str(raw.get("id", "")).strip() or None
                        raised_round = int(raw.get("raised_in_round", 0) or 0)
                        if description:
                            canonical_issue = tracker.add_issue(description, raised_round)
                            if issue_id and canonical_issue.id and canonical_issue.id != issue_id:
                                tracker.alias_duplicate(issue_id, canonical_issue.id)
                            if issue_id:
                                tracker.canonical_issue_id(issue_id)
            state = PlanningState(
                session_id=session_id,
                requirements=requirements,
                no_recall=no_recall,
                issue_tracker=tracker,
            )
            if isinstance(snapshot, dict):
                state.revision_round = int(snapshot.get("revision_round", 0) or 0)
                state.plan_markdown = (
                    str(snapshot.get("plan_markdown", "")) if snapshot.get("plan_markdown") is not None else None
                )
                accessed_paths = snapshot.get("accessed_paths", [])
                if isinstance(accessed_paths, list):
                    state.accessed_paths = {str(path) for path in accessed_paths if str(path).strip()}
                architecture_change_rounds = snapshot.get("architecture_change_rounds", [])
                if isinstance(architecture_change_rounds, list):
                    state.architecture_change_rounds = [bool(item) for item in architecture_change_rounds][-8:]
                    state.architecture_change_count = sum(1 for item in state.architecture_change_rounds if item)
                review_score_history = snapshot.get("review_score_history", [])
                if isinstance(review_score_history, list):
                    state.review_score_history = [float(item) for item in review_score_history][-8:]
                open_issue_history = snapshot.get("open_issue_history", [])
                if isinstance(open_issue_history, list):
                    state.open_issue_history = [int(item) for item in open_issue_history][-8:]
            self._states[session_id] = state
            if len(self._states) > MAX_STATE_CACHE:
                oldest_session_id = next(iter(self._states))
                self._states.pop(oldest_session_id, None)
        if not state.manifesto:
            state.manifesto = self.memory.load_manifesto()
        if not state.constraints:
            state.constraints = self._constraints()
        return state

    def _repo_memory(self, state: PlanningState) -> dict[str, str]:
        return self._context_assembler.repo_memory(state)

    @staticmethod
    def _design_record_from_payload(payload: dict[str, Any] | None) -> DesignRecord | None:
        if not payload:
            return None
        return DesignRecord(
            problem_summary=str(payload.get("problem_summary", "")),
            constraints=[str(item) for item in payload.get("constraints", [])],
            architecture=str(payload.get("architecture", "")),
            alternatives_considered=[str(item) for item in payload.get("alternatives_considered", [])],
            tradeoffs=[str(item) for item in payload.get("tradeoffs", [])],
            chosen_design=str(payload.get("chosen_design", "")),
            assumptions=[str(item) for item in payload.get("assumptions", [])],
            potential_failure_modes=[str(item) for item in payload.get("potential_failure_modes", [])],
        )

    @staticmethod
    def _design_record_payload(record: DesignRecord | None) -> dict[str, Any] | None:
        if record is None:
            return None
        return asdict(record)

    @staticmethod
    def _as_serializable(value: Any) -> Any:
        return RuntimeStateSnapshots.as_serializable(value)

    @staticmethod
    def _reset_round_telemetry(state: PlanningState) -> None:
        state.round_cost_usd = 0.0
        state.author_prompt_tokens = 0
        state.author_completion_tokens = 0
        state.critic_prompt_tokens = 0
        state.critic_completion_tokens = 0

    def _persist_state_snapshot(self, session_id: str) -> None:
        self._snapshot_io.persist_state_snapshot(session_id)

    def read_state_snapshot(self, session_id: str) -> dict[str, Any] | None:
        return self._snapshot_io.read_state_snapshot(session_id)

    def list_state_snapshots(self) -> list[dict[str, Any]]:
        return self._snapshot_io.list_state_snapshots()

    def _skills_context(self, session_id: str) -> str:
        return self._context_assembler.skills_context(session_id)

    def _recall_disabled(self, session_id: str) -> bool:
        return bool(self._state(session_id).no_recall)

    def _build_recall_context(self, session_id: str, query: str) -> str:
        return self._context_assembler.build_recall_context(session_id, query)

    def _session_reads(self, session_id: str) -> set[str]:
        return self._state(session_id).accessed_paths

    def _record_session_reads(self, session_id: str, read_paths: set[str]) -> None:
        if not read_paths:
            return
        self._session_reads(session_id).update(read_paths)

    def _critic_evidence_context(self, session_id: str) -> str:
        read_paths = sorted(path for path in self._session_reads(session_id) if path)
        if not read_paths:
            return "Author evidence reads this round: (none recorded)"
        limited = read_paths[:20]
        bullets = "\n".join(f"- {path}" for path in limited)
        suffix = "\n- ... (truncated)" if len(read_paths) > len(limited) else ""
        return f"Author evidence reads this round (for critic grounding):\n{bullets}{suffix}"

    @staticmethod
    def _has_concrete_critic_evidence_gap(critic: CriticResult) -> bool:
        """
        Treat evidence gaps as actionable only when they cite concrete code locations.
        This prevents generic clarification prompts that do not help convergence.
        """

        candidates = [*critic.missing_evidence, *critic.unsupported_claims]
        if not candidates:
            return False

        file_path_hint = re.compile(
            r"(`[^`\n]+`)|(\b[\w./-]+\.(py|ts|tsx|js|jsx|md|json|yml|yaml|toml|sql|sh)\b)",
            re.IGNORECASE,
        )
        change_hint = re.compile(
            r"\b(line|section|heading|function|class|field|file|path|endpoint|test)\b", re.IGNORECASE
        )
        for raw in candidates:
            text = str(raw).strip()
            if not text:
                continue
            if file_path_hint.search(text) and change_hint.search(text):
                return True
        return False

    def _clarification_gate(self, session_id: str) -> ClarificationGate:
        return self._chat_flow.clarification_gate(session_id)

    def provide_clarification(self, session_id: str, answers: list[str]) -> None:
        self._chat_flow.provide_clarification(session_id, answers)

    def abort_clarification(self, session_id: str) -> None:
        self._chat_flow.abort_clarification(session_id)

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
    def _context_window_limit(config: PlanningConfig) -> int:
        windows: list[int] = []
        for model in {config.author_model, config.critic_model}:
            window = MODEL_CONTEXT_WINDOWS.get(model)
            if isinstance(window, int) and window > 0:
                windows.append(window)
        if windows:
            return min(windows)
        return 128_000

    def _should_compact_context(
        self,
        *,
        session_id: str,
        round_number: int,
        critic_turns: list[str],
        current_plan: str,
    ) -> bool:
        if round_number < 2:
            return False
        if len(critic_turns) >= 4:
            return True
        context_window = self._context_window_limit(self.planning_config)
        recent_peak = int(self._state(session_id).max_prompt_tokens or 0)
        if recent_peak > int(context_window * 0.65):
            return True
        if len(current_plan) >= 14_000:
            return True
        return False

    def _build_working_summary(
        self,
        *,
        requirements: str,
        critic_turns: list[str],
        current_plan: str,
    ) -> str:
        critique_summary = "(none yet)"
        if critic_turns:
            try:
                critique_summary = self._compressor.summarize(critic_turns)
            except Exception:  # noqa: BLE001
                critique_summary = critic_turns[-1][:1200]
        objective = self._single_line(requirements or "Refine plan against critique.", limit=280)
        summary = (
            "WORKING SUMMARY (compact prior rounds)\n\n"
            f"Objective: {objective}\n\n"
            f"Current plan snapshot (excerpt):\n{current_plan[:1800]}\n\n"
            f"Prior critique trajectory:\n{critique_summary}"
        )
        return summary[:5000]

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
        major_remaining = int(
            getattr(critic, "major_issues_remaining", None) or len(getattr(critic, "blocking_issues", []) or [])
        )
        constraint_items = list(getattr(critic, "hard_constraint_violations", []) or [])
        if not constraint_items:
            constraint_items = [f"Constraint violation: {cid}" for cid in getattr(critic, "constraint_violations", [])]
        major_issues = [
            f"major_issues_remaining={major_remaining}",
            *[self._single_line(item) for item in constraint_items],
        ]
        major_issues = self._cap_lines([issue for issue in major_issues if issue.strip()], 5)
        clarifications = [
            self._single_line(str(item.get("answer", "")))
            for item in self._state(session_id).clarification_logs
            if str(item.get("answer", "")).strip()
        ]
        if len(clarifications) > 10:
            clarifications = clarifications[-10:]
        clarifications = self._cap_lines(clarifications, 5)
        inspected = sorted(self._session_reads(session_id))
        inspected = self._cap_lines(inspected, 10)
        return (
            "ROUND ANCHOR\n\n"
            f"Current Objective:\n{objective}\n\n"
            "Unresolved Major Issues:\n"
            + ("\n".join(f"- {item}" for item in major_issues) if major_issues else "- (none)")
            + "\n\nClarifications Received:\n"
            + ("\n".join(f"- {item}" for item in clarifications) if clarifications else "- (none)")
            + "\n\nFiles Inspected So Far:\n"
            + ("\n".join(f"- {item}" for item in inspected) if inspected else "- (none)")
        )

    def _manifesto_relative_path(self) -> str:
        try:
            return str(self.repo.resolved_manifesto.relative_to(self.repo.resolved_path))
        except ValueError:
            return str(self.repo.resolved_manifesto)

    def _build_context_index(self, blocks: dict[str, str]) -> str:
        return self._context_assembler.build_context_index(blocks)

    def _memory_block_for_tool(self, key: str) -> dict[str, Any]:
        return self._context_assembler.memory_block_for_tool(key, MEMORY_BLOCK_KEYS)

    def _next_version(self, session_id: str) -> int:
        return self._event_router.next_version(session_id)

    async def _emit_event(self, callback: Any | None, event: dict[str, Any], session_id: str) -> None:
        await self._event_router.emit_event(callback, event, session_id)

    def _persist_tool_event_state(
        self,
        session_id: str,
        event_type: str,
        event: dict[str, Any],
    ) -> dict[str, Any] | None:
        return self._event_router.persist_tool_event_state(session_id, event_type, event)

    async def _prepare_memory(
        self,
        rebuild_memory: bool = False,
        progress_callback: Any = None,
        event_callback: Any = None,
    ) -> dict[str, str]:
        return await self._initial_draft.prepare_memory(
            rebuild_memory=rebuild_memory,
            progress_callback=progress_callback,
            event_callback=event_callback,
        )

    def _constraints(self) -> list[ParsedConstraint]:
        return self.memory.load_constraints(self.repo.resolved_manifesto)

    @staticmethod
    def _plan_document_from_version(version_content: str, plan_json: str | None = None) -> PlanDocument:
        if plan_json:
            try:
                payload = json.loads(plan_json)
                if isinstance(payload, dict):
                    return PlanDocument(
                        title=str(payload.get("title", "Plan")),
                        summary=str(payload.get("summary", "")),
                        goals=str(payload.get("goals", "")),
                        non_goals=str(payload.get("non_goals", payload.get("non-goals", ""))),
                        files_changed=str(
                            payload.get(
                                "files_changed", payload.get("files changed", payload.get("execution_flow", ""))
                            )
                        ),
                        architecture=str(payload.get("architecture", "")),
                        implementation_steps=str(payload.get("implementation_steps", "")),
                        test_strategy=str(payload.get("test_strategy", "")),
                        rollback_plan=str(payload.get("rollback_plan", "")),
                        open_questions=str(payload.get("open_questions", "")),
                    )
            except Exception:  # noqa: BLE001
                pass

        def extract(heading: str) -> str:
            pattern = _section_heading_pattern(heading)
            match = pattern.search(version_content)
            return match.group(1).strip() if match else ""

        title_match = PLAN_TITLE_PATTERN.search(version_content)
        title = title_match.group(1).strip() if title_match else "Plan"
        return PlanDocument(
            title=title,
            summary=extract("Summary"),
            goals=extract("Goals"),
            non_goals=extract("Non-Goals"),
            files_changed=extract("Files Changed"),
            architecture=extract("Architecture"),
            implementation_steps=extract("Implementation Steps"),
            test_strategy=extract("Test Strategy"),
            rollback_plan=extract("Rollback Plan"),
            open_questions=extract("Open Questions"),
        )

    def _review_chat_summary(self, review: ReviewResult) -> str:
        lines = [
            f"Design review: score {review.design_quality_score:.1f}/10, confidence={review.confidence}.",
        ]
        if review.primary_issue:
            lines.append(f"Primary issue: {review.primary_issue}")
        if review.blocking_issues:
            lines.append("Blocking issues: " + "; ".join(review.blocking_issues[:3]))
        if review.recommended_changes:
            lines.append("Recommended changes: " + "; ".join(review.recommended_changes[:3]))
        if review.prose:
            lines.append(review.prose)
        return "\n\n".join(lines)

    @staticmethod
    def _author_chat_summary(plan_content: str, round_number: int) -> str:
        referenced = sorted(set(re.findall(r"`([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)`", plan_content)))
        file_count = len(referenced)
        max_listed_files = 3
        if file_count:
            listed = ", ".join(referenced[:max_listed_files])
            if file_count > max_listed_files:
                listed = f"{listed}, +{file_count - max_listed_files} more"
            referenced_summary = f"Referenced files: {file_count} ({listed}). "
        else:
            referenced_summary = f"Referenced files: {file_count}. "
        return (
            f"Draft ready (version {round_number + 1}). "
            + referenced_summary
            + "Open the plan panel to review and continue refining."
        )

    def _plan_artifact_payloads(
        self,
        *,
        plan_document: PlanDocument,
        plan_content: str,
        current_version_id: int | None,
        previous_graph_json: str | None = None,
    ) -> tuple[str, str]:
        compatibility_graph = decision_graph_from_plan(
            open_questions=getattr(plan_document, "open_questions", ""),
            plan_content=plan_content,
        )
        previous_graph = decision_graph_from_json(previous_graph_json)
        current_graph = previous_graph if previous_graph.nodes else compatibility_graph
        if compatibility_graph.nodes:
            current_graph = merge_decision_graphs(
                compatibility_graph,
                current_graph,
                carry_forward_unresolved=True,
            )
        followups = self._followup_engine.generate(
            current_graph=current_graph,
            plan_content=plan_content,
            plan_version_id=current_version_id,
        )
        return decision_graph_to_json(current_graph), followups_to_json(followups)

    def _attach_plan_version_artifacts(
        self,
        *,
        version_id: int | None,
        plan_document: PlanDocument,
        plan_content: str,
        previous_graph_json: str | None = None,
    ) -> None:
        if version_id is None:
            return
        decision_graph_json, followups_json = self._plan_artifact_payloads(
            plan_document=plan_document,
            plan_content=plan_content,
            current_version_id=version_id,
            previous_graph_json=previous_graph_json,
        )
        self.store.update_plan_version_artifacts(
            version_id=version_id,
            decision_graph_json=decision_graph_json,
            followups_json=followups_json,
        )

    def _critic_chat_summary(
        self,
        critic: CriticResult,
        constraints: list[ParsedConstraint] | None = None,
    ) -> str:
        if isinstance(critic, ReviewResult):
            return self._review_chat_summary(critic)
        lines = ["Design review result available."]
        prose = getattr(critic, "prose", "")
        if prose:
            lines.append(str(prose))
        return "\n\n".join(lines)

    async def _continue_initial_draft(
        self,
        session_id: str,
        requirements: str,
        author_model_override: str | None = None,
        event_callback: Any | None = None,
        timeout_seconds: int | None = None,
        rebuild_memory: bool = False,
    ) -> PlanningSession:
        return await self._initial_draft.continue_initial_draft(
            session_id=session_id,
            requirements=requirements,
            author_model_override=author_model_override,
            event_callback=event_callback,
            timeout_seconds=timeout_seconds,
            rebuild_memory=rebuild_memory,
        )

    async def create_requirements_session(
        self,
        requirements: str,
        author_model: str | None = None,
        critic_model: str | None = None,
        title: str | None = None,
        no_recall: bool = False,
        rebuild_memory: bool = False,
    ) -> PlanningSession:
        return await self._session_starts.create_requirements_session(
            requirements=requirements,
            author_model=author_model,
            critic_model=critic_model,
            title=title,
            no_recall=no_recall,
            rebuild_memory=rebuild_memory,
        )

    async def continue_requirements_draft(
        self,
        session_id: str,
        requirements: str,
        author_model_override: str | None = None,
        event_callback: Any | None = None,
        rebuild_memory: bool = False,
    ) -> PlanningSession:
        return await self._session_starts.continue_requirements_draft(
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
    ) -> PlanningSession:
        return await self._session_starts.start_from_requirements(
            requirements=requirements,
            author_model=author_model,
            critic_model=critic_model,
            title=title,
            no_recall=no_recall,
            rebuild_memory=rebuild_memory,
            event_callback=event_callback,
        )

    def _build_pr_seed_context(self, pr: PullRequest, evaluation: Any, files: list[Any]) -> str:
        return self._session_starts.build_pr_seed_context(pr, evaluation, files)

    async def start_from_pr(
        self,
        upstream_repo: str,
        pr_number: int,
        author_model: str | None = None,
        critic_model: str | None = None,
        no_recall: bool = False,
        rebuild_memory: bool = False,
        event_callback: Any | None = None,
    ) -> PlanningSession:
        return await self._session_starts.start_from_pr(
            upstream_repo=upstream_repo,
            pr_number=pr_number,
            author_model=author_model,
            critic_model=critic_model,
            no_recall=no_recall,
            rebuild_memory=rebuild_memory,
            event_callback=event_callback,
        )

    async def start_from_chat(
        self,
        author_model: str | None = None,
        critic_model: str | None = None,
        no_recall: bool = False,
        rebuild_memory: bool = False,
    ) -> tuple[PlanningSession, str | None]:
        return await self._session_starts.start_from_chat(
            author_model=author_model,
            critic_model=critic_model,
            no_recall=no_recall,
            rebuild_memory=rebuild_memory,
        )

    async def continue_chat_setup(
        self,
        session_id: str,
        rebuild_memory: bool = False,
        event_callback: Any | None = None,
    ) -> str:
        return await self._session_starts.continue_chat_setup(
            session_id=session_id,
            rebuild_memory=rebuild_memory,
            event_callback=event_callback,
        )

    async def handle_discovery_turn(
        self,
        session_id: str,
        user_message: str,
        author_model_override: str | None = None,
        critic_model_override: str | None = None,
        event_callback: Any | None = None,
    ) -> DiscoveryTurnResult:
        return await self._chat_flow.handle_discovery_turn(
            session_id=session_id,
            user_message=user_message,
            author_model_override=author_model_override,
            critic_model_override=critic_model_override,
            event_callback=event_callback,
        )

    async def chat_with_author(
        self,
        session_id: str,
        user_message: str,
        author_model_override: str | None = None,
        event_callback: Any | None = None,
    ) -> str:
        return await self._chat_flow.chat_with_author(
            session_id=session_id,
            user_message=user_message,
            author_model_override=author_model_override,
            event_callback=event_callback,
        )

    async def handle_refinement_message(
        self,
        session_id: str,
        user_message: str,
        author_model_override: str | None = None,
        critic_model_override: str | None = None,
        event_callback: Any | None = None,
    ) -> tuple[str, str | None]:
        return await self._chat_flow.handle_refinement_message(
            session_id=session_id,
            user_message=user_message,
            author_model_override=author_model_override,
            critic_model_override=critic_model_override,
            event_callback=event_callback,
        )

    async def apply_followup_answer(
        self,
        *,
        session_id: str,
        followup_id: str,
        followup_answer: str,
        target_sections: list[str] | None = None,
        author_model_override: str | None = None,
        event_callback: Any | None = None,
    ) -> tuple[str, str | None]:
        return await self._chat_flow.apply_followup_answer(
            session_id=session_id,
            followup_id=followup_id,
            followup_answer=followup_answer,
            target_sections=target_sections,
            author_model_override=author_model_override,
            event_callback=event_callback,
        )

    async def _run_initial_draft(
        self,
        core: PlanningCore,
        requirements: str,
        author_model_override: str | None = None,
        timeout_seconds_override: Callable[[], int] | int | None = None,
    ) -> AuthorResult:
        return await self._initial_draft.run_initial_draft(
            core=core,
            requirements=requirements,
            author_model_override=author_model_override,
            timeout_seconds_override=timeout_seconds_override,
        )

    async def _stage_design_review(
        self,
        *,
        ctx: PlanningRoundContext,
        current_plan_content: str,
        emit_tool: Callable[..., Any],
    ) -> ReviewResult:
        return await self._stages.design_review(
            ctx=ctx,
            current_plan_content=current_plan_content,
            emit_tool=emit_tool,
        )

    async def _stage_repair_plan(
        self,
        *,
        ctx: PlanningRoundContext,
        current_plan_doc: PlanDocument,
        review_result: ReviewResult,
        emit_tool: Callable[..., Any],
    ) -> RepairPlan:
        return await self._stages.repair_plan(
            ctx=ctx,
            current_plan_doc=current_plan_doc,
            review_result=review_result,
            emit_tool=emit_tool,
        )

    async def _stage_revise_plan(
        self,
        *,
        ctx: PlanningRoundContext,
        current_plan_doc: PlanDocument,
        repair_plan: RepairPlan,
        review_result: ReviewResult,
        emit_tool: Callable[..., Any],
    ) -> tuple[str, list[str], RevisionResult]:
        return await self._stages.revise_plan(
            ctx=ctx,
            current_plan_doc=current_plan_doc,
            repair_plan=repair_plan,
            review_result=review_result,
            emit_tool=emit_tool,
        )

    async def _stage_validation_review(
        self,
        *,
        ctx: PlanningRoundContext,
        updated_markdown: str,
        emit_tool: Callable[..., Any],
    ) -> ReviewResult:
        return await self._stages.validation_review(
            ctx=ctx,
            updated_markdown=updated_markdown,
            emit_tool=emit_tool,
        )

    async def _stage_convergence_check(
        self,
        *,
        ctx: PlanningRoundContext,
        updated_markdown: str,
        validation_review: ReviewResult,
        emit_tool: Callable[..., Any],
    ) -> tuple[ImplementabilityResult, ConvergenceResult]:
        return await self._stages.convergence_check(
            ctx=ctx,
            updated_markdown=updated_markdown,
            validation_review=validation_review,
            emit_tool=emit_tool,
        )

    async def run_adversarial_round(
        self,
        session_id: str,
        user_input: str | None = None,
        author_model_override: str | None = None,
        critic_model_override: str | None = None,
        event_callback: Any | None = None,
    ) -> tuple[CriticResult, AuthorResult, ConvergenceResult]:
        return await self._round_entry.run_adversarial_round(
            session_id=session_id,
            user_input=user_input,
            author_model_override=author_model_override,
            critic_model_override=critic_model_override,
            event_callback=event_callback,
        )

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
        return paths

    def status(self, session_id: str, merged_pr_files: set[str]) -> dict[str, Any]:
        core = self._core(session_id)
        session = core.get_session()
        plan = core.get_current_plan()
        if plan is None:
            raise ValueError("No plan to compare")
        state = self._state(session_id, session)
        issue_tracker = state.issue_tracker if isinstance(state.issue_tracker, IssueTracker) else None
        open_issues = issue_tracker.open_issues() if issue_tracker is not None else []
        root_open_issues = issue_tracker.root_open_issues() if issue_tracker is not None else []
        dependency_blocks = issue_tracker.unresolved_dependency_chains() if issue_tracker is not None else 0
        planned_files = set(re.findall(r"`([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)`", plan.plan_content))
        implemented = planned_files & merged_pr_files
        missing = planned_files - merged_pr_files
        unplanned = merged_pr_files - planned_files
        round_metrics = self.store.get_round_metrics(session_id)
        confidence_points = [
            float(metric.critic_confidence) for metric in round_metrics if metric.critic_confidence is not None
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
            "open_issues": len(open_issues),
            "root_open_issues": len(root_open_issues),
            "dependency_blocks": int(dependency_blocks),
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
