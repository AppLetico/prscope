"""
Planning runtime orchestration for start modes and adversarial rounds.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from ...config import PlanningConfig, PrscopeConfig, RepoProfile
from ...memory import MemoryStore, ParsedConstraint
from ...pricing import MODEL_CONTEXT_WINDOWS
from ...profile import build_profile
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
from .events import AnalyticsEmitter, ToolEventStateManager, apply_token_usage_event
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
# Keep startup responsive; deeper refinement can happen in subsequent rounds.
INITIAL_DRAFT_TIMEOUT_SECONDS = 90
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
        if value is None:
            return None
        if hasattr(value, "__dataclass_fields__"):
            return asdict(value)
        return value

    @staticmethod
    def _reset_round_telemetry(state: PlanningState) -> None:
        state.round_cost_usd = 0.0
        state.author_prompt_tokens = 0
        state.author_completion_tokens = 0
        state.critic_prompt_tokens = 0
        state.critic_completion_tokens = 0

    def _persist_state_snapshot(self, session_id: str) -> None:
        state = self._states.get(session_id)
        if state is None:
            return
        sessions_dir = self.repo.resolved_path / ".prscope" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        issue_entries: list[dict[str, Any]] = []
        issue_graph: dict[str, Any] = {
            "nodes": [],
            "edges": [],
            "duplicate_alias": {},
            "summary": {
                "open_total": 0,
                "root_open": 0,
                "resolved_total": 0,
                "open_major": 0,
                "open_minor": 0,
                "open_info": 0,
                "unresolved_dependency_chains": 0,
            },
        }
        if isinstance(state.issue_tracker, IssueTracker):
            issue_entries = state.issue_tracker.open_issue_dicts()
            snapshot = state.issue_tracker.graph_snapshot()
            if isinstance(snapshot, dict):
                issue_graph = snapshot
        payload: dict[str, Any] = {
            "schema_version": STATE_SNAPSHOT_SCHEMA_VERSION,
            "session_id": state.session_id,
            "requirements": state.requirements,
            "repo_memory_keys": sorted(list(state.repo_memory.keys())),
            "manifesto_excerpt": (state.manifesto or "")[:2000],
            "constraints": [asdict(constraint) for constraint in state.constraints],
            "plan_markdown": state.plan_markdown,
            "design_record": self._as_serializable(state.design_record),
            "review": self._as_serializable(state.review),
            "constraint_eval": self._as_serializable(state.constraint_eval),
            "revision_round": state.revision_round,
            "open_issues": issue_entries,
            "issue_graph": issue_graph,
            "accessed_paths": sorted(state.accessed_paths),
            "session_cost_usd": state.session_cost_usd,
            "round_cost_usd": state.round_cost_usd,
            "max_prompt_tokens": state.max_prompt_tokens,
            "author_prompt_tokens": state.author_prompt_tokens,
            "author_completion_tokens": state.author_completion_tokens,
            "critic_prompt_tokens": state.critic_prompt_tokens,
            "critic_completion_tokens": state.critic_completion_tokens,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        snapshot_path = sessions_dir / f"{session_id}.json"
        snapshot_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def read_state_snapshot(self, session_id: str) -> dict[str, Any] | None:
        snapshot_path = self.repo.resolved_path / ".prscope" / "sessions" / f"{session_id}.json"
        if not snapshot_path.exists():
            return None
        try:
            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def list_state_snapshots(self) -> list[dict[str, Any]]:
        snapshots_dir = self.repo.resolved_path / ".prscope" / "sessions"
        if not snapshots_dir.exists():
            return []
        items: list[dict[str, Any]] = []
        for path in sorted(snapshots_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            items.append(
                {
                    "session_id": str(payload.get("session_id", path.stem)),
                    "updated_at": str(payload.get("updated_at", "")),
                    "path": str(path),
                }
            )
        return items

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
        gate = self._clarification_gates.get(session_id)
        if gate is None:
            gate = ClarificationGate(timeout_seconds=self.planning_config.clarification_timeout_seconds)
            self._clarification_gates[session_id] = gate
        return gate

    def provide_clarification(self, session_id: str, answers: list[str]) -> None:
        state = self._state(session_id)
        if answers:
            logs = state.clarification_logs
            pending_indices = [idx for idx, item in enumerate(logs) if not str(item.get("answer", "")).strip()]
            for idx, answer in zip(pending_indices, answers):
                logs[idx]["answer"] = answer
                logs[idx]["timed_out"] = False
            self.store.update_planning_session(
                session_id,
                clarifications_log_json=json.dumps(logs),
            )
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
        v = self._session_version.get(session_id, 0) + 1
        self._session_version[session_id] = v
        return v

    async def _emit_event(self, callback: Any | None, event: dict[str, Any], session_id: str) -> None:
        emitter = AnalyticsEmitter(callback)
        event_type = str(event.get("type", "")).strip().lower()

        async def emit_versioned(payload: dict[str, Any]) -> None:
            # Increment once per emitted SSE message.
            version = self._next_version(session_id)
            await emitter.emit({**payload, "session_version": version})

        if event_type in {"tool_call", "tool_result", "tool_update", "complete"}:
            persist_type = event_type
            if event_type in {"tool_call", "tool_result"}:
                persist_type = "tool_update"
            snapshot = self._persist_tool_event_state(session_id, persist_type, event)
            if snapshot is not None:
                await emit_versioned(snapshot)

        if event.get("type") == "token_usage":
            event = apply_token_usage_event(
                state=self._state(session_id),
                store=self.store,
                session_id=session_id,
                event=event,
            )

        if event_type in {"tool_call", "tool_result"}:
            now_iso = datetime.now(timezone.utc).isoformat()
            call_id = str(event.get("call_id", ""))
            if not call_id:
                call_id = f"{int(time.time() * 1000)}-{event.get('name', 'tool')}"
            tool_entry: dict[str, Any] = {
                "call_id": call_id,
                "name": str(event.get("name", "tool")),
                "sessionStage": event.get("session_stage"),
                "path": event.get("path"),
                "query": event.get("query"),
                "status": "done" if event_type == "tool_result" else "running",
                "created_at": now_iso,
            }
            if event_type == "tool_result":
                tool_entry["durationMs"] = event.get("duration_ms")
            await emit_versioned(
                {
                    "type": "tool_update",
                    "tool": tool_entry,
                }
            )
        else:
            await emit_versioned(event)

    def _persist_tool_event_state(
        self,
        session_id: str,
        event_type: str,
        event: dict[str, Any],
    ) -> dict[str, Any] | None:
        return self._tool_event_state.persist_event(session_id=session_id, event_type=event_type, event=event)

    async def _prepare_memory(
        self,
        rebuild_memory: bool = False,
        progress_callback: Any = None,
        event_callback: Any = None,
    ) -> dict[str, str]:
        profile = build_profile(self.repo.resolved_path)
        await self.memory.ensure_memory(
            profile,
            rebuild=rebuild_memory,
            progress_callback=progress_callback,
            event_callback=event_callback,
        )
        blocks = self.memory.load_all_blocks()
        blocks["manifesto"] = self.memory.load_manifesto()
        return blocks

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
        author_model_override: str | None = None,
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
            snapshot = core.transition_and_snapshot(
                "draft",
                phase_message="Building initial plan draft...",
                pending_questions_json=None,
            )
            await wrapped_event(snapshot)
            await self._emit_event(
                event_callback,
                {"type": "thinking", "message": "Building initial plan draft..."},
                session_id,
            )
            memory_started = time.perf_counter()
            await self._prepare_memory(
                rebuild_memory=rebuild_memory,
                event_callback=wrapped_event,
            )
            await self._emit_event(
                event_callback,
                {
                    "type": "thinking",
                    "message": (f"Memory ready for initial draft ({time.perf_counter() - memory_started:.1f}s)."),
                },
                session_id,
            )
            await wrapped_event({"type": "tool_call", "name": "draft_plan", "session_stage": "planner"})
            _draft_started = time.perf_counter()
            try:
                author_result = await asyncio.wait_for(
                    self._run_initial_draft(
                        core,
                        requirements=requirements,
                        author_model_override=self._resolve_author_model(
                            session,
                            author_model_override,
                        ),
                    ),
                    timeout=timeout_seconds,
                )
                plan_content = author_result.plan
                self._record_session_reads(session.id, author_result.accessed_paths)
                self._state(session.id, session).design_record = self._design_record_from_payload(
                    author_result.design_record
                )
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
                            "message": (f"Initial draft produced fallback content. Reason: {fallback_details}"),
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
            await wrapped_event(
                {
                    "type": "tool_result",
                    "name": "draft_plan",
                    "session_stage": "planner",
                    "duration_ms": round((time.perf_counter() - _draft_started) * 1000),
                }
            )
            core.add_turn("author", self._author_chat_summary(plan_content, 0), round_number=0)
            core.save_plan_version(plan_content, round_number=0)
            state = self._state(session.id, session)
            state.plan_markdown = plan_content
            self._persist_state_snapshot(session.id)
            await self._emit_event(
                event_callback,
                {
                    "type": "plan_ready",
                    "round": 0,
                    "initial_draft": True,
                    "saved_at_unix_s": time.time(),
                },
                session_id,
            )
            snapshot = core.transition_and_snapshot("refining", phase_message=None)
            await self._emit_event(event_callback, snapshot, session_id)
            await self._emit_event(
                event_callback,
                {"type": "complete", "message": "Initial draft complete"},
                session_id,
            )
            return self.store.get_planning_session(session_id) or session

    async def create_requirements_session(
        self,
        requirements: str,
        author_model: str | None = None,
        critic_model: str | None = None,
        title: str | None = None,
        no_recall: bool = False,
        rebuild_memory: bool = False,
    ) -> PlanningSession:
        session = self.store.create_planning_session(
            repo_name=self.repo.name,
            title=title or (requirements.splitlines()[0][:80] if requirements.strip() else "New Plan"),
            requirements=requirements,
            author_model=author_model or self.planning_config.author_model,
            critic_model=critic_model or self.planning_config.critic_model,
            seed_type="requirements",
            no_recall=no_recall,
            status="draft",
        )
        self._state(session.id, session).no_recall = no_recall
        return session

    async def continue_requirements_draft(
        self,
        session_id: str,
        requirements: str,
        author_model_override: str | None = None,
        event_callback: Any | None = None,
        rebuild_memory: bool = False,
    ) -> PlanningSession:
        return await self._continue_initial_draft(
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
        session = await self.create_requirements_session(
            requirements=requirements,
            author_model=author_model,
            critic_model=critic_model,
            title=title,
            no_recall=no_recall,
            rebuild_memory=rebuild_memory,
        )
        return await self._continue_initial_draft(
            session_id=session.id,
            requirements=requirements,
            author_model_override=author_model,
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
        author_model: str | None = None,
        critic_model: str | None = None,
        no_recall: bool = False,
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
            author_model=author_model or self.planning_config.author_model,
            critic_model=critic_model or self.planning_config.critic_model,
            seed_type="upstream_pr",
            seed_ref=f"{upstream_repo}#{pr_number}",
            no_recall=no_recall,
            status="draft",
        )
        self._state(session.id, session).no_recall = no_recall
        core = self._core(session.id)
        self.tools.set_session(session.id)
        author_result = await self._run_initial_draft(
            core,
            requirements=requirements,
            author_model_override=self._resolve_author_model(session, author_model),
        )
        self._record_session_reads(session.id, author_result.accessed_paths)
        self._state(session.id, session).design_record = self._design_record_from_payload(author_result.design_record)
        core.add_turn("author", self._author_chat_summary(author_result.plan, 0), round_number=0)
        core.save_plan_version(author_result.plan, round_number=0)
        core.transition("refining")
        return self.store.get_planning_session(session.id) or session

    async def start_from_chat(
        self,
        author_model: str | None = None,
        critic_model: str | None = None,
        no_recall: bool = False,
        rebuild_memory: bool = False,
    ) -> tuple[PlanningSession, str | None]:
        """Create a chat session in draft. Setup runs in background with phase progress events."""
        session = self.store.create_planning_session(
            repo_name=self.repo.name,
            title="New Plan (discovery)",
            requirements="",
            author_model=author_model or self.planning_config.author_model,
            critic_model=critic_model or self.planning_config.critic_model,
            seed_type="chat",
            no_recall=no_recall,
            status="draft",
        )
        self._state(session.id, session).no_recall = no_recall
        # Mark draft setup as in-progress immediately so the UI can render
        # the setup state even before the first SSE progress event arrives.
        core = self._core(session.id)
        core.transition_and_snapshot("draft", phase_message="Preparing codebase memory...")
        updated = self.store.get_planning_session(session.id) or session
        return updated, None

    async def continue_chat_setup(
        self,
        session_id: str,
        rebuild_memory: bool = False,
        event_callback: Any | None = None,
    ) -> str:
        """Build memory (with progress), add opening turn, keep status at draft. Returns opening message."""

        async def wrapped_event(event: dict[str, Any]) -> None:
            await self._emit_event(event_callback, event, session_id)

        async def on_progress(step: str) -> None:
            await wrapped_event({"type": "setup_progress", "step": step})

        await self._prepare_memory(
            rebuild_memory=rebuild_memory,
            progress_callback=on_progress,
            event_callback=wrapped_event,
        )
        self.discovery.reset_session(session_id)
        opening = self.discovery.opening_prompt()
        core = self._core(session_id)
        core.add_turn("author", opening, round_number=0)
        snapshot = core.transition_and_snapshot("draft", phase_message=None)
        await wrapped_event(snapshot)
        await wrapped_event({"type": "discovery_ready", "opening": opening})
        return opening

    async def handle_discovery_turn(
        self,
        session_id: str,
        user_message: str,
        author_model_override: str | None = None,
        critic_model_override: str | None = None,
        event_callback: Any | None = None,
        defer_initial_draft: bool = False,
    ) -> DiscoveryTurnResult:
        async with self._session_lock(session_id):
            core = self._core(session_id)
            self.tools.set_session(session_id)
            session = core.get_session()
            if session.status != "draft":
                raise ValueError("Session is not in draft mode")
            core.validate_command("message", session)

            async def wrapped_event(event: dict[str, Any]) -> None:
                await self._emit_event(event_callback, event, session_id)

            self.discovery.event_callback = wrapped_event
            self.author.event_callback = wrapped_event

            current_round = session.current_round
            previous_pending_questions_json = session.pending_questions_json
            core.add_turn("user", user_message, round_number=current_round)
            processing_message = (
                "Analyzing your answers and preparing the first draft"
                if previous_pending_questions_json
                else "Analyzing your request and preparing clarifying questions"
            )
            snapshot = core.transition_and_snapshot(
                "draft",
                phase_message=processing_message,
                pending_questions_json=None,
            )
            await wrapped_event(snapshot)
            conversation = [{"role": turn.role, "content": turn.content} for turn in core.get_conversation()]
            discovery_context = "".join(
                [
                    self._skills_context(session_id),
                    self._build_recall_context(session_id, user_message),
                ]
            )
            try:
                selected_author_model = self._resolve_author_model(session, author_model_override)
                result = await self.discovery.handle_turn(
                    conversation,
                    session_id=session_id,
                    model_override=selected_author_model,
                    extra_context=discovery_context,
                )
                self._record_session_reads(session_id, self.tools.accessed_paths.copy())
                # Preserve full discovery question text in conversation history so
                # follow-up user answers keep enough context for the model.
                core.add_turn("author", result.reply, round_number=current_round)

                if result.complete:
                    summary = result.summary or user_message
                    self.store.update_planning_session(session_id, requirements=summary)
                    snapshot = core.transition_and_snapshot(
                        "draft",
                        phase_message="Building initial plan draft...",
                        pending_questions_json=None,
                    )
                    await wrapped_event(snapshot)
                    if defer_initial_draft:
                        pass
                    else:
                        author_result = await self._run_initial_draft(
                            core,
                            requirements=summary,
                            author_model_override=selected_author_model,
                        )
                        self._record_session_reads(session_id, author_result.accessed_paths)
                        self._state(session_id).design_record = self._design_record_from_payload(
                            author_result.design_record
                        )
                        core.add_turn(
                            "author",
                            self._author_chat_summary(author_result.plan, 0),
                            round_number=0,
                        )
                        core.save_plan_version(author_result.plan, round_number=0)
                        snapshot = core.transition_and_snapshot("refining", phase_message=None)
                        await wrapped_event(snapshot)
                else:
                    snapshot = core.transition_and_snapshot(
                        "draft",
                        phase_message=None,
                        pending_questions_json=(
                            json.dumps([asdict(question) for question in result.questions])
                            if result.questions
                            else None
                        ),
                    )
                    await wrapped_event(snapshot)

                await self._emit_event(
                    event_callback,
                    {"type": "complete", "message": "Discovery turn complete"},
                    session_id,
                )
                return result
            except Exception:
                latest = core.get_session()
                if latest.status == "draft" and bool(latest.is_processing):
                    recovery = core.transition_and_snapshot(
                        "draft",
                        phase_message=None,
                        pending_questions_json=previous_pending_questions_json,
                    )
                    await wrapped_event(recovery)
                if event_callback:
                    maybe = event_callback({"type": "error", "message": "Discovery turn failed"})
                    if asyncio.iscoroutine(maybe):
                        await maybe
                raise

    async def continue_discovery_draft(
        self,
        session_id: str,
        requirements: str,
        author_model_override: str | None = None,
        event_callback: Any | None = None,
    ) -> PlanningSession:
        """Continue drafting after discovery has prepared requirements in draft state."""
        return await self._continue_initial_draft(
            session_id=session_id,
            requirements=requirements,
            author_model_override=author_model_override,
            event_callback=event_callback,
            rebuild_memory=False,
        )

    async def chat_with_author(
        self,
        session_id: str,
        user_message: str,
        author_model_override: str | None = None,
        event_callback: Any | None = None,
    ) -> str:
        """
        Handle a conversational author reply in refinement/converged mode without running critique.
        """
        async with self._session_lock(session_id):
            core = self._core(session_id)
            session = core.get_session()
            if session.status not in {"refining", "converged"}:
                raise ValueError(f"Session is not in chat-refinement state: {session.status}")
            core.validate_command("message", session)

            status = session.status
            current_round = session.current_round
            current_plan = core.get_current_plan()
            if current_plan is None:
                raise ValueError("Cannot chat with author before an initial draft exists")

            snapshot = core.transition_and_snapshot(
                status,
                phase_message="Author is responding...",
                allow_round_stability=True,
            )
            await self._emit_event(event_callback, snapshot, session_id)

            completed = False
            try:
                core.add_turn("user", user_message, round_number=current_round)
                recent_turns = core.get_conversation()[-8:]
                history_lines = "\n".join(
                    f"{turn.role}: {turn.content.strip()}" for turn in recent_turns if turn.content.strip()
                )
                prompt = (
                    "You are the planning author in a refinement chat.\n"
                    "Answer the user's message conversationally and concisely.\n"
                    "Do not run critique yourself.\n"
                    "Do not claim the plan is updated unless a critique/refinement round is explicitly run.\n"
                    "If useful, propose what should be adjusted in the next critique run.\n\n"
                    f"Current plan (excerpt):\n{current_plan.plan_content[:3500]}\n\n"
                    f"Recent conversation:\n{history_lines}\n\n"
                    f"User message:\n{user_message}"
                )
                response, _ = await self.author._llm_call(  # noqa: SLF001
                    [
                        {"role": "system", "content": "You are a pragmatic software planning assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    allow_tools=False,
                    max_output_tokens=700,
                    model_override=self._resolve_author_model(session, author_model_override),
                )
                message = response.choices[0].message
                reply = str(getattr(message, "content", None) or "").strip()
                if not reply:
                    reply = (
                        "I hear you. I can discuss changes here, and when you're ready we can run "
                        "critique to apply updates to the plan."
                    )
                core.add_turn("author", reply, round_number=current_round)
                completed = True
                return reply
            finally:
                recovery = core.transition_and_snapshot(
                    status,
                    phase_message=None,
                    allow_round_stability=True,
                )
                await self._emit_event(event_callback, recovery, session_id)
                if completed:
                    await self._emit_event(
                        event_callback,
                        {"type": "complete", "message": "Author chat reply complete"},
                        session_id,
                    )

    async def _run_initial_draft(
        self,
        core: PlanningCore,
        requirements: str,
        author_model_override: str | None = None,
    ) -> AuthorResult:
        session = core.get_session()
        state = self._state(session.id, session)
        state.requirements = requirements
        blocks = self._repo_memory(state)
        manifesto = self._truncate_memory_block("manifesto", state.manifesto)
        skills_block = state.skills_context or self._skills_context(session.id)
        recall_block = self._build_recall_context(session.id, requirements)
        context_index = self._build_context_index(blocks)
        result = await self.author.run_initial_draft(
            requirements=requirements,
            manifesto=manifesto,
            manifesto_path=self._manifesto_relative_path(),
            skills_block=skills_block,
            recall_block=recall_block,
            context_index=context_index,
            grounding_paths=self._session_reads(session.id),
            model_override=author_model_override,
        )
        return result

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
        async with self._session_lock(session_id):
            core = self._core(session_id)
            self.tools.set_session(session_id)
            session = core.get_session()
            if session.status not in {"refining", "converged"}:
                raise ValueError(f"Session is not in refining state: {session.status}")
            core.validate_command("run_round", session)
            if session.status == "converged":
                snapshot = core.transition_and_snapshot("refining", phase_message=None)
                await self._emit_event(event_callback, snapshot, session_id)
                session = core.get_session()

            current = core.get_current_plan()
            if current is None:
                raise ValueError("Cannot run adversarial round without initial plan")

            round_number = session.current_round + 1
            requirements = (session.requirements or "") + (f"\n\nUser input:\n{user_input}" if user_input else "")
            state = self._state(session_id, session)
            state.requirements = requirements
            state.revision_round = round_number
            self._reset_round_telemetry(state)
            selected_author_model = self._resolve_author_model(session, author_model_override)
            selected_critic_model = self._resolve_critic_model(session, critic_model_override)

            issue_tracker = state.issue_tracker
            if not isinstance(issue_tracker, IssueTracker):
                raise RuntimeError("PlanningState issue_tracker must be an IssueTracker instance")
            state.issue_tracker = issue_tracker
            ctx = PlanningRoundContext(
                core=core,
                session_id=session_id,
                round_number=round_number,
                requirements=requirements,
                state=state,
                issue_tracker=issue_tracker,
                selected_author_model=selected_author_model,
                selected_critic_model=selected_critic_model,
                event_callback=event_callback,
            )
            return await self._adversarial_loop.run_round(ctx=ctx, current_plan=current, user_input=user_input)

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
