from __future__ import annotations

from collections import deque
from typing import Any

from ....model_catalog import model_provider
from ..pipeline import PlanningRoundContext

_ASSISTANT_CONTEXT_MAX_CHARS = 8000


def _is_short_followup_message(user_input: str) -> bool:
    """
    Short affirmations ("Yes do it") need the prior assistant turn in-context; the critic review
    blob alone does not carry what "it" refers to.
    """
    t = (user_input or "").strip()
    if not t or len(t) > 220:
        return False
    tl = t.lower()
    if len(t) <= 12 and len(t.split()) <= 4:
        return True
    affirm = (
        "yes do it",
        "do it",
        "do that",
        "go ahead",
        "please do",
        "sounds good",
        "make it so",
        "proceed",
        "continue",
        "apply",
        "go for it",
    )
    if any(phrase in tl for phrase in affirm):
        return True
    if tl in {"yes", "yes.", "yep", "yeah", "ok", "okay", "sure", "please"}:
        return True
    return False


def _last_assistant_turn_content(core: Any, *, max_chars: int = _ASSISTANT_CONTEXT_MAX_CHARS) -> str | None:
    """
    Text the user is likely replying to. Prefer the most recent author turn (plan / assistant prose);
    if none, use the most recent critic turn. Pending user message is not in the conversation yet.
    """
    conv = core.get_conversation()

    def _take(turn: Any) -> str | None:
        content = str(getattr(turn, "content", "") or "").strip()
        return content or None

    chosen: str | None = None
    for turn in reversed(conv):
        if str(getattr(turn, "role", "")).strip().lower() != "author":
            continue
        chosen = _take(turn)
        if chosen:
            break
    if not chosen:
        for turn in reversed(conv):
            if str(getattr(turn, "role", "")).strip().lower() != "critic":
                continue
            chosen = _take(turn)
            if chosen:
                break
    if not chosen:
        return None
    if len(chosen) > max_chars:
        return chosen[: max_chars - 120] + "\n...[truncated for planning context budget]"
    return chosen


class RuntimeRoundEntry:
    def __init__(self, runtime: Any):
        self._runtime = runtime

    @staticmethod
    def _effective_requirements(core: Any, session: Any, user_input: str | None) -> str:
        base = str(session.requirements or "")
        if user_input:
            tail = ""
            if _is_short_followup_message(user_input):
                probe = _last_assistant_turn_content(core)
                if probe:
                    tail = (
                        "\n\nReply context — the user's message is a short follow-up to this assistant message. "
                        "Interpret User input as agreeing to or requesting what the assistant offered below."
                        "\n\n---\n"
                        f"{probe}"
                        "\n---\n"
                    )
            return base + tail + f"\n\nUser input:\n{user_input}"

        recent_guidance: deque[str] = deque(maxlen=3)
        seen: set[str] = set()
        for turn in reversed(core.get_conversation()):
            if str(getattr(turn, "role", "")).strip() != "user":
                continue
            round_number = int(getattr(turn, "round", 0) or 0)
            if round_number <= 0:
                continue
            content = str(getattr(turn, "content", "") or "").strip()
            if not content or content in seen:
                continue
            seen.add(content)
            recent_guidance.appendleft(content)
            if len(recent_guidance) >= 3:
                break
        if not recent_guidance:
            return base
        guidance_block = "\n".join(f"- {item}" for item in recent_guidance)
        return base + f"\n\nLatest user guidance:\n{guidance_block}"

    async def run_adversarial_round(
        self,
        session_id: str,
        user_input: str | None = None,
        refinement_evidence: dict[str, Any] | None = None,
        author_model_override: str | None = None,
        critic_model_override: str | None = None,
        event_callback: Any | None = None,
    ) -> tuple[Any, Any, Any]:
        async with self._runtime._session_lock(session_id):
            core = self._runtime._core(session_id)
            self._runtime.tools.set_session(session_id)
            session = core.get_session()
            if session.status not in {"refining", "converged"}:
                raise ValueError(f"Session is not in refining state: {session.status}")
            core.validate_command("run_round", session)
            if session.status == "converged":
                snapshot = core.transition_and_snapshot("refining", phase_message=None)
                await self._runtime._emit_event(event_callback, snapshot, session_id)
                session = core.get_session()

            current = core.get_current_plan()
            if current is None:
                raise ValueError("Cannot run adversarial round without initial plan")

            round_number = session.current_round + 1
            requirements = self._effective_requirements(core, session, user_input)
            state = self._runtime._state(session_id, session)
            state.requirements = requirements
            state.revision_round = round_number
            self._runtime._reset_round_telemetry(state)
            model_policy = self._runtime._resolve_model_policy(
                session,
                author_model_override=author_model_override,
                critic_model_override=critic_model_override,
            )

            issue_tracker = state.issue_tracker
            if not hasattr(issue_tracker, "open_issues") or not hasattr(issue_tracker, "root_open_issues"):
                raise RuntimeError("PlanningState issue_tracker must provide issue-tracker methods")
            state.issue_tracker = issue_tracker
            ctx = PlanningRoundContext(
                core=core,
                session_id=session_id,
                round_number=round_number,
                requirements=requirements,
                state=state,
                issue_tracker=issue_tracker,
                selected_author_model=model_policy.author_refine.primary_model,
                selected_critic_model=model_policy.critic_review.primary_model,
                model_policy=model_policy,
                event_callback=event_callback,
                refinement_evidence=refinement_evidence,
            )
            await self._runtime._emit_event(
                event_callback,
                {
                    "type": "model_selection",
                    "model_stage": "author_refine",
                    "model": model_policy.author_refine.primary_model,
                    "provider": model_provider(model_policy.author_refine.primary_model),
                    "fallback_model": model_policy.author_refine.first_fallback_model,
                },
                session_id,
            )
            await self._runtime._emit_event(
                event_callback,
                {
                    "type": "model_selection",
                    "model_stage": "critic_review",
                    "model": model_policy.critic_review.primary_model,
                    "provider": model_provider(model_policy.critic_review.primary_model),
                    "fallback_model": model_policy.critic_review.first_fallback_model,
                },
                session_id,
            )
            return await self._runtime._adversarial_loop.run_round(ctx=ctx, current_plan=current, user_input=user_input)
