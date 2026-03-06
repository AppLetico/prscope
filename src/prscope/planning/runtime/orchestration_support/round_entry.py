from __future__ import annotations

from typing import Any

from ..pipeline import PlanningRoundContext


class RuntimeRoundEntry:
    def __init__(self, runtime: Any):
        self._runtime = runtime

    async def run_adversarial_round(
        self,
        session_id: str,
        user_input: str | None = None,
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
            requirements = (session.requirements or "") + (f"\n\nUser input:\n{user_input}" if user_input else "")
            state = self._runtime._state(session_id, session)
            state.requirements = requirements
            state.revision_round = round_number
            self._runtime._reset_round_telemetry(state)
            selected_author_model = self._runtime._resolve_author_model(session, author_model_override)
            selected_critic_model = self._runtime._resolve_critic_model(session, critic_model_override)

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
                selected_author_model=selected_author_model,
                selected_critic_model=selected_critic_model,
                event_callback=event_callback,
            )
            return await self._runtime._adversarial_loop.run_round(ctx=ctx, current_plan=current, user_input=user_input)
