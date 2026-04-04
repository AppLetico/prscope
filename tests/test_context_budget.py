from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prscope.config import PlanningConfig, PrscopeConfig, RepoProfile
from prscope.planning.runtime.context.budget import TokenBudgetManager, estimate_tokens
from prscope.planning.runtime.discovery_support.llm import DiscoveryLLMClient, trim_discovery_messages_for_budget
from prscope.planning.runtime.orchestration import PlanningRuntime
from prscope.pricing import DEFAULT_CONTEXT_WINDOW_FALLBACK, context_window_for_model
from prscope.store import Store


def test_context_window_for_model_known() -> None:
    assert context_window_for_model("gpt-4o") == 128_000
    assert context_window_for_model("gpt-5") == 200_000


def test_context_window_for_model_unknown_falls_back() -> None:
    assert context_window_for_model("unknown-model-xyz") == DEFAULT_CONTEXT_WINDOW_FALLBACK


def test_token_budget_manager_reserved_prompt() -> None:
    b = TokenBudgetManager(
        context_window=100_000,
        max_completion_tokens=4000,
        reserved_prompt_tokens=1000,
    )
    assert b.available_prompt_tokens == 100_000 - 4000 - 1000


def test_estimate_tokens_tiktoken_when_requested() -> None:
    h = estimate_tokens("hello world " * 50, model_id="gpt-4o", estimator="heuristic")
    try:
        import tiktoken  # noqa: F401
    except ImportError:
        pytest.skip("tiktoken not installed")
    t = estimate_tokens("hello world " * 50, model_id="gpt-4o", estimator="tiktoken")
    assert isinstance(t, int)
    assert t > 0
    assert t <= h + 50


def test_trim_discovery_messages_drops_old_tools() -> None:
    msgs = [
        {"role": "system", "content": "x" * 100},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": ""},
        {"role": "tool", "content": "a" * 5000},
        {"role": "assistant", "content": ""},
        {"role": "tool", "content": "b" * 5000},
    ]
    out, _removed = trim_discovery_messages_for_budget(msgs, max_chars=2000)
    assert len(out) < len(msgs) or sum(len(str(m.get("content", ""))) for m in out) <= 2500


def test_trim_discovery_respects_token_cap_when_chars_allow() -> None:
    msgs = [
        {"role": "system", "content": "x" * 100},
        {"role": "user", "content": "hi"},
        {"role": "tool", "content": "z" * 15_000},
    ]
    out, removed = trim_discovery_messages_for_budget(
        msgs,
        max_chars=500_000,
        max_input_tokens=100,
        model_id="gpt-4o",
        token_estimator="heuristic",
    )
    assert removed
    assert len(out) < len(msgs)


@pytest.mark.asyncio
async def test_discovery_compaction_circuit_breaker_emits_after_limit() -> None:
    events: list[dict[str, Any]] = []

    async def capture(ev: dict[str, Any]) -> None:
        events.append(ev)

    mgr = MagicMock()
    mgr.config = PlanningConfig(
        author_model="gpt-4o-mini",
        critic_model="gpt-4o-mini",
        discovery_compaction_failure_max=2,
        token_budget_estimator="heuristic",
    )
    mgr._emit = AsyncMock(side_effect=capture)
    mgr._normalize_roles.side_effect = lambda m: m
    mgr._extract_feature_intent.return_value = None
    mgr._latest_user_message.return_value = ""
    mgr._active_discovery_session_id = "s1"
    mgr.tool_executor.execute = MagicMock(
        return_value={"tool_call_id": "1", "name": "list_files", "result": {"ok": True}}
    )
    mgr._ingest_feature_evidence_from_tool = AsyncMock()

    client = DiscoveryLLMClient(mgr)

    async def fake_safe(*args: Any, **kwargs: Any) -> Any:
        r = MagicMock()
        r.choices = [MagicMock()]
        r.choices[0].message.content = ""
        tc = MagicMock()
        tc.id = "x"
        tc.function.name = "list_files"
        tc.function.arguments = "{}"
        r.choices[0].message.tool_calls = [tc]
        return r

    def always_trim(msgs: list, *a: Any, **k: Any) -> tuple[list, bool]:
        return list(msgs), True

    with patch(
        "prscope.planning.runtime.discovery_support.llm.trim_discovery_messages_for_budget",
        side_effect=always_trim,
    ):
        client.safe_completion_call = fake_safe
        out = await client.llm_call_with_tools(
            [{"role": "user", "content": "hi"}],
            max_tool_rounds=10,
        )
    assert "Discovery paused" in out
    assert any(e.get("type") == "discovery_circuit_breaker" for e in events)


@pytest.mark.asyncio
async def test_summarize_critiques_for_compaction_llm_path(tmp_path) -> None:
    store = Store(tmp_path / "t.db")
    cfg = PrscopeConfig(
        local_repo=str(tmp_path),
        planning=PlanningConfig(
            author_model="gpt-4o-mini",
            critic_model="gpt-4o-mini",
            critique_llm_summarize_enabled=True,
            critique_llm_summarize_prompt_tokens_threshold=1,
        ),
    )
    repo = RepoProfile(name="r", path=str(tmp_path))
    runtime = PlanningRuntime(store=store, config=cfg, repo=repo)
    session = store.create_planning_session(
        repo_name="r",
        title="t",
        requirements="req",
        seed_type="requirements",
        status="draft",
    )
    st = runtime._state(session.id, session)  # noqa: SLF001
    st.max_prompt_tokens = 500_000

    async def fake_call(*args, **kwargs):  # type: ignore[no-untyped-def]
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "LLM summary of critiques"
        return resp, "gpt-4o-mini"

    runtime.author._llm_client.call = fake_call  # type: ignore[method-assign]

    events: list[dict] = []

    async def capture(ev: dict) -> None:
        events.append(ev)

    out = await runtime.summarize_critiques_for_compaction(
        session_id=session.id,
        critic_turns=["first", "second critique"],
        requirements="Do the thing",
        current_plan="# Plan",
        event_callback=capture,
    )
    assert "LLM summary" in out
    assert any(e.get("type") == "context_compaction" and e.get("reason") == "critique_llm_summary" for e in events)


@pytest.mark.asyncio
async def test_summarize_critiques_for_compaction_heuristic_emits_event(tmp_path) -> None:
    store = Store(tmp_path / "t.db")
    cfg = PrscopeConfig(
        local_repo=str(tmp_path),
        planning=PlanningConfig(
            author_model="gpt-4o-mini",
            critic_model="gpt-4o-mini",
            critique_llm_summarize_enabled=False,
        ),
    )
    repo = RepoProfile(name="r", path=str(tmp_path))
    runtime = PlanningRuntime(store=store, config=cfg, repo=repo)
    session = store.create_planning_session(
        repo_name="r",
        title="t",
        requirements="req",
        seed_type="requirements",
        status="draft",
    )
    events: list[dict] = []

    async def capture(ev: dict) -> None:
        events.append(ev)

    out = await runtime.summarize_critiques_for_compaction(
        session_id=session.id,
        critic_turns=["a critique"],
        requirements="Do the thing",
        current_plan="# Plan",
        event_callback=capture,
    )
    assert "WORKING SUMMARY" in out
    assert any(
        e.get("type") == "context_compaction" and e.get("reason") == "critique_heuristic_summary" for e in events
    )


@pytest.mark.asyncio
async def test_discovery_llm_emits_context_compaction_on_trim() -> None:
    mgr = MagicMock()
    mgr.config = PlanningConfig(
        author_model="gpt-4o-mini",
        critic_model="gpt-4o-mini",
        discovery_conversation_max_chars=800,
    )
    mgr._emit = AsyncMock()
    mgr._normalize_roles = lambda msgs: msgs  # noqa: E731
    mgr._extract_feature_intent = MagicMock(return_value=None)
    mgr._latest_user_message = MagicMock(return_value="hi")
    mgr._active_discovery_session_id = "s1"

    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": ""},
        {"role": "tool", "content": "x" * 5000},
        {"role": "assistant", "content": ""},
        {"role": "tool", "content": "y" * 5000},
    ]
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = "final"
    resp.choices[0].message.tool_calls = None

    client = DiscoveryLLMClient(mgr)
    with patch.object(client, "safe_completion_call", new=AsyncMock(return_value=resp)):
        out = await client.llm_call_with_tools(msgs, max_tool_rounds=1)
    assert out == "final"
    mgr._emit.assert_any_call(
        {
            "type": "context_compaction",
            "enabled": True,
            "reason": "discovery_transcript_trim",
            "session_stage": "discovery",
        }
    )


@pytest.mark.asyncio
async def test_run_initial_draft_uses_dynamic_context_window(tmp_path) -> None:
    from prscope.planning.runtime.author import AuthorAgent
    from prscope.planning.runtime.authoring.models import AuthorResult

    tools = MagicMock()
    tools.accessed_paths = set()
    tools.read_history = {}
    agent = AuthorAgent(
        PlanningConfig(author_model="gpt-5", initial_draft_model="gpt-5"),
        tools,
    )

    async def fake_loop(*args, **kwargs):  # type: ignore[no-untyped-def]
        return AuthorResult(plan="# ok", unverified_references=set(), accessed_paths=set())

    with patch.object(AuthorAgent, "author_loop", new=fake_loop):
        with patch("prscope.planning.runtime.author.TokenBudgetManager") as tb_cls:
            tb_inst = MagicMock()
            tb_inst.available_prompt_tokens = 190_000
            tb_inst.enforce_required = MagicMock()
            tb_inst.allocate = MagicMock(side_effect=lambda text, rem: (text, min(100, rem)))
            tb_inst.injection_ratio = MagicMock(return_value=0.1)
            tb_inst.enforce_ratio = 0.5
            tb_inst.warn_ratio = 0.4
            tb_inst.context_window = 200_000
            tb_cls.return_value = tb_inst
            await agent.run_initial_draft(
                requirements="req",
                manifesto="m",
                manifesto_path=".prscope/manifesto.md",
                skills_block="",
                recall_block="",
                context_index="- x",
                grounding_paths=set(),
            )
            kw = tb_cls.call_args.kwargs
            assert kw["context_window"] == 200_000
            assert kw["reserved_prompt_tokens"] == 1000
