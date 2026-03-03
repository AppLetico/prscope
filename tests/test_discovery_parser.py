from types import SimpleNamespace

import pytest

from prscope.planning.runtime.discovery import DiscoveryManager, parse_questions


def test_parse_questions_handles_plain_q_format():
    reply = """
Q1: What should the health endpoint return?
A) A simple "OK" with 200 status
B) JSON with {"status":"ok"}
C) Include build metadata in payload
D) Other - describe preference
""".strip()

    questions = parse_questions(reply)
    assert len(questions) == 1
    assert questions[0].text == "What should the health endpoint return?"
    assert len(questions[0].options) == 4
    assert questions[0].options[0].letter == "A"
    assert questions[0].options[3].is_other is True


def test_parse_questions_handles_markdown_wrapped_blocks():
    reply = """
**Q2: Where should the endpoint be added?**
- A) In `prscope/web/api.py`
- B) In a dedicated health module
- C) In an existing router class
- D) Other — describe your preference
""".strip()

    questions = parse_questions(reply)
    assert len(questions) == 1
    assert questions[0].text == "Where should the endpoint be added?"
    assert [opt.letter for opt in questions[0].options] == ["A", "B", "C", "D"]


def test_extract_completion_accepts_fenced_nested_json_with_comments():
    raw = """
Based on your responses, here is the draft direction.
```json
{
  "discovery": "complete",
  "summary": {
    "endpoint_design": {
      "path": "/health",
      "methods": ["GET"]
    },
    "files_to_modify": [
      "prscope/web/api.py" // create if missing
    ]
  }
}
```
""".strip()

    manager = DiscoveryManager.__new__(DiscoveryManager)
    result = manager._try_extract_completion(raw)
    assert result.complete is True
    assert result.summary is not None
    assert '"path": "/health"' in result.summary
    assert "Based on your responses" in result.reply


def test_extract_completion_does_not_complete_for_incomplete_json():
    raw = """
We should keep discovery open until you confirm deployment constraints.
{
  "discovery": "in_progress",
  "summary": "not done yet"
}
""".strip()

    manager = DiscoveryManager.__new__(DiscoveryManager)
    result = manager._try_extract_completion(raw)
    assert result.complete is False
    assert result.summary is None


def test_discovery_turn_counts_are_isolated_per_session():
    manager = DiscoveryManager.__new__(DiscoveryManager)
    manager.turn_counts_by_session = {}

    assert manager._next_turn_count("session-a") == 1
    assert manager._next_turn_count("session-a") == 2
    assert manager._next_turn_count("session-b") == 1
    manager.reset_session("session-a")
    assert manager._next_turn_count("session-a") == 1
    manager.clear_session("session-b")
    assert manager._next_turn_count("session-b") == 1


@pytest.mark.asyncio
async def test_handle_turn_expands_singleton_questions_on_first_turn():
    manager = DiscoveryManager.__new__(DiscoveryManager)
    manager.turn_counts_by_session = {}
    manager.config = SimpleNamespace(discovery_max_turns=5, discovery_tool_rounds=3)
    manager._build_memory_context = lambda: ""

    async def _noop_emit(_event):
        return None

    async def _singleton_with_tools(_messages, max_tool_rounds=0, model_override=None):
        _ = max_tool_rounds, model_override
        return """
Q1: Which endpoint shape should we use?
A) /api/health
B) /healthz
C) /status
D) Other — describe your preference
""".strip()

    async def _expanded_without_tools(_messages, model_override=None):
        _ = model_override
        return """
Q1: Which endpoint shape should we use?
A) /api/health
B) /healthz
C) /status
D) Other — describe your preference

Q2: What response payload format should we return?
A) {"status":"healthy"}
B) {"ok":true}
C) Include version metadata
D) Other — describe your preference
""".strip()

    manager._emit = _noop_emit
    manager._llm_call_with_tools = _singleton_with_tools
    manager._llm_call = _expanded_without_tools

    result = await manager.handle_turn(
        conversation=[{"role": "user", "content": "Add a health endpoint"}],
        session_id="session-batch-test",
    )

    assert result.complete is False
    assert len(result.questions) == 2


@pytest.mark.asyncio
async def test_handle_turn_expands_singleton_questions_on_later_turns():
    manager = DiscoveryManager.__new__(DiscoveryManager)
    manager.turn_counts_by_session = {"session-batch-test-2": 1}
    manager.config = SimpleNamespace(discovery_max_turns=5, discovery_tool_rounds=3)
    manager._build_memory_context = lambda: ""

    async def _noop_emit(_event):
        return None

    async def _singleton_with_tools(_messages, max_tool_rounds=0, model_override=None):
        _ = max_tool_rounds, model_override
        return """
Q2: Where should tests be placed?
A) tests/test_health.py
B) tests/test_web_api_models.py
C) tests/api/test_health.py
D) Other — describe your preference
""".strip()

    async def _expanded_without_tools(_messages, model_override=None):
        _ = model_override
        return """
Q1: Where should tests be placed?
A) tests/test_health.py
B) tests/test_web_api_models.py
C) tests/api/test_health.py
D) Other — describe your preference

Q2: What should the initial assertion include?
A) 200 status only
B) status and JSON payload fields
C) include schema validation
D) Other — describe your preference
""".strip()

    manager._emit = _noop_emit
    manager._llm_call_with_tools = _singleton_with_tools
    manager._llm_call = _expanded_without_tools

    result = await manager.handle_turn(
        conversation=[{"role": "user", "content": "use unit tests"}],
        session_id="session-batch-test-2",
    )

    assert result.complete is False
    assert len(result.questions) == 2

