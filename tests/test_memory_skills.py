from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from prscope.memory import MemoryStore, load_skills


def test_load_skills_sorted_and_boundary_safe(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "b.md").write_text("B skill", encoding="utf-8")
    (skills_dir / "a.md").write_text("A skill", encoding="utf-8")

    content = load_skills(skills_dir, max_chars=10_000)
    assert "### a.md" in content
    assert "### b.md" in content
    assert content.index("### a.md") < content.index("### b.md")


def test_load_skills_skips_oversized_first_file(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "a.md").write_text("x" * 5000, encoding="utf-8")
    (skills_dir / "b.md").write_text("small", encoding="utf-8")

    content = load_skills(skills_dir, max_chars=100)
    assert "### a.md" not in content
    assert "### b.md" in content


def test_load_skills_truncates_at_boundaries(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "a.md").write_text("A", encoding="utf-8")
    (skills_dir / "b.md").write_text("B" * 1000, encoding="utf-8")

    content = load_skills(skills_dir, max_chars=120)
    assert "### a.md" in content
    assert "### b.md" not in content
    assert content.endswith("... (truncated due to token budget)")


@pytest.mark.asyncio
async def test_memory_complete_emits_usage_without_logging_format_error(monkeypatch):
    calls: list[dict[str, object]] = []
    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="summary block"))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )

    def _completion(**kwargs):
        calls.append(kwargs)
        return fake_response

    fake_litellm = SimpleNamespace(completion=_completion)
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    fake_store = SimpleNamespace(config=SimpleNamespace(memory_model="gemini-2.5-flash"))
    events: list[dict[str, object]] = []

    async def _capture(event: dict[str, object]) -> None:
        events.append(event)

    result = await MemoryStore._complete(
        fake_store,  # type: ignore[arg-type]
        "Summarize the repo.",
        block_name="modules",
        event_callback=_capture,
    )

    assert result == "summary block"
    assert calls[0]["model"] == "gemini/gemini-2.5-flash"
    assert len(events) == 1
    assert events[0]["type"] == "token_usage"
    assert events[0]["memory_block"] == "modules"
    assert events[0]["model"] == "gemini-2.5-flash"
