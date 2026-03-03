from __future__ import annotations

from prscope.memory import load_skills


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
