from __future__ import annotations

from prscope.profile import (
    IGNORE_DIRS,
    build_profile,
    extract_readme,
    scan_file_tree,
)


def test_extract_readme_finds_readme_md(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# My Project\n\nThis is a test project.")

    content = extract_readme(tmp_path)

    assert content is not None
    assert "# My Project" in content
    assert "test project" in content


def test_extract_readme_returns_none_when_missing(tmp_path):
    content = extract_readme(tmp_path)
    assert content is None


def test_extract_readme_truncates_long_content(tmp_path):
    readme = tmp_path / "README.md"
    # Create a very long README
    long_content = "# Project\n\n" + ("Lorem ipsum dolor sit amet. " * 500)
    readme.write_text(long_content)

    content = extract_readme(tmp_path, max_chars=1000)

    assert content is not None
    assert len(content) < 1100  # Some buffer for truncation message
    assert "truncated" in content


def test_extract_readme_prefers_readme_md(tmp_path):
    # Create multiple README files
    (tmp_path / "README.md").write_text("# Markdown README")
    (tmp_path / "README.txt").write_text("Text README")
    (tmp_path / "README").write_text("Plain README")

    content = extract_readme(tmp_path)

    assert content is not None
    assert "Markdown README" in content


def test_build_profile_includes_readme(tmp_path):
    # Create a minimal project
    (tmp_path / "README.md").write_text("# Test Project\n\nDescription here.")
    (tmp_path / "main.py").write_text("print('hello')")

    profile = build_profile(tmp_path)

    assert "readme" in profile
    assert profile["readme"] is not None
    assert "Test Project" in profile["readme"]


def test_build_profile_readme_none_when_missing(tmp_path):
    # Create project without README
    (tmp_path / "main.py").write_text("print('hello')")

    profile = build_profile(tmp_path)

    assert "readme" in profile
    assert profile["readme"] is None


def test_scan_file_tree_ignores_directories(tmp_path):
    # Create files in normal and ignored directories
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("code")

    node_modules = tmp_path / "node_modules"
    node_modules.mkdir()
    (node_modules / "pkg.js").write_text("ignored")

    tree = scan_file_tree(tmp_path)

    assert "src/main.py" in tree["files"]
    assert not any("node_modules" in f for f in tree["files"])


def test_ignore_dirs_contains_expected():
    assert ".git" in IGNORE_DIRS
    assert "node_modules" in IGNORE_DIRS
    assert "__pycache__" in IGNORE_DIRS
    assert ".prscope" in IGNORE_DIRS
