"""
Local repository profiling for Prscope.

Scans the filesystem to build a profile of the local codebase:
- File tree structure
- Package dependencies (from package.json, requirements.txt, etc.)
- Basic import/export statistics
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from .config import get_repo_root

# Directories to ignore when scanning
IGNORE_DIRS = {
    ".git",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".prscope",
    ".next",
    ".nuxt",
    "venv",
    ".venv",
    "env",
    ".env",
    "coverage",
    ".coverage",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "target",  # Rust
    "vendor",  # Go, PHP
}

# File extensions to analyze
CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".scala",
    ".rb",
    ".php",
    ".swift",
    ".c",
    ".cpp",
    ".h",
    ".cs",
    ".fs",
    ".vue",
    ".svelte",
}


def get_git_head_sha(repo_root: Path | None = None) -> str:
    """Get the current git HEAD SHA."""
    if repo_root is None:
        repo_root = get_repo_root()

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        # Not a git repo or git not available
        return "unknown"


def scan_file_tree(repo_root: Path | None = None) -> dict[str, Any]:
    """
    Scan the repository file tree.

    Returns a dictionary with:
    - files: list of relative file paths
    - directories: list of relative directory paths
    - extensions: count of files by extension
    - total_files: total file count
    """
    if repo_root is None:
        repo_root = get_repo_root()

    files: list[str] = []
    directories: set[str] = set()
    extensions: dict[str, int] = {}

    for path in repo_root.rglob("*"):
        # Get relative path
        try:
            rel_path = path.relative_to(repo_root)
        except ValueError:
            continue

        # Check if any parent is in ignore list
        parts = rel_path.parts
        if any(part in IGNORE_DIRS for part in parts):
            continue

        if path.is_file():
            files.append(str(rel_path))
            ext = path.suffix.lower()
            if ext:
                extensions[ext] = extensions.get(ext, 0) + 1
        elif path.is_dir():
            directories.add(str(rel_path))

    return {
        "files": sorted(files),
        "directories": sorted(directories),
        "extensions": extensions,
        "total_files": len(files),
    }


def parse_package_json(repo_root: Path | None = None) -> dict[str, Any] | None:
    """Parse package.json for Node.js dependencies."""
    if repo_root is None:
        repo_root = get_repo_root()

    package_path = repo_root / "package.json"
    if not package_path.exists():
        return None

    try:
        with open(package_path) as f:
            data = json.load(f)

        return {
            "name": data.get("name"),
            "version": data.get("version"),
            "dependencies": list(data.get("dependencies", {}).keys()),
            "devDependencies": list(data.get("devDependencies", {}).keys()),
        }
    except (OSError, json.JSONDecodeError):
        return None


def parse_requirements_txt(repo_root: Path | None = None) -> list[str] | None:
    """Parse requirements.txt for Python dependencies."""
    if repo_root is None:
        repo_root = get_repo_root()

    req_path = repo_root / "requirements.txt"
    if not req_path.exists():
        return None

    try:
        with open(req_path) as f:
            lines = f.readlines()

        deps = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                # Extract package name (before any version specifier)
                pkg = line.split("==")[0].split(">=")[0].split("<=")[0].split("[")[0]
                deps.append(pkg.strip())

        return deps
    except OSError:
        return None


def parse_pyproject_toml(repo_root: Path | None = None) -> dict[str, Any] | None:
    """Parse pyproject.toml for Python project info."""
    if repo_root is None:
        repo_root = get_repo_root()

    pyproject_path = repo_root / "pyproject.toml"
    if not pyproject_path.exists():
        return None

    try:
        # Simple TOML parsing for dependencies
        # (avoiding external dependency for now)
        with open(pyproject_path) as f:
            content = f.read()

        deps = []
        in_deps = False
        for line in content.split("\n"):
            if "dependencies" in line and "=" in line:
                in_deps = True
                # Handle inline array
                if "[" in line:
                    start = line.index("[")
                    if "]" in line:
                        end = line.index("]")
                        deps_str = line[start + 1 : end]
                        for dep in deps_str.split(","):
                            dep = dep.strip().strip('"').strip("'")
                            if dep:
                                pkg = dep.split(">=")[0].split("==")[0].split("[")[0]
                                deps.append(pkg.strip())
                        in_deps = False
            elif in_deps:
                if "]" in line:
                    in_deps = False
                else:
                    line = line.strip().strip(",").strip('"').strip("'")
                    if line:
                        pkg = line.split(">=")[0].split("==")[0].split("[")[0]
                        deps.append(pkg.strip())

        return {"dependencies": deps} if deps else None
    except OSError:
        return None


def compute_import_stats(repo_root: Path | None = None) -> dict[str, Any]:
    """
    Compute basic import/export statistics for code files.

    Returns counts of import statements by type.
    """
    if repo_root is None:
        repo_root = get_repo_root()

    stats = {
        "python_imports": 0,
        "js_imports": 0,
        "js_exports": 0,
        "files_analyzed": 0,
    }

    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue

        # Check ignore dirs
        try:
            rel_path = path.relative_to(repo_root)
            if any(part in IGNORE_DIRS for part in rel_path.parts):
                continue
        except ValueError:
            continue

        ext = path.suffix.lower()
        if ext not in CODE_EXTENSIONS:
            continue

        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                content = f.read()

            stats["files_analyzed"] += 1

            for line in content.split("\n"):
                line = line.strip()

                # Python imports
                if ext == ".py":
                    if line.startswith("import ") or line.startswith("from "):
                        stats["python_imports"] += 1

                # JS/TS imports and exports
                if ext in {".js", ".ts", ".tsx", ".jsx"}:
                    if line.startswith("import ") or "require(" in line:
                        stats["js_imports"] += 1
                    if line.startswith("export "):
                        stats["js_exports"] += 1
        except OSError:
            continue

    return stats


def extract_readme(repo_root: Path | None = None, max_chars: int = 4000) -> str | None:
    """
    Extract README content from the repository.

    Looks for README.md, README.rst, README.txt, or README in order.
    Truncates to max_chars for LLM context efficiency.
    """
    if repo_root is None:
        repo_root = get_repo_root()

    readme_names = ["README.md", "README.rst", "README.txt", "README", "readme.md"]

    for name in readme_names:
        readme_path = repo_root / name
        if readme_path.exists():
            try:
                with open(readme_path, encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Truncate if too long, but try to cut at a paragraph break
                if len(content) > max_chars:
                    # Find a good cut point
                    cut_point = content.rfind("\n\n", 0, max_chars)
                    if cut_point < max_chars // 2:
                        cut_point = max_chars
                    content = content[:cut_point] + "\n\n... (truncated)"

                return content
            except OSError:
                continue

    return None


def build_profile(repo_root: Path | None = None) -> dict[str, Any]:
    """
    Build a complete profile of the local repository.

    Returns a dictionary containing:
    - git_sha: Current HEAD SHA
    - file_tree: File structure info
    - dependencies: Package dependencies
    - import_stats: Import/export statistics
    - readme: README content (for LLM context)
    - timestamp: When the profile was created
    """
    if repo_root is None:
        repo_root = get_repo_root()

    from datetime import datetime

    profile = {
        "repo_root": str(repo_root),
        "git_sha": get_git_head_sha(repo_root),
        "file_tree": scan_file_tree(repo_root),
        "dependencies": {},
        "import_stats": compute_import_stats(repo_root),
        "readme": extract_readme(repo_root),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    # Add package dependencies
    pkg_json = parse_package_json(repo_root)
    if pkg_json:
        profile["dependencies"]["node"] = pkg_json

    req_txt = parse_requirements_txt(repo_root)
    if req_txt:
        profile["dependencies"]["python_requirements"] = req_txt

    pyproject = parse_pyproject_toml(repo_root)
    if pyproject:
        profile["dependencies"]["python_pyproject"] = pyproject

    return profile


def hash_profile(profile: dict[str, Any]) -> str:
    """
    Compute a hash of the profile for comparison.
    Uses the git SHA if available, otherwise hashes the content.
    """
    git_sha = profile.get("git_sha", "")
    if git_sha and git_sha != "unknown":
        return git_sha

    # Fallback: hash the profile content
    content = json.dumps(profile, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]
