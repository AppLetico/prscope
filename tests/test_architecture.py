"""Structural import boundary tests.

Enforces the layer dependency rules defined in ARCHITECTURE.md:

    Interface → Intelligence → Storage → Foundation
                                          (no upward imports)

Run with: pytest tests/test_architecture.py -v
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "prscope"

# ---------------------------------------------------------------------------
# Layer definitions
# ---------------------------------------------------------------------------

FOUNDATION = {
    "config",
    "pricing",
    "model_catalog",
    "profile",
    "semantic",
}

STORAGE = {
    "store",
}

INTELLIGENCE = {
    "memory",
    "llm",
    "scoring",
    "github",
    "planner",
    "planning",
    "planning.core",
    "planning.executor",
    "planning.render",
    "planning.scanners",
    "planning.scanners.base",
    "planning.scanners.grep",
    "planning.scanners.repomap",
    "planning.scanners.repomix",
    "planning.runtime",
    "planning.runtime.orchestration",
    "planning.runtime.discovery",
    "planning.runtime.author",
    "planning.runtime.authoring",
    "planning.runtime.authoring.models",
    "planning.runtime.authoring.discovery",
    "planning.runtime.authoring.validation",
    "planning.runtime.authoring.repair",
    "planning.runtime.authoring.pipeline",
    "planning.runtime.critic",
    "planning.runtime.tools",
    "planning.runtime.budget",
    "planning.runtime.telemetry",
    "planning.runtime.clarification",
    "planning.runtime.compression",
    "planning.runtime.round_controller",
    "planning.runtime.analytics_emitter",
    "planning.runtime.context",
    "planning.runtime.context.budget",
    "planning.runtime.context.clarification",
    "planning.runtime.context.compression",
    "planning.runtime.context.context_assembler",
    "planning.runtime.events",
    "planning.runtime.events.analytics_emitter",
    "planning.runtime.events.token_accounting",
    "planning.runtime.events.tool_event_state",
    "planning.runtime.pipeline",
    "planning.runtime.pipeline.adversarial_loop",
    "planning.runtime.pipeline.round_context",
    "planning.runtime.pipeline.stages",
    "planning.runtime.review",
    "planning.runtime.review.issue_causality",
    "planning.runtime.review.issue_graph",
    "planning.runtime.review.issue_similarity",
    "planning.runtime.review.manifesto_checker",
    "planning.runtime.state",
    "planning.runtime.transport",
    "planning.runtime.transport.llm_client",
}

INTERFACE = {
    "cli",
    "benchmark",
    "web",
    "web.api",
    "web.server",
    "web.events",
}


def _layer_of(module: str) -> str | None:
    """Return the layer name for a prscope module, or None if unknown."""
    if module in FOUNDATION:
        return "foundation"
    if module in STORAGE:
        return "storage"
    if module in INTELLIGENCE:
        return "intelligence"
    if module in INTERFACE:
        return "interface"
    return None


ALLOWED_IMPORTS: dict[str, set[str]] = {
    "foundation": {"foundation"},
    "storage": {"foundation", "storage"},
    "intelligence": {"foundation", "storage", "intelligence"},
    "interface": {"foundation", "storage", "intelligence", "interface"},
}

# ---------------------------------------------------------------------------
# Import extraction
# ---------------------------------------------------------------------------

_RELATIVE_IMPORT_RE = re.compile(r"^from\s+(\.+)(\S*)\s+import")


def _resolve_relative(dots: str, remainder: str, module_path: Path) -> str:
    """Resolve a relative import to a dotted module name relative to prscope/."""
    levels = len(dots)
    parts = list(module_path.relative_to(PACKAGE_ROOT).with_suffix("").parts)
    # Go up `levels` directories (first level goes to parent of the file's package)
    anchor = parts[:-1]  # directory containing the module
    for _ in range(levels - 1):
        if anchor:
            anchor.pop()
    target = ".".join(anchor + ([remainder] if remainder else []))
    return target


def _extract_prscope_imports(filepath: Path) -> list[str]:
    """Return dotted module names of all prscope-internal imports in a file."""
    source = filepath.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return []

    results: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level and node.level > 0:
            dots = "." * node.level
            resolved = _resolve_relative(dots, node.module, filepath)
            # Normalize to top-level module (first component or two for sub-packages)
            results.append(resolved)
    return results


def _normalize_to_known_module(dotted: str) -> str | None:
    """Map a resolved dotted path to the nearest known module in our layer map."""
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        if _layer_of(candidate) is not None:
            return candidate
    return None


def _collect_python_files() -> list[Path]:
    return sorted(PACKAGE_ROOT.rglob("*.py"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _module_name(filepath: Path) -> str:
    return ".".join(filepath.relative_to(PACKAGE_ROOT).with_suffix("").parts)


def _check_layer_boundary(filepath: Path) -> list[str]:
    """Return a list of violation messages for a single file."""
    mod_name = _module_name(filepath)
    if mod_name == "__init__":
        return []
    source_module = _normalize_to_known_module(mod_name)
    if source_module is None:
        return []
    source_layer = _layer_of(source_module)
    if source_layer is None:
        return []

    violations = []
    for imp in _extract_prscope_imports(filepath):
        target_module = _normalize_to_known_module(imp)
        if target_module is None:
            continue
        target_layer = _layer_of(target_module)
        if target_layer is None:
            continue
        if target_layer not in ALLOWED_IMPORTS[source_layer]:
            violations.append(
                f"{mod_name} ({source_layer}) imports {imp} ({target_layer}) — "
                f"VIOLATION: {source_layer} may only import from {sorted(ALLOWED_IMPORTS[source_layer])}"
            )
    return violations


def test_no_upward_layer_imports():
    """Every module respects the layer dependency direction defined in ARCHITECTURE.md."""
    all_violations = []
    for filepath in _collect_python_files():
        all_violations.extend(_check_layer_boundary(filepath))
    assert all_violations == [], "Import boundary violations found:\n" + "\n".join(f"  • {v}" for v in all_violations)


def test_intelligence_never_imports_interface():
    """Intelligence layer modules must never import from Interface layer."""
    violations = []
    for filepath in _collect_python_files():
        mod_name = _module_name(filepath)
        source_module = _normalize_to_known_module(mod_name)
        if source_module is None:
            continue
        if _layer_of(source_module) != "intelligence":
            continue
        for imp in _extract_prscope_imports(filepath):
            target = _normalize_to_known_module(imp)
            if target and _layer_of(target) == "interface":
                violations.append(f"{mod_name} imports {imp} (interface)")
    assert violations == [], "Intelligence → Interface violations:\n" + "\n".join(f"  • {v}" for v in violations)


def test_foundation_only_imports_foundation():
    """Foundation modules may only import from other Foundation modules."""
    violations = []
    for filepath in _collect_python_files():
        mod_name = _module_name(filepath)
        source_module = _normalize_to_known_module(mod_name)
        if source_module is None:
            continue
        if _layer_of(source_module) != "foundation":
            continue
        for imp in _extract_prscope_imports(filepath):
            target = _normalize_to_known_module(imp)
            if target and _layer_of(target) != "foundation":
                violations.append(f"{mod_name} imports {imp} ({_layer_of(target)})")
    assert violations == [], "Foundation boundary violations:\n" + "\n".join(f"  • {v}" for v in violations)


def test_storage_imports_only_foundation():
    """Storage modules may only import from Foundation."""
    violations = []
    for filepath in _collect_python_files():
        mod_name = _module_name(filepath)
        source_module = _normalize_to_known_module(mod_name)
        if source_module is None:
            continue
        if _layer_of(source_module) != "storage":
            continue
        for imp in _extract_prscope_imports(filepath):
            target = _normalize_to_known_module(imp)
            if target and _layer_of(target) not in ("foundation", "storage"):
                violations.append(f"{mod_name} imports {imp} ({_layer_of(target)})")
    assert violations == [], "Storage boundary violations:\n" + "\n".join(f"  • {v}" for v in violations)


def test_runtime_leaf_modules_stay_leaf():
    """Runtime leaf modules (tools, budget, analytics_emitter, clarification, compression)
    must not import from planning.core or planning.executor."""
    leaf_modules = {
        "planning.runtime.tools",
        "planning.runtime.budget",
        "planning.runtime.analytics_emitter",
        "planning.runtime.clarification",
        "planning.runtime.compression",
    }
    forbidden_targets = {"planning.core", "planning.executor"}
    violations = []
    for filepath in _collect_python_files():
        mod_name = _module_name(filepath)
        source_module = _normalize_to_known_module(mod_name)
        if source_module not in leaf_modules:
            continue
        for imp in _extract_prscope_imports(filepath):
            target = _normalize_to_known_module(imp)
            if target in forbidden_targets:
                violations.append(f"{mod_name} imports {imp} — leaf modules must not import core/executor")
    assert violations == [], "Leaf module violations:\n" + "\n".join(f"  • {v}" for v in violations)
