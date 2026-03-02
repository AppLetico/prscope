"""
Pluggable codebase scanner backends for prscope planning.

Usage in prscope.yml:
    planning:
      scanner: grep       # default — no extra deps
      scanner: repomap    # tree-sitter symbol map (pip install aider-chat)
      scanner: repomix    # full repo pack (npm install -g repomix)
"""

from __future__ import annotations

import logging

from .base import ScannerBackend
from .grep import GrepScanner
from .repomap import RepoMapScanner
from .repomix import RepomixScanner

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type[ScannerBackend]] = {
    "grep": GrepScanner,       # type: ignore[type-abstract]
    "repomap": RepoMapScanner, # type: ignore[type-abstract]
    "repomix": RepomixScanner, # type: ignore[type-abstract]
}


def get_scanner(name: str) -> ScannerBackend:
    """
    Return a scanner backend instance by name.

    If the requested backend is not available (missing dependency), falls
    back to GrepScanner with a logged warning.
    """
    cls = _REGISTRY.get(name)
    if cls is None:
        logger.warning(
            "Unknown scanner backend '%s'. Available: %s. Falling back to 'grep'.",
            name,
            ", ".join(_REGISTRY),
        )
        return GrepScanner()

    instance = cls()
    if not instance.is_available():
        logger.warning(
            "Scanner backend '%s' is not available (missing dependency). "
            "Falling back to 'grep'. See prscope docs for installation instructions.",
            name,
        )
        return GrepScanner()

    logger.debug("Using scanner backend: %s", name)
    return instance


def list_scanners() -> list[dict[str, object]]:
    """Return status of all registered scanner backends."""
    result = []
    for name, cls in _REGISTRY.items():
        instance = cls()
        result.append({
            "name": name,
            "available": instance.is_available(),
            "class": cls.__name__,
        })
    return result


__all__ = [
    "ScannerBackend",
    "GrepScanner",
    "RepoMapScanner",
    "RepomixScanner",
    "get_scanner",
    "list_scanners",
]
