from __future__ import annotations

from importlib import import_module
from pathlib import Path

from .connection import DB_FILENAME, StoreConnectionMixin
from .file_stats import StoreFileStatsMixin
from .planning_commands import StorePlanningCommandsMixin
from .planning_history import StorePlanningHistoryMixin
from .planning_sessions import StorePlanningSessionsMixin
from .repo_data import StoreRepoDataMixin
from .schema import StoreSchemaMixin


def _resolve_default_db_path() -> Path:
    # Keep compatibility with tests monkeypatching prscope.store.get_prscope_dir.
    module = import_module("prscope.store")
    getter = getattr(module, "get_prscope_dir", None)
    if callable(getter):
        return getter() / DB_FILENAME
    from prscope.config import get_prscope_dir

    return get_prscope_dir() / DB_FILENAME


class Store(
    StoreSchemaMixin,
    StoreConnectionMixin,
    StoreRepoDataMixin,
    StorePlanningCommandsMixin,
    StorePlanningSessionsMixin,
    StorePlanningHistoryMixin,
    StoreFileStatsMixin,
):
    """SQLite storage manager for Prscope."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or _resolve_default_db_path()
        self._ensure_schema()
