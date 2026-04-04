"""
Coerce LLM-emitted tool arguments (often JSON strings) to typed values.

Mirrors Claude Code-style semantic coercion for robust tool execution.
"""

from __future__ import annotations

from typing import Any


def coerce_int(value: Any, *, default: int | None = None) -> int | None:
    """Parse int from int, float, or numeric string. Booleans are treated as ints (True=1)."""
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return default
        try:
            return int(float(s))
        except ValueError:
            return default
    return default


def coerce_int_optional(value: Any) -> int | None:
    """Like coerce_int but preserves explicit None; empty string -> None."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return coerce_int(value, default=None)


def coerce_non_negative_int(value: Any, *, default: int = 0) -> int:
    v = coerce_int(value, default=default)
    if v is None:
        return default
    return max(0, v)


def coerce_positive_int(value: Any, *, default: int, minimum: int = 1) -> int:
    v = coerce_int(value, default=default)
    if v is None:
        return default
    return max(minimum, v)


def coerce_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off", ""):
            return False
    return default


def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))
