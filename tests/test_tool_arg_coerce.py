"""Tests for tool argument coercion."""

from __future__ import annotations

from prscope.planning.runtime.tool_arg_coerce import (
    clamp_int,
    coerce_bool,
    coerce_int,
    coerce_int_optional,
    coerce_non_negative_int,
    coerce_positive_int,
)


def test_coerce_int_string() -> None:
    assert coerce_int("10", default=0) == 10
    assert coerce_int("  -3 ", default=0) == -3


def test_coerce_int_invalid() -> None:
    assert coerce_int("x", default=None) is None
    assert coerce_int(None, default=42) == 42


def test_coerce_non_negative_int() -> None:
    assert coerce_non_negative_int("-1", default=0) == 0
    assert coerce_non_negative_int("5", default=0) == 5


def test_coerce_positive_int() -> None:
    assert coerce_positive_int(0, default=40, minimum=1) == 1
    assert coerce_positive_int("40", default=1, minimum=1) == 40


def test_coerce_bool_strings() -> None:
    assert coerce_bool("true", default=False) is True
    assert coerce_bool("0", default=True) is False


def test_coerce_int_optional_empty_str() -> None:
    assert coerce_int_optional("") is None
    assert coerce_int_optional("  ") is None


def test_clamp_int() -> None:
    assert clamp_int(500, 1, 200) == 200
