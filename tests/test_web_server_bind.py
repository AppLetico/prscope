from __future__ import annotations

import pytest

from prscope.web.server import PUBLIC_BIND_ENV, require_safe_bind_host


def test_require_safe_bind_host_allows_loopback() -> None:
    require_safe_bind_host("127.0.0.1")
    require_safe_bind_host("localhost")
    require_safe_bind_host("::1")


def test_require_safe_bind_host_blocks_public_without_env() -> None:
    for host in ("0.0.0.0", "::"):
        with pytest.raises(RuntimeError, match=PUBLIC_BIND_ENV):
            require_safe_bind_host(host)


def test_require_safe_bind_host_allows_public_with_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PUBLIC_BIND_ENV, "1")
    require_safe_bind_host("0.0.0.0")
    require_safe_bind_host("::")
