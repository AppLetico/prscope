"""Stable fingerprint for plan markdown (skip redundant design review)."""

from __future__ import annotations

import hashlib


def plan_content_fingerprint(plan_content: str) -> str:
    return hashlib.sha256(str(plan_content or "").encode("utf-8")).hexdigest()
