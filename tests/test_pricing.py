from __future__ import annotations

from prscope.model_catalog import CATALOG
from prscope.pricing import MODEL_CONTEXT_WINDOWS, MODEL_PRICING, estimate_cost_usd

# Keep this list intentionally small. If a model lacks a known context window,
# add it here with a clear decision rather than silently drifting.
ALLOWED_UNKNOWN_CONTEXT_WINDOWS = {
    "gpt-5.2",
    "gpt-5.3-codex",
}


def test_model_pricing_contains_new_frontier_models():
    assert MODEL_PRICING["o3"] == (2.00, 8.00)
    assert MODEL_PRICING["o4-mini"] == (1.10, 4.40)
    assert MODEL_PRICING["claude-sonnet-4-6"] == (3.00, 15.00)
    assert MODEL_PRICING["claude-opus-4-6"] == (5.00, 25.00)
    assert MODEL_PRICING["gemini-3.1-pro"] == (2.00, 12.00)


def test_estimate_cost_uses_per_million_rates():
    # 500k prompt + 250k completion on o3
    estimate = estimate_cost_usd("o3", prompt_tokens=500_000, completion_tokens=250_000)
    assert estimate.prompt_cost_usd == 1.0
    assert estimate.completion_cost_usd == 2.0
    assert estimate.total_cost_usd == 3.0


def test_estimate_cost_unknown_model_is_zero():
    estimate = estimate_cost_usd("not-a-real-model", prompt_tokens=123_456, completion_tokens=654_321)
    assert estimate.prompt_cost_usd == 0.0
    assert estimate.completion_cost_usd == 0.0
    assert estimate.total_cost_usd == 0.0


def test_catalog_models_have_context_window_metadata_or_are_explicitly_allowlisted():
    catalog_model_ids = {spec.model_id for spec in CATALOG}
    models_with_context_windows = set(MODEL_CONTEXT_WINDOWS.keys())
    missing = sorted(
        model_id
        for model_id in catalog_model_ids
        if model_id not in models_with_context_windows and model_id not in ALLOWED_UNKNOWN_CONTEXT_WINDOWS
    )
    assert missing == []
