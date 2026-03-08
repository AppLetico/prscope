from __future__ import annotations

from prscope.planning.runtime.review import build_impact_view


def test_build_impact_view_clusters_root_causes_and_pressure() -> None:
    decision_graph = {
        "nodes": {
            "architecture.database": {
                "id": "architecture.database",
                "description": "Which database should store the primary application data?",
                "value": "PostgreSQL",
                "section": "architecture",
            }
        }
    }
    issue_graph = {
        "nodes": [
            {
                "id": "issue_1",
                "description": "Database scaling limits remain unresolved.",
                "status": "open",
                "raised_round": 1,
                "severity": "major",
                "issue_type": "architecture",
                "related_decision_ids": ["architecture.database"],
                "tags": [],
            },
            {
                "id": "issue_2",
                "description": "Database throughput may bottleneck rollout traffic.",
                "status": "open",
                "raised_round": 1,
                "severity": "minor",
                "issue_type": "performance",
                "related_decision_ids": ["architecture.database"],
                "tags": [],
            },
        ],
        "edges": [{"source": "issue_1", "target": "issue_2", "relation": "causes"}],
        "duplicate_alias": {},
    }

    impact_view = build_impact_view(decision_graph=decision_graph, issue_graph=issue_graph)

    assert [decision["decision_id"] for decision in impact_view["decisions"]] == ["architecture.database"]
    decision = impact_view["decisions"][0]
    assert decision["decision_pressure"] == 4
    assert decision["pressure_breakdown"] == {"major": 1, "minor": 1, "info": 0, "clusters": 1}
    assert decision["risk_level"] == "medium"
    assert decision["dominant_cluster"]["root_issue_id"] == "issue_1"
    assert decision["dominant_cluster"]["symptom_issue_count"] == 2
    assert decision["dominant_cluster"]["affected_plan_sections"] == ["architecture"]


def test_build_impact_view_excludes_resolved_and_dependency_blocked_issues() -> None:
    decision_graph = {
        "nodes": {
            "architecture.cache_strategy": {
                "id": "architecture.cache_strategy",
                "description": "What caching strategy should this feature use?",
                "value": "shared cache",
                "section": "architecture",
            }
        }
    }
    issue_graph = {
        "nodes": [
            {
                "id": "issue_1",
                "description": "Caching strategy remains ambiguous.",
                "status": "resolved",
                "raised_round": 1,
                "severity": "major",
                "issue_type": "ambiguity",
                "related_decision_ids": ["architecture.cache_strategy"],
                "tags": ["decision:missing"],
            },
            {
                "id": "issue_2",
                "description": "Cache invalidation plan is blocked by missing consistency guarantees.",
                "status": "open",
                "raised_round": 2,
                "severity": "major",
                "issue_type": "correctness",
                "related_decision_ids": ["architecture.cache_strategy"],
                "tags": [],
            },
            {
                "id": "issue_3",
                "description": "Consistency guarantees are still undefined.",
                "status": "open",
                "raised_round": 2,
                "severity": "major",
                "issue_type": "ambiguity",
                "related_decision_ids": [],
                "tags": [],
            },
        ],
        "edges": [{"source": "issue_2", "target": "issue_3", "relation": "depends_on"}],
        "duplicate_alias": {},
    }

    impact_view = build_impact_view(decision_graph=decision_graph, issue_graph=issue_graph)

    assert impact_view["decisions"] == []


def test_build_impact_view_handles_causal_cycles_and_emits_reconsideration_candidates() -> None:
    decision_graph = {
        "nodes": {
            "architecture.database": {
                "id": "architecture.database",
                "description": "Which database should store the primary application data?",
                "value": "PostgreSQL",
                "section": "architecture",
            }
        }
    }
    previous_decision_graph = {
        "nodes": {
            "architecture.database": {
                "id": "architecture.database",
                "description": "Which database should store the primary application data?",
                "value": "PostgreSQL",
                "section": "architecture",
            }
        }
    }
    issue_graph = {
        "nodes": [
            {
                "id": "issue_1",
                "description": "Database choice conflicts with throughput requirements.",
                "status": "open",
                "raised_round": 1,
                "severity": "major",
                "issue_type": "architecture",
                "related_decision_ids": ["architecture.database"],
                "tags": ["decision:conflict"],
            },
            {
                "id": "issue_2",
                "description": "Database throughput remains below the scaling target.",
                "status": "open",
                "raised_round": 2,
                "severity": "major",
                "issue_type": "performance",
                "related_decision_ids": ["architecture.database"],
                "tags": [],
            },
        ],
        "edges": [
            {"source": "issue_1", "target": "issue_2", "relation": "causes"},
            {"source": "issue_2", "target": "issue_1", "relation": "causes"},
        ],
        "duplicate_alias": {},
    }

    impact_view = build_impact_view(
        decision_graph=decision_graph,
        issue_graph=issue_graph,
        previous_decision_graph=previous_decision_graph,
    )

    decision = impact_view["decisions"][0]
    assert decision["decision_pressure"] == 6
    assert decision["dominant_cluster"]["root_issue_id"] == "issue_1"
    assert decision["dominant_cluster"]["suggested_action"] == "reconsider architecture"
    assert impact_view["reconsideration_candidates"] == [
        {
            "decision_id": "architecture.database",
            "reason": "high_pressure_cluster",
            "decision_pressure": 6,
            "dominant_cluster": decision["dominant_cluster"],
            "suggested_action": "reconsider architecture",
            "recently_changed": False,
            "eligible": True,
        }
    ]
