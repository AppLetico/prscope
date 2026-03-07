from prscope.planning.runtime.followups import (
    apply_answer_to_graph,
    decision_graph_from_json,
    decision_graph_from_open_questions,
    decision_graph_from_plan,
    decision_graph_to_json,
    merge_decision_graphs,
)


def test_catalog_backed_decision_ids_survive_question_wording_changes() -> None:
    previous = decision_graph_from_open_questions("- Which database should store the primary application data?")
    previous = apply_answer_to_graph(previous, "architecture.database", "PostgreSQL")

    current = decision_graph_from_open_questions("- What database should persist the primary app data?")
    merged = merge_decision_graphs(current, previous)

    database = merged.nodes["architecture.database"]
    assert database.value == "PostgreSQL"
    assert database.concept == "primary_database"
    assert database.evidence == [
        "open_questions: What database should persist the primary app data?",
        "open_questions: Which database should store the primary application data?",
    ]


def test_merge_preserves_resolved_decisions_when_rewrite_omits_them() -> None:
    previous = decision_graph_from_open_questions("- Which database should store the primary application data?")
    previous = apply_answer_to_graph(previous, "architecture.database", "PostgreSQL")

    current = decision_graph_from_open_questions("- What logging strategy should this feature use?")
    merged = merge_decision_graphs(current, previous)

    assert "architecture.logging_strategy" in merged.nodes
    assert merged.nodes["architecture.database"].value == "PostgreSQL"


def test_uncataloged_resolved_decisions_survive_small_rewrites() -> None:
    previous = decision_graph_from_open_questions("- Which queue should handle email jobs?")
    question_id = next(iter(previous.nodes))
    previous = apply_answer_to_graph(previous, question_id, "worker-email")

    current = decision_graph_from_open_questions("- What queue should handle email jobs?")
    merged = merge_decision_graphs(current, previous)

    merged_node = next(
        node for node in merged.nodes.values() if "queue should handle email jobs" in node.description.lower()
    )
    assert merged_node.value == "worker-email"


def test_decision_graph_json_round_trip_preserves_evidence() -> None:
    graph = decision_graph_from_open_questions("- What database should persist the primary app data?")

    restored = decision_graph_from_json(decision_graph_to_json(graph))

    assert restored.nodes["architecture.database"].evidence == [
        "open_questions: What database should persist the primary app data?"
    ]


def test_plan_graph_extracts_resolved_catalog_decision_from_architecture_section() -> None:
    graph = decision_graph_from_plan(
        open_questions=None,
        plan_content=(
            "# Plan\n\n## Architecture\nUse PostgreSQL for the primary application database. Keep the API REST-style.\n"
        ),
    )

    assert graph.nodes["architecture.database"].value == "PostgreSQL"
    assert graph.nodes["architecture.api_protocol"].value == "REST-style"


def test_plan_graph_merges_open_questions_with_explicit_decision_blocks() -> None:
    graph = decision_graph_from_plan(
        open_questions="- Which database should store the primary application data?",
        plan_content=(
            "# Plan\n\n"
            "## Design Decision Records\n"
            "### Primary data store\n"
            "- Decision: Which database should store the primary application data?\n"
            "- Status: decided\n"
            "- Choice: PostgreSQL\n"
            "- Evidence: Existing deployments already run PostgreSQL.\n"
        ),
    )

    node = graph.nodes["architecture.database"]
    assert node.value == "PostgreSQL"
    assert any(item.startswith("open_questions:") for item in node.evidence)
    assert any(item.startswith("explicit_decision_block:") for item in node.evidence)


def test_plan_graph_keeps_explicit_open_decision_block_without_choice() -> None:
    graph = decision_graph_from_plan(
        open_questions=None,
        plan_content=(
            "# Plan\n\n"
            "## Design Decision Records\n"
            "### Rollout queue\n"
            "- Decision: Which queue should handle email jobs?\n"
            "- Status: open\n"
            "- Options: worker-email, worker-background\n"
        ),
    )

    node = next(iter(graph.nodes.values()))
    assert node.description == "Which queue should handle email jobs?"
    assert node.value is None
    assert node.options == ["worker-email", "worker-background"]


def test_plan_graph_extracts_explicit_dependency_edges() -> None:
    graph = decision_graph_from_plan(
        open_questions=None,
        plan_content=(
            "# Plan\n\n"
            "## Design Decision Records\n"
            "### Primary data store\n"
            "- Decision: Which database should store the primary application data?\n"
            "- Status: decided\n"
            "- Choice: PostgreSQL\n\n"
            "### API shape\n"
            "- Decision: What response schema should this feature use?\n"
            "- Status: decided\n"
            "- Choice: keep current format\n"
            "- Depends On: architecture.database\n"
        ),
    )

    assert graph.edges
    assert graph.edges[0].source == "architecture.response_schema"
    assert graph.edges[0].target == "architecture.database"


def test_decision_graph_json_round_trip_preserves_edges() -> None:
    graph = decision_graph_from_plan(
        open_questions=None,
        plan_content=(
            "# Plan\n\n"
            "## Design Decision Records\n"
            "### Primary data store\n"
            "- Decision: Which database should store the primary application data?\n"
            "- Status: decided\n"
            "- Choice: PostgreSQL\n\n"
            "### API shape\n"
            "- Decision: What response schema should this feature use?\n"
            "- Status: decided\n"
            "- Choice: keep current format\n"
            "- Depends On: architecture.database\n"
        ),
    )

    restored = decision_graph_from_json(decision_graph_to_json(graph))

    assert restored.edges
    assert restored.edges[0].source == "architecture.response_schema"
    assert restored.edges[0].target == "architecture.database"
