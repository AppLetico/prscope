from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class DecisionCatalogEntry:
    id: str
    section: str
    concept: str
    options: list[str] | None = None
    required: bool = True
    match_tokens: tuple[str, ...] = ()


@dataclass
class DecisionNode:
    id: str
    description: str
    options: list[str] | None
    value: str | None
    section: str
    required: bool = True
    concept: str | None = None


@dataclass
class DecisionGraph:
    nodes: dict[str, DecisionNode] = field(default_factory=dict)

    def unresolved_nodes(self) -> list[DecisionNode]:
        return [node for node in self.nodes.values() if node.required and not str(node.value or "").strip()]


@dataclass
class FollowupQuestionArtifact:
    id: str
    question: str
    options: list[str] | None
    target_sections: list[str]
    concept: str
    resolved: bool = False


@dataclass
class FollowupSuggestionArtifact:
    id: str
    suggestion: str


@dataclass
class PlanFollowupsArtifact:
    plan_version_id: int | None
    questions: list[FollowupQuestionArtifact]
    suggestions: list[FollowupSuggestionArtifact]


DEFAULT_DECISION_CATALOG: tuple[DecisionCatalogEntry, ...] = (
    DecisionCatalogEntry(
        id="logging_strategy",
        section="architecture",
        concept="logging_strategy",
        options=["log everything", "log only when issues occur", "configurable", "skip logging"],
        required=False,
        match_tokens=("log", "logging", "response time"),
    ),
    DecisionCatalogEntry(
        id="response_schema",
        section="architecture",
        concept="response_schema",
        options=["keep current format", "add timing to responses", "bundle in a metrics structure"],
        required=False,
        match_tokens=("response format", "response structure", "schema", "client"),
    ),
    DecisionCatalogEntry(
        id="metrics_scope",
        section="architecture",
        concept="metrics_scope",
        options=["health check only", "all routes", "specific routes"],
        required=False,
        match_tokens=("metrics", "endpoint", "scope"),
    ),
    DecisionCatalogEntry(
        id="cache_strategy",
        section="architecture",
        concept="cache_backend",
        options=["shared cache", "local cache", "no caching"],
        required=False,
        match_tokens=("cache", "caching"),
    ),
    DecisionCatalogEntry(
        id="database_type",
        section="architecture",
        concept="primary_database",
        options=["PostgreSQL", "MySQL", "SQLite"],
        required=False,
        match_tokens=("database", "storage"),
    ),
    DecisionCatalogEntry(
        id="api_protocol",
        section="architecture",
        concept="api_protocol",
        options=["REST-style", "gRPC", "GraphQL"],
        required=False,
        match_tokens=("api", "protocol", "rest", "grpc", "graphql"),
    ),
)


def _slugify(text: str) -> str:
    lowered = re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
    return lowered[:48] or "decision"


def _strip_bullet_prefix(line: str) -> str:
    return re.sub(r"^\s*[-*]\s*", "", line).strip()


def _normalized_question(line: str) -> str:
    stripped = _strip_bullet_prefix(line)
    return stripped[:-1].strip() if stripped.endswith("?") else stripped


def _stable_question_id(text: str) -> str:
    normalized = _normalized_question(text)
    digest = hashlib.sha1(normalized.lower().encode("utf-8")).hexdigest()[:8]
    return f"{_slugify(normalized)}_{digest}"


def _match_catalog_entry(text: str) -> DecisionCatalogEntry | None:
    normalized = text.lower()
    best: DecisionCatalogEntry | None = None
    best_score = 0
    for entry in DEFAULT_DECISION_CATALOG:
        score = sum(1 for token in entry.match_tokens if token in normalized)
        if score > best_score:
            best = entry
            best_score = score
    return best


def decision_graph_from_open_questions(open_questions: str | None) -> DecisionGraph:
    graph = DecisionGraph()
    if not open_questions:
        return graph
    for raw_line in str(open_questions).splitlines():
        line = _strip_bullet_prefix(raw_line)
        if not line:
            continue
        catalog_entry = _match_catalog_entry(line)
        node_id = catalog_entry.id if catalog_entry else _stable_question_id(line)
        graph.nodes[node_id] = DecisionNode(
            id=node_id,
            description=line,
            options=list(catalog_entry.options) if catalog_entry and catalog_entry.options else None,
            value=None,
            section=catalog_entry.section if catalog_entry else "architecture",
            required=catalog_entry.required if catalog_entry else True,
            concept=catalog_entry.concept if catalog_entry else node_id,
        )
    return graph


def graph_to_followup_questions(graph: DecisionGraph) -> list[FollowupQuestionArtifact]:
    return [
        FollowupQuestionArtifact(
            id=node.id,
            question=node.description,
            options=node.options,
            target_sections=[node.section],
            concept=node.concept or node.id,
            resolved=bool(str(node.value or "").strip()),
        )
        for node in graph.unresolved_nodes()
    ]


def apply_answer_to_graph(graph: DecisionGraph, question_id: str, value: str) -> DecisionGraph:
    current = graph.nodes.get(question_id)
    if current is None:
        raise ValueError(f"Unknown follow-up question: {question_id}")
    next_nodes = dict(graph.nodes)
    next_nodes[question_id] = DecisionNode(
        id=current.id,
        description=current.description,
        options=current.options,
        value=value.strip(),
        section=current.section,
        required=current.required,
        concept=current.concept,
    )
    return DecisionGraph(nodes=next_nodes)


def decision_graph_to_json(graph: DecisionGraph) -> str:
    return json_dumps({"nodes": {key: asdict(value) for key, value in graph.nodes.items()}})


def decision_graph_from_json(raw: str | None) -> DecisionGraph:
    if not raw:
        return DecisionGraph()
    try:
        payload = json_loads(raw)
    except ValueError:
        return DecisionGraph()
    nodes_raw = payload.get("nodes", {})
    if not isinstance(nodes_raw, dict):
        return DecisionGraph()
    nodes: dict[str, DecisionNode] = {}
    for key, value in nodes_raw.items():
        if not isinstance(value, dict):
            continue
        opts = value.get("options")
        options = [str(item) for item in opts] if isinstance(opts, list) else None
        raw_val = value.get("value")
        value_str = str(raw_val).strip() if raw_val is not None and str(raw_val).strip() else None
        nodes[str(key)] = DecisionNode(
            id=str(value.get("id", key)),
            description=str(value.get("description", "")),
            options=options,
            value=value_str,
            section=str(value.get("section", "architecture")),
            required=bool(value.get("required", True)),
            concept=str(value.get("concept")) if value.get("concept") is not None else None,
        )
    return DecisionGraph(nodes=nodes)


def followups_to_json(followups: PlanFollowupsArtifact) -> str:
    return json_dumps(asdict(followups))


def followups_from_json(raw: str | None) -> PlanFollowupsArtifact:
    if not raw:
        return PlanFollowupsArtifact(plan_version_id=None, questions=[], suggestions=[])
    try:
        payload = json_loads(raw)
    except ValueError:
        return PlanFollowupsArtifact(plan_version_id=None, questions=[], suggestions=[])
    questions_raw = payload.get("questions", [])
    suggestions_raw = payload.get("suggestions", [])
    questions = [
        FollowupQuestionArtifact(
            id=str(item.get("id", "")),
            question=str(item.get("question", "")),
            options=[str(opt) for opt in item.get("options", [])] if isinstance(item.get("options"), list) else None,
            target_sections=[str(section) for section in item.get("target_sections", [])]
            if isinstance(item.get("target_sections"), list)
            else [],
            concept=str(item.get("concept", item.get("id", ""))),
            resolved=bool(item.get("resolved", False)),
        )
        for item in questions_raw
        if isinstance(item, dict)
    ]
    suggestions = [
        FollowupSuggestionArtifact(
            id=str(item.get("id", "")),
            suggestion=str(item.get("suggestion", "")),
        )
        for item in suggestions_raw
        if isinstance(item, dict)
    ]
    plan_version_id = payload.get("plan_version_id")
    pv_int = None
    if isinstance(plan_version_id, int):
        pv_int = plan_version_id
    elif isinstance(plan_version_id, str) and plan_version_id.isdigit():
        pv_int = int(plan_version_id)
    return PlanFollowupsArtifact(
        plan_version_id=pv_int,
        questions=questions,
        suggestions=suggestions,
    )


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":"))


def json_loads(raw: str) -> dict[str, Any]:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Expected object payload")
    return payload
