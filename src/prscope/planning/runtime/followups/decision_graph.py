from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from ...decision_catalog import DEFAULT_DECISION_CATALOG, DecisionCatalogEntry


@dataclass
class DecisionNode:
    id: str
    description: str
    options: list[str] | None
    value: str | None
    section: str
    required: bool = True
    concept: str | None = None
    evidence: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DecisionEdge:
    source: str
    target: str
    relation: str = "depends_on"


@dataclass
class DecisionGraph:
    nodes: dict[str, DecisionNode] = field(default_factory=dict)
    edges: list[DecisionEdge] = field(default_factory=list)

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


def _extract_markdown_section(plan_content: str, heading: str) -> str:
    pattern = re.compile(
        rf"^##\s+{re.escape(heading)}\b(.*?)(?=^##\s+|\Z)",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(plan_content)
    return match.group(1).strip() if match else ""


def _pick_catalog_value(entry: DecisionCatalogEntry, text: str) -> str | None:
    lowered = text.lower()
    for option in entry.options or []:
        if option.lower() in lowered:
            return option
    return None


def _graph_from_catalog_sections(plan_content: str) -> DecisionGraph:
    graph = DecisionGraph()
    sections = {
        "architecture": _extract_markdown_section(plan_content, "Architecture"),
        "design_decision_records": _extract_markdown_section(plan_content, "Design Decision Records"),
    }
    for section_name, content in sections.items():
        lowered = content.lower()
        if not lowered:
            continue
        for entry in DEFAULT_DECISION_CATALOG:
            if not any(token in lowered for token in entry.match_tokens):
                continue
            value = _pick_catalog_value(entry, content)
            node = graph.nodes.get(entry.id)
            evidence = f"{section_name}: {content[:240].strip()}"
            if node is None:
                graph.nodes[entry.id] = DecisionNode(
                    id=entry.id,
                    description=entry.description,
                    options=list(entry.options) if entry.options else None,
                    value=value,
                    section=entry.section,
                    required=entry.required,
                    concept=entry.concept,
                    evidence=[evidence],
                )
                continue
            graph.nodes[entry.id] = DecisionNode(
                id=node.id,
                description=node.description,
                options=node.options,
                value=node.value if node.value is not None else value,
                section=node.section,
                required=node.required,
                concept=node.concept,
                evidence=_combine_evidence(node.evidence, [evidence]),
            )
    return graph


def _explicit_decision_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if re.match(r"^###\s+", stripped):
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            current.append(stripped)
            continue
        if not stripped and current:
            blocks.append("\n".join(current).strip())
            current = []
            continue
        if current:
            current.append(stripped)
    if current:
        blocks.append("\n".join(current).strip())
    return blocks


def _parse_key_value_line(line: str) -> tuple[str, str] | None:
    cleaned = _strip_bullet_prefix(line)
    if ":" not in cleaned:
        return None
    key, _, value = cleaned.partition(":")
    key = key.strip().lower()
    value = value.strip()
    if not key or not value:
        return None
    return key, value


def _graph_from_explicit_decisions(plan_content: str) -> DecisionGraph:
    graph = DecisionGraph()
    pending_edges: list[tuple[str, str]] = []
    for section_name in ("Architecture", "Design Decision Records"):
        section = _extract_markdown_section(plan_content, section_name)
        if not section:
            continue
        for block in _explicit_decision_blocks(section):
            lines = [line for line in block.splitlines() if line.strip()]
            if not lines:
                continue
            heading_match = re.match(r"^###\s+(.*)$", lines[0].strip())
            heading = heading_match.group(1).strip() if heading_match else ""
            fields: dict[str, str] = {}
            for line in lines[1:] if heading else lines:
                parsed = _parse_key_value_line(line)
                if parsed is None:
                    continue
                key, value = parsed
                fields[key] = value
            description = fields.get("decision") or heading
            if not description:
                continue
            catalog_entry = _match_catalog_entry(description)
            section_value = fields.get("section") or (catalog_entry.section if catalog_entry else "architecture")
            concept = fields.get("concept") or (catalog_entry.concept if catalog_entry else None)
            options_field = fields.get("options")
            options = (
                [item.strip() for item in options_field.split(",") if item.strip()]
                if options_field
                else list(catalog_entry.options)
                if catalog_entry and catalog_entry.options
                else None
            )
            status = fields.get("status", "").lower()
            value = fields.get("choice") or fields.get("selected") or fields.get("value") or fields.get("answer")
            if status in {"open", "unresolved", "pending", "tbd"} and not value:
                value = None
            node_id = catalog_entry.id if catalog_entry else _stable_question_id(description)
            evidence = [f"explicit_decision_block: {line}" for line in lines]
            graph.nodes[node_id] = DecisionNode(
                id=node_id,
                description=description,
                options=options,
                value=value,
                section=section_value,
                required=catalog_entry.required if catalog_entry else True,
                concept=concept or node_id,
                evidence=evidence,
            )
            depends_on = fields.get("depends on") or fields.get("depends_on") or fields.get("dependencies")
            if depends_on:
                pending_edges.append((node_id, depends_on))
    for node_id, raw_dependencies in pending_edges:
        for dependency in [item.strip() for item in raw_dependencies.split(",") if item.strip()]:
            target_id = None
            if dependency in graph.nodes:
                target_id = dependency
            else:
                dependency_entry = _match_catalog_entry(dependency)
                if dependency_entry is not None:
                    target_id = dependency_entry.id
                else:
                    normalized = dependency.lower().replace(" ", "_")
                    for candidate in graph.nodes.values():
                        if normalized in {candidate.id.lower(), str(candidate.concept or "").lower()}:
                            target_id = candidate.id
                            break
            if target_id and target_id in graph.nodes and target_id != node_id:
                edge = DecisionEdge(source=node_id, target=target_id)
                if edge not in graph.edges:
                    graph.edges.append(edge)
    return graph


def _node_identity(node: DecisionNode) -> tuple[str, str] | None:
    concept = str(node.concept or "").strip().lower()
    section = str(node.section or "").strip().lower()
    if not concept or not section:
        return None
    return (section, concept)


def _normalized_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _semantic_similarity(left: str, right: str) -> float:
    left_tokens = set(_normalized_text(left).split())
    right_tokens = set(_normalized_text(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    return overlap / max(len(left_tokens), len(right_tokens))


def _combine_evidence(*groups: list[str] | None) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group or []:
            evidence = str(item).strip()
            if evidence and evidence not in seen:
                seen.add(evidence)
                merged.append(evidence)
    return merged


def _merge_node(current: DecisionNode, previous: DecisionNode) -> DecisionNode:
    current_value = str(current.value or "").strip() or None
    previous_value = str(previous.value or "").strip() or None
    return DecisionNode(
        id=current.id,
        description=current.description or previous.description,
        options=current.options or previous.options,
        value=current_value if current_value is not None else previous_value,
        section=current.section or previous.section,
        required=current.required if current.description else previous.required,
        concept=current.concept or previous.concept,
        evidence=_combine_evidence(current.evidence, previous.evidence),
    )


def _merge_edges(
    *,
    current_graph: DecisionGraph,
    previous_graph: DecisionGraph,
    merged_nodes: dict[str, DecisionNode],
) -> list[DecisionEdge]:
    merged: list[DecisionEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for edge in [*current_graph.edges, *previous_graph.edges]:
        key = (edge.source, edge.target, edge.relation)
        if key in seen:
            continue
        if edge.source not in merged_nodes or edge.target not in merged_nodes:
            continue
        seen.add(key)
        merged.append(edge)
    return merged


def _open_question_descriptions(open_questions: str | None) -> set[str]:
    if not open_questions:
        return set()
    lines = [line.strip() for line in str(open_questions).splitlines() if line.strip()]
    return {_strip_bullet_prefix(line) for line in lines if line.lower() not in {"none", "none."}}


def merge_decision_graphs(
    current_graph: DecisionGraph,
    previous_graph: DecisionGraph,
    *,
    carry_forward_unresolved: bool = False,
    open_questions_current: str | None = None,
) -> DecisionGraph:
    if not previous_graph.nodes:
        return current_graph

    merged_nodes = dict(current_graph.nodes)
    matched_previous: set[str] = set()

    previous_by_identity = {
        identity: node for node in previous_graph.nodes.values() if (identity := _node_identity(node)) is not None
    }

    for node_id, current_node in list(merged_nodes.items()):
        previous_node = previous_graph.nodes.get(node_id)
        if previous_node is None:
            identity = _node_identity(current_node)
            if identity is not None:
                previous_node = previous_by_identity.get(identity)
        if previous_node is None:
            best_match: DecisionNode | None = None
            best_score = 0.0
            for candidate_id, candidate in previous_graph.nodes.items():
                if candidate_id in matched_previous:
                    continue
                score = _semantic_similarity(current_node.description, candidate.description)
                if score > best_score:
                    best_match = candidate
                    best_score = score
            if best_match is not None and best_score >= 0.6:
                previous_node = best_match
        if previous_node is None:
            continue
        matched_previous.add(previous_node.id)
        merged_nodes[node_id] = _merge_node(current_node, previous_node)

    current_open_set = (
        _open_question_descriptions(open_questions_current) if open_questions_current is not None else None
    )
    for previous_id, previous_node in previous_graph.nodes.items():
        previous_value = str(previous_node.value or "").strip()
        if previous_id in matched_previous or (not previous_value and not carry_forward_unresolved):
            continue
        if current_open_set is not None and any(
            e.startswith("open_questions:") for e in (previous_node.evidence or [])
        ):
            if previous_node.description not in current_open_set:
                continue
        if previous_id not in merged_nodes:
            merged_nodes[previous_id] = DecisionNode(
                id=previous_node.id,
                description=previous_node.description,
                options=previous_node.options,
                value=previous_node.value,
                section=previous_node.section,
                required=previous_node.required,
                concept=previous_node.concept,
                evidence=list(previous_node.evidence),
            )

    return DecisionGraph(
        nodes=merged_nodes,
        edges=_merge_edges(
            current_graph=current_graph,
            previous_graph=previous_graph,
            merged_nodes=merged_nodes,
        ),
    )


def decision_graph_from_open_questions(open_questions: str | None) -> DecisionGraph:
    graph = DecisionGraph()
    if not open_questions:
        return graph
    for raw_line in str(open_questions).splitlines():
        line = _strip_bullet_prefix(raw_line)
        if not line:
            continue
        if line.lower() in {"none", "none."}:
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
            evidence=[f"open_questions: {line}"],
        )
    return graph


def decision_graph_from_plan(*, open_questions: str | None, plan_content: str) -> DecisionGraph:
    graph = decision_graph_from_open_questions(open_questions)
    graph = merge_decision_graphs(
        graph,
        _graph_from_catalog_sections(plan_content),
        carry_forward_unresolved=True,
    )
    graph = merge_decision_graphs(
        graph,
        _graph_from_explicit_decisions(plan_content),
        carry_forward_unresolved=True,
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
        evidence=list(current.evidence),
    )
    return DecisionGraph(nodes=next_nodes)


def decision_graph_to_json(graph: DecisionGraph) -> str:
    return json_dumps(
        {
            "nodes": {key: asdict(value) for key, value in graph.nodes.items()},
            "edges": [asdict(edge) for edge in graph.edges],
        }
    )


def decision_graph_from_json(raw: str | None) -> DecisionGraph:
    if not raw:
        return DecisionGraph()
    try:
        payload = json_loads(raw)
    except ValueError:
        return DecisionGraph()
    nodes_raw = payload.get("nodes", {})
    edges_raw = payload.get("edges", [])
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
        evidence_raw = value.get("evidence")
        evidence = [str(item).strip() for item in evidence_raw] if isinstance(evidence_raw, list) else []
        nodes[str(key)] = DecisionNode(
            id=str(value.get("id", key)),
            description=str(value.get("description", "")),
            options=options,
            value=value_str,
            section=str(value.get("section", "architecture")),
            required=bool(value.get("required", True)),
            concept=str(value.get("concept")) if value.get("concept") is not None else None,
            evidence=[item for item in evidence if item],
        )
    edges: list[DecisionEdge] = []
    if isinstance(edges_raw, list):
        for item in edges_raw:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "")).strip()
            target = str(item.get("target", "")).strip()
            relation = str(item.get("relation", "depends_on")).strip() or "depends_on"
            if not source or not target or source not in nodes or target not in nodes:
                continue
            edges.append(DecisionEdge(source=source, target=target, relation=relation))
    return DecisionGraph(nodes=nodes, edges=edges)


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
