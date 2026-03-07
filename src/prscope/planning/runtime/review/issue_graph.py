"""
Graph-backed issue tracker with canonical IDs and propagation semantics.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .issue_similarity import IssueSimilarityService

logger = logging.getLogger(__name__)

IssueStatus = Literal["open", "resolved"]
IssueSeverity = Literal["major", "minor", "info"]
IssueSource = Literal["critic", "validation", "inference"]
IssueRelation = Literal["causes", "depends_on", "duplicate"]
IssueResolutionSource = Literal["review", "lightweight"]


@dataclass
class Issue:
    id: str
    description: str
    status: str
    raised_in_round: int
    resolved_in_round: int | None = None
    resolution_source: IssueResolutionSource | None = None


@dataclass
class IssueNode:
    id: str
    description: str
    status: IssueStatus
    raised_round: int
    resolved_round: int | None = None
    resolution_source: IssueResolutionSource | None = None
    severity: IssueSeverity = "major"
    source: IssueSource = "critic"
    embedding: list[float] | None = None
    tags: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class IssueEdge:
    source: str
    target: str
    relation: IssueRelation


@dataclass
class IssueGraph:
    nodes: dict[str, IssueNode] = field(default_factory=dict)
    edges: list[IssueEdge] = field(default_factory=list)
    duplicate_alias: dict[str, str] = field(default_factory=dict)


class IssueGraphTracker:
    def __init__(
        self,
        similarity: IssueSimilarityService,
        *,
        max_nodes: int = 50,
        max_edges: int = 100,
    ) -> None:
        self._similarity = similarity
        self._graph = IssueGraph()
        self._next_id = 1
        self._next_alias = 1
        self._max_nodes = max(1, int(max_nodes))
        self._max_edges = max(1, int(max_edges))
        self._children_causes: dict[str, set[str]] = defaultdict(set)
        self._parents_causes: dict[str, set[str]] = defaultdict(set)
        self._dependencies: dict[str, set[str]] = defaultdict(set)

    def canonical_issue_id(self, issue_id: str) -> str:
        canonical = str(issue_id).strip()
        if not canonical:
            return canonical
        seen: set[str] = set()
        while canonical in self._graph.duplicate_alias:
            if canonical in seen:
                break
            seen.add(canonical)
            canonical = self._graph.duplicate_alias[canonical]
        return canonical

    def canonicalize_text(self, description: str) -> str:
        return " ".join(description.strip().split())

    def is_duplicate(self, description: str) -> str | None:
        duplicate = self._similarity.find_duplicate(
            description=description,
            open_issues=[(node.id, node.description) for node in self._graph.nodes.values() if node.status == "open"],
        )
        if duplicate is None:
            return None
        return self.canonical_issue_id(duplicate)

    def add_issue(
        self,
        description: str,
        round_number: int,
        *,
        severity: IssueSeverity = "major",
        source: IssueSource = "critic",
        preferred_id: str | None = None,
    ) -> Issue:
        normalized = self.canonicalize_text(description)
        if not normalized:
            return Issue(
                id="",
                description="",
                status="resolved",
                raised_in_round=round_number,
                resolved_in_round=round_number,
            )
        duplicate = self.is_duplicate(normalized)
        if duplicate is not None and duplicate in self._graph.nodes:
            alias_id = f"issue_alias_{self._next_alias}"
            self._next_alias += 1
            self.alias_duplicate(alias_id, duplicate)
            canonical = self._graph.nodes[duplicate]
            return self._to_issue(canonical)

        issue_id = self.canonical_issue_id(preferred_id or "")
        if not issue_id:
            issue_id = self._allocate_issue_id()
        if issue_id in self._graph.nodes:
            existing = self._graph.nodes[issue_id]
            if existing.status == "resolved":
                existing.status = "open"
                existing.resolved_round = None
                existing.resolution_source = None
            return self._to_issue(existing)

        node = IssueNode(
            id=issue_id,
            description=normalized,
            status="open",
            raised_round=int(round_number),
            severity=severity,
            source=source,
        )
        self._graph.nodes[issue_id] = node
        self._enforce_graph_caps()
        return self._to_issue(node)

    def alias_duplicate(self, alias_id: str, canonical_id: str) -> str:
        alias = str(alias_id).strip()
        canonical = self.canonical_issue_id(canonical_id)
        if not alias or not canonical or alias == canonical:
            return canonical
        self._graph.duplicate_alias[alias] = canonical
        return canonical

    def add_edge(self, source: str, target: str, relation: IssueRelation) -> None:
        canonical_source = self.canonical_issue_id(source)
        canonical_target = self.canonical_issue_id(target)
        if not canonical_source or not canonical_target:
            return
        if canonical_source == canonical_target:
            return
        if canonical_source not in self._graph.nodes or canonical_target not in self._graph.nodes:
            return
        edge = IssueEdge(source=canonical_source, target=canonical_target, relation=relation)
        if edge in self._graph.edges:
            return
        self._graph.edges.append(edge)
        self._index_edge(edge)
        self._enforce_graph_caps()

    def resolve_issue(
        self,
        issue_id: str,
        round_number: int,
        *,
        propagate_causes: bool = True,
        resolution_source: IssueResolutionSource = "review",
    ) -> None:
        root_id = self.canonical_issue_id(issue_id)
        if root_id not in self._graph.nodes:
            return
        if self._has_unresolved_dependencies(root_id):
            return
        if not propagate_causes:
            node = self._graph.nodes.get(root_id)
            if node is None or node.status == "resolved":
                return
            node.status = "resolved"
            node.resolved_round = int(round_number)
            node.resolution_source = resolution_source
            return
        queue: deque[str] = deque([root_id])
        while queue:
            current_id = queue.popleft()
            node = self._graph.nodes.get(current_id)
            if node is None:
                continue
            if self._has_unresolved_dependencies(current_id):
                continue
            if node.status != "resolved":
                node.status = "resolved"
                node.resolved_round = int(round_number)
                node.resolution_source = resolution_source
            for child_id in sorted(self._children_causes.get(current_id, set())):
                child = self._graph.nodes.get(child_id)
                if child is None or child.status == "resolved":
                    continue
                parent_ids = self._parents_causes.get(child_id, set())
                if all(
                    (self._graph.nodes.get(parent_id) is not None and self._graph.nodes[parent_id].status == "resolved")
                    for parent_id in parent_ids
                ):
                    queue.append(child_id)

    def open_nodes(self) -> list[IssueNode]:
        return [node for node in self._graph.nodes.values() if node.status == "open"]

    def open_issues(self) -> list[Issue]:
        return [self._to_issue(node) for node in self._sorted_nodes(self.open_nodes())]

    def open_issue_dicts(self) -> list[dict[str, Any]]:
        return [
            {
                "id": issue.id,
                "description": issue.description,
                "status": issue.status,
                "raised_in_round": issue.raised_in_round,
                "resolved_in_round": issue.resolved_in_round,
            }
            for issue in self.open_issues()
        ]

    def root_open_nodes(self) -> list[IssueNode]:
        roots: list[IssueNode] = []
        for node in self.open_nodes():
            if len(self._parents_causes.get(node.id, set())) == 0:
                roots.append(node)
        return self._sorted_nodes(roots)

    def root_open_issues(self) -> list[Issue]:
        return [self._to_issue(node) for node in self.root_open_nodes()]

    def unresolved_dependency_chains(self) -> int:
        count = 0
        for node in self.open_nodes():
            dependencies = self._dependencies.get(node.id, set())
            for dep_id in dependencies:
                dep = self._graph.nodes.get(dep_id)
                if dep is None or dep.status != "resolved":
                    count += 1
        return count

    def descendants(self, issue_id: str) -> list[Issue]:
        root = self.canonical_issue_id(issue_id)
        if root not in self._graph.nodes:
            return []
        visited: set[str] = set()
        queue: deque[str] = deque(sorted(self._children_causes.get(root, set())))
        results: list[Issue] = []
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            node = self._graph.nodes.get(current)
            if node is None:
                continue
            results.append(self._to_issue(node))
            for nxt in sorted(self._children_causes.get(current, set())):
                if nxt not in visited:
                    queue.append(nxt)
        return results

    def parents(self, issue_id: str) -> list[Issue]:
        canonical = self.canonical_issue_id(issue_id)
        parent_ids = sorted(self._parents_causes.get(canonical, set()))
        return [
            self._to_issue(self._graph.nodes[parent_id]) for parent_id in parent_ids if parent_id in self._graph.nodes
        ]

    def dependencies_for(self, issue_id: str) -> list[Issue]:
        canonical = self.canonical_issue_id(issue_id)
        dependency_ids = sorted(self._dependencies.get(canonical, set()))
        return [self._to_issue(self._graph.nodes[dep_id]) for dep_id in dependency_ids if dep_id in self._graph.nodes]

    def distilled_context(self) -> str:
        open_issues = self.open_issues()
        if not open_issues:
            return "(none)"
        return "\n".join(f"- {issue.id}: {issue.description}" for issue in open_issues)

    def graph_snapshot(self) -> dict[str, Any]:
        nodes = [
            {
                "id": node.id,
                "description": node.description,
                "status": node.status,
                "raised_round": node.raised_round,
                "resolved_round": node.resolved_round,
                "resolution_source": node.resolution_source,
                "severity": node.severity,
                "source": node.source,
                "embedding": node.embedding,
                "tags": sorted(node.tags),
            }
            for node in self._sorted_nodes(self._graph.nodes.values())
        ]
        edges = [
            {"source": edge.source, "target": edge.target, "relation": edge.relation}
            for edge in sorted(self._graph.edges, key=lambda edge: (edge.source, edge.target, edge.relation))
        ]
        open_nodes = self.open_nodes()
        root_open_nodes = self.root_open_nodes()
        summary = {
            "open_total": len(open_nodes),
            "root_open": len(root_open_nodes),
            "resolved_total": len([node for node in self._graph.nodes.values() if node.status == "resolved"]),
            "open_major": len([node for node in open_nodes if node.severity == "major"]),
            "open_minor": len([node for node in open_nodes if node.severity == "minor"]),
            "open_info": len([node for node in open_nodes if node.severity == "info"]),
            "unresolved_dependency_chains": self.unresolved_dependency_chains(),
        }
        return {
            "nodes": nodes,
            "edges": edges,
            "duplicate_alias": dict(sorted(self._graph.duplicate_alias.items())),
            "summary": summary,
        }

    def load_snapshot(self, payload: dict[str, Any]) -> None:
        self._graph = IssueGraph()
        self._children_causes = defaultdict(set)
        self._parents_causes = defaultdict(set)
        self._dependencies = defaultdict(set)
        nodes = payload.get("nodes", [])
        edges = payload.get("edges", [])
        duplicate_alias = payload.get("duplicate_alias", {})
        for raw in nodes:
            if not isinstance(raw, dict):
                continue
            issue_id = str(raw.get("id", "")).strip()
            if not issue_id:
                continue
            status_value = str(raw.get("status", "open")).strip().lower()
            status: IssueStatus = "resolved" if status_value == "resolved" else "open"
            severity_value = str(raw.get("severity", "major")).strip().lower()
            severity: IssueSeverity = (
                "minor" if severity_value == "minor" else "info" if severity_value == "info" else "major"
            )
            source_value = str(raw.get("source", "critic")).strip().lower()
            source: IssueSource = (
                "validation"
                if source_value == "validation"
                else "inference"
                if source_value == "inference"
                else "critic"
            )
            self._graph.nodes[issue_id] = IssueNode(
                id=issue_id,
                description=self.canonicalize_text(str(raw.get("description", ""))),
                status=status,
                raised_round=int(raw.get("raised_round", 0) or 0),
                resolved_round=(int(raw["resolved_round"]) if raw.get("resolved_round") is not None else None),
                resolution_source=(
                    "lightweight"
                    if str(raw.get("resolution_source", "")).strip().lower() == "lightweight"
                    else "review"
                    if str(raw.get("resolution_source", "")).strip().lower() == "review"
                    else None
                ),
                severity=severity,
                source=source,
                embedding=raw.get("embedding") if isinstance(raw.get("embedding"), list) else None,
                tags={str(tag) for tag in raw.get("tags", []) if str(tag).strip()},
            )
        for alias, canonical in duplicate_alias.items():
            alias_id = str(alias).strip()
            canonical_id = str(canonical).strip()
            if alias_id and canonical_id:
                self._graph.duplicate_alias[alias_id] = canonical_id
        for raw in edges:
            if not isinstance(raw, dict):
                continue
            relation = str(raw.get("relation", "")).strip().lower()
            if relation not in {"causes", "depends_on", "duplicate"}:
                continue
            source = self.canonical_issue_id(str(raw.get("source", "")))
            target = self.canonical_issue_id(str(raw.get("target", "")))
            if source in self._graph.nodes and target in self._graph.nodes and source != target:
                edge = IssueEdge(source=source, target=target, relation=relation)  # type: ignore[arg-type]
                if edge not in self._graph.edges:
                    self._graph.edges.append(edge)
                    self._index_edge(edge)
        self._next_id = self._infer_next_id()
        self._next_alias = self._infer_next_alias()

    def _to_issue(self, node: IssueNode) -> Issue:
        return Issue(
            id=node.id,
            description=node.description,
            status=node.status,
            raised_in_round=node.raised_round,
            resolved_in_round=node.resolved_round,
            resolution_source=node.resolution_source,
        )

    @staticmethod
    def _sorted_nodes(nodes: Any) -> list[IssueNode]:
        def sort_key(node: IssueNode) -> tuple[int, str]:
            try:
                number = int(str(node.id).split("_")[-1])
            except ValueError:
                number = 0
            return (number, node.id)

        return sorted(list(nodes), key=sort_key)

    def _allocate_issue_id(self) -> str:
        issue_id = f"issue_{self._next_id}"
        self._next_id += 1
        return issue_id

    def _infer_next_id(self) -> int:
        max_seen = 0
        for issue_id in self._graph.nodes:
            if not issue_id.startswith("issue_"):
                continue
            try:
                max_seen = max(max_seen, int(issue_id.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
        return max_seen + 1

    def _infer_next_alias(self) -> int:
        max_seen = 0
        for alias_id in self._graph.duplicate_alias:
            if not alias_id.startswith("issue_alias_"):
                continue
            try:
                max_seen = max(max_seen, int(alias_id.split("_", 2)[2]))
            except (IndexError, ValueError):
                continue
        return max_seen + 1

    def _index_edge(self, edge: IssueEdge) -> None:
        source = self.canonical_issue_id(edge.source)
        target = self.canonical_issue_id(edge.target)
        if edge.relation == "causes":
            self._children_causes[source].add(target)
            self._parents_causes[target].add(source)
        elif edge.relation == "depends_on":
            self._dependencies[source].add(target)

    def _has_unresolved_dependencies(self, issue_id: str) -> bool:
        canonical = self.canonical_issue_id(issue_id)
        for dep_id in self._dependencies.get(canonical, set()):
            dep = self._graph.nodes.get(dep_id)
            if dep is None or dep.status != "resolved":
                return True
        return False

    def _enforce_graph_caps(self) -> None:
        if len(self._graph.nodes) <= self._max_nodes and len(self._graph.edges) <= self._max_edges:
            return
        if len(self._graph.nodes) > self._max_nodes:
            removable = [
                node
                for node in self._graph.nodes.values()
                if node.status == "open"
                and node.severity in {"minor", "info"}
                and node.source in {"inference", "validation"}
                and len(self._parents_causes.get(node.id, set())) > 0
            ]
            removable = self._sorted_nodes(removable)
            while len(self._graph.nodes) > self._max_nodes and removable:
                doomed = removable.pop(0)
                self._remove_node(doomed.id)
            if len(self._graph.nodes) > self._max_nodes:
                logger.warning(
                    "Issue graph node cap reached; retaining %s nodes without further pruning",
                    len(self._graph.nodes),
                )
        if len(self._graph.edges) > self._max_edges:
            self._graph.edges = sorted(
                self._graph.edges,
                key=lambda edge: (edge.relation != "causes", edge.source, edge.target),
            )[: self._max_edges]
            self._rebuild_indexes()

    def _remove_node(self, issue_id: str) -> None:
        canonical_id = self.canonical_issue_id(issue_id)
        if canonical_id not in self._graph.nodes:
            return
        del self._graph.nodes[canonical_id]
        self._graph.edges = [
            edge for edge in self._graph.edges if edge.source != canonical_id and edge.target != canonical_id
        ]
        aliases_to_drop = [alias for alias, target in self._graph.duplicate_alias.items() if target == canonical_id]
        for alias in aliases_to_drop:
            self._graph.duplicate_alias.pop(alias, None)
        self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        self._children_causes = defaultdict(set)
        self._parents_causes = defaultdict(set)
        self._dependencies = defaultdict(set)
        for edge in self._graph.edges:
            self._index_edge(edge)
