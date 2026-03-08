from __future__ import annotations

from collections import defaultdict
from typing import Any

from .issue_types import infer_issue_type

_CLUSTER_SEVERITY_WEIGHT = {"major": 5, "minor": 2, "info": 1}
_DECISION_PRESSURE_WEIGHT = {"major": 3, "minor": 1, "info": 0}
_MAX_ROOT_DEPTH = 10
_RECONSIDERATION_THRESHOLD = 6


def build_impact_view(
    *,
    decision_graph: dict[str, Any] | None,
    issue_graph: dict[str, Any] | None,
    previous_decision_graph: dict[str, Any] | None = None,
    reconsideration_threshold: int = _RECONSIDERATION_THRESHOLD,
) -> dict[str, Any]:
    decisions_payload = (
        decision_graph.get("nodes", {}) if isinstance(decision_graph, dict) else {}
    )
    issue_nodes_raw = issue_graph.get("nodes", []) if isinstance(issue_graph, dict) else []
    issue_edges_raw = issue_graph.get("edges", []) if isinstance(issue_graph, dict) else []
    duplicate_alias_raw = issue_graph.get("duplicate_alias", {}) if isinstance(issue_graph, dict) else {}
    if not isinstance(decisions_payload, dict) or not isinstance(issue_nodes_raw, list):
        return {"decisions": [], "reconsideration_candidates": []}

    decisions = {str(key): value for key, value in decisions_payload.items() if isinstance(value, dict)}
    issue_nodes = {str(item.get("id", "")).strip(): item for item in issue_nodes_raw if isinstance(item, dict)}
    duplicate_alias = {
        str(alias).strip(): str(target).strip()
        for alias, target in duplicate_alias_raw.items()
        if str(alias).strip() and str(target).strip()
    }
    parents, dependencies = _build_indexes(issue_edges_raw, issue_nodes, duplicate_alias)
    blocked_by_dependencies = {
        issue_id: _is_blocked(issue_id, issue_nodes, dependencies) for issue_id in issue_nodes
    }
    root_by_issue = {
        issue_id: _canonical_root_for_issue(
            issue_id,
            issue_nodes=issue_nodes,
            parents=parents,
            blocked_by_dependencies=blocked_by_dependencies,
        )
        for issue_id in issue_nodes
    }
    decision_to_linked = _decision_to_linked_issue_ids(issue_nodes, decisions.keys())

    decision_entries: list[dict[str, Any]] = []
    reconsideration_candidates: list[dict[str, Any]] = []
    previous_nodes = (
        previous_decision_graph.get("nodes", {})
        if isinstance(previous_decision_graph, dict) and isinstance(previous_decision_graph.get("nodes"), dict)
        else {}
    )

    for decision_id in sorted(decisions):
        linked_issue_ids = [
            issue_id
            for issue_id in decision_to_linked.get(decision_id, [])
            if _is_contributing_issue(issue_id, issue_nodes, blocked_by_dependencies)
        ]
        if not linked_issue_ids:
            continue
        clusters = _build_clusters(
            decision_id=decision_id,
            decision_node=decisions[decision_id],
            linked_issue_ids=linked_issue_ids,
            issue_nodes=issue_nodes,
            root_by_issue=root_by_issue,
            blocked_by_dependencies=blocked_by_dependencies,
        )
        if not clusters:
            continue
        dominant_cluster = max(
            clusters,
            key=lambda cluster: (cluster["cluster_pressure"], -_issue_id_sort_key(cluster["root_issue_id"]), cluster["root_issue_id"]),
        )
        pressure_breakdown = {
            "major": sum(1 for issue_id in linked_issue_ids if _severity_of(issue_nodes[issue_id]) == "major"),
            "minor": sum(1 for issue_id in linked_issue_ids if _severity_of(issue_nodes[issue_id]) == "minor"),
            "info": sum(1 for issue_id in linked_issue_ids if _severity_of(issue_nodes[issue_id]) == "info"),
            "clusters": len(clusters),
        }
        decision_pressure = (
            pressure_breakdown["major"] * _DECISION_PRESSURE_WEIGHT["major"]
            + pressure_breakdown["minor"] * _DECISION_PRESSURE_WEIGHT["minor"]
        )
        risk_level = _risk_level(decision_pressure)
        previous_node = previous_nodes.get(decision_id) if isinstance(previous_nodes.get(decision_id), dict) else None
        recently_changed = _decision_recently_changed(decisions[decision_id], previous_node)
        entry = {
            "decision_id": decision_id,
            "linked_issue_ids": linked_issue_ids,
            "decision_pressure": decision_pressure,
            "pressure_breakdown": pressure_breakdown,
            "risk_level": risk_level,
            "highest_severity": dominant_cluster["severity"],
            "dominant_cluster": _public_cluster_payload(dominant_cluster),
            "issue_clusters": [_public_cluster_payload(cluster) for cluster in clusters],
        }
        decision_entries.append(entry)

        if (
            decision_pressure >= reconsideration_threshold
            and dominant_cluster["severity"] == "major"
            and not recently_changed
            and str(decisions[decision_id].get("value", "")).strip()
        ):
            reconsideration_candidates.append(
                {
                    "decision_id": decision_id,
                    "reason": "high_pressure_cluster",
                    "decision_pressure": decision_pressure,
                    "dominant_cluster": _public_cluster_payload(dominant_cluster),
                    "suggested_action": dominant_cluster["suggested_action"],
                    "recently_changed": recently_changed,
                    "eligible": True,
                }
            )

    return {
        "decisions": decision_entries,
        "reconsideration_candidates": reconsideration_candidates,
    }


def _build_clusters(
    *,
    decision_id: str,
    decision_node: dict[str, Any],
    linked_issue_ids: list[str],
    issue_nodes: dict[str, dict[str, Any]],
    root_by_issue: dict[str, str],
    blocked_by_dependencies: dict[str, bool],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for issue_id in linked_issue_ids:
        grouped[root_by_issue.get(issue_id, issue_id)].append(issue_id)

    clusters: list[dict[str, Any]] = []
    for root_issue_id, issue_ids in grouped.items():
        root_issue = issue_nodes.get(root_issue_id)
        if root_issue is None:
            continue
        sorted_issue_ids = sorted(set(issue_ids), key=_issue_id_sort_key)
        severities = [_severity_of(issue_nodes[issue_id]) for issue_id in sorted_issue_ids if issue_id in issue_nodes]
        highest_severity = max(severities, key=_severity_rank) if severities else "info"
        issue_types = [
            _issue_type_of(issue_nodes[issue_id]) for issue_id in sorted_issue_ids if issue_id in issue_nodes
        ]
        affected_sections = _affected_plan_sections(
            decision_node=decision_node,
            cluster_issue_ids=sorted_issue_ids,
            issue_nodes=issue_nodes,
        )
        cluster = {
            "root_issue_id": root_issue_id,
            "root_issue": str(root_issue.get("description", "")).strip(),
            "severity": highest_severity,
            "issue_ids": sorted_issue_ids,
            "symptom_issue_count": len(sorted_issue_ids),
            "affected_plan_sections": affected_sections,
            "suggested_action": _suggested_action(
                root_issue=root_issue,
                issue_ids=sorted_issue_ids,
                issue_nodes=issue_nodes,
                blocked_by_dependencies=blocked_by_dependencies,
                issue_types=issue_types,
            ),
            "cluster_pressure": _CLUSTER_SEVERITY_WEIGHT.get(highest_severity, 1) + len(sorted_issue_ids),
        }
        clusters.append(cluster)
    return sorted(
        clusters,
        key=lambda cluster: (
            -_severity_rank(cluster["severity"]),
            -int(cluster["cluster_pressure"]),
            cluster["root_issue_id"],
        ),
    )


def _public_cluster_payload(cluster: dict[str, Any]) -> dict[str, Any]:
    return {
        "root_issue_id": cluster["root_issue_id"],
        "root_issue": cluster["root_issue"],
        "severity": cluster["severity"],
        "issue_ids": cluster["issue_ids"],
        "symptom_issue_count": cluster["symptom_issue_count"],
        "affected_plan_sections": cluster["affected_plan_sections"],
        "suggested_action": cluster["suggested_action"],
    }


def _build_indexes(
    edges_raw: list[Any],
    issue_nodes: dict[str, dict[str, Any]],
    duplicate_alias: dict[str, str],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    parents: dict[str, set[str]] = defaultdict(set)
    dependencies: dict[str, set[str]] = defaultdict(set)
    for raw in edges_raw:
        if not isinstance(raw, dict):
            continue
        relation = str(raw.get("relation", "")).strip().lower()
        source = _canonical_issue_id(str(raw.get("source", "")), duplicate_alias)
        target = _canonical_issue_id(str(raw.get("target", "")), duplicate_alias)
        if source not in issue_nodes or target not in issue_nodes or source == target:
            continue
        if relation == "causes":
            parents[target].add(source)
        elif relation == "depends_on":
            dependencies[source].add(target)
    return parents, dependencies


def _canonical_root_for_issue(
    issue_id: str,
    *,
    issue_nodes: dict[str, dict[str, Any]],
    parents: dict[str, set[str]],
    blocked_by_dependencies: dict[str, bool],
) -> str:
    current_id = str(issue_id).strip()
    if current_id not in issue_nodes:
        return current_id
    roots = _collect_roots(
        current_id,
        issue_nodes=issue_nodes,
        parents=parents,
        depth=0,
        visited=set(),
    )
    if not roots:
        return current_id
    return sorted(
        roots,
        key=lambda candidate: (
            -_severity_rank(_severity_of(issue_nodes[candidate])),
            int(issue_nodes[candidate].get("raised_round", 0) or 0),
            _issue_id_sort_key(candidate),
            candidate,
        ),
    )[0]


def _collect_roots(
    issue_id: str,
    *,
    issue_nodes: dict[str, dict[str, Any]],
    parents: dict[str, set[str]],
    depth: int,
    visited: set[str],
) -> set[str]:
    if issue_id in visited or depth >= _MAX_ROOT_DEPTH:
        return {issue_id}
    visited.add(issue_id)
    parent_ids = [parent_id for parent_id in parents.get(issue_id, set()) if parent_id in issue_nodes]
    if not parent_ids:
        return {issue_id}
    roots: set[str] = set()
    for parent_id in parent_ids:
        roots.update(
            _collect_roots(
                parent_id,
                issue_nodes=issue_nodes,
                parents=parents,
                depth=depth + 1,
                visited=set(visited),
            )
        )
    return roots or {issue_id}


def _decision_to_linked_issue_ids(
    issue_nodes: dict[str, dict[str, Any]],
    decision_ids: Any,
) -> dict[str, list[str]]:
    decision_set = {str(item) for item in decision_ids}
    linked: dict[str, list[str]] = defaultdict(list)
    for issue_id, node in issue_nodes.items():
        related_ids = node.get("related_decision_ids")
        if not isinstance(related_ids, list):
            continue
        for decision_id in related_ids:
            normalized = str(decision_id).strip()
            if normalized and normalized in decision_set:
                linked[normalized].append(issue_id)
    return {key: sorted(set(value), key=_issue_id_sort_key) for key, value in linked.items()}


def _affected_plan_sections(
    *,
    decision_node: dict[str, Any],
    cluster_issue_ids: list[str],
    issue_nodes: dict[str, dict[str, Any]],
) -> list[str]:
    sections = {str(decision_node.get("section", "architecture")).strip() or "architecture"}
    for issue_id in cluster_issue_ids:
        node = issue_nodes.get(issue_id)
        if node is None:
            continue
        for tag in node.get("tags", []):
            text = str(tag).strip()
            if text.startswith("section:"):
                section = text.partition(":")[2].strip()
                if section:
                    sections.add(section)
    return sorted(sections)


def _suggested_action(
    *,
    root_issue: dict[str, Any],
    issue_ids: list[str],
    issue_nodes: dict[str, dict[str, Any]],
    blocked_by_dependencies: dict[str, bool],
    issue_types: list[str],
) -> str:
    if any(blocked_by_dependencies.get(issue_id, False) for issue_id in issue_ids):
        return "resolve upstream issue"
    tags = {str(tag).strip().lower() for issue_id in issue_ids for tag in issue_nodes.get(issue_id, {}).get("tags", [])}
    if "decision:missing" in tags:
        return "revisit decision"
    if "decision:conflict" in tags:
        return "reconsider architecture"
    if issue_types.count("ambiguity") >= max(1, len(issue_types) // 2):
        return "clarify decision"
    if issue_types.count("performance") >= max(1, len(issue_types) // 2):
        return "evaluate alternatives"
    if issue_types.count("architecture") >= max(1, len(issue_types) // 2):
        return "reconsider architecture"
    return "revisit decision"


def _is_contributing_issue(
    issue_id: str,
    issue_nodes: dict[str, dict[str, Any]],
    blocked_by_dependencies: dict[str, bool],
) -> bool:
    node = issue_nodes.get(issue_id)
    if node is None:
        return False
    return str(node.get("status", "open")).strip().lower() == "open" and not blocked_by_dependencies.get(issue_id, False)


def _is_blocked(issue_id: str, issue_nodes: dict[str, dict[str, Any]], dependencies: dict[str, set[str]]) -> bool:
    for dep_id in dependencies.get(issue_id, set()):
        dep = issue_nodes.get(dep_id)
        if dep is None or str(dep.get("status", "open")).strip().lower() != "resolved":
            return True
    return False


def _canonical_issue_id(issue_id: str, duplicate_alias: dict[str, str]) -> str:
    current = str(issue_id).strip()
    visited: set[str] = set()
    while current and current in duplicate_alias and current not in visited:
        visited.add(current)
        current = duplicate_alias[current]
    return current


def _severity_of(node: dict[str, Any]) -> str:
    raw = str(node.get("severity", "major")).strip().lower()
    if raw in {"major", "minor", "info"}:
        return raw
    return "major"


def _issue_type_of(node: dict[str, Any]) -> str:
    raw = str(node.get("issue_type", "")).strip().lower()
    if raw in {"architecture", "ambiguity", "correctness", "performance"}:
        return raw
    relation = ""
    for tag in node.get("tags", []):
        text = str(tag).strip().lower()
        if text.startswith("decision:"):
            relation = text.partition(":")[2]
            break
    return infer_issue_type(
        str(node.get("description", "")),
        decision_relation=relation,
    )


def _decision_recently_changed(current_node: dict[str, Any], previous_node: dict[str, Any] | None) -> bool:
    current_value = str(current_node.get("value", "")).strip()
    if not current_value:
        return False
    previous_value = str(previous_node.get("value", "")).strip() if isinstance(previous_node, dict) else ""
    return current_value != previous_value


def _risk_level(pressure: int) -> str:
    if pressure >= 6:
        return "high"
    if pressure >= 3:
        return "medium"
    return "low"


def _severity_rank(severity: str) -> int:
    return _CLUSTER_SEVERITY_WEIGHT.get(str(severity).strip().lower(), 0)


def _issue_id_sort_key(issue_id: str) -> int:
    try:
        return int(str(issue_id).split("_")[-1])
    except ValueError:
        return 0
