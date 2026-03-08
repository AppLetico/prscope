import type { ArchitectureImpactView, DecisionGraph, ImpactDecision, IssueGraphNode } from "../types";

export interface RelatedDecisionSummary {
  decisionId: string;
  label: string;
  riskLevel: "low" | "medium" | "high";
  pressure: number;
  dominantCluster?: {
    rootIssue: string;
    severity: "major" | "minor" | "info";
    suggestedAction: string;
  };
}

export function getPressuredDecisions(impactView?: ArchitectureImpactView | null): ImpactDecision[] {
  if (!impactView?.decisions?.length) return [];
  return [...impactView.decisions].sort((left, right) => {
    if (right.decision_pressure !== left.decision_pressure) {
      return right.decision_pressure - left.decision_pressure;
    }
    if (left.decision_id !== right.decision_id) {
      return left.decision_id.localeCompare(right.decision_id);
    }
    return 0;
  });
}

export function getTopPressureSummary(
  impactView?: ArchitectureImpactView | null,
  decisionGraph?: DecisionGraph | null,
): RelatedDecisionSummary | null {
  const [top] = getPressuredDecisions(impactView);
  if (!top) return null;
  return {
    decisionId: top.decision_id,
    label: decisionLabel(top.decision_id, decisionGraph),
    riskLevel: top.risk_level,
    pressure: top.decision_pressure,
    dominantCluster: top.dominant_cluster
      ? {
          rootIssue: top.dominant_cluster.root_issue,
          severity: top.dominant_cluster.severity,
          suggestedAction: top.dominant_cluster.suggested_action,
        }
      : undefined,
  };
}

export function getRelatedDecisionSummaries(
  issue: Pick<IssueGraphNode, "id" | "related_decision_ids">,
  impactView?: ArchitectureImpactView | null,
  decisionGraph?: DecisionGraph | null,
): RelatedDecisionSummary[] {
  const relatedIds = issue.related_decision_ids ?? [];
  if (!relatedIds.length) return [];
  const byDecisionId = new Map((impactView?.decisions ?? []).map((entry) => [entry.decision_id, entry]));
  return relatedIds
    .map((decisionId) => {
      const impact = byDecisionId.get(decisionId);
      return {
        decisionId,
        label: decisionLabel(decisionId, decisionGraph),
        riskLevel: impact?.risk_level ?? "low",
        pressure: impact?.decision_pressure ?? 0,
        dominantCluster: impact?.dominant_cluster
          ? {
              rootIssue: impact.dominant_cluster.root_issue,
              severity: impact.dominant_cluster.severity,
              suggestedAction: impact.dominant_cluster.suggested_action,
            }
          : undefined,
      };
    })
    .sort((left, right) => {
      if (right.pressure !== left.pressure) return right.pressure - left.pressure;
      return left.decisionId.localeCompare(right.decisionId);
    });
}

function decisionLabel(decisionId: string, decisionGraph?: DecisionGraph | null): string {
  const node = decisionGraph?.nodes?.[decisionId];
  return node?.description?.trim() || decisionId;
}
