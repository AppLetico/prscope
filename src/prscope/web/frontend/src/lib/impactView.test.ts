import { describe, expect, it } from "vitest";
import { getPressuredDecisions, getRelatedDecisionSummaries, getTopPressureSummary } from "./impactView";
import type { ArchitectureImpactView, DecisionGraph, IssueGraphNode } from "../types";

const decisionGraph: DecisionGraph = {
  nodes: {
    "architecture.database": {
      id: "architecture.database",
      description: "Which database should store the primary application data?",
      section: "architecture",
    },
    "architecture.cache_strategy": {
      id: "architecture.cache_strategy",
      description: "What caching strategy should this feature use?",
      section: "architecture",
    },
  },
};

const impactView: ArchitectureImpactView = {
  decisions: [
    {
      decision_id: "architecture.database",
      linked_issue_ids: ["issue_1", "issue_2"],
      decision_pressure: 6,
      pressure_breakdown: { major: 2, minor: 0, info: 0, clusters: 1 },
      risk_level: "high",
      highest_severity: "major",
      dominant_cluster: {
        root_issue_id: "issue_1",
        root_issue: "Database scaling limits remain unresolved.",
        severity: "major",
        issue_ids: ["issue_1", "issue_2"],
        symptom_issue_count: 2,
        affected_plan_sections: ["architecture"],
        suggested_action: "reconsider architecture",
      },
      issue_clusters: [],
    },
    {
      decision_id: "architecture.cache_strategy",
      linked_issue_ids: ["issue_3"],
      decision_pressure: 1,
      pressure_breakdown: { major: 0, minor: 1, info: 0, clusters: 1 },
      risk_level: "low",
      highest_severity: "minor",
      dominant_cluster: {
        root_issue_id: "issue_3",
        root_issue: "Caching strategy remains underspecified.",
        severity: "minor",
        issue_ids: ["issue_3"],
        symptom_issue_count: 1,
        affected_plan_sections: ["architecture"],
        suggested_action: "revisit decision",
      },
      issue_clusters: [],
    },
  ],
  reconsideration_candidates: [],
};

describe("impactView helpers", () => {
  it("sorts pressured decisions by pressure and id", () => {
    expect(getPressuredDecisions(impactView).map((item) => item.decision_id)).toEqual([
      "architecture.database",
      "architecture.cache_strategy",
    ]);
  });

  it("returns the top pressure summary with decision label and dominant cluster", () => {
    expect(getTopPressureSummary(impactView, decisionGraph)).toEqual({
      decisionId: "architecture.database",
      label: "Which database should store the primary application data?",
      riskLevel: "high",
      pressure: 6,
      dominantCluster: {
        rootIssue: "Database scaling limits remain unresolved.",
        severity: "major",
        suggestedAction: "reconsider architecture",
      },
    });
  });

  it("maps issue related decisions to labeled summaries ordered by pressure", () => {
    const issue: Pick<IssueGraphNode, "id" | "related_decision_ids"> = {
      id: "issue_2",
      related_decision_ids: ["architecture.cache_strategy", "architecture.database"],
    };

    expect(getRelatedDecisionSummaries(issue, impactView, decisionGraph)).toEqual([
      {
        decisionId: "architecture.database",
        label: "Which database should store the primary application data?",
        riskLevel: "high",
        pressure: 6,
        dominantCluster: {
          rootIssue: "Database scaling limits remain unresolved.",
          severity: "major",
          suggestedAction: "reconsider architecture",
        },
      },
      {
        decisionId: "architecture.cache_strategy",
        label: "What caching strategy should this feature use?",
        riskLevel: "low",
        pressure: 1,
        dominantCluster: {
          rootIssue: "Caching strategy remains underspecified.",
          severity: "minor",
          suggestedAction: "revisit decision",
        },
      },
    ]);
  });
});
