import { describe, expect, it } from "vitest";
import { augmentPlanMarkdownWithDecisionGraph } from "./decisionGraphRender";
import type { DecisionGraph } from "../types";

describe("augmentPlanMarkdownWithDecisionGraph", () => {
  it("appends decision state to the architecture section", () => {
    const graph: DecisionGraph = {
      nodes: {
        "architecture.database": {
          id: "architecture.database",
          description: "Which database should store the primary application data?",
          value: "PostgreSQL",
          section: "architecture",
          concept: "primary_database",
        },
      },
    };

    const rendered = augmentPlanMarkdownWithDecisionGraph(
      "# Plan\n\n## Architecture\nUse a service layer.\n",
      graph,
    );

    expect(rendered).toContain("## Architecture");
    expect(rendered).toContain("### Decision State");
    expect(rendered).toContain("Which database should store the primary application data?: PostgreSQL");
  });

  it("adds open questions when the stored markdown omits them", () => {
    const graph: DecisionGraph = {
      nodes: {
        "question_1": {
          id: "question_1",
          description: "Which environment should rollout target first?",
          section: "architecture",
          required: true,
        },
      },
    };

    const rendered = augmentPlanMarkdownWithDecisionGraph("# Plan\n\n## Summary\nReady.\n", graph);

    expect(rendered).toContain("## Open Questions");
    expect(rendered).toContain("- Which environment should rollout target first?");
  });

  it("does not duplicate open questions already present in the markdown", () => {
    const graph: DecisionGraph = {
      nodes: {
        "question_1": {
          id: "question_1",
          description: "Which environment should rollout target first?",
          section: "architecture",
          required: true,
        },
      },
    };

    const rendered = augmentPlanMarkdownWithDecisionGraph(
      "# Plan\n\n## Open Questions\n- Which environment should rollout target first?\n",
      graph,
    );

    const matches = rendered.match(/Which environment should rollout target first\?/g) ?? [];
    expect(matches).toHaveLength(1);
  });
});
