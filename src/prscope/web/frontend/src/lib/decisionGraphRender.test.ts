import { describe, expect, it } from "vitest";
import {
  augmentPlanMarkdownWithDecisionGraph,
  decisionGraphToMermaid,
  hasMermaidInArchitectureSection,
  sanitizeMermaidNodeId,
} from "./decisionGraphRender";
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

  it("does not inject Decision map (auto) mermaid into the plan draft view", () => {
    const graph: DecisionGraph = {
      nodes: {
        a: {
          id: "a",
          description: "Service A",
          section: "architecture",
          value: "ok",
        },
        b: {
          id: "b",
          description: "Service B",
          section: "architecture",
          value: "ok",
        },
      },
      edges: [{ source: "a", target: "b", relation: "calls" }],
    };

    const rendered = augmentPlanMarkdownWithDecisionGraph(
      "# Plan\n\n## Architecture\nSome prose.\n",
      graph,
    );

    expect(rendered).not.toContain("### Decision map (auto)");
    expect(rendered).not.toContain("flowchart LR");
  });
});

describe("decisionGraphToMermaid", () => {
  it("returns flowchart with edges", () => {
    const graph: DecisionGraph = {
      nodes: {
        n1: { id: "n1", description: "Alpha", section: "architecture" },
        n2: { id: "n2", description: "Beta", section: "architecture" },
      },
      edges: [{ source: "n1", target: "n2", relation: "next" }],
    };
    const out = decisionGraphToMermaid(graph);
    expect(out).toContain("flowchart LR");
    expect(out).toContain("Alpha");
    expect(out).toContain("next");
  });

  it("uses subgraphs when there are no edges and multiple nodes", () => {
    const graph: DecisionGraph = {
      nodes: {
        a: { id: "a", description: "Q1", section: "architecture" },
        b: { id: "b", description: "Q2", section: "architecture" },
      },
    };
    const out = decisionGraphToMermaid(graph);
    expect(out).toContain("flowchart TB");
    expect(out).toContain("subgraph");
  });

  it("returns null for empty nodes", () => {
    expect(decisionGraphToMermaid({ nodes: {} })).toBeNull();
  });
});

describe("hasMermaidInArchitectureSection", () => {
  it("detects mermaid fence under Architecture", () => {
    expect(
      hasMermaidInArchitectureSection("## Architecture\n\n```mermaid\nx\n```"),
    ).toBe(true);
  });

  it("is false when mermaid only outside Architecture", () => {
    expect(
      hasMermaidInArchitectureSection("## Summary\n```mermaid\nx\n```\n## Architecture\nText"),
    ).toBe(false);
  });
});

describe("sanitizeMermaidNodeId", () => {
  it("prefixes leading digits", () => {
    expect(sanitizeMermaidNodeId("1a")).toMatch(/^n_/);
  });
});
