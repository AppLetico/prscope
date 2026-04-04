import { describe, expect, it } from "vitest";
import {
  dedupeSimilarCriticIssues,
  dedupeOpenIssueNodesByDescription,
  issueSimilarityScore,
  jaccardIssueSimilarity,
} from "./issueDedupe";

describe("dedupeSimilarCriticIssues", () => {
  it("merges problem vs recommended fix phrasing about the same topic", () => {
    const items = [
      { text: "Lack of detailed rollback steps in the rollback plan.", severity: "major" as const },
      { text: "No explicit verification steps outlined for testing the new behavior.", severity: "major" as const },
      {
        text: "Expand the rollback plan with detailed steps and commands.",
        severity: "minor" as const,
      },
    ];
    const out = dedupeSimilarCriticIssues(items);
    expect(out).toHaveLength(2);
    expect(out.some((i) => i.text.includes("rollback"))).toBe(true);
    expect(out.some((i) => i.text.includes("verification"))).toBe(true);
  });

  it("keeps distinct issues with low token overlap", () => {
    const items = [
      { text: "Missing API authentication headers in the client module.", severity: "major" as const },
      { text: "Database migration lacks a down migration script.", severity: "major" as const },
    ];
    expect(dedupeSimilarCriticIssues(items)).toHaveLength(2);
  });

  it("matches backend-style Jaccard on overlapping rollback wording", () => {
    const a = "Lack of detailed rollback steps in the rollback plan.";
    const b = "Expand the rollback plan with detailed steps and commands.";
    expect(jaccardIssueSimilarity(a, b)).toBeGreaterThanOrEqual(0.5);
  });

  it("merges performance monitoring paraphrases via recall (subset tokens)", () => {
    const a = "Performance monitoring strategy is still vague and lacks detail.";
    const b = "Lack of specific performance monitoring strategy for middleware.";
    expect(jaccardIssueSimilarity(a, b)).toBeLessThan(0.5);
    expect(issueSimilarityScore(a, b)).toBeGreaterThanOrEqual(0.5);
    const nodes = [
      { id: "issue_17", description: a },
      { id: "issue_19", description: b },
    ];
    const out = dedupeOpenIssueNodesByDescription(nodes);
    expect(out).toHaveLength(1);
    expect(out[0].id).toBe("issue_17");
  });

  it("dedupeOpenIssueNodesByDescription collapses minor wording drift across fallback graph rows", () => {
    const a =
      "The plan should explicitly reference how the new middleware layer coordinates with existing authentication flows and failure handling.";
    const b =
      "The plan should explicitly reference how new middleware coordinates with existing authentication flows and failure handling.";
    const nodes = [
      { id: "fallback_issue_1", description: a },
      { id: "fallback_issue_2", description: b },
    ];
    const out = dedupeOpenIssueNodesByDescription(nodes, 0.5);
    expect(out).toHaveLength(1);
    expect(out[0].id).toBe("fallback_issue_1");
  });
});
