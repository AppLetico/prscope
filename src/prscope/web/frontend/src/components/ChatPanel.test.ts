import { describe, expect, it } from "vitest";
import {
  buildRefinementRoundSummaries,
  collapseTimelineForDisplay,
  compactTimelineToRecentRounds,
  contextGaugeLabels,
  peakContextUsageRatio,
  getLiveStatusMessage,
  hasRunningToolCalls,
  normalizeChatMessageForDedup,
  shouldHideCompletedToolGroup,
  shouldShowActiveToolStream,
} from "./chatPanelUtils";
import type { PlanningTurn, ToolCallEntry, ToolCallGroup } from "../types";

describe("normalizeChatMessageForDedup", () => {
  it("matches optimistic and persisted user text across whitespace differences", () => {
    const a = "Hello\n\nworld";
    const b = "  Hello  \n\n  world  ";
    expect(normalizeChatMessageForDedup(a)).toBe(normalizeChatMessageForDedup(b));
  });
});

describe("contextGaugeLabels", () => {
  it("shows peak and window when both are available", () => {
    const { tooltip, ariaLabel } = contextGaugeLabels(25, 2048, 8192);
    expect(tooltip).toContain("2,048");
    expect(tooltip).toContain("8,192");
    expect(tooltip).toContain("(25%)");
    expect(ariaLabel).toContain("2,048");
    expect(ariaLabel).toContain("25");
  });

  it("falls back to percent-only when window is unknown", () => {
    const { tooltip } = contextGaugeLabels(40, undefined, null);
    expect(tooltip).toContain("40%");
    expect(tooltip).not.toContain("/");
  });

  it("derives peak from percent when max prompt is missing but window is set", () => {
    const { tooltip } = contextGaugeLabels(50, undefined, 10000);
    expect(tooltip).toContain("5,000");
    expect(tooltip).toContain("10,000");
  });
});

describe("peakContextUsageRatio", () => {
  it("returns peak fill vs window", () => {
    expect(peakContextUsageRatio(32000, 128000)).toBeCloseTo(0.25);
  });
  it("returns null when window missing", () => {
    expect(peakContextUsageRatio(1000, null)).toBeNull();
  });
});

describe("getLiveStatusMessage", () => {
  const base = {
    questionsLength: 0,
    pendingClarification: false,
    isProcessing: true,
    activeToolCalls: [] as ToolCallEntry[],
    animatedThinkingMessage: "Thinking",
  };

  it("prefers server phase over generic thinking when processing", () => {
    expect(
      getLiveStatusMessage({
        ...base,
        phaseMessage: "Revising design",
        animatedThinkingMessage: "Preparing the next step",
      }),
    ).toBe("Revising design");
  });

  it("falls back to animated thinking when no phase", () => {
    expect(
      getLiveStatusMessage({
        ...base,
        phaseMessage: null,
        animatedThinkingMessage: "Thinking",
      }),
    ).toBe("Thinking");
  });

  it("prefers running plan-phase tools over phase message", () => {
    expect(
      getLiveStatusMessage({
        ...base,
        phaseMessage: "Revising design",
        animatedThinkingMessage: "Thinking",
        activeToolCalls: [
          { id: "1", name: "design_review", status: "running" },
        ],
      }),
    ).toBe("Design review...");
  });
});

describe("hasRunningToolCalls", () => {
  it("returns false for completed-only tool calls", () => {
    const completedOnly: ToolCallEntry[] = [
      { id: "cmd-a-1", name: "list_files", status: "done" },
      { id: "cmd-a-2", name: "read_file", status: "done", durationMs: 23 },
    ];
    expect(hasRunningToolCalls(completedOnly)).toBe(false);
  });

  it("returns true when at least one tool call is running", () => {
    const mixed: ToolCallEntry[] = [
      { id: "cmd-b-1", name: "list_files", status: "done" },
      { id: "cmd-b-2", name: "read_file", status: "running" },
    ];
    expect(hasRunningToolCalls(mixed)).toBe(true);
  });
});

describe("shouldShowActiveToolStream", () => {
  const completedOnly: ToolCallEntry[] = [
    { id: "cmd-c-1", name: "list_files", status: "done" },
    { id: "cmd-c-2", name: "read_file", status: "done", durationMs: 23 },
  ];

  it("hides completed-only tool calls once processing is over", () => {
    expect(shouldShowActiveToolStream(completedOnly, false)).toBe(false);
  });

  it("shows completed-only tool calls while processing is still active", () => {
    expect(shouldShowActiveToolStream(completedOnly, true)).toBe(true);
  });
});

function makeTurn(overrides: Partial<PlanningTurn>): PlanningTurn {
  return {
    id: overrides.id ?? 1,
    session_id: "s1",
    role: overrides.role ?? "author",
    content: overrides.content ?? "test",
    round: overrides.round ?? 1,
    created_at: overrides.created_at ?? new Date().toISOString(),
    sequence: overrides.sequence ?? 1,
  };
}

function makeGroup(toolNames: string[]): ToolCallGroup {
  return {
    sequence: 1,
    created_at: new Date().toISOString(),
    tools: toolNames.map((name, index) => ({
      id: `${name}-${index}`,
      name,
      status: "done" as const,
    })),
  };
}

describe("refinement timeline collapsing", () => {
  it("hides completed tool groups from history", () => {
    expect(shouldHideCompletedToolGroup(makeGroup(["design_review", "apply_critique"]))).toBe(true);
    expect(shouldHideCompletedToolGroup(makeGroup(["read_file"]))).toBe(true);
  });

  it("keeps critic reviews visible while still hiding repair chatter", () => {
    const timeline = [
      { kind: "tool_group" as const, key: "g1", group: makeGroup(["design_review"]) },
      {
        kind: "turn" as const,
        key: "t1",
        turn: makeTurn({
          role: "critic",
          content: "Design review: score 6.0/10\n\nPrimary issue: Missing test strategy.",
          sequence: 2,
        }),
      },
      {
        kind: "turn" as const,
        key: "t2",
        turn: makeTurn({
          role: "author",
          content: "Repair planning complete.\n\nProblem understanding: Missing test strategy.",
          sequence: 3,
        }),
      },
      {
        kind: "turn" as const,
        key: "t3",
        turn: makeTurn({
          role: "author",
          content: "Updated sections: test_strategy",
          sequence: 4,
        }),
      },
    ];

    const collapsed = collapseTimelineForDisplay(timeline);
    expect(collapsed).toHaveLength(2);
    expect(collapsed[0]?.kind).toBe("turn");
    expect((collapsed[0] as { kind: "turn"; turn: PlanningTurn }).turn.role).toBe("critic");
    expect(collapsed[0]?.kind).toBe("turn");
    expect((collapsed[1] as { kind: "turn"; turn: PlanningTurn }).turn.content).toContain("Updated sections:");
  });

  it("extracts the primary critic issue by round", () => {
    const timeline = [
      {
        kind: "turn" as const,
        key: "t1",
        turn: makeTurn({
          role: "critic",
          round: 3,
          content: "Design review: score 6.0/10\n\nPrimary issue: Missing rollback plan.",
        }),
      },
    ];

    const summaries = buildRefinementRoundSummaries(timeline);
    expect(summaries[3]?.primaryIssue).toBe("Missing rollback plan.");
  });

  it("keeps the first user request and the most recent rounds", () => {
    const timeline = [
      { kind: "turn" as const, key: "u0", turn: makeTurn({ role: "user", round: 0, sequence: 1, content: "initial request" }) },
      { kind: "turn" as const, key: "a0", turn: makeTurn({ role: "author", round: 0, sequence: 2, content: "first reply" }) },
      { kind: "turn" as const, key: "u1", turn: makeTurn({ role: "user", round: 1, sequence: 3, content: "revise once" }) },
      { kind: "turn" as const, key: "a1", turn: makeTurn({ role: "author", round: 1, sequence: 4, content: "reply 1" }) },
      { kind: "turn" as const, key: "u2", turn: makeTurn({ role: "user", round: 2, sequence: 5, content: "revise twice" }) },
      { kind: "turn" as const, key: "a2", turn: makeTurn({ role: "author", round: 2, sequence: 6, content: "reply 2" }) },
      { kind: "turn" as const, key: "u3", turn: makeTurn({ role: "user", round: 3, sequence: 7, content: "latest ask" }) },
      { kind: "turn" as const, key: "a3", turn: makeTurn({ role: "author", round: 3, sequence: 8, content: "latest reply" }) },
    ];

    const compacted = compactTimelineToRecentRounds(timeline);
    expect(compacted.map((item) => item.key)).toEqual(["u0", "u2", "a2", "u3", "a3"]);
  });
});
