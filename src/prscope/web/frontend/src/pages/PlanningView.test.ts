import { describe, expect, it } from "vitest";
import {
  buildTimeline,
  formatPhaseTimingLabel,
  formatToolActivityLabel,
  upsertLiveActivity,
  upsertToolCall,
  timelineReducer,
  INITIAL_TIMELINE_STATE,
} from "../components/chatPanelUtils";
import type { LiveActivityEntry, PlanningTurn, ToolCallEntry, ToolCallGroup } from "../types";

function makeTurn(overrides: Partial<PlanningTurn> & { sequence: number }): PlanningTurn {
  return {
    id: overrides.id ?? Math.floor(Math.random() * 1000),
    session_id: "s1",
    role: overrides.role ?? "author",
    content: overrides.content ?? "test",
    round: overrides.round ?? 0,
    created_at: overrides.created_at ?? new Date().toISOString(),
    sequence: overrides.sequence,
  };
}

function makeGroup(seq: number, toolNames: string[] = ["read_file"]): ToolCallGroup {
  return {
    sequence: seq,
    created_at: new Date().toISOString(),
    tools: toolNames.map((name, i) => ({
      id: `tool-${seq}-${i}`,
      call_id: `call-${seq}-${i}`,
      name,
      status: "done" as const,
    })),
  };
}

describe("buildTimeline", () => {
  it("merges turns and groups by sequence", () => {
    const turns = [makeTurn({ sequence: 2 }), makeTurn({ sequence: 5, role: "critic" })];
    const groups = [makeGroup(1), makeGroup(4)];
    const result = buildTimeline(turns, groups);
    expect(result.map((r) => r.kind)).toEqual(["tool_group", "turn", "tool_group", "turn"]);
  });

  it("tool groups with same sequence as turn appear first", () => {
    const turns = [makeTurn({ sequence: 3 })];
    const groups = [makeGroup(3)];
    const result = buildTimeline(turns, groups);
    expect(result[0]!.kind).toBe("tool_group");
    expect(result[1]!.kind).toBe("turn");
  });

  it("handles empty inputs", () => {
    expect(buildTimeline([], [])).toEqual([]);
    const turns = [makeTurn({ sequence: 1 })];
    expect(buildTimeline(turns, []).length).toBe(1);
    expect(buildTimeline([], [makeGroup(1)]).length).toBe(1);
  });
});

describe("upsertToolCall", () => {
  const existing: ToolCallEntry[] = [
    { id: "1", call_id: "abc", name: "read_file", status: "running" },
    { id: "2", call_id: "def", name: "grep", status: "running" },
  ];

  it("updates existing tool by call_id", () => {
    const updated: ToolCallEntry = { id: "1", call_id: "abc", name: "read_file", status: "done", durationMs: 42 };
    const result = upsertToolCall(existing, updated);
    expect(result.length).toBe(2);
    expect(result[0]!.status).toBe("done");
    expect(result[0]!.durationMs).toBe(42);
  });

  it("appends new tool when call_id not found", () => {
    const newTool: ToolCallEntry = { id: "3", call_id: "ghi", name: "search", status: "running" };
    const result = upsertToolCall(existing, newTool);
    expect(result.length).toBe(3);
    expect(result[2]!.call_id).toBe("ghi");
  });
});

describe("timelineReducer", () => {
  it("sync_turns rebuilds timeline from stored groups", () => {
    const groups = [makeGroup(1)];
    const withGroups = timelineReducer(INITIAL_TIMELINE_STATE, {
      type: "session_state",
      groups,
      activeTools: [],
    });
    const turns = [makeTurn({ sequence: 2 })];
    const result = timelineReducer(withGroups, { type: "sync_turns", turns });
    expect(result.timeline.map((t) => t.kind)).toEqual(["tool_group", "turn"]);
    expect(result.turns).toStrictEqual(turns);
  });

  it("session_state rebuilds timeline from stored turns", () => {
    const turns = [makeTurn({ sequence: 1 })];
    const withTurns = timelineReducer(INITIAL_TIMELINE_STATE, { type: "sync_turns", turns });
    const groups = [makeGroup(2)];
    const active: ToolCallEntry[] = [{ id: "x", call_id: "x1", name: "grep", status: "running" }];
    const result = timelineReducer(withTurns, { type: "session_state", groups, activeTools: active });
    expect(result.timeline.map((t) => t.kind)).toEqual(["tool_group", "turn"]);
    expect(result.activeTools).toBe(active);
  });

  it("tool_update upserts into activeTools", () => {
    const tool: ToolCallEntry = { id: "1", call_id: "abc", name: "read", status: "running" };
    const s1 = timelineReducer(INITIAL_TIMELINE_STATE, { type: "tool_update", tool });
    expect(s1.activeTools).toHaveLength(1);
    const updated = { ...tool, status: "done" as const };
    const s2 = timelineReducer(s1, { type: "tool_update", tool: updated });
    expect(s2.activeTools).toHaveLength(1);
    expect(s2.activeTools[0]!.status).toBe("done");
  });

  it("complete clears activeTools", () => {
    const tool: ToolCallEntry = { id: "1", call_id: "abc", name: "read", status: "running" };
    const s1 = timelineReducer(INITIAL_TIMELINE_STATE, { type: "tool_update", tool });
    const s2 = timelineReducer(s1, { type: "complete" });
    expect(s2.activeTools).toHaveLength(0);
  });

  it("sync_turns is a no-op when turns reference is unchanged", () => {
    const turns = [makeTurn({ sequence: 1 })];
    const s1 = timelineReducer(INITIAL_TIMELINE_STATE, { type: "sync_turns", turns });
    const s2 = timelineReducer(s1, { type: "sync_turns", turns });
    expect(s2).toStrictEqual(s1);
  });
});

describe("live activity helpers", () => {
  it("formats phase timing labels with elapsed seconds", () => {
    expect(formatPhaseTimingLabel("initial_draft", "start")).toBe("Initial draft started");
    expect(formatPhaseTimingLabel("discovery", "complete", 3400)).toBe("Discovery completed in 3.4s");
    expect(formatPhaseTimingLabel("planner_redraft", "failed", 12000)).toBe("Planner redraft failed after 12s");
  });

  it("formats tool activity labels for running and completed tools", () => {
    expect(formatToolActivityLabel({ id: "1", name: "draft_plan", status: "running" })).toBe("Draft plan running");
    expect(
      formatToolActivityLabel({ id: "2", name: "read_file", status: "done", durationMs: 41 }),
    ).toBe("Read file completed in 41ms");
  });

  it("upserts activity by id and collapses repeated notes", () => {
    const running: LiveActivityEntry = {
      id: "tool:1",
      kind: "tool",
      message: "Draft plan running",
      stage: "planner",
      status: "running",
      created_at: "2026-01-01T00:00:00.000Z",
    };
    const done: LiveActivityEntry = {
      ...running,
      message: "Draft plan completed in 900ms",
      status: "done",
    };
    const first = upsertLiveActivity([], running);
    const second = upsertLiveActivity(first, done);
    expect(second).toHaveLength(1);
    expect(second[0]?.status).toBe("done");

    const repeated = upsertLiveActivity(second, {
      id: "thought:scanning",
      kind: "thought",
      message: "Scanning context",
      status: "running",
      created_at: "2026-01-01T00:00:01.000Z",
    });
    const collapsed = upsertLiveActivity(repeated, {
      id: "thought:scanning:2",
      kind: "thought",
      message: "Scanning context",
      status: "running",
      created_at: "2026-01-01T00:00:02.000Z",
    });
    expect(collapsed).toHaveLength(2);
    expect(collapsed[1]?.count).toBe(2);
  });
});
