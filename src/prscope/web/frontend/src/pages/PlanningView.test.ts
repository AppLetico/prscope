import { describe, expect, it } from "vitest";
import { buildTimeline, upsertToolCall, timelineReducer, INITIAL_TIMELINE_STATE } from "../components/chatPanelUtils";
import type { PlanningTurn, ToolCallEntry, ToolCallGroup } from "../types";

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
    expect(result.turns).toBe(turns);
  });

  it("session_state rebuilds timeline from stored turns", () => {
    const turns = [makeTurn({ sequence: 1 })];
    const withTurns = timelineReducer(INITIAL_TIMELINE_STATE, { type: "sync_turns", turns });
    const groups = [makeGroup(2)];
    const active: ToolCallEntry[] = [{ id: "x", call_id: "x1", name: "grep", status: "running" }];
    const result = timelineReducer(withTurns, { type: "session_state", groups, activeTools: active });
    expect(result.timeline.map((t) => t.kind)).toEqual(["turn", "tool_group"]);
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
    expect(s2).toBe(s1);
  });
});
