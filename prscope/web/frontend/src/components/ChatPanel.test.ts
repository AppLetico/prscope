import { describe, expect, it } from "vitest";
import { hasRunningToolCalls } from "./chatPanelUtils";
import type { ToolCallEntry } from "../types";

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
