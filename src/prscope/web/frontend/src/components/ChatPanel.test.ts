import { describe, expect, it } from "vitest";
import { hasRunningToolCalls, shouldShowActiveToolStream } from "./chatPanelUtils";
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
