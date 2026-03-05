import { describe, expect, it } from "vitest";
import {
  isTerminalCompletedSnapshot,
  shouldFinalizeFromTerminalSnapshot,
} from "./planningViewUtils";
import type { ToolCallEntry } from "../types";

describe("shouldFinalizeFromTerminalSnapshot", () => {
  const doneSnapshot: ToolCallEntry[] = [
    { id: "cmd-1-tool-1", name: "list_files", status: "done" },
    { id: "cmd-1-tool-2", name: "read_file", status: "done", durationMs: 12 },
  ];

  it("returns false when active calls were already cleared", () => {
    // Regression: after a prior 'complete' event has already finalized/cleared state,
    // a follow-up terminal session_state with done-only calls must not finalize again.
    expect(shouldFinalizeFromTerminalSnapshot(false, doneSnapshot, 0)).toBe(false);
  });

  it("returns true for recovery when terminal snapshot has done-only calls and active calls remain", () => {
    expect(shouldFinalizeFromTerminalSnapshot(false, doneSnapshot, 2)).toBe(true);
  });

  it("returns false while still processing or if snapshot includes running calls", () => {
    expect(shouldFinalizeFromTerminalSnapshot(true, doneSnapshot, 2)).toBe(false);
    expect(
      shouldFinalizeFromTerminalSnapshot(false, [{ id: "r1", name: "grep_code", status: "running" }], 2),
    ).toBe(false);
  });
});

describe("isTerminalCompletedSnapshot", () => {
  it("detects terminal done-only snapshots", () => {
    const doneOnly: ToolCallEntry[] = [{ id: "d1", name: "read_file", status: "done" }];
    expect(isTerminalCompletedSnapshot(false, doneOnly)).toBe(true);
  });

  it("returns false for processing or running snapshots", () => {
    const running: ToolCallEntry[] = [{ id: "r1", name: "read_file", status: "running" }];
    expect(isTerminalCompletedSnapshot(true, running)).toBe(false);
    expect(isTerminalCompletedSnapshot(false, running)).toBe(false);
    expect(isTerminalCompletedSnapshot(false, [])).toBe(false);
  });
});
