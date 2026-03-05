import type { ToolCallEntry } from "../types";

export function shouldFinalizeFromTerminalSnapshot(
  eventIsProcessing: boolean,
  snapshotCalls: ToolCallEntry[],
  activeInMemoryCount: number,
): boolean {
  if (eventIsProcessing) return false;
  if (activeInMemoryCount <= 0) return false;
  if (snapshotCalls.length === 0) return false;
  const snapshotHasRunning = snapshotCalls.some((call) => call.status === "running");
  return !snapshotHasRunning;
}

export function isTerminalCompletedSnapshot(
  eventIsProcessing: boolean,
  snapshotCalls: ToolCallEntry[],
): boolean {
  if (eventIsProcessing) return false;
  if (snapshotCalls.length === 0) return false;
  return !snapshotCalls.some((call) => call.status === "running");
}
