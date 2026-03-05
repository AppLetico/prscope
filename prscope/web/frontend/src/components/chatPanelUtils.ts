import type { ToolCallEntry } from "../types";

export function hasRunningToolCalls(toolCalls: ToolCallEntry[]): boolean {
  return toolCalls.some((call) => call.status === "running");
}

export function extractFirstJsonObject(
  raw: string,
): { parsed: Record<string, unknown>; start: number; end: number } | null {
  const start = raw.indexOf("{");
  if (start < 0) return null;
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let idx = start; idx < raw.length; idx += 1) {
    const ch = raw[idx];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (ch === "\\") {
      escaped = true;
      continue;
    }
    if (ch === '"') {
      inString = !inString;
      continue;
    }
    if (inString) continue;
    if (ch === "{") depth += 1;
    if (ch === "}") {
      depth -= 1;
      if (depth === 0) {
        const candidate = raw.slice(start, idx + 1);
        try {
          const parsed = JSON.parse(candidate) as Record<string, unknown>;
          return { parsed, start, end: idx + 1 };
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}
