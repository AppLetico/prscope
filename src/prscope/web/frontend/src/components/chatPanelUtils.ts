import type { PlanningTurn, ToolCallEntry, ToolCallGroup } from "../types";

export type TimelineItem =
  | { kind: "turn"; turn: PlanningTurn; key: string }
  | { kind: "tool_group"; group: ToolCallGroup; key: string };

export type TimelineState = {
  turns: PlanningTurn[];
  groups: ToolCallGroup[];
  timeline: TimelineItem[];
  activeTools: ToolCallEntry[];
};

export type TimelineAction =
  | { type: "sync_turns"; turns: PlanningTurn[] }
  | { type: "session_state"; groups: ToolCallGroup[]; activeTools: ToolCallEntry[] }
  | { type: "tool_update"; tool: ToolCallEntry }
  | { type: "complete" }
  | { type: "plan_ready" };

export const INITIAL_TIMELINE_STATE: TimelineState = {
  turns: [],
  groups: [],
  timeline: [],
  activeTools: [],
};

export function timelineReducer(state: TimelineState, action: TimelineAction): TimelineState {
  switch (action.type) {
    case "sync_turns": {
      if (action.turns === state.turns) return state;
      const turns = [...action.turns].sort((a, b) => (a.sequence ?? 0) - (b.sequence ?? 0));
      const timeline = buildTimeline(turns, state.groups);
      return { ...state, turns, timeline };
    }
    case "session_state": {
      const groups = [...action.groups].sort((a, b) => (a.sequence ?? 0) - (b.sequence ?? 0));
      const timeline = buildTimeline(state.turns, groups);
      return { ...state, groups, timeline, activeTools: action.activeTools };
    }
    case "tool_update":
      return { ...state, activeTools: upsertToolCall(state.activeTools, action.tool) };
    case "complete":
    case "plan_ready":
      return { ...state, activeTools: [] };
    default:
      return state;
  }
}

/**
 * O(n) linear merge of pre-sorted turns and tool groups by sequence number.
 * Tool groups with sequence <= turn.sequence sort first, guaranteeing tools
 * appear before the response they produced.
 */
export function buildTimeline(
  turns: PlanningTurn[],
  groups: ToolCallGroup[],
): TimelineItem[] {
  const timeline: TimelineItem[] = [];
  let t = 0;
  let g = 0;
  while (t < turns.length && g < groups.length) {
    const turnSeq = turns[t]?.sequence ?? 0;
    const groupSeq = groups[g]?.sequence ?? 0;
    const turnRole = String(turns[t]?.role ?? "");
    // Backend sequencing occasionally emits the tool-group one tick after the
    // assistant turn it produced. Keep that group visually above that turn.
    const belongsToUpcomingAssistantTurn = turnRole !== "user" && groupSeq === turnSeq + 1;
    if (groupSeq <= turnSeq || belongsToUpcomingAssistantTurn) {
      timeline.push({
        kind: "tool_group",
        group: groups[g]!,
        key: `tg-${groups[g]!.sequence}`,
      });
      g++;
    } else {
      timeline.push({
        kind: "turn",
        turn: turns[t]!,
        key: `turn-${turns[t]!.id ?? t}`,
      });
      t++;
    }
  }
  while (t < turns.length) {
    timeline.push({
      kind: "turn",
      turn: turns[t]!,
      key: `turn-${turns[t]!.id ?? t}`,
    });
    t++;
  }
  while (g < groups.length) {
    timeline.push({
      kind: "tool_group",
      group: groups[g]!,
      key: `tg-${groups[g]!.sequence}`,
    });
    g++;
  }
  return timeline;
}

/** Upsert a tool call by call_id. Insert if new, replace if existing. */
export function upsertToolCall(tools: ToolCallEntry[], tool: ToolCallEntry): ToolCallEntry[] {
  if (tool.call_id) {
    const idx = tools.findIndex((t) => t.call_id === tool.call_id);
    if (idx !== -1) {
      const next = [...tools];
      next[idx] = tool;
      return next;
    }
  }
  return [...tools, tool];
}

export function hasRunningToolCalls(toolCalls: ToolCallEntry[]): boolean {
  return toolCalls.some((call) => call.status === "running");
}

export function shouldShowActiveToolStream(
  toolCalls: ToolCallEntry[],
  isProcessing: boolean,
): boolean {
  return hasRunningToolCalls(toolCalls) || (isProcessing && toolCalls.length > 0);
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
