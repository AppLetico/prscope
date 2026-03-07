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

export function shouldHideCompletedToolGroup(group: ToolCallGroup): boolean {
  return group.tools.length > 0;
}

export function isCollapsedCriticTurn(turn: PlanningTurn): boolean {
  return turn.role === "critic" && turn.content.trim().toLowerCase().startsWith("design review:");
}

export function isCollapsedRepairTurn(turn: PlanningTurn): boolean {
  return turn.role === "author" && turn.content.trim().startsWith("Repair planning complete.");
}

export function extractCriticPrimaryIssue(content: string): string | null {
  const match = content.match(/^Primary issue:\s*(.+)$/im);
  return match?.[1]?.trim() || null;
}

export function buildRefinementRoundSummaries(
  timeline: TimelineItem[],
): Record<number, { primaryIssue: string | null }> {
  const summaries: Record<number, { primaryIssue: string | null }> = {};
  for (const item of timeline) {
    if (item.kind !== "turn" || !isCollapsedCriticTurn(item.turn)) continue;
    summaries[item.turn.round] = {
      primaryIssue: extractCriticPrimaryIssue(item.turn.content),
    };
  }
  return summaries;
}

export function collapseTimelineForDisplay(timeline: TimelineItem[]): TimelineItem[] {
  return timeline.filter((item) => {
    if (item.kind === "tool_group") {
      return !shouldHideCompletedToolGroup(item.group);
    }
    return !isCollapsedCriticTurn(item.turn) && !isCollapsedRepairTurn(item.turn);
  });
}

export function compactTimelineToRecentRounds(
  timeline: TimelineItem[],
  roundsToKeep = 2,
): TimelineItem[] {
  const turns = timeline.filter((item): item is Extract<TimelineItem, { kind: "turn" }> => item.kind === "turn");
  if (turns.length === 0) return timeline;
  const latestRound = Math.max(...turns.map((item) => item.turn.round ?? 0));
  if (latestRound <= roundsToKeep - 1) return timeline;
  const minRoundToKeep = Math.max(0, latestRound - roundsToKeep + 1);
  const firstUserKey = turns.find((item) => item.turn.role === "user")?.key;

  return timeline.filter((item) => {
    if (item.kind !== "turn") return true;
    if (item.turn.round >= minRoundToKeep) return true;
    return item.key === firstUserKey;
  });
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
