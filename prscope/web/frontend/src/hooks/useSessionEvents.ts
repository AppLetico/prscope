import { useEffect } from "react";
import type { DiscoveryQuestion, SessionStatus, ToolCallEntry, ToolCallGroup, UIEvent } from "../types";

const SESSION_STATUS_VALUES: ReadonlySet<SessionStatus> = new Set([
  "draft",
  "refining",
  "converged",
  "approved",
  "error",
]);

function parseSessionStatus(value: unknown): SessionStatus {
  if (typeof value === "string" && SESSION_STATUS_VALUES.has(value as SessionStatus)) {
    return value as SessionStatus;
  }
  return "error";
}

function normalizeEvent(rawType: string, rawPayload: Record<string, unknown>): UIEvent | null {
  if (rawType === "session_state") {
    const activeCommandRaw = rawPayload.active_command;
    const activeCommand =
      typeof activeCommandRaw === "object" && activeCommandRaw !== null
        ? (activeCommandRaw as Record<string, unknown>)
        : null;
    return {
      type: "session_state",
      v: 1,
      status: parseSessionStatus(rawPayload.status),
      phase_message: rawPayload.phase_message ? String(rawPayload.phase_message) : null,
      is_processing: Boolean(rawPayload.is_processing),
      current_round: Number(rawPayload.current_round ?? 0),
      pending_questions: Array.isArray(rawPayload.pending_questions)
        ? (rawPayload.pending_questions as DiscoveryQuestion[])
        : null,
      active_tool_calls: Array.isArray(rawPayload.active_tool_calls)
        ? (rawPayload.active_tool_calls as ToolCallEntry[])
        : [],
      completed_tool_call_groups: Array.isArray(rawPayload.completed_tool_call_groups)
        ? (rawPayload.completed_tool_call_groups as ToolCallGroup[])
        : [],
      active_command_id:
        activeCommand && typeof activeCommand.command_id === "string"
          ? activeCommand.command_id
          : null,
    };
  }
  if (rawType === "thinking") {
    return { type: "thinking", message: String(rawPayload.message ?? "Thinking...") };
  }
  if (rawType === "tool_update") {
    const tool = (rawPayload.tool ?? rawPayload) as Record<string, unknown>;
    return {
      type: "tool_update",
      tool: {
        id: String(tool.call_id ?? tool.id ?? `${Date.now()}`),
        call_id: String(tool.call_id ?? tool.id ?? `${Date.now()}`),
        name: String(tool.name ?? "tool"),
        sessionStage: tool.sessionStage ? String(tool.sessionStage) : (tool.session_stage ? String(tool.session_stage) : undefined),
        path: tool.path ? String(tool.path) : undefined,
        query: tool.query ? String(tool.query) : undefined,
        status: String(tool.status ?? "running") as "running" | "done",
        durationMs: tool.durationMs !== undefined ? Number(tool.durationMs) : (tool.duration_ms !== undefined ? Number(tool.duration_ms) : undefined),
        created_at: tool.created_at ? String(tool.created_at) : new Date().toISOString(),
      },
    };
  }
  if (rawType === "complete") {
    return { type: "complete", message: rawPayload.message ? String(rawPayload.message) : undefined };
  }
  if (rawType === "plan_ready") {
    return {
      type: "plan_ready",
      round: rawPayload.round !== undefined ? Number(rawPayload.round) : undefined,
      initial_draft:
        rawPayload.initial_draft !== undefined
          ? Boolean(rawPayload.initial_draft)
          : undefined,
      saved_at_unix_s:
        rawPayload.saved_at_unix_s !== undefined
          ? Number(rawPayload.saved_at_unix_s)
          : undefined,
    };
  }
  if (rawType === "context_compaction") {
    return {
      type: "context_compaction",
      enabled: Boolean(rawPayload.enabled),
      reason: rawPayload.reason ? String(rawPayload.reason) : undefined,
    };
  }
  if (rawType === "error") {
    return { type: "error", message: String(rawPayload.message ?? "Unknown error") };
  }
  if (rawType === "warning") {
    return { type: "warning", message: String(rawPayload.message ?? "Warning") };
  }
  if (rawType === "token_usage") {
    return {
      type: "token_usage",
      model: String(rawPayload.model ?? "unknown"),
      prompt_tokens: Number(rawPayload.prompt_tokens ?? 0),
      completion_tokens: Number(rawPayload.completion_tokens ?? 0),
      call_cost_usd: Number(rawPayload.call_cost_usd ?? 0),
      session_total_usd:
        rawPayload.session_total_usd !== undefined
          ? Number(rawPayload.session_total_usd)
          : undefined,
      max_prompt_tokens:
        rawPayload.max_prompt_tokens !== undefined
          ? Number(rawPayload.max_prompt_tokens)
          : undefined,
      context_window_tokens:
        rawPayload.context_window_tokens !== undefined
          ? Number(rawPayload.context_window_tokens)
          : undefined,
      context_usage_ratio:
        rawPayload.context_usage_ratio !== undefined
          ? Number(rawPayload.context_usage_ratio)
          : undefined,
    };
  }
  if (rawType === "clarification_needed") {
    const recommendations = Array.isArray(rawPayload.recommendations)
      ? rawPayload.recommendations
        .filter((value): value is string => typeof value === "string")
      : undefined;
    return {
      type: "clarification_needed",
      question: String(rawPayload.question ?? ""),
      context: rawPayload.context ? String(rawPayload.context) : undefined,
      source: rawPayload.source ? String(rawPayload.source) : undefined,
      recommendations,
    };
  }
  if (rawType === "setup_progress") {
    return { type: "setup_progress", step: String(rawPayload.step ?? "") };
  }
  if (rawType === "discovery_ready") {
    return { type: "discovery_ready", opening: String(rawPayload.opening ?? "") };
  }
  return null;
}

export function useSessionEvents(
  sessionId: string,
  onEvent: (event: UIEvent) => void,
  enabled = true,
  onReconnect?: () => void,
) {
  useEffect(() => {
    if (!enabled || !sessionId) {
      return;
    }
    const params = new URLSearchParams(window.location.search);
    const repo = params.get("repo") ?? window.localStorage.getItem("prscope.web.repo");
    const basePath = `/api/sessions/${sessionId}/events`;
    const url = repo ? `${basePath}?repo=${encodeURIComponent(repo)}` : basePath;
    const eventSource = new EventSource(url);
    let initialConnection = true;

    eventSource.onopen = () => {
      if (initialConnection) {
        initialConnection = false;
        return;
      }
      onReconnect?.();
    };

    const bind = (eventName: string) => {
      eventSource.addEventListener(eventName, (event) => {
        try {
          const payload = JSON.parse((event as MessageEvent).data) as Record<string, unknown>;
          const normalized = normalizeEvent(eventName, payload);
          if (normalized) {
            if (typeof payload.session_version === "number") {
              normalized.session_version = payload.session_version;
            }
            onEvent(normalized);
          }
        } catch {
          // Ignore malformed events
        }
      });
    };

    bind("thinking");
    bind("session_state");
    bind("tool_update");
    bind("context_compaction");
    bind("plan_ready");
    bind("complete");
    bind("error");
    bind("warning");
    bind("token_usage");
    bind("clarification_needed");
    bind("setup_progress");
    bind("discovery_ready");

    eventSource.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data) as Record<string, unknown>;
        const normalized = normalizeEvent("thinking", payload);
        if (normalized) onEvent(normalized);
      } catch {
        // Ignore malformed fallback events
      }
    };

    return () => {
      eventSource.close();
    };
  }, [sessionId, onEvent, enabled, onReconnect]);
}
