import { useEffect } from "react";
import type { UIEvent } from "../types";

function normalizeEvent(rawType: string, rawPayload: Record<string, unknown>): UIEvent | null {
  if (rawType === "thinking") {
    return { type: "thinking", message: String(rawPayload.message ?? "Thinking...") };
  }
  if (rawType === "tool_call") {
    return {
      type: "tool_call",
      name: String(rawPayload.name ?? "tool"),
      path: rawPayload.path ? String(rawPayload.path) : undefined,
      session_stage: rawPayload.session_stage ? String(rawPayload.session_stage) : undefined,
    };
  }
  if (rawType === "complete") {
    return { type: "complete", message: rawPayload.message ? String(rawPayload.message) : undefined };
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
    };
  }
  if (rawType === "clarification_needed") {
    return {
      type: "clarification_needed",
      question: String(rawPayload.question ?? ""),
      context: rawPayload.context ? String(rawPayload.context) : undefined,
      source: rawPayload.source ? String(rawPayload.source) : undefined,
    };
  }
  return null;
}

export function useSessionEvents(sessionId: string, onEvent: (event: UIEvent) => void) {
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const repo = params.get("repo") ?? window.localStorage.getItem("prscope.web.repo");
    const basePath = `/api/sessions/${sessionId}/events`;
    const url = repo ? `${basePath}?repo=${encodeURIComponent(repo)}` : basePath;
    const eventSource = new EventSource(url);

    const bind = (eventName: string) => {
      eventSource.addEventListener(eventName, (event) => {
        try {
          const payload = JSON.parse((event as MessageEvent).data) as Record<string, unknown>;
          const normalized = normalizeEvent(eventName, payload);
          if (normalized) onEvent(normalized);
        } catch {
          // Ignore malformed events
        }
      });
    };

    bind("thinking");
    bind("tool_call");
    bind("complete");
    bind("error");
    bind("warning");
    bind("token_usage");
    bind("clarification_needed");

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
  }, [sessionId, onEvent]);
}
