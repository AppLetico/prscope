import type {
  DiscoveryTurnResult,
  PlanVersion,
  PlanningSession,
  PlanningTurn,
} from "../types";

const REPO_STORAGE_KEY = "prscope.web.repo";

function getRepoContext(): string | null {
  const params = new URLSearchParams(window.location.search);
  const repo = params.get("repo");
  if (repo) {
    window.localStorage.setItem(REPO_STORAGE_KEY, repo);
    return repo;
  }
  return window.localStorage.getItem(REPO_STORAGE_KEY);
}

export function getActiveRepoContext(): string | null {
  return getRepoContext();
}

function withRepoQuery(path: string): string {
  const repo = getRepoContext();
  if (!repo) return path;
  const separator = path.includes("?") ? "&" : "?";
  return `${path}${separator}repo=${encodeURIComponent(repo)}`;
}

function withRepoBody<T extends Record<string, unknown>>(body: T): T & { repo?: string } {
  const repo = getRepoContext();
  return repo ? { ...body, repo } : body;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    const text = await response.text();
    let detail = text;
    try {
      const payload = JSON.parse(text) as { detail?: string };
      if (payload.detail) detail = payload.detail;
    } catch {
      // Keep raw text fallback for non-JSON errors
    }
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

export function listSessions() {
  return request<{ items: PlanningSession[] }>(withRepoQuery("/api/sessions"));
}

export function createChatSession() {
  return request<{ session: PlanningSession; opening?: string }>(withRepoQuery("/api/sessions"), {
    method: "POST",
    body: JSON.stringify(withRepoBody({ mode: "chat" })),
  });
}

export function createRequirementsSession(requirements: string) {
  return request<{ session: PlanningSession }>(withRepoQuery("/api/sessions"), {
    method: "POST",
    body: JSON.stringify(withRepoBody({ mode: "requirements", requirements })),
  });
}

export function getSession(sessionId: string) {
  return request<{
    session: PlanningSession;
    conversation: PlanningTurn[];
    plan_versions: PlanVersion[];
    current_plan: PlanVersion | null;
    tool_summary: { recent_tool_calls: string[] };
  }>(withRepoQuery(`/api/sessions/${sessionId}`));
}

export function sendDiscoveryMessage(sessionId: string, message: string) {
  return request<{ result: DiscoveryTurnResult }>(withRepoQuery(`/api/sessions/${sessionId}/message`), {
    method: "POST",
    body: JSON.stringify(withRepoBody({ message })),
  });
}

export function runRound(sessionId: string, user_input?: string) {
  return request<{
    critic: Record<string, unknown>;
    author: Record<string, unknown>;
    convergence: { converged: boolean; reason: string; change_pct: number };
  }>(withRepoQuery(`/api/sessions/${sessionId}/round`), {
    method: "POST",
    body: JSON.stringify(withRepoBody({ user_input })),
  });
}

export function submitClarification(sessionId: string, answers: string[]) {
  return request<{ ok: boolean }>(withRepoQuery(`/api/sessions/${sessionId}/clarify`), {
    method: "POST",
    body: JSON.stringify(withRepoBody({ answers })),
  });
}

export function approveSession(sessionId: string) {
  return request<{ approved: boolean }>(withRepoQuery(`/api/sessions/${sessionId}/approve`), {
    method: "POST",
    body: JSON.stringify(withRepoBody({})),
  });
}

export function exportSession(sessionId: string) {
  return request<{ files: Array<{ name: string; kind: string; url: string }> }>(
    withRepoQuery(`/api/sessions/${sessionId}/export`),
    {
      method: "POST",
      body: JSON.stringify(withRepoBody({})),
    },
  );
}

export async function downloadFile(url: string, filename: string): Promise<void> {
  const response = await fetch(withRepoQuery(url));
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Download failed: ${response.status}`);
  }

  const blob = await response.blob();
  const objectUrl = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = objectUrl;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(objectUrl);
}

export function getDiff(sessionId: string) {
  return request<{ diff: string }>(withRepoQuery(`/api/sessions/${sessionId}/diff`));
}
