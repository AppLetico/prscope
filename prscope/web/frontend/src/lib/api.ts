import type {
  DiscoveryTurnResult,
  ModelCatalogItem,
  PlanVersion,
  PlanningSession,
  PlanningTurn,
  RepoProfileSummary,
  RoundMetric,
} from "../types";

const REPO_STORAGE_KEY = "prscope.web.repo";
const MODEL_SELECTION_STORAGE_PREFIX = "prscope.web.models.";

export class ConflictError extends Error {
  status?: string;
  phase_message?: string | null;
  allowed_commands?: string[];
  reason?: string;
}

function getRepoContext(): string | null {
  const params = new URLSearchParams(window.location.search);
  const repo = params.get("repo");
  if (repo) {
    window.localStorage.setItem(REPO_STORAGE_KEY, repo);
    return repo;
  }
  return window.localStorage.getItem(REPO_STORAGE_KEY);
}

/** Set repo context (localStorage). Caller should also update URL search params if needed. */
export function setRepoContext(repo: string | null): void {
  if (repo) {
    window.localStorage.setItem(REPO_STORAGE_KEY, repo);
  } else {
    window.localStorage.removeItem(REPO_STORAGE_KEY);
  }
}

function pathWithRepo(path: string, repo: string | null): string {
  if (repo == null) return path;
  const separator = path.includes("?") ? "&" : "?";
  return `${path}${separator}repo=${encodeURIComponent(repo)}`;
}

export function getActiveRepoContext(): string | null {
  return getRepoContext();
}

function modelSelectionStorageKey(repo: string | null): string {
  return `${MODEL_SELECTION_STORAGE_PREFIX}${repo ?? "__default__"}`;
}

export function getStoredModelSelection(repo?: string | null): {
  author_model?: string;
  critic_model?: string;
} {
  const key = modelSelectionStorageKey(repo ?? getRepoContext());
  const raw = window.localStorage.getItem(key);
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw) as { author_model?: string; critic_model?: string };
    return {
      author_model: parsed.author_model,
      critic_model: parsed.critic_model,
    };
  } catch {
    return {};
  }
}

export function setStoredModelSelection(
  selection: { author_model?: string; critic_model?: string },
  repo?: string | null,
): void {
  const key = modelSelectionStorageKey(repo ?? getRepoContext());
  window.localStorage.setItem(key, JSON.stringify(selection));
}

function withRepoQuery(path: string): string {
  const repo = getRepoContext();
  return pathWithRepo(path, repo);
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
    let payload: Record<string, unknown> | null = null;
    try {
      payload = JSON.parse(text) as Record<string, unknown>;
      if (typeof payload.detail === "string") detail = payload.detail;
      if (typeof payload.detail === "object" && payload.detail !== null) {
        payload = payload.detail as Record<string, unknown>;
        if (typeof payload.detail === "string") detail = payload.detail;
      }
    } catch {
      // Keep raw text fallback for non-JSON errors
    }
    if (response.status === 409) {
      const error = new ConflictError(detail || "Operation in progress");
      if (payload) {
        error.status = typeof payload.status === "string" ? payload.status : undefined;
        error.reason = typeof payload.reason === "string" ? payload.reason : undefined;
        error.phase_message =
          payload.phase_message === null || typeof payload.phase_message === "string"
            ? (payload.phase_message as string | null)
            : undefined;
        if (Array.isArray(payload.allowed_commands)) {
          error.allowed_commands = payload.allowed_commands
            .filter((value): value is string => typeof value === "string");
        }
      }
      throw error;
    }
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

export function listSessions(repo?: string | null) {
  const effectiveRepo = repo !== undefined ? repo : getRepoContext();
  return request<{ items: PlanningSession[] }>(pathWithRepo("/api/sessions", effectiveRepo));
}

export function listRepos() {
  return request<{ cwd: { name: string; path: string }; items: RepoProfileSummary[] }>("/api/repos");
}

export function createChatSession(models?: { author_model?: string; critic_model?: string }) {
  return request<{ session: PlanningSession; opening?: string }>(withRepoQuery("/api/sessions"), {
    method: "POST",
    body: JSON.stringify(withRepoBody({ mode: "chat", ...models })),
  });
}

export function createRequirementsSession(
  requirements: string,
  models?: { author_model?: string; critic_model?: string },
) {
  return request<{ session: PlanningSession }>(withRepoQuery("/api/sessions"), {
    method: "POST",
    body: JSON.stringify(withRepoBody({ mode: "requirements", requirements, ...models })),
  });
}

export function getSession(sessionId: string) {
  return request<{
    session: PlanningSession;
    conversation: PlanningTurn[];
    plan_versions: PlanVersion[];
    current_plan: PlanVersion | null;
    tool_summary: { recent_tool_calls: string[] };
    round_metrics?: RoundMetric[];
  }>(withRepoQuery(`/api/sessions/${sessionId}`));
}

export function sendDiscoveryMessage(
  sessionId: string,
  message: string,
  models?: { author_model?: string; critic_model?: string },
) {
  const command_id = crypto.randomUUID();
  return request<{ result?: DiscoveryTurnResult; accepted?: boolean; mode?: string }>(`/api/sessions/${sessionId}/command`, {
    method: "POST",
    body: JSON.stringify({ command: "message", command_id, message, ...models }),
  });
}

export function runRound(
  sessionId: string,
  user_input?: string,
  models?: { author_model?: string; critic_model?: string },
) {
  const command_id = crypto.randomUUID();
  return request<{
    status: string;
    allowed_commands?: string[];
    idempotent_replay?: boolean;
  }>(`/api/sessions/${sessionId}/command`, {
    method: "POST",
    body: JSON.stringify({ command: "run_round", user_input, command_id, ...models }),
  });
}

export function listModels() {
  return request<{ items: ModelCatalogItem[] }>(withRepoQuery("/api/models"));
}

export function submitClarification(sessionId: string, answers: string[]) {
  return request<{ ok: boolean }>(`/api/sessions/${sessionId}/clarify`, {
    method: "POST",
    body: JSON.stringify({ answers }),
  });
}

export function approveSession(sessionId: string) {
  const command_id = crypto.randomUUID();
  return request<{ approved?: boolean; status?: string }>(`/api/sessions/${sessionId}/command`, {
    method: "POST",
    body: JSON.stringify({ command: "approve", command_id }),
  });
}

export function exportSession(sessionId: string) {
  const command_id = crypto.randomUUID();
  return request<{ files: Array<{ name: string; kind: string; url: string }> }>(`/api/sessions/${sessionId}/command`, {
    method: "POST",
    body: JSON.stringify({ command: "export", command_id }),
  });
}

export async function downloadFile(url: string, filename: string): Promise<void> {
  const response = await fetch(url);
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

export function deleteSession(sessionId: string) {
  return request<{ deleted: boolean }>(`/api/sessions/${sessionId}`, {
    method: "DELETE",
  });
}

export function stopSession(sessionId: string) {
  return request<{ stopped: boolean; reason?: string }>(`/api/sessions/${sessionId}/stop`, {
    method: "POST",
  });
}
