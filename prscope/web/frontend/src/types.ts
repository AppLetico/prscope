export type SessionStatus =
  | "created"
  | "preparing"
  | "discovering"
  | "drafting"
  | "refining"
  | "converged"
  | "approved"
  | "exported"
  | "error";

export interface PlanningSession {
  id: string;
  repo_name: string;
  title: string;
  requirements: string;
  author_model?: string | null;
  critic_model?: string | null;
  status: SessionStatus;
  seed_type: string;
  seed_ref: string | null;
  current_round: number;
  pending_questions?: DiscoveryQuestion[] | null;
  phase_message?: string | null;
  is_processing?: boolean;
  active_tool_calls?: ToolCallEntry[];
  created_at: string;
  updated_at: string;
  session_total_cost_usd?: number | null;
  max_prompt_tokens?: number | null;
}

export interface PlanningTurn {
  id: number | null;
  session_id: string;
  role: string;
  content: string;
  round: number;
  major_issues_remaining?: number | null;
  minor_issues_remaining?: number | null;
  hard_constraint_violations?: string[] | null;
  parse_error?: string | null;
  created_at: string;
}

export interface PlanVersion {
  id: number | null;
  session_id: string;
  round: number;
  plan_content: string;
  plan_sha: string;
  created_at: string;
  diff_from_previous?: string | null;
  convergence_score?: number | null;
}

export interface QuestionOption {
  letter: string;
  text: string;
  is_other: boolean;
}

export interface DiscoveryQuestion {
  index: number;
  text: string;
  options: QuestionOption[];
}

export interface DiscoveryTurnResult {
  reply: string;
  complete: boolean;
  summary?: string | null;
  questions: DiscoveryQuestion[];
}

export interface ClarificationPrompt {
  question: string;
  context?: string;
  source?: string;
}

export interface ToolCallEntry {
  id: number;
  name: string;
  sessionStage?: string;
  path?: string;
  query?: string;
  status: "running" | "done";
  durationMs?: number;
}

export interface ModelCatalogItem {
  provider: string;
  model_id: string;
  available: boolean;
  unavailable_reason?: string | null;
  required_env_keys: string[];
  context_window_tokens?: number | null;
}

export interface RepoProfileSummary {
  name: string;
  path: string;
}

export type UIEvent =
  | { type: "thinking"; message: string }
  | {
      type: "session_state";
      v: 1;
      status: SessionStatus;
      phase_message: string | null;
      is_processing: boolean;
      current_round: number;
      pending_questions: DiscoveryQuestion[] | null;
      active_tool_calls: ToolCallEntry[];
    }
  | {
      type: "tool_call";
      name: string;
      path?: string;
      query?: string;
      session_stage?: string;
    }
  | {
      type: "tool_result";
      name: string;
      session_stage?: string;
      duration_ms?: number;
    }
  | {
      type: "context_compaction";
      enabled: boolean;
      reason?: string;
    }
  | { type: "plan_ready"; round?: number; initial_draft?: boolean; saved_at_unix_s?: number }
  | { type: "complete"; message?: string }
  | { type: "error"; message: string }
  | { type: "warning"; message: string }
  | {
      type: "token_usage";
      model: string;
      prompt_tokens: number;
      completion_tokens: number;
      call_cost_usd: number;
      session_total_usd?: number;
      max_prompt_tokens?: number;
      context_window_tokens?: number;
      context_usage_ratio?: number;
    }
  | { type: "clarification_needed"; question: string; context?: string; source?: string }
  | { type: "setup_progress"; step: string }
  | { type: "discovery_ready"; opening: string };
