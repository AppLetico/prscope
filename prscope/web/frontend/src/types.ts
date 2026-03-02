export type SessionStatus =
  | "created"
  | "discovering"
  | "discovery"
  | "drafting"
  | "refining"
  | "converged"
  | "approved"
  | "exported";

export interface PlanningSession {
  id: string;
  repo_name: string;
  title: string;
  requirements: string;
  status: SessionStatus;
  seed_type: string;
  seed_ref: string | null;
  current_round: number;
  created_at: string;
  updated_at: string;
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

export type UIEvent =
  | { type: "thinking"; message: string }
  | { type: "tool_call"; name: string; path?: string; session_stage?: string }
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
    }
  | { type: "clarification_needed"; question: string; context?: string; source?: string };
