export type SessionStatus =
  | "draft"
  | "refining"
  | "converged"
  | "approved"
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
  completed_tool_call_groups?: ToolCallGroup[];
  created_at: string;
  updated_at: string;
  session_total_cost_usd?: number | null;
  max_prompt_tokens?: number | null;
}

export interface DraftTimingDiagnostics {
  warnings_total?: number;
  errors_total?: number;
  initial_draft_model?: string | null;
  initial_draft_provider?: string | null;
  initial_draft_fallback_model?: string | null;
  author_refine_model?: string | null;
  author_refine_provider?: string | null;
  author_refine_fallback_model?: string | null;
  author_refine_json_retries?: number;
  author_refine_fallbacks?: number;
  author_refine_last_failure_category?: string | null;
  critic_review_model?: string | null;
  critic_review_provider?: string | null;
  critic_review_fallback_model?: string | null;
  critic_review_json_retries?: number;
  critic_review_fallbacks?: number;
  critic_review_last_failure_category?: string | null;
  discovery_model?: string | null;
  discovery_provider?: string | null;
  memory_model?: string | null;
  memory_provider?: string | null;
  routing_decisions_total?: number;
  routing_heuristic_decisions?: number;
  routing_model_decisions?: number;
  routing_fallback_decisions?: number;
  route_author_chat_total?: number;
  route_lightweight_refine_total?: number;
  route_full_refine_total?: number;
  route_existing_feature_total?: number;
  [key: string]: string | number | boolean | null | undefined;
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
  sequence?: number;
}

export interface PlanVersion {
  id: number | null;
  session_id: string;
  round: number;
  plan_content: string;
  plan_json?: string | null;
  decision_graph_json?: string | null;
  followups_json?: string | null;
  decision_graph?: DecisionGraph | null;
  followups?: PlanFollowups | null;
  plan_sha: string;
  created_at: string;
  changed_sections?: string | null;
  diff_from_previous?: string | null;
  convergence_score?: number | null;
}

export interface DecisionNode {
  id: string;
  description: string;
  options?: string[] | null;
  value?: string | null;
  section: string;
  required?: boolean;
  concept?: string | null;
  evidence?: string[] | null;
}

export interface DecisionEdge {
  source: string;
  target: string;
  relation: string;
}

export interface DecisionGraph {
  nodes: Record<string, DecisionNode>;
  edges?: DecisionEdge[];
}

export interface FollowupQuestion {
  id: string;
  question: string;
  options?: string[] | null;
  target_sections: string[];
  concept: string;
  resolved?: boolean;
}

export interface FollowupSuggestion {
  id: string;
  suggestion: string;
}

export interface PlanFollowups {
  plan_version_id?: number | null;
  questions: FollowupQuestion[];
  suggestions: FollowupSuggestion[];
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
  recommendations?: string[];
}

export interface ToolCallEntry {
  id: number | string;
  call_id?: string;
  name: string;
  sessionStage?: string;
  path?: string;
  query?: string;
  status: "running" | "done";
  durationMs?: number;
  created_at?: string;
}

export interface ToolCallGroup {
  sequence: number;
  created_at: string;
  tools: ToolCallEntry[];
}

export interface LiveActivityEntry {
  id: string;
  kind: "thought" | "update" | "tool";
  message: string;
  stage?: string;
  status?: "running" | "done" | "start" | "complete" | "failed";
  created_at: string;
  count?: number;
}

export interface RoundMetric {
  round: number;
  major_issues?: number | null;
  minor_issues?: number | null;
  critic_confidence?: number | null;
  convergence_score?: number | null;
  call_cost_usd?: number | null;
  issue_graph_summary?: {
    open_total: number;
    root_open: number;
    resolved_total: number;
    unresolved_dependency_chains?: number;
  } | null;
}

export type IssueType = "architecture" | "ambiguity" | "correctness" | "performance";

export interface SessionSnapshotSummary {
  session_id: string;
  updated_at: string;
  path: string;
}

export interface IssueGraphNode {
  id: string;
  description: string;
  status: "open" | "resolved";
  raised_round: number;
  resolved_round?: number | null;
  resolution_source?: "review" | "lightweight" | null;
  severity?: "major" | "minor" | "info";
  source?: "critic" | "validation" | "inference";
  issue_type?: IssueType | null;
  tags?: string[];
  related_decision_ids?: string[];
}

export interface IssueGraphEdge {
  source: string;
  target: string;
  relation: "causes" | "depends_on" | "duplicate";
}

export interface IssueGraphSnapshot {
  nodes: IssueGraphNode[];
  edges: IssueGraphEdge[];
  duplicate_alias?: Record<string, string>;
  summary?: {
    open_total?: number;
    root_open?: number;
    resolved_total?: number;
    open_major?: number;
    open_minor?: number;
    open_info?: number;
    unresolved_dependency_chains?: number;
  };
}

export interface SessionStateSnapshot {
  session_id: string;
  updated_at: string;
  open_issues?: Array<{ id: string; description: string; status: string }>;
  issue_graph?: IssueGraphSnapshot | null;
  constraint_eval?: { constraint_violations?: string[] } | null;
  author_prompt_tokens?: number;
  author_completion_tokens?: number;
  critic_prompt_tokens?: number;
  critic_completion_tokens?: number;
  round_cost_usd?: number;
}

export interface ImpactIssueCluster {
  root_issue_id: string;
  root_issue: string;
  severity: "major" | "minor" | "info";
  issue_ids: string[];
  symptom_issue_count: number;
  affected_plan_sections: string[];
  suggested_action: string;
}

export interface ImpactDecision {
  decision_id: string;
  linked_issue_ids: string[];
  decision_pressure: number;
  pressure_breakdown: {
    major: number;
    minor: number;
    info: number;
    clusters: number;
  };
  risk_level: "low" | "medium" | "high";
  highest_severity: "major" | "minor" | "info";
  dominant_cluster: ImpactIssueCluster;
  issue_clusters: ImpactIssueCluster[];
}

export interface ReconsiderationCandidate {
  decision_id: string;
  reason: string;
  decision_pressure: number;
  dominant_cluster: ImpactIssueCluster;
  suggested_action: string;
  recently_changed: boolean;
  eligible: boolean;
}

export interface ArchitectureImpactView {
  decisions: ImpactDecision[];
  reconsideration_candidates: ReconsiderationCandidate[];
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

export type UIEventBase = { session_version?: number };

export type UIEvent = UIEventBase & (
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
      completed_tool_call_groups: ToolCallGroup[];
      active_command_id?: string | null;
    }
  | { type: "tool_update"; tool: ToolCallEntry }
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
      session_stage?: string;
      llm_call_latency_ms?: number;
      memory_block?: string;
      session_total_usd?: number;
      max_prompt_tokens?: number;
      context_window_tokens?: number;
      context_usage_ratio?: number;
    }
  | {
      type: "clarification_needed";
      question: string;
      context?: string;
      source?: string;
      recommendations?: string[];
    }
  | {
      type: "setup_progress";
      step: string;
      draft_stage?: string;
      session_stage?: string;
      memory_elapsed_s?: number;
      planner_elapsed_s?: number;
      cache_hit?: boolean;
      complexity?: string;
    }
  | {
      type: "phase_timing";
      session_stage?: string;
      state?: "start" | "complete" | "failed";
      elapsed_ms?: number;
    }
  | { type: "discovery_ready"; opening: string }
);
