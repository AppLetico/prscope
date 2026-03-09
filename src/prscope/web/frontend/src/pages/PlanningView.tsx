import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { useLocation, useParams, useNavigate } from "react-router-dom";
import { ActionBar } from "../components/ActionBar";
import { ChatPanel } from "../components/ChatPanel";
import { PlanPanel } from "../components/PlanPanel";
import { ResizableLayout } from "../components/ResizableLayout";
import { useSessionEvents } from "../hooks/useSessionEvents";
import {
  answerPlanFollowup,
  approveSession,
  ConflictError,
  deleteSession,
  downloadFile,
  exportSession,
  getActiveRepoContext,
  getStoredModelSelection,
  getSessionSnapshot,
  listModels,
  getSession,
  runRound,
  sendDiscoveryMessage,
  setStoredModelSelection,
  stopSession,
  submitClarification,
} from "../lib/api";
import { cleanPlanTitle } from "../lib/planTitle";
import type { ClarificationPrompt, LiveActivityEntry, PlanningSession, UIEvent } from "../types";
import { AlertCircle, Check, Loader2 } from "lucide-react";
import {
  formatPhaseTimingLabel,
  formatToolActivityLabel,
  INITIAL_TIMELINE_STATE,
  timelineReducer,
  upsertLiveActivity,
} from "../components/chatPanelUtils";

function normalizeActivityText(value: string): string {
  return value.replace(/\.{3,}\s*$/u, "").trim();
}

export function PlanningViewPage() {
  const { id = "" } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const [tl, dispatchTl] = useReducer(timelineReducer, INITIAL_TIMELINE_STATE);
  const [thinkingMessage, setThinkingMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [authorModel, setAuthorModel] = useState("");
  const [criticModel, setCriticModel] = useState("");
  const [sessionCostUsd, setSessionCostUsd] = useState<number>(0);
  const [maxPromptTokens, setMaxPromptTokens] = useState<number>(0);
  const [contextWindowTokens, setContextWindowTokens] = useState<number | null>(null);
  const [contextUsageRatio, setContextUsageRatio] = useState<number | null>(null);
  const [contextCompactionEnabled, setContextCompactionEnabled] = useState<boolean>(false);
  const [pendingClarification, setPendingClarification] = useState<ClarificationPrompt | null>(null);
  const [showCritiquePrompt, setShowCritiquePrompt] = useState(false);
  const [setupSteps, setSetupSteps] = useState<string[]>([]);
  const [initialSetupDone, setInitialSetupDone] = useState(false);
  const [deletingOrphan, setDeletingOrphan] = useState(false);
  const [liveActivities, setLiveActivities] = useState<LiveActivityEntry[]>([]);
  const [sessionState, setSessionState] = useState<{
    status: PlanningSession["status"];
    current_round: number;
    pending_questions: PlanningSession["pending_questions"];
    phase_message: string | null;
    is_processing: boolean;
  } | null>(null);
  const [chatInputAppendRequest, setChatInputAppendRequest] = useState<{ id: number; text: string } | null>(null);
  const [latestResponseMode, setLatestResponseMode] = useState<"author_chat" | "refine_round" | null>(null);
  const lastEventAtMs = useRef<number>(0);
  const lastRefetchAtMs = useRef<number>(0);
  const lastContextNoticeAtMs = useRef<number>(0);
  const lastVersionSeen = useRef<number>(0);
  const queryClient = useQueryClient();

  useEffect(() => {
    if (lastEventAtMs.current === 0) {
      lastEventAtMs.current = Date.now();
    }
  }, []);

  useEffect(() => {
    // Session route changed: clear transient setup UI so old events/steps cannot bleed through.
    setSetupSteps([]);
    setInitialSetupDone(false);
    setLiveActivities([]);
    lastVersionSeen.current = 0;
  }, [id]);

  const appendLiveActivity = useCallback((activity: LiveActivityEntry) => {
    setLiveActivities((prev) => upsertLiveActivity(prev, activity));
  }, []);

  const isSessionNotFoundError = useCallback((err: unknown) => {
    return String(err).toLowerCase().includes("session not found");
  }, []);

  const sessionQuery = useQuery({
    queryKey: ["session", id],
    queryFn: () => getSession(id),
    enabled: Boolean(id),
    retry: (failureCount, err) => {
      if (isSessionNotFoundError(err)) return false;
      return failureCount < 3;
    },
  });
  const modelQuery = useQuery({
    queryKey: ["models"],
    queryFn: listModels,
    staleTime: 60_000,
  });
  const session = sessionQuery.data?.session;
  const hasCurrentPlan = Boolean((sessionQuery.data?.current_plan?.plan_content ?? "").trim());
  const shouldPollSnapshot =
    Boolean(id)
    && (
      session?.status === "refining"
      || session?.status === "converged"
      || session?.status === "approved"
      || hasCurrentPlan
    );
  const snapshotQuery = useQuery({
    queryKey: ["session-snapshot", id],
    queryFn: () => getSessionSnapshot(id),
    enabled: shouldPollSnapshot,
    retry: false,
  });
  const turns = useMemo(
    () => sessionQuery.data?.conversation ?? [],
    [sessionQuery.data?.conversation],
  );
  const modelItems = modelQuery.data?.items ?? [];
  const availableModelItems = modelItems.filter((item) => item.available);
  const fallbackModel = availableModelItems[0]?.model_id || modelItems[0]?.model_id || "";
  const storedModels = useMemo(
    () => getStoredModelSelection(session?.repo_name),
    [session?.repo_name],
  );
  const selectedAuthorModel =
    authorModel || session?.author_model || storedModels.author_model || fallbackModel;
  const selectedCriticModel =
    criticModel || session?.critic_model || storedModels.critic_model || fallbackModel;
  const refetchSession = sessionQuery.refetch;
  const questions = sessionState?.pending_questions ?? [];
  const phaseMessage = sessionState?.phase_message
    ? normalizeActivityText(sessionState.phase_message)
    : null;
  const isProcessing = sessionState?.is_processing ?? false;
  const effectiveStatus = sessionState?.status ?? session?.status;
  const effectiveRound = sessionState?.current_round ?? session?.current_round ?? 0;

  useEffect(() => {
    if (!session) return;
    setSessionState({
      status: session.status,
      current_round: session.current_round,
      pending_questions: session.pending_questions ?? null,
      phase_message: session.phase_message ?? null,
      is_processing: Boolean(session.is_processing),
    });
    dispatchTl({
      type: "session_state",
      groups: session.completed_tool_call_groups ?? [],
      activeTools: session.active_tool_calls ?? [],
    });
    if (session.session_total_cost_usd != null) setSessionCostUsd(session.session_total_cost_usd);
    if (session.max_prompt_tokens != null) setMaxPromptTokens(session.max_prompt_tokens);
    if ((sessionQuery.data?.conversation?.length ?? 0) > 0) {
      setInitialSetupDone(true);
    }
  }, [
    session,
    session?.id,
    session?.status,
    session?.current_round,
    session?.pending_questions,
    session?.phase_message,
    session?.is_processing,
    session?.active_tool_calls,
    session?.completed_tool_call_groups,
    session?.session_total_cost_usd,
    session?.max_prompt_tokens,
    sessionQuery.data?.conversation?.length,
  ]);

  useEffect(() => {
    if (hasCurrentPlan || turns.length > 0 || effectiveStatus === "refining" || effectiveStatus === "converged") {
      setInitialSetupDone(true);
    }
  }, [effectiveStatus, hasCurrentPlan, turns.length]);

  useEffect(() => {
    if (!id || initialSetupDone || hasCurrentPlan) {
      return;
    }
    const intervalId = window.setInterval(() => {
      void refetchSession();
    }, 1500);
    return () => window.clearInterval(intervalId);
  }, [hasCurrentPlan, id, initialSetupDone, refetchSession]);

  useEffect(() => {
    dispatchTl({ type: "sync_turns", turns });
  }, [turns]);

  const handleEvent = useCallback((event: UIEvent) => {
    lastEventAtMs.current = Date.now();

    const v = event.session_version;
    if (typeof v === "number") {
      if (v <= lastVersionSeen.current) return;
      lastVersionSeen.current = v;
    }

    if (event.type === "session_state") {
      // SSE reconnects emit an initial unversioned snapshot. Once we've seen
      // versioned events, ignore unversioned snapshots to avoid stale rewinds.
      if (typeof event.session_version !== "number" && lastVersionSeen.current > 0) {
        return;
      }
      setSessionState({
        status: event.status,
        current_round: event.current_round,
        pending_questions: event.pending_questions,
        phase_message: event.phase_message,
        is_processing: event.is_processing,
      });
      dispatchTl({
        type: "session_state",
        groups: event.completed_tool_call_groups,
        activeTools: event.active_tool_calls,
      });
      if (!event.is_processing || (event.pending_questions?.length ?? 0) > 0) {
        setThinkingMessage(null);
      }
      // If we receive a clean draft state (not processing, no pending questions,
      // no active/completed tools), mark initial setup as done to skip the setup modal.
      // This handles page reloads after reset and avoids waiting for setup_progress events
      // that won't come for already-initialized sessions.
      if (
        event.status === "draft"
        && !event.is_processing
        && (event.pending_questions?.length ?? 0) === 0
        && event.active_tool_calls.length === 0
        && event.completed_tool_call_groups.length === 0
      ) {
        setInitialSetupDone(true);
      }
      return;
    }
    if (event.type === "thinking") {
      setThinkingMessage(event.message);
      appendLiveActivity({
        id: `thought:${normalizeActivityText(event.message).toLowerCase()}`,
        kind: "thought",
        message: normalizeActivityText(event.message) || "Thinking",
        created_at: new Date().toISOString(),
        status: "running",
      });
      return;
    }
    if (event.type === "tool_update") {
      dispatchTl({ type: "tool_update", tool: event.tool });
      appendLiveActivity({
        id: `tool:${String(event.tool.call_id ?? event.tool.id)}`,
        kind: "tool",
        message: formatToolActivityLabel(event.tool),
        stage: event.tool.sessionStage,
        status: event.tool.status,
        created_at: event.tool.created_at ?? new Date().toISOString(),
      });
      return;
    }
    if (event.type === "routing_decision") {
      appendLiveActivity({
        id: `route:${event.route}:${event.investigation_reason ?? "none"}`,
        kind: "update",
        message: event.evidence_refresh_used
          ? `Routing chose ${event.route} after targeted evidence refresh`
          : `Routing chose ${event.route}`,
        stage: "refinement",
        status: "complete",
        created_at: new Date().toISOString(),
      });
      return;
    }
    if (event.type === "refinement_investigation") {
      appendLiveActivity({
        id: `investigation:${event.trigger_reason ?? "skip"}`,
        kind: "update",
        message: event.used
          ? `Checked more repo evidence${event.trigger_reason ? ` for ${event.trigger_reason}` : ""}`
          : "Skipped extra repo investigation",
        stage: event.session_stage ?? "author_refine",
        status: event.used ? "running" : "complete",
        created_at: new Date().toISOString(),
      });
      return;
    }
    if (event.type === "complete") {
      dispatchTl({ type: "complete" });
      setThinkingMessage(null);
      setWarnings([]);
      setInitialSetupDone(true);
      const now = Date.now();
      if (now - lastRefetchAtMs.current >= 800) {
        lastRefetchAtMs.current = now;
        void refetchSession();
      }
      return;
    }
    if (event.type === "plan_ready") {
      dispatchTl({ type: "plan_ready" });
      setThinkingMessage(null);
      setInitialSetupDone(true);
      if (event.round === 0) {
        setShowCritiquePrompt(true);
      }
      const now = Date.now();
      if (now - lastRefetchAtMs.current >= 800) {
        lastRefetchAtMs.current = now;
        void refetchSession();
      }
      void queryClient.invalidateQueries({ queryKey: ["session-snapshot", id] });
      return;
    }
    if (event.type === "error") {
      setThinkingMessage(null);
      setError(event.message);
      return;
    }
    if (event.type === "warning") {
      const normalized = event.message.toLowerCase();
      // Internal gate failures are quality signals, not user-visible errors — suppress them.
      const isInternalGate = normalized.includes("explorer gate")
        || normalized.includes("grounding validation")
        || normalized.includes("fallback plan")
        || normalized.includes("fallback draft")
        || normalized.includes("fallback content");
      if (isInternalGate) return;
      const looksLikeContextSummary = normalized.includes("context compaction enabled")
        || normalized.includes("chat context summarized")
        || normalized.includes("summarized chat context");
      if (looksLikeContextSummary) {
        setContextCompactionEnabled(true);
        const now = Date.now();
        if (now - lastContextNoticeAtMs.current > 2500) {
          lastContextNoticeAtMs.current = now;
          setWarnings((prev) => [
            ...prev,
            "Context got long, so I summarized earlier messages to keep working smoothly.",
          ].slice(-12));
        }
      } else {
        setWarnings((prev) => [...prev, event.message].slice(-12));
      }
      return;
    }
    if (event.type === "context_compaction") {
      setContextCompactionEnabled(event.enabled);
      return;
    }
    if (event.type === "token_usage") {
      setSessionCostUsd(event.session_total_usd ?? sessionCostUsd);
      setMaxPromptTokens(event.max_prompt_tokens ?? maxPromptTokens);
      if (event.context_window_tokens !== undefined) {
        setContextWindowTokens(event.context_window_tokens);
      }
      if (event.context_usage_ratio !== undefined) {
        setContextUsageRatio(event.context_usage_ratio);
      } else if (event.context_window_tokens && event.max_prompt_tokens) {
        setContextUsageRatio(event.max_prompt_tokens / event.context_window_tokens);
      }
      return;
    }
    if (event.type === "clarification_needed") {
      setPendingClarification({
        question: event.question,
        context: event.context,
        source: event.source,
        recommendations: event.recommendations,
      });
      return;
    }
    if (event.type === "setup_progress") {
      setSetupSteps((prev) => [...prev, event.step]);
      appendLiveActivity({
        id: `setup:${event.session_stage ?? "setup"}:${event.draft_stage ?? normalizeActivityText(event.step).toLowerCase()}`,
        kind: "update",
        message: normalizeActivityText(event.step),
        stage: event.draft_stage ?? event.session_stage,
        status: "running",
        created_at: new Date().toISOString(),
      });
      if ((sessionState?.pending_questions?.length ?? 0) === 0 && !pendingClarification) {
        setThinkingMessage(normalizeActivityText(event.step));
      }
      return;
    }
    if (event.type === "phase_timing") {
      appendLiveActivity({
        id: `phase:${event.session_stage ?? "activity"}:${event.state ?? "update"}`,
        kind: "update",
        message: formatPhaseTimingLabel(event.session_stage, event.state, event.elapsed_ms),
        stage: event.session_stage,
        status: event.state,
        created_at: new Date().toISOString(),
      });
      return;
    }
    if (event.type === "discovery_ready") {
      setInitialSetupDone(true);
      lastRefetchAtMs.current = Date.now();
      void refetchSession();
      return;
    }
  }, [appendLiveActivity, id, maxPromptTokens, pendingClarification, queryClient, refetchSession, sessionCostUsd, sessionState?.pending_questions]);

  const handleReconnect = useCallback(() => {
    lastRefetchAtMs.current = Date.now();
    void refetchSession();
    if (shouldPollSnapshot) {
      void queryClient.invalidateQueries({ queryKey: ["session-snapshot", id] });
    }
  }, [id, queryClient, refetchSession, shouldPollSnapshot]);

  useSessionEvents(id, handleEvent, Boolean(session), handleReconnect);

  const submitMessage = async (text: string) => {
    try {
      setError(null);
      setThinkingMessage("Thinking...");
      setLiveActivities([]);
      const status = effectiveStatus;
      const models = {
        author_model: selectedAuthorModel || undefined,
        critic_model: selectedCriticModel || undefined,
      };
      if (status === "draft") {
        await sendDiscoveryMessage(id, text, models);
        setLatestResponseMode(null);
        const refreshed = await sessionQuery.refetch();
        if (!refreshed.data?.session.is_processing) {
          setThinkingMessage(null);
        }
      } else if (status === "refining" || status === "converged") {
        const response = await sendDiscoveryMessage(id, text, models);
        if (response.mode === "author_chat" || response.mode === "refine_round") {
          setLatestResponseMode(response.mode);
        } else {
          setLatestResponseMode(null);
        }
        const refreshed = await sessionQuery.refetch();
        if (!refreshed.data?.session.is_processing) {
          setThinkingMessage(null);
        }
      } else {
        setError("Messages are only accepted during draft, refinement, or converged feedback.");
        return;
      }
      if (sessionQuery.data?.session.repo_name) {
        setStoredModelSelection(models, sessionQuery.data.session.repo_name);
      }
    } catch (err) {
      if (err instanceof ConflictError) {
        setError(err.phase_message || err.message);
        return;
      }
      setError(String(err));
    }
  };

  const handleFollowupAnswer = async (followupId: string, answer: string) => {
    const currentPlan = sessionQuery.data?.current_plan;
    if (!currentPlan?.id) {
      throw new Error("No current plan version available.");
    }
    try {
      setError(null);
      setThinkingMessage("Applying follow-up answer...");
      setLiveActivities([]);
      await answerPlanFollowup(id, {
        plan_version_id: currentPlan.id,
        followup_id: followupId,
        followup_answer: answer,
        author_model: selectedAuthorModel || undefined,
        critic_model: selectedCriticModel || undefined,
      });
      const refreshed = await sessionQuery.refetch();
      if (!refreshed.data?.session.is_processing) {
        setThinkingMessage(null);
      }
    } catch (err) {
      if (err instanceof ConflictError) {
        setError(err.message);
      } else {
        setError(String(err));
      }
      setThinkingMessage(null);
      throw err;
    }
  };

  const handleClarificationSubmit = async (answer: string) => {
    try {
      setError(null);
      const normalized = answer.trim();
      await submitClarification(id, normalized ? [normalized] : []);
      setPendingClarification(null);
    } catch (err) {
      setError(String(err));
    }
  };

  const onCritique = async () => {
    try {
      setError(null);
      setShowCritiquePrompt(false);
      setLiveActivities([]);
      await runRound(id, undefined, {
        author_model: selectedAuthorModel || undefined,
        critic_model: selectedCriticModel || undefined,
      });
    } catch (err) {
      setError(String(err));
    } finally {
      await sessionQuery.refetch();
    }
  };

  const onApprove = async () => {
    try {
      setError(null);
      await approveSession(id);
      await sessionQuery.refetch();
    } catch (err) {
      setError(String(err));
    }
  };

  const onExport = async () => {
    try {
      setError(null);
      const response = await exportSession(id);
      for (const file of response.files) {
        await downloadFile(file.url, file.name);
      }
      await sessionQuery.refetch();
    } catch (err) {
      setError(String(err));
    }
  };

  const onDelete = async () => {
    if (!window.confirm("Delete this plan? This cannot be undone.")) return;
    try {
      setError(null);
      await deleteSession(id);
      navigate("/", { replace: true });
    } catch (err) {
      setError(String(err));
    }
  };

  const onStop = async () => {
    try {
      setError(null);
      await stopSession(id);
      await sessionQuery.refetch();
    } catch (err) {
      setError(String(err));
    }
  };

  const contextPercent = contextUsageRatio !== null ? contextUsageRatio * 100 : null;
  const planContent = sessionQuery.data?.current_plan?.plan_content ?? "";
  const currentPlanFollowups = sessionQuery.data?.current_plan?.followups ?? null;
  const snapshot = snapshotQuery.data?.snapshot;
  const openIssuesCount = snapshot?.issue_graph?.summary?.open_total
    ?? (Array.isArray(snapshot?.open_issues) ? snapshot.open_issues.length : 0);
  const constraintViolationsCount = Array.isArray(snapshot?.constraint_eval?.constraint_violations)
    ? snapshot?.constraint_eval?.constraint_violations.length
    : 0;
  const versionedTitle = useMemo(() => {
    if (!session) return "";
    return cleanPlanTitle(session.title);
  }, [session]);
  const activeRepoContext = useMemo(() => {
    const params = new URLSearchParams(location.search);
    return params.get("repo") ?? getActiveRepoContext();
  }, [location.search]);
  const sessionsHref = activeRepoContext
    ? `/?repo=${encodeURIComponent(activeRepoContext)}`
    : "/";
  const newPlanHref = activeRepoContext
    ? `/new?repo=${encodeURIComponent(activeRepoContext)}`
    : "/new";
  const deleteStaleSession = async () => {
    try {
      setDeletingOrphan(true);
      // Use delete by session id without repo scoping so orphan records are removable.
      await deleteSession(id);
      window.location.href = "/";
    } catch {
      // If already gone, still recover the user back to sessions.
      window.location.href = "/";
    } finally {
      setDeletingOrphan(false);
    }
  };

  if (!session && !sessionQuery.isLoading) {
    return (
      <main className="h-screen flex items-center justify-center bg-zinc-950 px-6">
        <div className="w-full max-w-lg rounded-2xl border border-zinc-800 bg-zinc-900/70 p-8 shadow-xl">
          <div className="mb-5 inline-flex h-10 w-10 items-center justify-center rounded-lg bg-zinc-800/80 text-zinc-300">
            <AlertCircle className="h-5 w-5" />
          </div>
          <h1 className="text-xl font-semibold text-zinc-100">Session no longer exists</h1>
          <p className="mt-2 text-sm leading-6 text-zinc-400">
            This usually happens after `make reset` or deleting sessions. Open your sessions list to continue.
          </p>
          {activeRepoContext ? (
            <p className="mt-3 text-xs text-zinc-500">
              Repo context: <span className="font-mono text-zinc-400">{activeRepoContext}</span>
            </p>
          ) : null}
          <div className="mt-7 flex flex-wrap gap-3">
            <a
              href={sessionsHref}
              className="inline-flex items-center rounded-md bg-indigo-500 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-400"
            >
              Back to sessions
            </a>
            <a
              href={newPlanHref}
              className="inline-flex items-center rounded-md border border-zinc-700 bg-zinc-900 px-4 py-2 text-sm font-medium text-zinc-200 transition-colors hover:bg-zinc-800"
            >
              Start new plan
            </a>
            <button
              type="button"
              onClick={() => void deleteStaleSession()}
              disabled={deletingOrphan}
              className="inline-flex items-center gap-2 rounded-md border border-rose-700/60 bg-rose-900/20 px-4 py-2 text-sm font-medium text-rose-200 transition-colors hover:bg-rose-900/35 disabled:cursor-not-allowed disabled:opacity-70"
            >
              {deletingOrphan ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
              {deletingOrphan ? "Deleting..." : "Delete stale session"}
            </button>
          </div>
        </div>
      </main>
    );
  }

  const isInInitialSetup =
    !initialSetupDone
    && effectiveStatus === "draft"
    && turns.length === 0
    && !(sessionQuery.data?.current_plan?.plan_content ?? "").trim();

  if (isInInitialSetup) {
    return (
      <main className="h-screen flex flex-col bg-zinc-950 overflow-hidden">
        {session ? (
          <ActionBar
            repoName={session.repo_name}
            title={versionedTitle}
            round={session.current_round}
            status={session.status}
            convergenceScore={undefined}
            sessionCostUsd={0}
            maxPromptTokens={0}
            contextWindowTokens={null}
            contextPercent={null}
            contextCompactionEnabled={false}
            routingDiagnostics={sessionQuery.data?.draft_timing ?? null}
          />
        ) : null}
        <div className="flex-1 flex flex-col items-center justify-center p-8">
          <div className="w-full max-w-md rounded-xl border border-zinc-800 bg-zinc-900/50 p-8 shadow-lg">
            <h2 className="text-lg font-semibold text-zinc-100 mb-1">Setting up your session</h2>
            <p className="text-sm text-zinc-500 mb-6">
              Preparing codebase memory so the AI can understand your project. This is a one-time setup per new repo and may take a minute.
            </p>
            {setupSteps.length > 0 ? (
              <ul className="space-y-3">
                {setupSteps.map((step, i) => (
                  <li
                    key={i}
                    className="flex items-center gap-3 text-sm text-zinc-300"
                  >
                    {i < setupSteps.length - 1 ? (
                      <Check className="w-4 h-4 shrink-0 text-emerald-500" />
                    ) : (
                      <Loader2 className="w-4 h-4 shrink-0 text-indigo-400 animate-spin" />
                    )}
                    <span>{step}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="flex items-center gap-2 text-zinc-500 text-sm">
                <Loader2 className="w-4 h-4 shrink-0 animate-spin" />
                Connecting...
              </div>
            )}
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="h-screen flex flex-col bg-zinc-950 overflow-hidden">
      {session ? (
        <>
          <ActionBar
            repoName={session.repo_name}
            title={versionedTitle}
            round={effectiveRound}
            status={effectiveStatus ?? session.status}
            convergenceScore={sessionQuery.data?.current_plan?.convergence_score ?? undefined}
            sessionCostUsd={sessionCostUsd}
            maxPromptTokens={maxPromptTokens}
            contextWindowTokens={contextWindowTokens}
            contextPercent={contextPercent}
            contextCompactionEnabled={contextCompactionEnabled}
            routingDiagnostics={sessionQuery.data?.draft_timing ?? null}
            onDelete={onDelete}
            roundMetrics={sessionQuery.data?.round_metrics}
          />
          
          {error && (
            <div className="bg-rose-500/10 border-b border-rose-500/20 px-6 py-3 flex items-center gap-3 text-rose-400 text-sm z-40">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <p>{error}</p>
              <button 
                onClick={() => setError(null)}
                className="ml-auto text-rose-400 hover:text-rose-300"
              >
                Dismiss
              </button>
            </div>
          )}

          <div className="flex-1 min-h-0 relative">
            <ResizableLayout
              left={(
                <PlanPanel
                  content={planContent}
                  decisionGraph={sessionQuery.data?.current_plan?.decision_graph ?? null}
                  impactView={sessionQuery.data?.impact_view ?? null}
                  status={effectiveStatus ?? session.status}
                  isProcessing={sessionState?.is_processing ?? false}
                  canExport={Boolean(sessionQuery.data?.current_plan)}
                  onExport={() => void onExport()}
                  onAppendIssuePrompt={(text) => {
                    setChatInputAppendRequest({ id: Date.now(), text });
                  }}
                  health={{
                    snapshotUpdatedAt: snapshot?.updated_at,
                    openIssuesCount,
                    constraintViolationsCount,
                    constraintViolations: snapshot?.constraint_eval?.constraint_violations ?? [],
                    issueGraph: snapshot?.issue_graph ?? null,
                  }}
                />
              )}
              right={
                <ChatPanel
                  timeline={tl.timeline}
                  questions={questions}
                  activeToolCalls={tl.activeTools}
                  liveActivities={liveActivities}
                  thinkingMessage={thinkingMessage}
                  phaseMessage={phaseMessage}
                  warnings={warnings}
                  pendingClarification={pendingClarification}
                  planFollowups={currentPlanFollowups}
                  inputDisabled={isProcessing}
                  isProcessing={isProcessing}
                  authorModel={selectedAuthorModel}
                  criticModel={selectedCriticModel}
                  modelOptions={modelItems}
                  onAuthorModelChange={(modelId) => {
                    setAuthorModel(modelId);
                    if (session.repo_name) {
                      setStoredModelSelection(
                        { author_model: modelId, critic_model: selectedCriticModel || undefined },
                        session.repo_name,
                      );
                    }
                  }}
                  onCriticModelChange={(modelId) => {
                    setCriticModel(modelId);
                    if (session.repo_name) {
                      setStoredModelSelection(
                        { author_model: selectedAuthorModel || undefined, critic_model: modelId },
                        session.repo_name,
                      );
                    }
                  }}
                  canCritique={!isProcessing && (effectiveStatus === "refining" || effectiveStatus === "converged")}
                  canApprove={!isProcessing && effectiveStatus === "converged"}
                  critiquePending={showCritiquePrompt}
                  contextPercent={contextPercent}
                  contextWindowTokens={contextWindowTokens}
                  onCritique={() => void onCritique()}
                  onApprove={() => void onApprove()}
                  onStop={() => void onStop()}
                  onSubmit={submitMessage}
                  onSubmitFollowup={handleFollowupAnswer}
                  onSubmitClarification={handleClarificationSubmit}
                  externalInputAppend={chatInputAppendRequest}
                  latestResponseMode={latestResponseMode}
                />
              }
            />
          </div>
        </>
      ) : (
        <div className="h-full flex items-center justify-center">
          <div className="flex items-center gap-3 text-zinc-500">
            <div className="w-4 h-4 border-2 border-zinc-500 border-t-transparent rounded-full animate-spin"></div>
            Loading session...
          </div>
        </div>
      )}
    </main>
  );
}
