import { useQuery } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams, useNavigate } from "react-router-dom";
import { ActionBar } from "../components/ActionBar";
import { ChatPanel } from "../components/ChatPanel";
import { PlanPanel } from "../components/PlanPanel";
import { ResizableLayout } from "../components/ResizableLayout";
import { useSessionEvents } from "../hooks/useSessionEvents";
import {
  approveSession,
  ConflictError,
  deleteSession,
  downloadFile,
  exportSession,
  getStoredModelSelection,
  getDiff,
  listModels,
  getSession,
  runRound,
  sendDiscoveryMessage,
  setStoredModelSelection,
  stopSession,
  submitClarification,
} from "../lib/api";
import { cleanPlanTitle } from "../lib/planTitle";
import type { ClarificationPrompt, PlanningSession, ToolCallEntry, UIEvent } from "../types";
import { AlertCircle, Check, Loader2 } from "lucide-react";
import {
  isTerminalCompletedSnapshot,
  shouldFinalizeFromTerminalSnapshot,
} from "./planningViewUtils";

function normalizeActivityText(value: string): string {
  return value.replace(/\.{3,}\s*$/u, "").trim();
}

export function PlanningViewPage() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const [activeToolCalls, setActiveToolCalls] = useState<ToolCallEntry[]>([]);
  const [toolCallGroups, setToolCallGroups] = useState<ToolCallEntry[][]>([]);
  const [thinkingMessage, setThinkingMessage] = useState<string | null>(null);
  const [showDiff, setShowDiff] = useState(false);
  const [diff, setDiff] = useState("");
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
  const [sessionState, setSessionState] = useState<{
    status: PlanningSession["status"];
    current_round: number;
    pending_questions: PlanningSession["pending_questions"];
    phase_message: string | null;
    is_processing: boolean;
    active_tool_calls: ToolCallEntry[];
    completed_tool_call_groups: ToolCallEntry[][];
  } | null>(null);
  const activeToolCallsRef = useRef<ToolCallEntry[]>([]);
  const activeToolRunIdRef = useRef<string | null>(null);
  const nextLocalToolRunIdRef = useRef<number>(0);
  const lastFinalizedToolGroupSignatureRef = useRef<string>("");
  const lastEventAtMs = useRef<number>(0);
  const lastRefetchAtMs = useRef<number>(0);
  const lastContextNoticeAtMs = useRef<number>(0);

  const ensureRunId = useCallback((preferred?: string | null) => {
    if (preferred && preferred.trim()) return preferred;
    if (activeToolRunIdRef.current) return activeToolRunIdRef.current;
    nextLocalToolRunIdRef.current += 1;
    return `local-tool-run-${nextLocalToolRunIdRef.current}`;
  }, []);

  const finalizeToolCalls = useCallback((calls: ToolCallEntry[], runIdHint?: string | null) => {
    if (calls.length === 0) return false;
    const runId = runIdHint ?? activeToolRunIdRef.current ?? "unknown-run";
    const finalized = calls.map((call) => (
      call.status === "running" ? { ...call, status: "done" as const } : call
    ));
    const signature = `${runId}:${JSON.stringify(finalized)}`;
    if (signature === lastFinalizedToolGroupSignatureRef.current) {
      return false;
    }
    lastFinalizedToolGroupSignatureRef.current = signature;
    setToolCallGroups((groups) => [...groups, finalized].slice(-50));
    return true;
  }, []);

  useEffect(() => {
    if (lastEventAtMs.current === 0) {
      lastEventAtMs.current = Date.now();
    }
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
  const turns = sessionQuery.data?.conversation ?? [];
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
    const nextState = {
      status: session.status,
      current_round: session.current_round,
      pending_questions: session.pending_questions ?? null,
      phase_message: session.phase_message ?? null,
      is_processing: Boolean(session.is_processing),
      active_tool_calls: session.active_tool_calls ?? [],
      completed_tool_call_groups: session.completed_tool_call_groups ?? [],
    };
    const nextCalls = session.active_tool_calls ?? [];
    const nextCompletedGroups = session.completed_tool_call_groups ?? [];
    queueMicrotask(() => {
      setSessionState(nextState);
      setActiveToolCalls(nextCalls);
      activeToolCallsRef.current = nextCalls;
      setToolCallGroups((prev) => (prev.length > 0 ? prev : nextCompletedGroups));
    });
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
  ]);

  const flushActiveToolCalls = useCallback(() => {
    setActiveToolCalls((prev) => {
      if (prev.length === 0) return prev;
      finalizeToolCalls(prev, activeToolRunIdRef.current);
      activeToolCallsRef.current = [];
      activeToolRunIdRef.current = null;
      return [];
    });
  }, [finalizeToolCalls]);

  // Hydrate cost/tokens from session when loading an existing session (SSE only sends live events)
  useEffect(() => {
    if (!session) return;
    const cost = session.session_total_cost_usd;
    const tokens = session.max_prompt_tokens;
    queueMicrotask(() => {
      if (cost != null) setSessionCostUsd(cost);
      if (tokens != null) setMaxPromptTokens(tokens);
    });
  }, [session, session?.id, session?.session_total_cost_usd, session?.max_prompt_tokens]);

  const handleEvent = useCallback((event: UIEvent) => {
    lastEventAtMs.current = Date.now();
    if (event.type === "session_state") {
      const eventRunId = event.active_command_id?.trim() || null;
      const snapshotCalls = event.active_tool_calls ?? [];
      if (activeToolCallsRef.current.length > 0 && snapshotCalls.length === 0) {
        // Some backend flows clear active calls in session_state before a
        // terminal complete event is processed on the client.
        finalizeToolCalls(activeToolCallsRef.current, eventRunId ?? activeToolRunIdRef.current);
        activeToolCallsRef.current = [];
      }
      if (
        eventRunId
        && activeToolRunIdRef.current
        && eventRunId !== activeToolRunIdRef.current
        && activeToolCallsRef.current.length > 0
      ) {
        flushActiveToolCalls();
      }
      activeToolRunIdRef.current = eventRunId;
      setSessionState({
        status: event.status,
        current_round: event.current_round,
        pending_questions: event.pending_questions,
        phase_message: event.phase_message,
        is_processing: event.is_processing,
        active_tool_calls: event.active_tool_calls,
        completed_tool_call_groups: event.completed_tool_call_groups,
      });
      if ((event.completed_tool_call_groups?.length ?? 0) > 0) {
        setToolCallGroups((prev) => (prev.length > 0 ? prev : event.completed_tool_call_groups));
      }
      const terminalCompletedSnapshot = isTerminalCompletedSnapshot(event.is_processing, snapshotCalls);
      if (terminalCompletedSnapshot) {
        if (shouldFinalizeFromTerminalSnapshot(event.is_processing, snapshotCalls, activeToolCallsRef.current.length)) {
          // Recovery path: some flows end without explicit "complete".
          // Only finalize from terminal snapshot when we still have active
          // in-memory calls; otherwise this would duplicate an already-finalized group.
          finalizeToolCalls(snapshotCalls, eventRunId);
        }
        // Never keep done-only terminal snapshots in active state.
        setActiveToolCalls([]);
        activeToolCallsRef.current = [];
        activeToolRunIdRef.current = null;
      } else {
        setActiveToolCalls(snapshotCalls);
        activeToolCallsRef.current = snapshotCalls;
      }
      // session_state is canonical; clear transient thinking text once
      // the server reports stable/non-processing state.
      if (!event.is_processing) {
        setThinkingMessage(null);
      }
      return;
    }
    if (event.type === "thinking") {
      setThinkingMessage(event.message);
      return;
    }
    if (event.type === "tool_call") {
      const eventRunId = ensureRunId(event.command_id);
      if (
        activeToolRunIdRef.current
        && activeToolRunIdRef.current !== eventRunId
        && activeToolCallsRef.current.length > 0
      ) {
        flushActiveToolCalls();
      }
      activeToolRunIdRef.current = eventRunId;
      setActiveToolCalls((prev) => {
        const next = [
          ...prev,
          {
            id: `${eventRunId}-${Date.now()}-${prev.length}`,
            name: event.name,
            sessionStage: event.session_stage,
            path: event.path,
            query: event.query,
            status: "running" as const,
          },
        ].slice(-200);
        activeToolCallsRef.current = next;
        return next;
      });
      return;
    }
    if (event.type === "tool_result") {
      const eventRunId = ensureRunId(event.command_id);
      if (
        activeToolRunIdRef.current
        && activeToolRunIdRef.current !== eventRunId
        && activeToolCallsRef.current.length > 0
      ) {
        flushActiveToolCalls();
      }
      activeToolRunIdRef.current = eventRunId;
      setActiveToolCalls((prev) => {
        const updated = [...prev];
        for (let idx = updated.length - 1; idx >= 0; idx -= 1) {
          const candidate = updated[idx];
          const sameName = candidate.name === event.name;
          const sameStage = (candidate.sessionStage ?? "") === (event.session_stage ?? "");
          if (candidate.status === "running" && sameName && sameStage) {
            updated[idx] = {
              ...candidate,
              status: "done",
              durationMs: event.duration_ms,
            };
            activeToolCallsRef.current = updated;
            return updated;
          }
        }
        const next = [
          ...updated,
          {
            id: `${eventRunId}-${Date.now()}-${updated.length}`,
            name: event.name,
            sessionStage: event.session_stage,
            status: "done" as const,
            durationMs: event.duration_ms,
          },
        ].slice(-200);
        activeToolCallsRef.current = next;
        return next;
      });
      return;
    }
    if (event.type === "complete") {
      setThinkingMessage(null);
      flushActiveToolCalls();
      setWarnings([]);
      return;
    }
    if (event.type === "plan_ready") {
      setThinkingMessage(null);
      flushActiveToolCalls();
      if (event.round === 0) {
        setShowCritiquePrompt(true);
      }
      const now = Date.now();
      if (now - lastRefetchAtMs.current >= 800) {
        lastRefetchAtMs.current = now;
        void refetchSession();
      }
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
      return;
    }
    if (event.type === "discovery_ready") {
      lastRefetchAtMs.current = Date.now();
      void refetchSession();
      return;
    }
  }, [ensureRunId, finalizeToolCalls, flushActiveToolCalls, maxPromptTokens, refetchSession, sessionCostUsd]);

  useSessionEvents(id, handleEvent, Boolean(session));

  useEffect(() => {
    const status = effectiveStatus;
    const isActive = status === "draft" || status === "refining";
    if (!isActive) {
      return;
    }
    const timer = window.setInterval(() => {
      const now = Date.now();
      const eventStale = now - lastEventAtMs.current > 8000;
      const recentlyRefetched = now - lastRefetchAtMs.current < 4000;
      const shouldForceRefresh = (status === "draft" || status === "refining") && !recentlyRefetched;
      if ((eventStale || shouldForceRefresh) && !recentlyRefetched) {
        lastRefetchAtMs.current = now;
        void sessionQuery.refetch();
      }
    }, 3000);
    return () => {
      window.clearInterval(timer);
    };
  }, [effectiveStatus, sessionQuery]);

  const submitMessage = async (text: string) => {
    try {
      setError(null);
      const status = effectiveStatus;
      const models = {
        author_model: selectedAuthorModel || undefined,
        critic_model: selectedCriticModel || undefined,
      };
      if (status === "draft") {
        await sendDiscoveryMessage(id, text, models);
        await sessionQuery.refetch();
      } else if (status === "refining" || status === "converged") {
        await sendDiscoveryMessage(id, text, models);
        await sessionQuery.refetch();
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
      await runRound(id, undefined, {
        author_model: selectedAuthorModel || undefined,
        critic_model: selectedCriticModel || undefined,
      });
      await sessionQuery.refetch();
    } catch (err) {
      setError(String(err));
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

  const onToggleDiff = async () => {
    if (!showDiff) {
      try {
        const response = await getDiff(id);
        setDiff(response.diff || "No diff available.");
      } catch (err) {
        setError(String(err));
        return;
      }
    }
    setShowDiff((v) => !v);
  };

  useEffect(() => {
    if (!showDiff || !id) return;
    void (async () => {
      try {
        const response = await getDiff(id);
        setDiff(response.diff || "No diff available.");
      } catch (err) {
        setError(String(err));
      }
    })();
  }, [showDiff, id, sessionQuery.data?.current_plan?.round]);

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
  const planContent = useMemo(() => {
    if (showDiff) return diff;
    return sessionQuery.data?.current_plan?.plan_content ?? "";
  }, [showDiff, diff, sessionQuery.data?.current_plan?.plan_content]);
  const versionedTitle = useMemo(() => {
    if (!session) return "";
    return cleanPlanTitle(session.title);
  }, [session]);

  if (!session && !sessionQuery.isLoading) {
    return (
      <main className="h-screen flex flex-col items-center justify-center bg-zinc-950 text-zinc-400">
        <p>Session not found.</p>
        <Link to="/" className="mt-4 text-indigo-400 hover:text-indigo-300">Return to sessions</Link>
      </main>
    );
  }

  if (
    effectiveStatus === "draft"
    && isProcessing
    && setupSteps.length > 0
    && turns.length === 0
    && !(sessionQuery.data?.current_plan?.plan_content ?? "").trim()
  ) {
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
          />
        ) : null}
        <div className="flex-1 flex flex-col items-center justify-center p-8">
          <div className="w-full max-w-md rounded-xl border border-zinc-800 bg-zinc-900/50 p-8 shadow-lg">
            <h2 className="text-lg font-semibold text-zinc-100 mb-1">Setting up your session</h2>
            <p className="text-sm text-zinc-500 mb-6">
              Preparing codebase memory so the AI can understand your project. This may take a minute.
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
                  isDiffMode={showDiff}
                  status={effectiveStatus ?? session.status}
                  canExport={
                    Boolean(sessionQuery.data?.current_plan)
                    && effectiveStatus === "approved"
                  }
                  onToggleDiff={() => void onToggleDiff()}
                  onExport={() => void onExport()}
                />
              )}
              right={
                <ChatPanel
                  turns={turns}
                  questions={questions}
                  activeToolCalls={activeToolCalls}
                  toolCallGroups={toolCallGroups}
                  thinkingMessage={thinkingMessage}
                  phaseMessage={phaseMessage}
                  warnings={warnings}
                  pendingClarification={pendingClarification}
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
                  onSubmitClarification={handleClarificationSubmit}
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
