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
  submitClarification,
} from "../lib/api";
import type { ClarificationPrompt, PlanningSession, ToolCallEntry, UIEvent } from "../types";
import { AlertCircle, Check, Loader2 } from "lucide-react";

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
  } | null>(null);
  const activeToolCallsRef = useRef<ToolCallEntry[]>([]);
  const lastEventAtMs = useRef<number>(0);
  const lastRefetchAtMs = useRef<number>(0);
  const lastContextNoticeAtMs = useRef<number>(0);

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
  const phaseMessage = sessionState?.phase_message ?? null;
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
      active_tool_calls: session.active_tool_calls ?? [],
    });
    setActiveToolCalls(session.active_tool_calls ?? []);
  }, [
    session?.id,
    session?.status,
    session?.current_round,
    session?.pending_questions,
    session?.phase_message,
    session?.is_processing,
    session?.active_tool_calls,
  ]);

  const flushActiveToolCalls = useCallback(() => {
    setActiveToolCalls((prev) => {
      if (prev.length === 0) return prev;
      const finalized = prev.map((call) => (
        call.status === "running" ? { ...call, status: "done" as const } : call
      ));
      activeToolCallsRef.current = [];
      setToolCallGroups((groups) => [...groups, finalized].slice(-50));
      return [];
    });
  }, []);

  // Hydrate cost/tokens from session when loading an existing session (SSE only sends live events)
  useEffect(() => {
    if (!session) return;
    if (session.session_total_cost_usd != null) {
      setSessionCostUsd(session.session_total_cost_usd);
    }
    if (session.max_prompt_tokens != null) {
      setMaxPromptTokens(session.max_prompt_tokens);
    }
  }, [session?.id, session?.session_total_cost_usd, session?.max_prompt_tokens]);

  const handleEvent = useCallback((event: UIEvent) => {
    lastEventAtMs.current = Date.now();
    if (event.type === "session_state") {
      setSessionState({
        status: event.status,
        current_round: event.current_round,
        pending_questions: event.pending_questions,
        phase_message: event.phase_message,
        is_processing: event.is_processing,
        active_tool_calls: event.active_tool_calls,
      });
      setActiveToolCalls(event.active_tool_calls ?? []);
      return;
    }
    if (event.type === "thinking") {
      setThinkingMessage(event.message);
      return;
    }
    if (event.type === "tool_call") {
      setActiveToolCalls((prev) => {
        const next = [
          ...prev,
          {
            id: Date.now() + prev.length,
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
            id: Date.now() + updated.length,
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
      setPendingClarification(null);
      setWarnings([]);
      return;
    }
    if (event.type === "plan_ready") {
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
  }, [flushActiveToolCalls, maxPromptTokens, refetchSession, sessionCostUsd]);

  useSessionEvents(id, handleEvent, Boolean(session));

  useEffect(() => {
    const status = effectiveStatus;
    const isActive = status === "drafting"
      || status === "discovering"
      || status === "refining";
    if (!isActive) {
      return;
    }
    const timer = window.setInterval(() => {
      const now = Date.now();
      const eventStale = now - lastEventAtMs.current > 8000;
      const recentlyRefetched = now - lastRefetchAtMs.current < 4000;
      const shouldForceRefresh = (status === "drafting" || status === "refining") && !recentlyRefetched;
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
      if (status === "discovering") {
        await sendDiscoveryMessage(id, text, models);
      } else {
        setShowCritiquePrompt(false);
        await runRound(id, text, models);
      }
      if (sessionQuery.data?.session.repo_name) {
        setStoredModelSelection(models, sessionQuery.data.session.repo_name);
      }
    } catch (err) {
      if (err instanceof ConflictError) {
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
      const response = await getDiff(id);
      setDiff(response.diff || "No diff available.");
    }
    setShowDiff((v) => !v);
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

  const contextPercent = contextUsageRatio !== null ? contextUsageRatio * 100 : null;
  const activeStatus = effectiveStatus;
  const isActiveSession = activeStatus === "discovering"
    || activeStatus === "drafting"
    || activeStatus === "refining";
  const hasRunningToolCalls = activeToolCalls.some((call) => call.status === "running");
  const isPlanActivelyUpdating = Boolean(phaseMessage || thinkingMessage || hasRunningToolCalls);
  const leftPanelActivityMessage = useMemo(() => {
    if (!isActiveSession || !isPlanActivelyUpdating) return null;
    if (phaseMessage) return phaseMessage;
    if (thinkingMessage) return thinkingMessage;
    if (activeStatus === "drafting") return "Drafting first plan";
    return null;
  }, [activeStatus, isActiveSession, isPlanActivelyUpdating, phaseMessage, thinkingMessage]);
  const planContent = useMemo(() => {
    if (showDiff) return diff;
    return sessionQuery.data?.current_plan?.plan_content ?? "";
  }, [showDiff, diff, sessionQuery.data?.current_plan?.plan_content]);

  if (!session && !sessionQuery.isLoading) {
    return (
      <main className="h-screen flex flex-col items-center justify-center bg-zinc-950 text-zinc-400">
        <p>Session not found.</p>
        <Link to="/" className="mt-4 text-indigo-400 hover:text-indigo-300">Return to sessions</Link>
      </main>
    );
  }

  if (session?.status === "preparing") {
    return (
      <main className="h-screen flex flex-col bg-zinc-950 overflow-hidden">
        {session ? (
          <ActionBar
            repoName={session.repo_name}
            title={session.title}
            round={session.current_round}
            status={session.status}
            convergenceScore={undefined}
            canCritique={false}
            canApprove={false}
            canExport={false}
            onCritique={() => {}}
            onApprove={() => {}}
            onExport={() => {}}
            onToggleDiff={() => onToggleDiff()}
            isDiffMode={showDiff}
            authorModel={selectedAuthorModel}
            criticModel={selectedCriticModel}
            modelOptions={modelItems}
            onAuthorModelChange={setAuthorModel}
            onCriticModelChange={setCriticModel}
            sessionCostUsd={0}
            maxPromptTokens={0}
            contextWindowTokens={null}
            contextPercent={null}
            contextCompactionEnabled={false}
            critiquePending={false}
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
            title={session.title}
            round={effectiveRound}
            status={effectiveStatus ?? session.status}
            convergenceScore={sessionQuery.data?.current_plan?.convergence_score ?? undefined}
            canCritique={!isProcessing && (effectiveStatus === "refining" || effectiveStatus === "converged")}
            canApprove={!isProcessing && (effectiveStatus === "converged" || effectiveStatus === "approved")}
            canExport={
              Boolean(sessionQuery.data?.current_plan)
              && (effectiveStatus === "approved" || effectiveStatus === "exported")
            }
            onCritique={() => void onCritique()}
            onApprove={() => void onApprove()}
            onExport={() => void onExport()}
            onToggleDiff={() => void onToggleDiff()}
            isDiffMode={showDiff}
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
            sessionCostUsd={sessionCostUsd}
            maxPromptTokens={maxPromptTokens}
            contextWindowTokens={contextWindowTokens}
            contextPercent={contextPercent}
            contextCompactionEnabled={contextCompactionEnabled}
            critiquePending={showCritiquePrompt}
            onDelete={onDelete}
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
                  activityMessage={leftPanelActivityMessage}
                  isRefreshing={sessionQuery.isFetching && isPlanActivelyUpdating}
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
                  showCritiquePrompt={showCritiquePrompt}
                  inputDisabled={isProcessing}
                  onSubmit={submitMessage}
                  onCritique={onCritique}
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
