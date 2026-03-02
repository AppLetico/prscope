import { useQuery } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ActionBar } from "../components/ActionBar";
import { ChatPanel } from "../components/ChatPanel";
import { PlanPanel } from "../components/PlanPanel";
import { ResizableLayout } from "../components/ResizableLayout";
import { useSessionEvents } from "../hooks/useSessionEvents";
import {
  approveSession,
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
import type { ClarificationPrompt, DiscoveryQuestion, ToolCallEntry, UIEvent } from "../types";
import { AlertCircle } from "lucide-react";

export function PlanningViewPage() {
  const { id = "" } = useParams();
  const [questions, setQuestions] = useState<DiscoveryQuestion[]>([]);
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
  const lastEventAtMs = useRef<number>(0);
  const lastRefetchAtMs = useRef<number>(0);

  useEffect(() => {
    if (lastEventAtMs.current === 0) {
      lastEventAtMs.current = Date.now();
    }
  }, []);

  const sessionQuery = useQuery({
    queryKey: ["session", id],
    queryFn: () => getSession(id),
    enabled: Boolean(id),
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

  const handleEvent = useCallback((event: UIEvent) => {
    lastEventAtMs.current = Date.now();
    if (event.type === "thinking") {
      setThinkingMessage(event.message);
      return;
    }
    if (event.type === "tool_call") {
      setActiveToolCalls((prev) => [
        ...prev,
        {
          id: Date.now() + prev.length,
          name: event.name,
          sessionStage: event.session_stage,
          path: event.path,
          query: event.query,
          status: "running" as const,
        },
      ].slice(-200));
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
            return updated;
          }
        }
        return [
          ...updated,
          {
            id: Date.now() + updated.length,
            name: event.name,
            sessionStage: event.session_stage,
            status: "done" as const,
            durationMs: event.duration_ms,
          },
        ].slice(-200);
      });
      return;
    }
    if (event.type === "complete") {
      setThinkingMessage(null);
      setToolCallGroups((prev) => (activeToolCalls.length > 0 ? [...prev, activeToolCalls] : prev));
      setActiveToolCalls([]);
      setQuestions([]);
      setPendingClarification(null);
      setWarnings([]);
      lastRefetchAtMs.current = Date.now();
      void sessionQuery.refetch();
      return;
    }
    if (event.type === "plan_ready") {
      const now = Date.now();
      if (now - lastRefetchAtMs.current >= 800) {
        lastRefetchAtMs.current = now;
        void sessionQuery.refetch();
      }
      return;
    }
    if (event.type === "error") {
      setThinkingMessage(null);
      setError(event.message);
      return;
    }
    if (event.type === "warning") {
      setWarnings((prev) => [...prev, event.message].slice(-12));
      if (event.message.toLowerCase().includes("context compaction enabled")) {
        setContextCompactionEnabled(true);
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
    }
  }, [activeToolCalls, maxPromptTokens, sessionCostUsd, sessionQuery]);

  useSessionEvents(id, handleEvent);

  useEffect(() => {
    const status = sessionQuery.data?.session.status;
    const isActive = status === "drafting" || status === "discovering" || status === "discovery";
    if (!isActive) {
      return;
    }
    const timer = window.setInterval(() => {
      const now = Date.now();
      const eventStale = now - lastEventAtMs.current > 8000;
      const recentlyRefetched = now - lastRefetchAtMs.current < 4000;
      if (eventStale && !recentlyRefetched) {
        lastRefetchAtMs.current = now;
        void sessionQuery.refetch();
      }
    }, 3000);
    return () => {
      window.clearInterval(timer);
    };
  }, [sessionQuery, sessionQuery.data?.session.status]);

  const submitMessage = async (text: string) => {
    try {
      setError(null);
      const status = sessionQuery.data?.session.status;
      const models = {
        author_model: selectedAuthorModel || undefined,
        critic_model: selectedCriticModel || undefined,
      };
      if (status === "discovering" || status === "discovery") {
        const response = await sendDiscoveryMessage(id, text, models);
        setQuestions(response.result.questions ?? []);
      } else {
        await runRound(id, text, models);
      }
      if (sessionQuery.data?.session.repo_name) {
        setStoredModelSelection(models, sessionQuery.data.session.repo_name);
      }
      await sessionQuery.refetch();
    } catch (err) {
      setError(String(err));
    }
  };

  const handleSelectOption = async (_questionIndex: number, optionText: string, isOther: boolean) => {
    if (isOther) {
      return;
    }
    setQuestions([]);
    await submitMessage(optionText);
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

  const contextPercent = contextUsageRatio !== null ? contextUsageRatio * 100 : null;
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

  return (
    <main className="h-screen flex flex-col bg-zinc-950 overflow-hidden">
      {session ? (
        <>
          <ActionBar
            repoName={session.repo_name}
            title={session.title}
            round={session.current_round}
            status={session.status}
            convergenceScore={sessionQuery.data?.current_plan?.convergence_score ?? undefined}
            canCritique={session.status === "refining" || session.status === "converged"}
            canApprove={session.status === "converged" || session.status === "approved"}
            canExport={
              Boolean(sessionQuery.data?.current_plan)
              && (session.status === "approved" || session.status === "exported")
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
          />
          <div className="px-6 py-2 border-b border-zinc-800 text-xs text-zinc-400 flex items-center justify-end gap-4">
            <span
              className={
                sessionCostUsd > 0.5
                  ? "text-rose-400"
                  : sessionCostUsd > 0.1
                    ? "text-amber-400"
                    : "text-zinc-400"
              }
            >
              Session cost: ${sessionCostUsd.toFixed(4)}
            </span>
            <span>Max prompt: {maxPromptTokens}</span>
            <span
              className={
                contextPercent !== null && contextPercent >= 85
                  ? "text-rose-400"
                  : contextPercent !== null && contextPercent >= 70
                    ? "text-amber-400"
                    : "text-zinc-400"
              }
            >
              Context: {maxPromptTokens}
              {contextWindowTokens ? ` / ${contextWindowTokens}` : ""}
              {contextPercent !== null ? ` (${contextPercent.toFixed(1)}%)` : ""}
            </span>
            {contextCompactionEnabled && (
              <span className="text-indigo-300">
                Compaction: on
              </span>
            )}
          </div>
          
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
              left={<PlanPanel content={planContent} isDiffMode={showDiff} status={session.status} />}
              right={
                <ChatPanel
                  turns={turns}
                  questions={questions}
                  activeToolCalls={activeToolCalls}
                  toolCallGroups={toolCallGroups}
                  thinkingMessage={thinkingMessage}
                  warnings={warnings}
                  pendingClarification={pendingClarification}
                  onSubmit={submitMessage}
                  onSelectOption={handleSelectOption}
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
