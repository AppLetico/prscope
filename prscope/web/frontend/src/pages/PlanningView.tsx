import { useQuery } from "@tanstack/react-query";
import { useCallback, useMemo, useState } from "react";
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
  getDiff,
  getSession,
  runRound,
  sendDiscoveryMessage,
  submitClarification,
} from "../lib/api";
import type { ClarificationPrompt, DiscoveryQuestion, UIEvent } from "../types";
import { AlertCircle } from "lucide-react";

export function PlanningViewPage() {
  const { id = "" } = useParams();
  const [questions, setQuestions] = useState<DiscoveryQuestion[]>([]);
  const [toolCalls, setToolCalls] = useState<string[]>([]);
  const [thinkingMessage, setThinkingMessage] = useState<string | null>(null);
  const [showDiff, setShowDiff] = useState(false);
  const [diff, setDiff] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [sessionCostUsd, setSessionCostUsd] = useState<number>(0);
  const [maxPromptTokens, setMaxPromptTokens] = useState<number>(0);
  const [pendingClarification, setPendingClarification] = useState<ClarificationPrompt | null>(null);

  const sessionQuery = useQuery({
    queryKey: ["session", id],
    queryFn: () => getSession(id),
    enabled: Boolean(id),
  });

  const handleEvent = useCallback((event: UIEvent) => {
    if (event.type === "thinking") {
      setThinkingMessage(event.message);
      return;
    }
    if (event.type === "tool_call") {
      setToolCalls((prev) => [...prev, `${event.name}${event.path ? ` (${event.path})` : ""}`].slice(-200));
      return;
    }
    if (event.type === "complete") {
      setThinkingMessage(null);
      setToolCalls([]);
      setQuestions([]);
      setPendingClarification(null);
      void sessionQuery.refetch();
      return;
    }
    if (event.type === "error") {
      setThinkingMessage(null);
      setError(event.message);
      return;
    }
    if (event.type === "warning") {
      setError(event.message);
      return;
    }
    if (event.type === "token_usage") {
      setSessionCostUsd(event.session_total_usd ?? sessionCostUsd);
      setMaxPromptTokens(event.max_prompt_tokens ?? maxPromptTokens);
      return;
    }
    if (event.type === "clarification_needed") {
      setPendingClarification({
        question: event.question,
        context: event.context,
        source: event.source,
      });
    }
  }, [id, maxPromptTokens, sessionCostUsd, sessionQuery]);

  useSessionEvents(id, handleEvent);

  const submitMessage = async (text: string) => {
    try {
      setError(null);
      const status = sessionQuery.data?.session.status;
      if (status === "discovering" || status === "discovery") {
        const response = await sendDiscoveryMessage(id, text);
        setQuestions(response.result.questions ?? []);
      } else {
        await runRound(id, text);
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
      await runRound(id);
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

  const session = sessionQuery.data?.session;
  const turns = sessionQuery.data?.conversation ?? [];
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
              left={<PlanPanel content={planContent} isDiffMode={showDiff} />}
              right={
                <ChatPanel
                  turns={turns}
                  questions={questions}
                  toolCalls={toolCalls}
                  thinkingMessage={thinkingMessage}
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
