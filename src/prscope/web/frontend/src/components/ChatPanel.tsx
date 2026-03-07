import { useEffect, useMemo, useRef, useState } from "react";
import type { ClarificationPrompt, DiscoveryQuestion, PlanFollowups, PlanningTurn, ToolCallEntry } from "../types";
import { OptionButtons } from "./OptionButtons";
import { ToolCallStream } from "./ToolCallStream";
import { ModelSelector } from "./ModelSelector";
import { Send, Bot, User, Sparkles, Microscope, CheckCircle2, Copy, Check, Square, Loader2, ArrowRight, Zap, MessageSquare, X } from "lucide-react";
import { clsx } from "clsx";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Tooltip } from "./ui/Tooltip";
import { chatMarkdownComponents } from "../lib/markdownComponents";
import {
  buildRefinementRoundSummaries,
  collapseTimelineForDisplay,
  compactTimelineToRecentRounds,
  extractFirstJsonObject,
  shouldShowActiveToolStream,
} from "./chatPanelUtils";
import type { TimelineItem } from "./chatPanelUtils";

interface ModelOption {
  model_id: string;
  provider: string;
  available: boolean;
  unavailable_reason?: string | null;
}

interface FocusPrompt {
  label: string;
  text: string;
}

function deriveFocusPrompt(text: string): FocusPrompt {
  const normalized = text.trim();
  if (!normalized) {
    return { label: "Focus", text: "" };
  }
  if (normalized.startsWith("Please fix the following open issues")) {
    return { label: "Focus: review notes", text: normalized };
  }
  const firstSentence = normalized.split(/\n+/)[0]?.trim() || normalized;
  const compactLabel = firstSentence
    .replace(/^Please update the plan to address\s*/i, "")
    .replace(/^Please fix\s*/i, "")
    .replace(/\.$/, "")
    .trim();
  return {
    label: `Focus: ${compactLabel || "selected issue"}`,
    text: normalized,
  };
}

interface ChatPanelProps {
  timeline: TimelineItem[];
  questions: DiscoveryQuestion[];
  activeToolCalls: ToolCallEntry[];
  thinkingMessage: string | null;
  phaseMessage?: string | null;
  warnings: string[];
  pendingClarification: ClarificationPrompt | null;
  planFollowups?: PlanFollowups | null;
  inputDisabled?: boolean;
  isProcessing?: boolean;
  authorModel?: string;
  criticModel?: string;
  modelOptions?: ModelOption[];
  onAuthorModelChange?: (modelId: string) => void;
  onCriticModelChange?: (modelId: string) => void;
  canCritique?: boolean;
  canApprove?: boolean;
  critiquePending?: boolean;
  contextPercent?: number | null;
  contextWindowTokens?: number | null;
  onCritique?: () => void;
  onApprove?: () => void;
  onStop?: () => void;
  onSubmit: (text: string) => Promise<void>;
  onSubmitFollowup?: (followupId: string, answer: string) => Promise<void>;
  onSubmitClarification: (answer: string) => Promise<void>;
  externalInputAppend?: { id: number; text: string } | null;
  latestResponseMode?: "author_chat" | "refine_round" | null;
}

function isDiscoveryQuestionBlock(content: string): boolean {
  const normalized = content.trim();
  if (!normalized) return false;
  const hasQuestionHeader = /(^|\n)\s*(Q\s*\d+|Question\s+\d+|\d+)\s*[:.)-]\s+/i.test(normalized);
  const hasOptions = /(^|\n)\s*(?:[-*]\s*)?[A-D][).:-]\s+/im.test(normalized);
  return hasQuestionHeader && hasOptions;
}

export function ChatPanel({
  timeline,
  questions,
  activeToolCalls,
  thinkingMessage,
  phaseMessage = null,
  warnings,
  pendingClarification,
  planFollowups = null,
  inputDisabled = false,
  isProcessing = false,
  authorModel,
  criticModel,
  modelOptions = [],
  onAuthorModelChange,
  onCriticModelChange,
  canCritique = false,
  canApprove = false,
  critiquePending = false,
  contextPercent = null,
  contextWindowTokens = null,
  onCritique,
  onApprove,
  onStop,
  onSubmit,
  onSubmitFollowup,
  onSubmitClarification,
  externalInputAppend = null,
  latestResponseMode = null,
}: ChatPanelProps) {
  const [input, setInput] = useState("");
  const [optimisticUserMessage, setOptimisticUserMessage] = useState<string | null>(null);
  const [clarificationInput, setClarificationInput] = useState("");
  const [selectedClarificationRecommendation, setSelectedClarificationRecommendation] = useState<string | null>(null);
  const [useCustomClarification, setUseCustomClarification] = useState(false);
  const [selectedAnswers, setSelectedAnswers] = useState<Record<number, string>>({});
  const [selectedIsOther, setSelectedIsOther] = useState<Record<number, boolean>>({});
  const [otherInputs, setOtherInputs] = useState<Record<number, string>>({});
  const [submittingAnswers, setSubmittingAnswers] = useState(false);
  const [submittingFollowupId, setSubmittingFollowupId] = useState<string | null>(null);
  const [submittingSuggestionId, setSubmittingSuggestionId] = useState<string | null>(null);
  const [focusPrompt, setFocusPrompt] = useState<FocusPrompt | null>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const [thinkingStatusTick, setThinkingStatusTick] = useState(0);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const lastScrollTopRef = useRef(0);
  const lastActivitySignatureRef = useRef("");
  const lastExternalAppendIdRef = useRef<number | null>(null);

  const isNearBottom = () => {
    const el = scrollRef.current;
    if (!el) return true;
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    return distance <= 100;
  };

  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [timeline, activeToolCalls, thinkingMessage, questions, autoScroll]);

  useEffect(() => {
    const signature = [
      timeline.length,
      activeToolCalls.length,
      warnings.length,
      Boolean(thinkingMessage),
      questions.length,
      pendingClarification ? 1 : 0,
    ].join("|");
    lastActivitySignatureRef.current = signature;
  }, [
    autoScroll,
    timeline.length,
    activeToolCalls.length,
    warnings.length,
    thinkingMessage,
    questions.length,
    pendingClarification,
  ]);

  useEffect(() => {
    if (!optimisticUserMessage) return;
    const normalizedOptimistic = optimisticUserMessage.trim();
    if (!normalizedOptimistic) {
      setOptimisticUserMessage(null);
      return;
    }
    const confirmed = timeline.some(
      (item) => item.kind === "turn" && item.turn.role === "user" && item.turn.content.trim() === normalizedOptimistic,
    );
    if (confirmed) {
      setOptimisticUserMessage(null);
    }
  }, [optimisticUserMessage, timeline]);

  useEffect(() => {
    if (!externalInputAppend) return;
    if (lastExternalAppendIdRef.current === externalInputAppend.id) return;
    lastExternalAppendIdRef.current = externalInputAppend.id;
    const nextText = externalInputAppend.text.trim();
    if (!nextText) return;
    setFocusPrompt(deriveFocusPrompt(nextText));
    if (inputRef.current) {
      inputRef.current.focus();
      inputRef.current.style.height = "auto";
      inputRef.current.style.height = `${Math.min(inputRef.current.scrollHeight, 256)}px`;
    }
  }, [externalInputAppend]);

  useEffect(() => {
    const validIndexes = new Set(questions.map((q) => q.index));
    setSelectedAnswers((prev) => {
      const next: Record<number, string> = {};
      for (const [key, value] of Object.entries(prev)) {
        const idx = Number(key);
        if (validIndexes.has(idx)) {
          next[idx] = value;
        }
      }
      return next;
    });
    setSelectedIsOther((prev) => {
      const next: Record<number, boolean> = {};
      for (const [key, value] of Object.entries(prev)) {
        const idx = Number(key);
        if (validIndexes.has(idx)) {
          next[idx] = value;
        }
      }
      return next;
    });
    setOtherInputs((prev) => {
      const next: Record<number, string> = {};
      for (const [key, value] of Object.entries(prev)) {
        const idx = Number(key);
        if (validIndexes.has(idx)) {
          next[idx] = value;
        }
      }
      return next;
    });
  }, [questions]);

  useEffect(() => {
    if (!thinkingMessage) {
      setThinkingStatusTick(0);
      return;
    }
    setThinkingStatusTick(0);
    const startedAt = Date.now();
    const interval = window.setInterval(() => {
      const elapsedMs = Date.now() - startedAt;
      setThinkingStatusTick(Math.floor(elapsedMs / 4000));
    }, 1000);
    return () => {
      window.clearInterval(interval);
    };
  }, [thinkingMessage]);

  useEffect(() => {
    setClarificationInput("");
    setSelectedClarificationRecommendation(null);
    setUseCustomClarification(false);
  }, [pendingClarification?.question, pendingClarification?.context, pendingClarification?.source]);

  // Handle resize events to recalculate scroll anchoring
  useEffect(() => {
    const handleResize = () => {
      if (autoScroll && scrollRef.current) {
        scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
      }
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [autoScroll]);

  const submitMessage = async (rawText: string) => {
    if (inputDisabled) return;
    const text = rawText.trim();
    if (!text) return;
    setOptimisticUserMessage(text);
    try {
      await onSubmit(text);
    } catch (err) {
      setOptimisticUserMessage(null);
      throw err;
    }
  };

  const handleSubmit = async () => {
    const text = input.trim();
    const combined = [focusPrompt?.text?.trim() || "", text].filter(Boolean).join("\n\n");
    if (!combined) return;
    setInput("");
    setFocusPrompt(null);
    await submitMessage(combined);
  };

  const handleFollowupSubmit = async (followupId: string, answer: string) => {
    if (!onSubmitFollowup || inputDisabled) return;
    setSubmittingFollowupId(followupId);
    try {
      await onSubmitFollowup(followupId, answer);
    } finally {
      setSubmittingFollowupId((current) => (current === followupId ? null : current));
    }
  };

  const handleSuggestionSubmit = async (suggestionId: string, text: string) => {
    if (inputDisabled) return;
    setSubmittingSuggestionId(suggestionId);
    try {
      await submitMessage(text);
    } finally {
      setSubmittingSuggestionId((current) => (current === suggestionId ? null : current));
    }
  };

  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const currentTop = el.scrollTop;
    const scrolledUp = currentTop < lastScrollTopRef.current;
    lastScrollTopRef.current = currentTop;

    // If user scrolls upward, immediately stop auto-follow so the view
    // does not snap back to bottom while new events stream in.
    if (scrolledUp) {
      if (autoScroll) setAutoScroll(false);
      return;
    }

    if (isNearBottom()) {
      if (!autoScroll) setAutoScroll(true);
    }
  };

  const copyMessage = async (key: string, content: string) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedKey(key);
      window.setTimeout(() => {
        setCopiedKey((current) => (current === key ? null : current));
      }, 1500);
    } catch {
      // Best-effort clipboard behavior; keep UI stable if denied.
    }
  };

  const allQuestionsAnswered = questions.length > 0 && questions.every((q) => {
    const selected = selectedAnswers[q.index];
    if (!selected) return false;
    if (!selectedIsOther[q.index]) return true;
    return Boolean((otherInputs[q.index] ?? "").trim());
  });
  const answeredCount = questions.filter((q) => {
    const selected = selectedAnswers[q.index];
    if (!selected) return false;
    if (!selectedIsOther[q.index]) return true;
    return Boolean((otherInputs[q.index] ?? "").trim());
  }).length;
  const visiblePlanFollowups = useMemo(
    () => (planFollowups?.questions ?? []).filter((question) => !question.resolved),
    [planFollowups],
  );
  const hasPlanFollowupPanel = visiblePlanFollowups.length > 0 || (planFollowups?.suggestions?.length ?? 0) > 0;
  const canSendMessage = Boolean(input.trim() || focusPrompt?.text.trim());
  const displayedTimeline = useMemo(
    () => compactTimelineToRecentRounds(collapseTimelineForDisplay(timeline)),
    [timeline],
  );
  const refinementRoundSummaries = useMemo(() => buildRefinementRoundSummaries(timeline), [timeline]);

  const submitSelectedAnswers = async () => {
    if (!allQuestionsAnswered || submittingAnswers) return;
    const ordered = [...questions].sort((a, b) => a.index - b.index);
    const combined = ordered
      .map((q, idx) => {
        const answer = selectedIsOther[q.index]
          ? (otherInputs[q.index] ?? "").trim()
          : selectedAnswers[q.index];
        return `${idx + 1}. ${answer}`;
      })
      .join("\n");
    setSubmittingAnswers(true);
    try {
      await onSubmit(combined);
      setSelectedAnswers({});
      setSelectedIsOther({});
      setOtherInputs({});
    } finally {
      setSubmittingAnswers(false);
    }
  };

  const renderTurnContent = (turn: PlanningTurn): string => {
    if (turn.role === "user") return turn.content;
    const raw = turn.content.trim();
    // Avoid duplicating full plan content in chat; plan panel is canonical.
    const looksLikeFullPlan =
      raw.includes("## Goals")
      && raw.includes("## Non-Goals")
      && raw.includes("## Architecture");
    if (looksLikeFullPlan) {
      return "Plan updated. Review the full plan in the left panel.";
    }
    if (turn.role === "critic" && raw.toLowerCase().startsWith("design review:")) {
      const primaryIssueLine = raw
        .split("\n")
        .find((line) => line.toLowerCase().startsWith("primary issue:"));
      return primaryIssueLine ? `${raw.split("\n")[0]}\n\n${primaryIssueLine}` : raw.split("\n")[0];
    }
    if (turn.role === "author" && raw.startsWith("Repair planning complete.")) {
      return "Prepared the next revision based on the review feedback.";
    }
    if (turn.role === "author" && raw.startsWith("Updated sections:")) {
      const updatedSections = raw.split("\n")[0];
      const primaryIssue = refinementRoundSummaries[turn.round]?.primaryIssue;
      const reviewLine = primaryIssue
        ? `Addressed review feedback: ${primaryIssue}`
        : "Review and revision complete.";
      return `${updatedSections}\n\n${reviewLine}\n\nReview the latest draft in the plan panel.`;
    }
    return raw;
  };

  const warningSummary = warnings
    .filter((message) => {
      if (!pendingClarification) return true;
      const normalized = message.toLowerCase();
      // Clarification card is the primary UX; suppress duplicate clarification warnings.
      return !normalized.includes("clarification");
    })
    .reduce<Array<{ message: string; count: number }>>((acc, message) => {
      const existing = acc.find((item) => item.message === message);
    if (existing) {
      existing.count += 1;
    } else {
      acc.push({ message, count: 1 });
    }
    return acc;
    }, []);
  const clarificationRecommendations = useMemo(() => {
    if (!pendingClarification) return [];
    if (pendingClarification.recommendations && pendingClarification.recommendations.length > 0) {
      return pendingClarification.recommendations.slice(0, 5);
    }
    const question = pendingClarification.question.toLowerCase();
    if (question.includes("rate limit") || question.includes("rate limiting") || question.includes("auth")) {
      return [
        "Keep it public, no endpoint-specific rate limit",
        "Keep it public, rely on gateway/global rate limiting",
        "Require auth and add endpoint-level rate limiting",
      ];
    }
    if (question.startsWith("is ") || question.startsWith("are ") || question.startsWith("should ")) {
      return ["Yes", "No", "Depends / partially"];
    }
    return ["Keep current approach", "Adjust scope", "Need more context"];
  }, [pendingClarification]);
  const clarificationAnswer = useMemo(() => {
    if (useCustomClarification) return clarificationInput.trim();
    return selectedClarificationRecommendation ?? clarificationInput.trim();
  }, [clarificationInput, selectedClarificationRecommendation, useCustomClarification]);
  const showSubmittingStatus = submittingAnswers && !phaseMessage;
  const animatedThinkingMessage = useMemo(() => {
    if (!thinkingMessage) return null;
    const normalized = thinkingMessage.toLowerCase();
    if (normalized.includes("drafting plan")) {
      if (thinkingStatusTick < 2) return thinkingMessage;
      const elapsedSeconds = Math.max(4, thinkingStatusTick * 4);
      return `Drafting plan (${elapsedSeconds}s elapsed)`;
    }
    if (thinkingStatusTick < 2) return thinkingMessage;
    const activityPhrases = [
      "Thinking",
      "Planning next moves",
      "Scanning context",
      "Validating assumptions",
      "Preparing the next step",
    ];
    return activityPhrases[(thinkingStatusTick - 1) % activityPhrases.length];
  }, [thinkingMessage, thinkingStatusTick]);
  // const hasRunningActiveToolCalls = useMemo(
  //   () => hasRunningToolCalls(activeToolCalls),
  //   [activeToolCalls],
  // );
  const hasVisibleActiveToolStream = shouldShowActiveToolStream(activeToolCalls, isProcessing);
  const draftingInProgress = useMemo(() => {
    const phase = (phaseMessage ?? "").toLowerCase();
    const thinking = (thinkingMessage ?? "").toLowerCase();
    return phase.includes("draft") || thinking.includes("draft");
  }, [phaseMessage, thinkingMessage]);
  const placeholderDraftToolCalls = useMemo<ToolCallEntry[]>(
    () => ([
      {
        id: "draft-placeholder",
        call_id: "draft-placeholder",
        name: "draft_plan",
        sessionStage: "planner",
        status: "running",
        created_at: new Date().toISOString(),
      },
    ]),
    [],
  );
  const showDraftPlaceholderToolStream = isProcessing && !hasVisibleActiveToolStream && draftingInProgress;
  const latestAuthorTurnKey = useMemo(() => {
    for (let idx = displayedTimeline.length - 1; idx >= 0; idx -= 1) {
      const item = displayedTimeline[idx];
      if (item.kind === "turn" && item.turn.role === "author") return item.key;
    }
    return null;
  }, [displayedTimeline]);
  const liveStatusMessage = useMemo(() => {
    if (questions.length > 0 || pendingClarification) return null;
    const hasRunning = activeToolCalls.some((c) => c.status === "running");
    if (hasVisibleActiveToolStream) return null;
    if (animatedThinkingMessage) return animatedThinkingMessage;
    if (phaseMessage && isProcessing && !hasRunning) return phaseMessage;
    if (hasRunning) return "Running tools...";
    if (isProcessing) return "Working...";
    return null;
  }, [questions.length, pendingClarification, phaseMessage, animatedThinkingMessage, activeToolCalls, isProcessing, hasVisibleActiveToolStream]);

  return (
    <div className="h-full flex flex-col bg-zinc-950 relative">
      
      {/* Scrollable Message Area */}
      <div
        ref={scrollRef}
        className={clsx(
          "flex-1 overflow-y-auto px-4 md:px-6 pt-6",
          hasPlanFollowupPanel ? "pb-[42rem]" : "pb-[28rem]",
        )}
        onScroll={onScroll}
      >
        <div className="max-w-2xl mx-auto space-y-6">
          {displayedTimeline.length === 0 && !thinkingMessage && (
            <div className="flex flex-col items-center justify-center py-12 text-zinc-500">
              <Sparkles className="w-8 h-8 mb-3 opacity-50" />
              <p className="text-sm">Start the conversation to begin planning.</p>
            </div>
          )}

          {displayedTimeline.map((item) => {
            if (item.kind === "tool_group") {
              return (
                <div key={item.key} className="flex justify-start w-full animate-in slide-in-from-bottom-2 fade-in duration-300 ease-out">
                  <ToolCallStream toolCalls={item.group.tools} />
                </div>
              );
            }
            const turn = item.turn;
            const isUser = turn.role === "user";
            const turnKey = item.key;
            const shouldCollapseDiscoveryQuestionText = (
              turn.role === "author"
              && questions.length > 0
              && isDiscoveryQuestionBlock(turn.content)
            );
            const displayContent = renderTurnContent(turn);
            const criticJson = turn.role === "critic" ? extractFirstJsonObject(turn.content) : null;
            const criticPrelude = criticJson ? turn.content.slice(0, criticJson.start).trim() : "";
            const criticPostlude = criticJson ? turn.content.slice(criticJson.end).trim() : "";
            const majorIssues = typeof criticJson?.parsed.major_issues_remaining === "number"
              ? criticJson.parsed.major_issues_remaining
              : null;
            const minorIssues = typeof criticJson?.parsed.minor_issues_remaining === "number"
              ? criticJson.parsed.minor_issues_remaining
              : null;
            const hardViolations = Array.isArray(criticJson?.parsed.hard_constraint_violations)
              ? criticJson?.parsed.hard_constraint_violations.map((i) => String(i))
              : [];
            return (
              <div 
                key={turnKey}
                className="flex flex-col gap-2 animate-in slide-in-from-bottom-2 fade-in duration-300 ease-out"
              >
                <div className={clsx("flex gap-4", isUser ? "flex-row-reverse" : "flex-row")}>
                  <div className="shrink-0 mt-1">
                    {isUser ? (
                      <div className="w-8 h-8 rounded-full bg-indigo-500/20 border border-indigo-500/30 flex items-center justify-center">
                        <User className="w-4 h-4 text-indigo-400" />
                      </div>
                    ) : (
                      <div className="w-8 h-8 rounded-full bg-zinc-800 border border-zinc-700 flex items-center justify-center">
                        <Bot className="w-4 h-4 text-zinc-400" />
                      </div>
                    )}
                  </div>
                  
                  <div className="flex min-w-0 max-w-[85%] flex-col">
                    <div className={clsx(
                      "relative rounded-2xl px-4 py-3 text-sm shadow-sm overflow-hidden",
                      !isUser && "pr-11",
                      isUser 
                        ? "bg-indigo-500/10 border border-indigo-500/20 text-indigo-100 rounded-tr-sm" 
                        : "bg-zinc-800/50 border border-zinc-700/50 text-zinc-200 rounded-tl-sm"
                    )}>
                      {!isUser && turn.role === "author" && turnKey === latestAuthorTurnKey && latestResponseMode && (
                        <div className="mb-2">
                          <span
                            className={clsx(
                              "inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide",
                              latestResponseMode === "refine_round"
                                ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
                                : "border-blue-500/30 bg-blue-500/10 text-blue-300",
                            )}
                          >
                            {latestResponseMode === "refine_round" ? "Edited plan" : "Answered question"}
                          </span>
                        </div>
                      )}
                      {!isUser && (
                        <Tooltip content={copiedKey === turnKey ? "Copied" : "Copy response"}>
                          <button
                            type="button"
                            onClick={() => void copyMessage(turnKey, turn.content)}
                            className="absolute top-2 right-2 p-1 rounded-md text-zinc-500 hover:text-zinc-300 hover:bg-zinc-700/60 transition-colors"
                            aria-label="Copy message"
                          >
                            {copiedKey === turnKey ? (
                              <Check className="w-3.5 h-3.5" />
                            ) : (
                              <Copy className="w-3.5 h-3.5" />
                            )}
                          </button>
                        </Tooltip>
                      )}
                      {criticJson ? (
                        <div className="space-y-3">
                          {criticPrelude && (
                            <div className="chat-message-prose prose dark:prose-invert prose-zinc prose-sm max-w-none">
                              <ReactMarkdown remarkPlugins={[remarkGfm]} components={chatMarkdownComponents}>{criticPrelude}</ReactMarkdown>
                            </div>
                          )}
                          <div className="rounded-lg border border-zinc-700/70 bg-zinc-900/40 p-3">
                            <div className="flex flex-wrap items-center gap-2 text-xs">
                              {majorIssues !== null ? (
                                <span className="rounded bg-rose-500/20 px-2 py-1 text-rose-300">
                                  major: {majorIssues}
                                </span>
                              ) : null}
                              {minorIssues !== null ? (
                                <span className="rounded bg-amber-500/20 px-2 py-1 text-amber-300">
                                  minor: {minorIssues}
                                </span>
                              ) : null}
                              {hardViolations.length > 0 ? (
                                <span className="rounded bg-violet-500/20 px-2 py-1 text-violet-300">
                                  hard constraints: {hardViolations.join(", ")}
                                </span>
                              ) : null}
                            </div>
                            <details className="mt-2 text-xs text-zinc-400">
                              <summary className="cursor-pointer hover:text-zinc-200">
                                View structured critic payload
                              </summary>
                              <pre className="mt-2 overflow-x-auto whitespace-pre-wrap rounded bg-zinc-950/80 p-2 text-[11px] text-zinc-300">
                                {JSON.stringify(criticJson.parsed, null, 2)}
                              </pre>
                            </details>
                          </div>
                          {criticPostlude && (
                            <div className="chat-message-prose prose dark:prose-invert prose-zinc prose-sm max-w-none">
                              <ReactMarkdown>{criticPostlude}</ReactMarkdown>
                            </div>
                          )}
                        </div>
                      ) : (
                        <div className="chat-message-prose prose dark:prose-invert prose-zinc prose-sm max-w-none">
                          <ReactMarkdown remarkPlugins={[remarkGfm]} components={chatMarkdownComponents}>
                            {shouldCollapseDiscoveryQuestionText
                              ? `I found ${questions.length} clarifying question${questions.length === 1 ? "" : "s"}. Please answer the options below so I can draft the plan.`
                              : displayContent}
                          </ReactMarkdown>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
          {optimisticUserMessage && (
            <div className="flex flex-col gap-2 animate-in slide-in-from-bottom-2 fade-in duration-200 ease-out">
              <div className="flex gap-4 flex-row-reverse">
                <div className="shrink-0 mt-1">
                  <div className="w-8 h-8 rounded-full bg-indigo-500/20 border border-indigo-500/30 flex items-center justify-center">
                    <User className="w-4 h-4 text-indigo-400" />
                  </div>
                </div>
                <div className="flex min-w-0 max-w-[85%] flex-col items-end">
                  <div className="relative rounded-2xl px-4 py-3 text-sm shadow-sm overflow-hidden bg-indigo-500/10 border border-indigo-500/20 text-indigo-100 rounded-tr-sm opacity-60">
                    <div className="chat-message-prose prose dark:prose-invert prose-zinc prose-sm max-w-none">
                      <ReactMarkdown remarkPlugins={[remarkGfm]} components={chatMarkdownComponents}>{optimisticUserMessage}</ReactMarkdown>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
          {hasVisibleActiveToolStream && (
            <div className="flex justify-start w-full animate-in slide-in-from-bottom-2 fade-in duration-200 ease-out">
              <ToolCallStream toolCalls={activeToolCalls} forceRunning={isProcessing} />
            </div>
          )}
          {showDraftPlaceholderToolStream && (
            <div className="flex justify-start w-full animate-in slide-in-from-bottom-2 fade-in duration-200 ease-out">
              <ToolCallStream
                toolCalls={placeholderDraftToolCalls}
                forceRunning
              />
            </div>
          )}
          <div className="relative z-[60]">
            <OptionButtons
              questions={questions}
              selectedAnswers={selectedAnswers}
              selectedIsOther={selectedIsOther}
              otherInputs={otherInputs}
              onSelect={(questionIndex, optionText, isOther) => {
                setSelectedAnswers((prev) => ({ ...prev, [questionIndex]: optionText }));
                setSelectedIsOther((prev) => ({ ...prev, [questionIndex]: isOther }));
                if (!isOther) {
                  setOtherInputs((prev) => ({ ...prev, [questionIndex]: "" }));
                }
              }}
              onOtherInputChange={(questionIndex, value) => {
                setOtherInputs((prev) => ({ ...prev, [questionIndex]: value }));
              }}
            />
            {questions.length > 0 && (
              <>
                <div className="pl-12 mt-3 mb-24">
                  <div className="flex items-center justify-between gap-3">
                    <p className="text-xs text-zinc-500">
                      {answeredCount}/{questions.length} answered
                    </p>
                    <button
                      type="button"
                      disabled={inputDisabled || !allQuestionsAnswered || submittingAnswers}
                      onClick={() => void submitSelectedAnswers()}
                      className="rounded-md bg-indigo-500 px-3 py-1.5 text-xs font-medium text-white hover:bg-indigo-400 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {submittingAnswers ? "Submitting..." : "Submit all answers"}
                    </button>
                  </div>
                </div>
                {showSubmittingStatus && (
                  <p className="pl-12 mt-2 text-xs text-zinc-400 llm-status-glow">
                    <span className="llm-status-motion">Analyzing your answers and preparing the first draft</span>
                  </p>
                )}
              </>
            )}
            {pendingClarification && (
              <div className="ml-12 rounded-xl border border-indigo-500/30 bg-indigo-500/10 p-4 space-y-3">
              <p className="text-xs uppercase tracking-wide text-indigo-300">Clarification needed</p>
              <p className="text-sm text-zinc-100">{pendingClarification.question}</p>
              {pendingClarification.context && (
                <p className="text-xs text-zinc-400">{pendingClarification.context}</p>
              )}
              {clarificationRecommendations.length > 0 && (
                <div className="space-y-2">
                  <p className="text-[11px] uppercase tracking-wide text-zinc-400">
                    Recommended answers
                  </p>
                  <div className="grid gap-2">
                    {clarificationRecommendations.map((recommendation) => {
                      const isSelected = !useCustomClarification
                        && selectedClarificationRecommendation === recommendation;
                      return (
                        <button
                          key={recommendation}
                          type="button"
                          onClick={() => {
                            setUseCustomClarification(false);
                            setClarificationInput("");
                            setSelectedClarificationRecommendation(recommendation);
                          }}
                          className={clsx(
                            "w-full rounded-md border px-3 py-2 text-left text-xs transition-colors",
                            isSelected
                              ? "border-indigo-400 bg-indigo-500/20 text-indigo-100"
                              : "border-zinc-700 bg-zinc-900/50 text-zinc-300 hover:border-zinc-500 hover:text-zinc-100",
                          )}
                        >
                          {recommendation}
                        </button>
                      );
                    })}
                    <button
                      type="button"
                      onClick={() => {
                        setUseCustomClarification(true);
                        setSelectedClarificationRecommendation(null);
                      }}
                      className={clsx(
                        "w-full rounded-md border px-3 py-2 text-left text-xs transition-colors",
                        useCustomClarification
                          ? "border-indigo-400 bg-indigo-500/20 text-indigo-100"
                          : "border-zinc-700 bg-zinc-900/50 text-zinc-300 hover:border-zinc-500 hover:text-zinc-100",
                      )}
                    >
                      Other (custom answer)
                    </button>
                  </div>
                </div>
              )}
              <div className="flex gap-2">
                <input
                  value={clarificationInput}
                  onChange={(e) => setClarificationInput(e.target.value)}
                  placeholder="Type a custom answer..."
                  disabled={!useCustomClarification && Boolean(selectedClarificationRecommendation)}
                  className="flex-1 rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100"
                />
                <button
                  type="button"
                  disabled={!clarificationAnswer}
                  className="rounded-md bg-indigo-500 px-3 py-2 text-sm text-white hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-50"
                  onClick={() => {
                    const answer = clarificationAnswer;
                    setClarificationInput("");
                    setSelectedClarificationRecommendation(null);
                    setUseCustomClarification(false);
                    void onSubmitClarification(answer);
                  }}
                >
                  Submit
                </button>
                <button
                  type="button"
                  className="rounded-md border border-zinc-700 px-3 py-2 text-sm text-zinc-300 hover:bg-zinc-800"
                  onClick={() => {
                    setClarificationInput("");
                    setSelectedClarificationRecommendation(null);
                    setUseCustomClarification(false);
                    void onSubmitClarification("");
                  }}
                >
                  Skip
                </button>
              </div>
              </div>
            )}
            {!pendingClarification && hasPlanFollowupPanel && (
              <div className="ml-12 overflow-hidden rounded-2xl border border-emerald-500/20 bg-gradient-to-br from-emerald-950/30 to-zinc-900/50 p-0 shadow-lg shadow-emerald-900/10 backdrop-blur-sm animate-in fade-in slide-in-from-bottom-2 duration-500">
                {/* Header */}
                <div className="flex items-center gap-2 border-b border-emerald-500/10 bg-emerald-500/5 px-5 py-3">
                  <div className="flex h-6 w-6 items-center justify-center rounded-md bg-emerald-500/10 text-emerald-400 shadow-[0_0_10px_rgba(16,185,129,0.2)]">
                    <Zap className="h-3.5 w-3.5" />
                  </div>
                  <div>
                    <p className="text-xs font-bold uppercase tracking-wider text-emerald-400 drop-shadow-sm">Decisions to make</p>
                  </div>
                </div>

                <div className="p-5 space-y-5">
                  {/* Followups */}
                  {visiblePlanFollowups.length > 0 && (
                    <div className="space-y-4">
                      {visiblePlanFollowups.map((followup) => (
                        <div key={followup.id} className="group relative overflow-hidden rounded-xl border border-zinc-800 bg-zinc-900/40 p-4 transition-all duration-300 hover:border-emerald-500/30 hover:bg-zinc-900/60 hover:shadow-md hover:shadow-emerald-900/5">
                          <div className="mb-3 space-y-1">
                            <p className="text-sm font-medium text-zinc-100 leading-relaxed">{followup.question}</p>
                            {followup.target_sections.length > 0 && (
                              <p className="text-[10px] uppercase tracking-wide text-zinc-500 font-medium">
                                Updates: <span className="text-zinc-400">{followup.target_sections.join(", ")}</span>
                              </p>
                            )}
                          </div>
                          
                          {followup.options && followup.options.length > 0 ? (
                            <div className="flex flex-wrap gap-2">
                              {followup.options.map((option) => (
                                <button
                                  key={`${followup.id}-${option}`}
                                  type="button"
                                  disabled={Boolean(submittingFollowupId) || inputDisabled}
                                  onClick={() => void handleFollowupSubmit(followup.id, option)}
                                  className={clsx(
                                    "relative overflow-hidden rounded-lg border px-3.5 py-2 text-xs font-medium transition-all duration-200 active:scale-95",
                                    submittingFollowupId === followup.id
                                      ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300 cursor-wait"
                                      : "border-zinc-700/50 bg-zinc-800/50 text-zinc-300 hover:border-emerald-500/40 hover:bg-emerald-500/10 hover:text-emerald-300 hover:shadow-[0_0_10px_rgba(16,185,129,0.1)]"
                                  )}
                                >
                                  {submittingFollowupId === followup.id ? (
                                    <span className="flex items-center gap-1.5">
                                      <Loader2 className="h-3 w-3 animate-spin" />
                                      Applying...
                                    </span>
                                  ) : (
                                    option
                                  )}
                                </button>
                              ))}
                            </div>
                          ) : (
                            <button
                              type="button"
                              disabled={inputDisabled}
                              onClick={() => {
                                const seeded = `Follow-up: ${followup.question}`;
                                setInput((prev) => (prev.trim() ? `${prev}\n\n${seeded}` : seeded));
                                inputRef.current?.focus();
                              }}
                              className="flex items-center gap-2 rounded-lg border border-zinc-700/50 bg-zinc-800/30 px-3 py-2 text-xs font-medium text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
                            >
                              <MessageSquare className="h-3.5 w-3.5" />
                              Describe in chat
                            </button>
                          )}
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Optional Next Steps */}
                  {(planFollowups?.suggestions?.length ?? 0) > 0 && (
                    <div className={clsx("space-y-3", visiblePlanFollowups.length > 0 && "border-t border-zinc-800/60 pt-4")}>
                      <div className="flex items-center justify-between">
                        <p className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">Optional next steps</p>
                      </div>
                      <p className="text-xs text-zinc-400">
                        Click one to apply that small update to the current plan.
                      </p>
                      <div className="flex flex-wrap gap-2.5">
                        {planFollowups?.suggestions.map((suggestion) => (
                          <button
                            key={suggestion.id}
                            type="button"
                            disabled={inputDisabled || Boolean(submittingSuggestionId)}
                            onClick={() => void handleSuggestionSubmit(suggestion.id, suggestion.suggestion)}
                            className="group flex items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900/50 px-3 py-2 text-xs text-zinc-300 transition-all hover:border-zinc-600 hover:bg-zinc-800 hover:text-zinc-100 active:scale-95"
                          >
                            {submittingSuggestionId === suggestion.id ? (
                              <>
                                <Loader2 className="h-3 w-3 animate-spin" />
                                <span>Applying...</span>
                              </>
                            ) : (
                              <>
                                <span>{suggestion.suggestion}</span>
                                <ArrowRight className="h-3 w-3 text-zinc-500 opacity-0 transition-all group-hover:translate-x-0.5 group-hover:text-zinc-400 group-hover:opacity-100" />
                              </>
                            )}
                          </button>
                        ))}
                        {!isProcessing && canCritique && onCritique && (
                          <button
                            type="button"
                            onClick={() => void onCritique()}
                            className="group flex items-center gap-2 rounded-lg border border-indigo-500/20 bg-indigo-500/5 px-3 py-2 text-xs font-medium text-indigo-300 transition-all hover:border-indigo-500/40 hover:bg-indigo-500/10 hover:text-indigo-200 hover:shadow-[0_0_12px_rgba(99,102,241,0.15)] active:scale-95"
                          >
                            <Microscope className="h-3.5 w-3.5" />
                            Ask critic to review
                          </button>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
          {(liveStatusMessage || warningSummary.length > 0) && (
            <div className="mt-2 space-y-2 animate-in fade-in duration-200">
              {liveStatusMessage && (
                <div className="pl-12 flex items-center gap-2 text-xs text-zinc-300 llm-status-glow">
                  <div className="flex gap-1">
                    <div className="w-1 h-1 rounded-full bg-zinc-500 animate-bounce [animation-delay:-0.3s]"></div>
                    <div className="w-1 h-1 rounded-full bg-zinc-500 animate-bounce [animation-delay:-0.15s]"></div>
                    <div className="w-1 h-1 rounded-full bg-zinc-500 animate-bounce"></div>
                  </div>
                  <span className="llm-status-motion">{liveStatusMessage}</span>
                </div>
              )}
              {warningSummary.slice(-2).map((warning, idx) => (
                <div
                  key={`warning-${idx}`}
                  className="ml-12 max-w-[90%] rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-200/90"
                >
                  {warning.message}
                  {warning.count > 1 ? ` (x${warning.count})` : ""}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Floating Input Area */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 w-full max-w-3xl px-4 z-[80]">
        <div className={clsx(
          "bg-zinc-900 border rounded-2xl transition-all duration-300 chat-input-container",
          isProcessing
            ? "border-amber-500/30"
            : "border-zinc-700 focus-within:border-indigo-500/50"
        )}>
          {/* Textarea row */}
          {focusPrompt && (
            <div className="flex items-center gap-2 border-b border-zinc-700/80 px-4 pt-3 pb-2">
              <div className="min-w-0 flex items-center gap-2 rounded-full border border-indigo-500/20 bg-indigo-500/8 px-3 py-1.5 text-xs text-indigo-200">
                <span className="truncate">{focusPrompt.label}</span>
              </div>
              <div className="flex-1" />
              <button
                type="button"
                onClick={() => setFocusPrompt(null)}
                className="rounded-md p-1 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-300"
                aria-label="Remove focus"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>
          )}
          <textarea
            ref={inputRef}
            className="w-full max-h-64 min-h-[56px] bg-transparent text-zinc-100 placeholder:text-zinc-500/80 px-5 pt-4 pb-3 text-[15px] focus:outline-none resize-none overflow-y-auto leading-relaxed"
            value={input}
            disabled={inputDisabled}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                e.preventDefault();
                void handleSubmit();
              }
            }}
            placeholder={isProcessing ? "Working on it..." : focusPrompt ? "Add any extra guidance..." : "Type a message..."}
            rows={1}
            style={{ height: "auto" }}
            onInput={(e) => {
              const target = e.target as HTMLTextAreaElement;
              target.style.height = "auto";
              target.style.height = `${Math.min(target.scrollHeight, 256)}px`;
            }}
          />
          {/* Footer row */}
          <div className="flex flex-wrap sm:flex-nowrap items-center justify-between gap-2 sm:gap-3 px-3 pb-3 pt-2 border-t border-zinc-700">
            {/* Left: model selectors or processing status */}
            <div className="flex items-center gap-2 min-w-0 order-2 sm:order-1 w-full sm:w-auto">
              {isProcessing ? (
                <div className="flex items-center gap-2.5 px-3 py-1.5 bg-amber-500/10 border border-amber-500/20 rounded-xl shadow-[0_0_15px_rgba(245,158,11,0.1)] animate-in fade-in duration-300">
                  <Loader2 className="w-3.5 h-3.5 text-amber-400 animate-spin" />
                  <span className="text-[11px] text-amber-400/90 font-bold tracking-widest uppercase">Working</span>
                </div>
              ) : (
                <div className="flex items-center bg-zinc-900 border border-zinc-700/40 rounded-xl p-1 shadow-inner min-w-0 w-full sm:w-auto">
                  {authorModel !== undefined && onAuthorModelChange && (
                    <ModelSelector
                      label="Author"
                      value={authorModel}
                      onChange={onAuthorModelChange}
                      options={modelOptions}
                      icon={<Sparkles className="w-3.5 h-3.5" />}
                      className="hover:bg-zinc-800/80 rounded-lg px-2 py-1 flex-1 sm:flex-none justify-center sm:justify-start"
                      dropUp
                    />
                  )}
                  {(canCritique || canApprove) && criticModel !== undefined && onCriticModelChange && authorModel !== undefined && (
                    <>
                      <div className="w-px h-5 bg-zinc-700/50 mx-1 shrink-0" />
                      <ModelSelector
                        label="Critic"
                        value={criticModel}
                        onChange={onCriticModelChange}
                        options={modelOptions}
                        icon={<Microscope className="w-3.5 h-3.5" />}
                        className="hover:bg-zinc-800/80 rounded-lg px-2 py-1 flex-1 sm:flex-none justify-center sm:justify-start"
                        dropUp
                      />
                    </>
                  )}
                </div>
              )}
            </div>

            {/* Center: context dial */}
            <div className="flex items-center justify-end shrink-0 order-1 sm:order-2 mr-auto sm:mr-0 sm:flex-1 pr-1">
              <Tooltip content={`Context usage: ${(contextPercent ?? 0).toFixed(1)}%${contextWindowTokens ? ` of ${contextWindowTokens} max` : ''}`}>
                <div className="relative w-5 h-5 flex items-center justify-center">
                  <svg className="w-full h-full -rotate-90" viewBox="0 0 36 36">
                    {/* Background track */}
                    <circle
                      cx="18"
                      cy="18"
                      r="14"
                      fill="none"
                      className={clsx(
                        "transition-all duration-500",
                        isProcessing ? "stroke-amber-500/20 animate-pulse" : "stroke-zinc-700/50"
                      )}
                      strokeWidth="4"
                    />
                    {/* Progress arc */}
                    <circle
                      cx="18"
                      cy="18"
                      r="14"
                      fill="none"
                      className={clsx(
                        "transition-all duration-500 ease-out",
                        (contextPercent ?? 0) >= 85 ? "stroke-rose-500" : (contextPercent ?? 0) >= 70 ? "stroke-amber-500" : "stroke-zinc-400"
                      )}
                      strokeWidth="4"
                      strokeDasharray="88"
                      strokeDashoffset={88 - (88 * Math.min(100, contextPercent ?? 0)) / 100}
                      strokeLinecap="round"
                    />
                  </svg>
                </div>
              </Tooltip>
            </div>

            {/* Right: action buttons */}
            <div className="flex items-center justify-end gap-2 shrink-0 order-1 sm:order-3">
              {!isProcessing && canCritique && onCritique && (
                <Tooltip content="Run critique pass">
                  <button
                    type="button"
                    disabled={critiquePending}
                    onClick={onCritique}
                    className={clsx(
                      "flex items-center gap-2 px-3.5 py-2 text-sm font-medium rounded-xl border transition-all duration-200 active:scale-95",
                      critiquePending
                        ? "border-zinc-700 text-zinc-500 cursor-not-allowed"
                        : "border-zinc-700 bg-zinc-800/80 text-zinc-200 hover:bg-zinc-800 hover:border-zinc-600"
                    )}
                  >
                    {critiquePending ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Microscope className="w-4 h-4" />
                    )}
                    Review
                  </button>
                </Tooltip>
              )}
              {!isProcessing && canApprove && onApprove && (
                <Tooltip content="Approve plan">
                  <button
                    type="button"
                    onClick={onApprove}
                    className="flex items-center gap-2 px-4 py-2 text-sm font-semibold rounded-xl bg-zinc-100 text-zinc-900 hover:bg-zinc-50 transition-all duration-300 shadow-[0_4px_12px_rgba(255,255,255,0.1)] hover:shadow-[0_6px_16px_rgba(255,255,255,0.15)] active:scale-95"
                  >
                    <CheckCircle2 className="w-4 h-4" />
                    Approve
                  </button>
                </Tooltip>
              )}
              {isProcessing && onStop ? (
                <Tooltip content="Stop current process">
                  <button
                    type="button"
                    onClick={onStop}
                    className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-zinc-800/80 border border-zinc-700/50 text-zinc-300 hover:bg-rose-500/10 hover:text-rose-400 hover:border-rose-500/30 transition-all duration-300 active:scale-95 animate-in fade-in duration-300"
                  >
                    <Square className="w-3.5 h-3.5 fill-current" />
                    <span className="text-xs font-semibold tracking-wide uppercase">Stop</span>
                  </button>
                </Tooltip>
              ) : (
                <Tooltip content="Send message">
                  <button
                    type="button"
                    aria-label="Send message"
                    disabled={inputDisabled || !canSendMessage}
                    className={clsx(
                      "flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all duration-200 active:scale-95",
                      canSendMessage && !inputDisabled
                        ? "bg-indigo-500 text-white hover:bg-indigo-400 shadow-[0_4px_12px_rgba(99,102,241,0.3)]"
                        : "bg-zinc-800/80 text-zinc-500 cursor-not-allowed"
                    )}
                    onClick={() => void handleSubmit()}
                  >
                    <Send className="w-4 h-4" />
                    <span>Send</span>
                  </button>
                </Tooltip>
              )}
            </div>
          </div>
        </div>
      </div>

    </div>
  );
}
