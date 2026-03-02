import { useEffect, useMemo, useRef, useState } from "react";
import type { ClarificationPrompt, DiscoveryQuestion, PlanningTurn, ToolCallEntry } from "../types";
import { OptionButtons } from "./OptionButtons";
import { ToolCallStream } from "./ToolCallStream";
import { Send, ArrowDown, Bot, User, Sparkles, Copy, Check } from "lucide-react";
import { clsx } from "clsx";
import ReactMarkdown from "react-markdown";
import { Tooltip } from "./ui/Tooltip";

interface ChatPanelProps {
  turns: PlanningTurn[];
  questions: DiscoveryQuestion[];
  activeToolCalls: ToolCallEntry[];
  toolCallGroups: ToolCallEntry[][];
  thinkingMessage: string | null;
  warnings: string[];
  pendingClarification: ClarificationPrompt | null;
  onSubmit: (text: string) => Promise<void>;
  onSelectOption: (questionIndex: number, optionText: string, isOther: boolean) => Promise<void>;
  onSubmitClarification: (answer: string) => Promise<void>;
}

function extractFirstJsonObject(raw: string): { parsed: Record<string, unknown>; start: number; end: number } | null {
  const start = raw.indexOf("{");
  if (start < 0) return null;
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let idx = start; idx < raw.length; idx += 1) {
    const ch = raw[idx];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (ch === "\\") {
      escaped = true;
      continue;
    }
    if (ch === "\"") {
      inString = !inString;
      continue;
    }
    if (inString) continue;
    if (ch === "{") depth += 1;
    if (ch === "}") {
      depth -= 1;
      if (depth === 0) {
        const candidate = raw.slice(start, idx + 1);
        try {
          const parsed = JSON.parse(candidate) as Record<string, unknown>;
          return { parsed, start, end: idx + 1 };
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}

export function ChatPanel({
  turns,
  questions,
  activeToolCalls,
  toolCallGroups,
  thinkingMessage,
  warnings,
  pendingClarification,
  onSubmit,
  onSelectOption,
  onSubmitClarification,
}: ChatPanelProps) {
  const [input, setInput] = useState("");
  const [clarificationInput, setClarificationInput] = useState("");
  const [autoScroll, setAutoScroll] = useState(true);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

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
  }, [turns, activeToolCalls, thinkingMessage, questions, autoScroll]);

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

  const handleSubmit = async () => {
    const text = input.trim();
    if (!text) return;
    setInput("");
    await onSubmit(text);
  };

  const onScroll = () => {
    setAutoScroll(isNearBottom());
  };

  const scrollToBottom = () => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
    setAutoScroll(true);
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
    return raw;
  };

  const warningSummary = warnings.reduce<Array<{ message: string; count: number }>>((acc, message) => {
    const existing = acc.find((item) => item.message === message);
    if (existing) {
      existing.count += 1;
    } else {
      acc.push({ message, count: 1 });
    }
    return acc;
  }, []);

  const toolCallsByTurnIndex = useMemo(() => {
    const mapping = new Map<number, ToolCallEntry[]>();
    const nonUserIndexes = turns
      .map((turn, idx) => ({ turn, idx }))
      .filter(({ turn }) => turn.role !== "user")
      .map(({ idx }) => idx);
    if (nonUserIndexes.length === 0 || toolCallGroups.length === 0) {
      return mapping;
    }
    const start = Math.max(0, nonUserIndexes.length - toolCallGroups.length);
    for (let groupIdx = 0; groupIdx < toolCallGroups.length; groupIdx += 1) {
      const turnIdx = nonUserIndexes[start + groupIdx];
      if (turnIdx !== undefined) {
        mapping.set(turnIdx, toolCallGroups[groupIdx]);
      }
    }
    return mapping;
  }, [toolCallGroups, turns]);

  return (
    <div className="h-full flex flex-col bg-zinc-900/30 relative">
      
      {/* Scrollable Message Area */}
      <div 
        ref={scrollRef} 
        className="flex-1 overflow-y-auto px-6 pt-6 pb-40 scroll-smooth" 
        onScroll={onScroll}
      >
        <div className="max-w-2xl mx-auto space-y-6">
          {turns.length === 0 && !thinkingMessage && (
            <div className="flex flex-col items-center justify-center py-12 text-zinc-500">
              <Sparkles className="w-8 h-8 mb-3 opacity-50" />
              <p className="text-sm">Start the conversation to begin planning.</p>
            </div>
          )}

          {turns.map((turn, idx) => {
            const isUser = turn.role === "user";
            const turnKey = `${turn.role}-${idx}`;
            const displayContent = renderTurnContent(turn);
            const turnToolCalls = toolCallsByTurnIndex.get(idx) ?? [];
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
              ? criticJson?.parsed.hard_constraint_violations.map((item) => String(item))
              : [];
            return (
              <div 
                key={turnKey}
                className={clsx(
                  "flex gap-4 animate-in slide-in-from-bottom-2 fade-in duration-300 ease-out",
                  isUser ? "flex-row-reverse" : "flex-row"
                )}
              >
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
                
                <div className={clsx(
                  "relative",
                  "max-w-[85%] rounded-2xl px-4 py-3 text-sm shadow-sm overflow-hidden",
                  !isUser && "pr-11",
                  isUser 
                    ? "bg-indigo-500/10 border border-indigo-500/20 text-indigo-100 rounded-tr-sm" 
                    : "bg-zinc-800/50 border border-zinc-700/50 text-zinc-200 rounded-tl-sm"
                )}>
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
                        <div className="chat-message-prose prose prose-invert prose-sm max-w-none">
                          <ReactMarkdown>{criticPrelude}</ReactMarkdown>
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
                        <div className="chat-message-prose prose prose-invert prose-sm max-w-none">
                          <ReactMarkdown>{criticPostlude}</ReactMarkdown>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="chat-message-prose prose prose-invert prose-sm max-w-none">
                      <ReactMarkdown>{displayContent}</ReactMarkdown>
                    </div>
                  )}
                </div>
                {!isUser && turnToolCalls.length > 0 && (
                  <div className="pl-12">
                    <ToolCallStream toolCalls={turnToolCalls} />
                  </div>
                )}
              </div>
            );
          })}

          <ToolCallStream toolCalls={activeToolCalls} />
          <OptionButtons questions={questions} onSelect={onSelectOption} />
          {pendingClarification && (
            <div className="rounded-xl border border-indigo-500/30 bg-indigo-500/10 p-4 space-y-3">
              <p className="text-xs uppercase tracking-wide text-indigo-300">Clarification needed</p>
              <p className="text-sm text-zinc-100">{pendingClarification.question}</p>
              {pendingClarification.context && (
                <p className="text-xs text-zinc-400">{pendingClarification.context}</p>
              )}
              <div className="flex gap-2">
                <input
                  value={clarificationInput}
                  onChange={(e) => setClarificationInput(e.target.value)}
                  placeholder="Type your answer..."
                  className="flex-1 rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100"
                />
                <button
                  type="button"
                  className="rounded-md bg-indigo-500 px-3 py-2 text-sm text-white hover:bg-indigo-400"
                  onClick={() => {
                    const answer = clarificationInput;
                    setClarificationInput("");
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
                    void onSubmitClarification("");
                  }}
                >
                  Skip
                </button>
              </div>
            </div>
          )}

          {(warnings.length > 0 || thinkingMessage) && (
            <div className="space-y-1 animate-in fade-in duration-200">
              {warningSummary.slice(-4).map((warning, idx) => (
                <div key={`warning-${idx}`} className="pl-12 text-xs text-amber-300/80">
                  {warning.message}
                  {warning.count > 1 ? ` (x${warning.count})` : ""}
                </div>
              ))}
              {thinkingMessage && (
                <div className="pl-12 flex items-center gap-2 text-xs text-zinc-400">
                  <div className="flex gap-1">
                    <div className="w-1 h-1 rounded-full bg-zinc-500 animate-bounce [animation-delay:-0.3s]"></div>
                    <div className="w-1 h-1 rounded-full bg-zinc-500 animate-bounce [animation-delay:-0.15s]"></div>
                    <div className="w-1 h-1 rounded-full bg-zinc-500 animate-bounce"></div>
                  </div>
                  <span>{thinkingMessage}</span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Floating "Jump to latest" button - above input, clearly outside message flow */}
      {!autoScroll && (
        <div className="absolute bottom-32 left-1/2 -translate-x-1/2 z-50 pointer-events-auto animate-in slide-in-from-bottom-2 fade-in duration-200">
          <button
            type="button"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-zinc-800 border border-zinc-600 text-xs font-medium text-zinc-200 hover:text-white hover:bg-zinc-700 shadow-lg ring-1 ring-black/20 transition-all"
            onClick={scrollToBottom}
          >
            <ArrowDown className="w-3.5 h-3.5" />
            New activity
          </button>
        </div>
      )}

      {/* Floating Input Area */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 w-full max-w-2xl px-6">
        <div className="bg-zinc-800/80 backdrop-blur-xl border border-zinc-700/80 rounded-2xl shadow-2xl p-2 flex items-end focus-within:ring-2 ring-indigo-500/50 focus-within:border-indigo-500/50 transition-all">
          <textarea
            ref={inputRef}
            className="w-full max-h-48 min-h-[44px] bg-transparent text-zinc-100 placeholder:text-zinc-500 px-3 py-2.5 text-sm focus:outline-none resize-none overflow-y-auto"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                e.preventDefault();
                void handleSubmit();
              }
            }}
            placeholder="Type a message..."
            rows={1}
            style={{ height: "auto" }}
            onInput={(e) => {
              const target = e.target as HTMLTextAreaElement;
              target.style.height = "auto";
              target.style.height = `${Math.min(target.scrollHeight, 192)}px`;
            }}
          />
          
          <div className="flex items-center gap-2 shrink-0 p-1">
            <Tooltip content="Send message (⌘ + ↵)">
              <div className="hidden sm:flex items-center gap-1 text-[10px] font-mono text-zinc-500 select-none mr-1">
                <kbd className={clsx("px-1.5 py-0.5 rounded border border-zinc-700 transition-colors", input.trim() && "bg-zinc-700 text-zinc-300")}>⌘</kbd>
                <kbd className={clsx("px-1.5 py-0.5 rounded border border-zinc-700 transition-colors", input.trim() && "bg-zinc-700 text-zinc-300")}>↵</kbd>
              </div>
            </Tooltip>
            <Tooltip content="Send message">
              <button
                type="button"
                disabled={!input.trim()}
                className="p-2 rounded-xl bg-indigo-500 text-white hover:bg-indigo-400 disabled:opacity-50 disabled:hover:bg-indigo-500 transition-colors shadow-sm active:scale-95"
                onClick={() => void handleSubmit()}
              >
                <Send className="w-4 h-4" />
              </button>
            </Tooltip>
          </div>
        </div>
      </div>

    </div>
  );
}
