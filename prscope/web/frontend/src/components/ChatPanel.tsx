import { useEffect, useRef, useState } from "react";
import type { ClarificationPrompt, DiscoveryQuestion, PlanningTurn } from "../types";
import { OptionButtons } from "./OptionButtons";
import { ToolCallStream } from "./ToolCallStream";
import { Send, ArrowDown, Bot, User, Sparkles, Copy, Check } from "lucide-react";
import { clsx } from "clsx";
import ReactMarkdown from "react-markdown";
import { Tooltip } from "./ui/Tooltip";

interface ChatPanelProps {
  turns: PlanningTurn[];
  questions: DiscoveryQuestion[];
  toolCalls: string[];
  thinkingMessage: string | null;
  pendingClarification: ClarificationPrompt | null;
  onSubmit: (text: string) => Promise<void>;
  onSelectOption: (questionIndex: number, optionText: string, isOther: boolean) => Promise<void>;
  onSubmitClarification: (answer: string) => Promise<void>;
}

export function ChatPanel({
  turns,
  questions,
  toolCalls,
  thinkingMessage,
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
  }, [turns, toolCalls, thinkingMessage, questions, autoScroll]);

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
    // Legacy critic messages included raw JSON contract then prose; show prose only.
    if (turn.role === "critic" && raw.startsWith("{")) {
      const split = raw.split("\n\n");
      if (split.length > 1) {
        return split.slice(1).join("\n\n");
      }
    }
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
                  isUser 
                    ? "bg-indigo-500/10 border border-indigo-500/20 text-indigo-100 rounded-tr-sm" 
                    : "bg-zinc-800/50 border border-zinc-700/50 text-zinc-200 rounded-tl-sm"
                )}>
                  {!isUser && (
                    <Tooltip content={copiedKey === turnKey ? "Copied" : "Copy response"}>
                      <button
                        type="button"
                        onClick={() => void copyMessage(turnKey, displayContent)}
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
                  <div className="chat-message-prose prose prose-invert prose-sm max-w-none">
                    <ReactMarkdown>{displayContent}</ReactMarkdown>
                  </div>
                </div>
              </div>
            );
          })}

          <ToolCallStream toolCalls={toolCalls} />
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

          {thinkingMessage && (
            <div className="flex gap-4 animate-in fade-in duration-300">
              <div className="shrink-0 mt-1">
                <div className="w-8 h-8 rounded-full bg-zinc-800 border border-zinc-700 flex items-center justify-center">
                  <Bot className="w-4 h-4 text-zinc-400" />
                </div>
              </div>
              <div className="flex items-center gap-3 px-4 py-3 rounded-2xl rounded-tl-sm bg-zinc-800/30 border border-zinc-800">
                <div className="flex gap-1">
                  <div className="w-1.5 h-1.5 rounded-full bg-zinc-500 animate-bounce [animation-delay:-0.3s]"></div>
                  <div className="w-1.5 h-1.5 rounded-full bg-zinc-500 animate-bounce [animation-delay:-0.15s]"></div>
                  <div className="w-1.5 h-1.5 rounded-full bg-zinc-500 animate-bounce"></div>
                </div>
                <span className="text-sm font-medium text-transparent bg-clip-text bg-gradient-to-r from-zinc-400 via-zinc-200 to-zinc-400 animate-pulse">
                  {thinkingMessage}
                </span>
              </div>
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
      <div className="absolute bottom-6 left-6 right-6 max-w-2xl mx-auto w-full">
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
