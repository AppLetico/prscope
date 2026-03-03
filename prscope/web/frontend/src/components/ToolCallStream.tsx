import { ChevronRight, Search, Loader2, CheckCircle2 } from "lucide-react";
import { useState } from "react";
import { clsx } from "clsx";
import type { ToolCallEntry } from "../types";

interface ToolCallStreamProps {
  toolCalls: ToolCallEntry[];
  defaultOpen?: boolean;
  className?: string;
}

export function ToolCallStream({ toolCalls, defaultOpen = false, className }: ToolCallStreamProps) {
  const [userOpen, setUserOpen] = useState<boolean | null>(null);
  const hasRunningCalls = toolCalls.some((call) => call.status === "running");
  const count = toolCalls.length;
  const isOpen = hasRunningCalls ? true : (userOpen ?? defaultOpen);

  if (toolCalls.length === 0) return null;

  return (
    <div className={clsx("mt-2 mb-4", className)}>
      <button
        onClick={() => setUserOpen(!isOpen)}
        className="inline-flex items-center gap-2 rounded-md border border-zinc-800/80 bg-zinc-900/55 px-2.5 py-1 text-[11px] text-zinc-300 hover:border-zinc-700 hover:bg-zinc-900/75 transition-colors"
      >
        <ChevronRight className={clsx("w-3 h-3 text-zinc-500 transition-transform duration-200", isOpen && "rotate-90")} />
        {hasRunningCalls ? (
          <Loader2 className="w-3.5 h-3.5 text-indigo-400 animate-spin" />
        ) : (
          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
        )}
        <Search className="w-3.5 h-3.5 text-zinc-500" />
        <span>
          {hasRunningCalls
            ? `Scanning codebase (${count})`
            : `Scanned codebase (${count})`}
        </span>
      </button>
      
      {isOpen && (
        <div className="mt-2 bg-zinc-950 border border-zinc-800 rounded-md p-3 text-[11px] text-zinc-400 font-mono overflow-x-auto shadow-inner">
          <ul className="space-y-1">
            {toolCalls.map((call) => (
              <li key={call.id} className="flex items-start gap-2">
                <span
                  className={clsx(
                    "mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full",
                    call.status === "done" ? "bg-emerald-400/80" : "bg-indigo-400/80 animate-pulse",
                  )}
                />
                <span>
                  <span className="text-zinc-200">{call.name}</span>
                  {call.sessionStage ? <span className="text-zinc-500"> [{call.sessionStage}]</span> : null}
                  {call.path ? <span className="text-zinc-400"> path={call.path}</span> : null}
                  {call.query ? <span className="text-zinc-400"> query=&quot;{call.query}&quot;</span> : null}
                  <span className="text-zinc-500 ml-1">
                    {call.status === "done"
                      ? call.durationMs !== undefined
                        ? `${call.durationMs}ms`
                        : "done"
                      : "..."}
                  </span>
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
