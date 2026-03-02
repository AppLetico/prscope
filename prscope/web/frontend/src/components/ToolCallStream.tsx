import { Terminal, ChevronRight } from "lucide-react";
import { useState } from "react";
import { clsx } from "clsx";
import type { ToolCallEntry } from "../types";

interface ToolCallStreamProps {
  toolCalls: ToolCallEntry[];
}

export function ToolCallStream({ toolCalls }: ToolCallStreamProps) {
  const [isOpen, setIsOpen] = useState(false);

  if (toolCalls.length === 0) return null;

  return (
    <div className="mt-2 mb-4">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 text-xs text-zinc-500 font-mono hover:text-zinc-300 cursor-pointer py-1 transition-colors group"
      >
        <ChevronRight className={clsx("w-3 h-3 transition-transform duration-200", isOpen && "rotate-90")} />
        <Terminal className="w-3 h-3" />
        Executed {toolCalls.length} codebase quer{toolCalls.length === 1 ? 'y' : 'ies'}...
      </button>
      
      {isOpen && (
        <div className="mt-2 bg-zinc-950 border border-zinc-800 rounded-md p-3 text-[11px] text-zinc-400 font-mono overflow-x-auto shadow-inner">
          <ul className="space-y-1">
            {toolCalls.map((call) => (
              <li key={call.id} className="flex items-start gap-2">
                <span className="text-zinc-600 select-none">{'>'}</span>
                <span>
                  <span className="text-zinc-200">{call.name}</span>
                  {call.sessionStage ? <span className="text-zinc-500"> [{call.sessionStage}]</span> : null}
                  {call.path ? <span className="text-zinc-400"> path={call.path}</span> : null}
                  {call.query ? <span className="text-zinc-400"> query=&quot;{call.query}&quot;</span> : null}
                  {call.status === "done" ? (
                    <span className="text-emerald-400"> ✓ {call.durationMs !== undefined ? `${call.durationMs}ms` : ""}</span>
                  ) : (
                    <span className="text-zinc-500"> ...</span>
                  )}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
