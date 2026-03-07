import { ChevronRight, Search, Loader2, CheckCircle2, Bot, Pencil } from "lucide-react";
import { useState, useRef, useEffect } from "react";
import { clsx } from "clsx";
import type { ToolCallEntry } from "../types";

const PLAN_PHASE_NAMES: Record<string, { label: string; icon: "draft" | "critique" | "refine" | "search" }> = {
  draft_plan: { label: "Drafting plan", icon: "draft" },
  design_review: { label: "Design review", icon: "critique" },
  repair_planning: { label: "Planning repair", icon: "draft" },
  apply_critique: { label: "Revising design", icon: "refine" },
  review_validation: { label: "Validating changes", icon: "critique" },
  implementability_check: { label: "Checking implementability", icon: "critique" },
  lightweight_edit: { label: "Lightweight plan edit", icon: "refine" },
};

interface ToolCallStreamProps {
  toolCalls: ToolCallEntry[];
  defaultOpen?: boolean;
  className?: string;
  forceRunning?: boolean;
}

export function ToolCallStream({
  toolCalls,
  defaultOpen = false,
  className,
  forceRunning = false,
}: ToolCallStreamProps) {
  const [userOpen, setUserOpen] = useState<boolean | null>(null);
  const hasRunningCalls = toolCalls.some((call) => call.status === "running");
  const isEffectivelyRunning = forceRunning || hasRunningCalls;
  const count = toolCalls.length;
  
  // Stabilize the running state to prevent flickering during rapid transitions
  const lastRunningStateRef = useRef(isEffectivelyRunning);
  const runningStateChangeTimeRef = useRef(0);
  const [stableRunningState, setStableRunningState] = useState(isEffectivelyRunning);

  useEffect(() => {
    if (isEffectivelyRunning !== lastRunningStateRef.current) {
      lastRunningStateRef.current = isEffectivelyRunning;
      runningStateChangeTimeRef.current = Date.now();
      
      if (isEffectivelyRunning) {
        // Defer so we don't call setState synchronously in the effect
        const t = setTimeout(() => setStableRunningState(true), 0);
        return () => clearTimeout(t);
      } else {
        // Delay the transition to "not running" to prevent flicker
        const timer = setTimeout(() => {
          setStableRunningState(false);
        }, 150);
        return () => clearTimeout(timer);
      }
    }
  }, [isEffectivelyRunning]);
  
  const isOpen = userOpen ?? defaultOpen;

  if (toolCalls.length === 0) return null;

  const hasPlanPhase = toolCalls.some((c) => c.name in PLAN_PHASE_NAMES);
  const onlyPlanPhases = toolCalls.every((c) => c.name in PLAN_PHASE_NAMES);
  const runningPlanPhase = toolCalls.find((c) => c.name in PLAN_PHASE_NAMES && c.status === "running");

  // When exclusively plan-phase calls, use a planner-oriented summary label
  const summaryLabel = (() => {
    if (runningPlanPhase) return PLAN_PHASE_NAMES[runningPlanPhase.name]?.label ?? "Planning...";
    if (stableRunningState && !hasRunningCalls) return "Working...";
    if (onlyPlanPhases) return "Planning complete";
    if (stableRunningState) return `Running tools (${count})`;
    return `Tools completed (${count})`;
  })();

  const SummaryIcon = (() => {
    if (runningPlanPhase) {
      const iconKey = PLAN_PHASE_NAMES[runningPlanPhase.name]?.icon;
      if (iconKey === "critique") return <Loader2 className="w-3.5 h-3.5 text-amber-400 animate-spin" />;
      if (iconKey === "draft" || iconKey === "refine") {
        return <Loader2 className="w-3.5 h-3.5 text-indigo-400 animate-spin" />;
      }
    }
    if (stableRunningState) return <Loader2 className="w-3.5 h-3.5 text-indigo-400 animate-spin" />;
    return <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />;
  })();

  return (
    <div className={clsx("mt-2 mb-4", className)}>
      <div className="flex items-start gap-3">
        <div className="shrink-0 mt-0.5">
          <div
            className={clsx(
              "relative w-8 h-8 rounded-full border flex items-center justify-center transition-colors",
              stableRunningState
                ? "bg-indigo-500/15 border-indigo-400/50"
                : "bg-zinc-800 border-zinc-700",
            )}
          >
            <Bot className="w-4 h-4 text-zinc-400" />
          </div>
        </div>
        <div className="min-w-0">
          <button
            onClick={() => setUserOpen(!isOpen)}
            className="inline-flex items-center gap-2 rounded-md border border-zinc-800/80 bg-zinc-900/55 px-3 py-1.5 text-sm text-zinc-300 hover:border-zinc-700 hover:bg-zinc-900/75 transition-colors w-full min-w-0"
          >
            <ChevronRight className={clsx("w-3.5 h-3.5 text-zinc-500 transition-transform duration-200 shrink-0", isOpen && "rotate-90")} />
            {!hasPlanPhase && <Search className="w-4 h-4 text-zinc-500 shrink-0" />}
            {hasPlanPhase && toolCalls.some((c) => c.name === "draft_plan") && (
              <Pencil className="w-4 h-4 text-zinc-500 shrink-0" />
            )}
            <span className="min-w-0 truncate">{summaryLabel}</span>
            <span className="ml-auto shrink-0">{SummaryIcon}</span>
          </button>

          {isOpen && (
            <div className="mt-2 bg-zinc-950 border border-zinc-800 rounded-md p-3 text-[11px] text-zinc-400 font-mono overflow-x-auto shadow-inner">
          <ul className="space-y-1">
            {toolCalls.map((call) => {
              const planPhase = PLAN_PHASE_NAMES[call.name];
              return (
                <li key={call.id} className="flex items-start gap-2">
                  <span
                    className={clsx(
                      "mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full",
                      call.status === "done" ? "bg-emerald-400/80" : "bg-indigo-400/80 animate-pulse",
                    )}
                  />
                  <span>
                    <span className={clsx("text-zinc-200", planPhase && "font-semibold")}>
                      {planPhase ? planPhase.label : call.name}
                    </span>
                    {call.sessionStage && !planPhase ? <span className="text-zinc-500"> [{call.sessionStage}]</span> : null}
                    {planPhase && call.sessionStage ? <span className="text-zinc-500"> [{call.sessionStage}]</span> : null}
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
              );
            })}
          </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
