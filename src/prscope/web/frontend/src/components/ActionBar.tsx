import type { DraftTimingDiagnostics, RoundMetric, SessionStatus } from "../types";
import { ChevronDown, Trash2 } from "lucide-react";
import { Link } from "react-router-dom";
import { clsx } from "clsx";
import { useRef, useEffect, useState, useMemo } from "react";
import { Tooltip } from "./ui/Tooltip";

import { ThemeToggle } from "./ThemeToggle";

interface ActionBarProps {
  repoName: string;
  title: string;
  round: number;
  status: SessionStatus;
  convergenceScore?: number;
  sessionCostUsd?: number;
  maxPromptTokens?: number;
  contextWindowTokens?: number | null;
  contextPercent?: number | null;
  contextCompactionEnabled?: boolean;
  routingDiagnostics?: DraftTimingDiagnostics | null;
  onDelete?: () => void;
  roundMetrics?: RoundMetric[];
}

function scoreColor(score: number | null | undefined): string {
  if (score == null) return "text-zinc-500";
  if (score >= 0.85) return "text-emerald-400";
  if (score >= 0.65) return "text-amber-400";
  return "text-rose-400";
}

export function ActionBar({
  repoName,
  title,
  round,
  status,
  convergenceScore = 0,
  sessionCostUsd = 0,
  maxPromptTokens = 0,
  contextCompactionEnabled = false,
  routingDiagnostics = null,
  onDelete,
  roundMetrics,
}: ActionBarProps) {
  const [moreOpen, setMoreOpen] = useState(false);
  const [convOpen, setConvOpen] = useState(false);
  const moreRef = useRef<HTMLDivElement>(null);
  const convRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!moreOpen) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (moreRef.current && !moreRef.current.contains(e.target as Node)) {
        setMoreOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [moreOpen]);

  useEffect(() => {
    if (!convOpen) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (convRef.current && !convRef.current.contains(e.target as Node)) {
        setConvOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [convOpen]);

  const sortedMetrics = useMemo(
    () => (roundMetrics ? [...roundMetrics].sort((a, b) => b.round - a.round) : []),
    [roundMetrics],
  );
  const displayRevision = Math.max(1, round + 1);
  const latestScoredRound = sortedMetrics.length > 0 ? Math.max(1, sortedMetrics[0].round + 1) : null;
  const hasUnscoredRevision = latestScoredRound !== null && latestScoredRound < displayRevision;
  const routingDecisionCount = Number(routingDiagnostics?.routing_decisions_total ?? 0);
  const routingHeuristicCount = Number(routingDiagnostics?.routing_heuristic_decisions ?? 0);
  const routingModelCount = Number(routingDiagnostics?.routing_model_decisions ?? 0);
  const routingFallbackCount = Number(routingDiagnostics?.routing_fallback_decisions ?? 0);
  const routeAuthorChatCount = Number(routingDiagnostics?.route_author_chat_total ?? 0);
  const routeLightweightCount = Number(routingDiagnostics?.route_lightweight_refine_total ?? 0);
  const routeFullRefineCount = Number(routingDiagnostics?.route_full_refine_total ?? 0);
  const routeExistingFeatureCount = Number(routingDiagnostics?.route_existing_feature_total ?? 0);

  const totalCost = useMemo(
    () => sortedMetrics.reduce((sum, m) => sum + (m.call_cost_usd ?? 0), 0),
    [sortedMetrics],
  );
  const isConverged = status === "converged" || status === "approved";
  const isRefining = status === "refining" || status === "draft";
  const statusTooltip = {
    draft: "Collecting requirements and preparing the first plan draft.",
    refining: "Refining plan with critique rounds.",
    converged: "Plan has converged and is ready for approval.",
    approved: "Plan approved.",
    error: "Session encountered an error and needs attention.",
  }[status];
  
  return (
    <div className="h-14 flex items-center justify-between px-4 border-b border-zinc-800 bg-zinc-950/80 backdrop-blur-md sticky top-0 z-[100]">
      
      {/* Left: Context */}
      <div className="flex items-center gap-3 flex-1 min-w-0">
        <Link
          to="/"
          className="flex items-center justify-center p-1 -ml-1 text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800/50 rounded-md transition-colors"
          title="Back to sessions"
        >
          <img src="/logo.svg" alt="prscope" className="w-5 h-5 rounded-md" />
        </Link>
        <div className="flex items-center gap-2 text-sm truncate">
          <span className="font-mono text-zinc-500 truncate max-w-[120px]">{repoName}</span>
          <span className="text-zinc-700">/</span>
          <span className="font-semibold text-zinc-100 truncate max-w-[300px] text-base">{title}</span>
        </div>
      </div>

      {/* Center: Status Pill */}
      <div className="flex items-center justify-center flex-1">
        <div
          ref={convRef}
          role={(status === "refining" || status === "converged") ? "button" : undefined}
          tabIndex={(status === "refining" || status === "converged") ? 0 : undefined}
          onClick={(status === "refining" || status === "converged") ? () => setConvOpen((v) => !v) : undefined}
          onKeyDown={
            (status === "refining" || status === "converged")
              ? (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setConvOpen((v) => !v); } }
              : undefined
          }
          className={clsx(
            "flex items-center gap-2 sm:gap-3 bg-zinc-900/90 border border-zinc-800 rounded-full px-2 sm:px-3 py-1 sm:py-1.5 shadow-sm",
            (status === "refining" || status === "converged") && "cursor-pointer hover:bg-zinc-800/60 transition-colors"
          )}
          aria-expanded={(status === "refining" || status === "converged") ? convOpen : undefined}
        >
          <div className="flex items-center gap-1 sm:gap-2">
            <Tooltip content={statusTooltip}>
              <div className="flex items-center gap-2 cursor-default">
                <div className="flex items-center h-5 px-2 rounded-full bg-zinc-800/50 border border-zinc-700/30 text-[10px] font-mono font-medium text-zinc-400 shadow-sm">
                  Rev <span className="ml-1 text-zinc-200">{displayRevision}</span>
                </div>
                <div className="flex items-center gap-1.5">
                  {isConverged ? (
                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]"></div>
                  ) : isRefining ? (
                    <div className="w-1.5 h-1.5 rounded-full bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.5)] animate-pulse"></div>
                  ) : (
                    <div className="w-1.5 h-1.5 rounded-full bg-zinc-500"></div>
                  )}
                  <span className={clsx(
                    "text-[10px] font-bold uppercase tracking-widest hidden sm:inline-block",
                    isConverged ? "text-emerald-400" : isRefining ? "text-amber-400" : "text-zinc-400"
                  )}>
                    {status}
                  </span>
                </div>
              </div>
            </Tooltip>
          </div>
          
          {(status === "refining" || status === "converged") && (
            <>
              <div className="w-px h-3 bg-zinc-800"></div>
              <div className="relative">
                <Tooltip
                  content={
                    hasUnscoredRevision && latestScoredRound !== null
                      ? `Current revision: R${displayRevision}. Latest scored critique: R${latestScoredRound}. Run Review to refresh the score.`
                      : `Convergence: ${Math.round(convergenceScore * 100)}% — click for round breakdown`
                  }
                >
                  <div
                    className={clsx(
                      "flex items-center gap-1 sm:gap-2 px-1 sm:px-2 py-1 -my-1 rounded-md transition-colors pointer-events-none",
                      convOpen ? "bg-zinc-800/80" : ""
                    )}
                  >
                    {hasUnscoredRevision && latestScoredRound !== null ? (
                      <>
                        <span className="text-[10px] font-mono text-zinc-500 hidden sm:inline-block">review</span>
                        <span className="text-[10px] font-mono text-zinc-300">R{latestScoredRound}</span>
                        <span className="hidden sm:inline-flex rounded-full border border-amber-500/30 bg-amber-500/10 px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wider text-amber-400">
                          stale
                        </span>
                      </>
                    ) : (
                      <>
                        <span className="text-[10px] font-mono text-zinc-500 hidden sm:inline-block">conv</span>
                        <div className="w-8 sm:w-16 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                          <div
                            className={clsx(
                              "h-full transition-all duration-1000 ease-out rounded-full",
                              isConverged ? "bg-emerald-500" : "bg-gradient-to-r from-amber-500 to-amber-400",
                            )}
                            style={{ width: `${Math.max(10, Math.min(100, convergenceScore * 100))}%` }}
                          ></div>
                        </div>
                      </>
                    )}
                  </div>
                </Tooltip>

                {convOpen && (
                  <div className="absolute left-1/2 -translate-x-1/2 top-full mt-3 w-[300px] sm:w-[480px] rounded-xl border border-zinc-800 bg-zinc-900/95 backdrop-blur-xl shadow-2xl z-[100] p-4 overflow-hidden">
                    <div className="flex items-center justify-between mb-3">
                      <div className="text-[11px] font-semibold text-zinc-300 uppercase tracking-widest">
                        Revision and Critique History
                      </div>
                    </div>
                    {hasUnscoredRevision && latestScoredRound !== null && (
                      <div className="mb-4 rounded-lg border border-amber-500/20 bg-gradient-to-br from-amber-500/10 to-amber-500/5 p-3">
                        <div className="flex flex-wrap items-center gap-2 mb-2">
                          <span className="flex items-center gap-1.5 rounded-md bg-zinc-950/50 px-2 py-1 text-[11px] font-medium text-zinc-300 shadow-sm ring-1 ring-inset ring-white/10">
                            <span className="text-zinc-500">Revision</span>
                            <span className="font-mono text-zinc-200">R{displayRevision}</span>
                          </span>
                          <span className="text-zinc-600">→</span>
                          <span className="flex items-center gap-1.5 rounded-md bg-zinc-950/50 px-2 py-1 text-[11px] font-medium text-zinc-300 shadow-sm ring-1 ring-inset ring-white/10">
                            <span className="text-zinc-500">Last Critique</span>
                            <span className="font-mono text-zinc-200">R{latestScoredRound}</span>
                          </span>
                          <span className="ml-auto inline-flex items-center rounded-full bg-amber-400/10 px-2 py-0.5 text-[10px] font-medium text-amber-400 ring-1 ring-inset ring-amber-400/20">
                            Stale Score
                          </span>
                        </div>
                        <p className="text-[11px] leading-relaxed text-amber-200/80">
                          Lightweight edits changed the draft after the last full critique. This panel shows the most recent review score, not a score for the current revision.
                        </p>
                      </div>
                    )}
                    {sortedMetrics.length === 0 ? (
                      <div className="text-xs text-zinc-500 py-4 text-center bg-zinc-900/30 rounded-lg border border-zinc-800">No round data yet</div>
                    ) : (
                      <>
                        <div className="overflow-x-auto">
                          <table className="w-full text-xs border-collapse">
                            <thead>
                              <tr className="text-[10px] text-zinc-500 uppercase tracking-wider border-b border-zinc-800">
                                <th className="text-left pb-2.5 font-medium pl-2">Critique</th>
                                <th className="text-left pb-2.5 font-medium">Progress</th>
                                <th className="text-right pb-2.5 font-medium">Score</th>
                                <th className="text-right pb-2.5 font-medium hidden sm:table-cell">Major</th>
                                <th className="text-right pb-2.5 font-medium hidden sm:table-cell">Minor</th>
                                <th className="text-right pb-2.5 font-medium hidden sm:table-cell">Conf</th>
                                <th className="text-right pb-2.5 font-medium pr-2">Cost</th>
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-zinc-800/50">
                              {sortedMetrics.map((m) => {
                                const pct = m.convergence_score != null ? m.convergence_score : null;
                                const barWidth = pct != null ? Math.round(pct * 100) : 0;
                                const isLatestScored = latestScoredRound === Math.max(1, m.round + 1);
                                return (
                                  <tr key={m.round} className="group hover:bg-zinc-900/50 transition-colors">
                                    <td className="py-2.5 pl-2 pr-3 text-zinc-400 font-mono text-[11px]">
                                      <div className="flex items-center gap-2">
                                        <span>R{Math.max(1, m.round + 1)}</span>
                                        {isLatestScored && (
                                          <span className="rounded-full border border-zinc-700/60 bg-zinc-800/60 px-1.5 py-0.5 text-[9px] uppercase tracking-wider text-zinc-400">
                                            latest
                                          </span>
                                        )}
                                      </div>
                                    </td>
                                    <td className="py-2.5 pr-3">
                                      <div className="w-12 sm:w-20 bg-zinc-800/80 rounded-full h-1.5 overflow-hidden">
                                        <div
                                          style={{ width: `${barWidth}%` }}
                                          className={clsx(
                                            "h-full rounded-full transition-all duration-500",
                                            pct != null && pct >= 0.85
                                              ? "bg-emerald-500"
                                              : pct != null && pct >= 0.65
                                                ? "bg-amber-500"
                                                : "bg-rose-500",
                                          )}
                                        />
                                      </div>
                                    </td>
                                    <td className={clsx("py-2.5 pr-3 text-right font-mono text-[11px]", scoreColor(pct))}>
                                      {pct != null ? `${Math.round(pct * 100)}%` : "—"}
                                    </td>
                                    <td className="py-2.5 pr-3 text-right text-zinc-400 font-mono text-[11px] hidden sm:table-cell">
                                      {m.major_issues ?? "—"}
                                    </td>
                                    <td className="py-2.5 pr-3 text-right text-zinc-500 font-mono text-[11px] hidden sm:table-cell">
                                      {m.minor_issues ?? "—"}
                                    </td>
                                    <td className="py-2.5 pr-3 text-right text-zinc-500 font-mono text-[11px] hidden sm:table-cell">
                                      {m.critic_confidence != null
                                        ? `${Math.round(m.critic_confidence * 100)}%`
                                        : "—"}
                                    </td>
                                    <td className="py-2.5 pr-2 text-right text-zinc-400 font-mono text-[11px]">
                                      {m.call_cost_usd != null ? `$${m.call_cost_usd.toFixed(3)}` : "—"}
                                    </td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                        {totalCost > 0 && (
                          <div className="flex justify-between items-center border-t border-zinc-800 pt-3 mt-2">
                            <span className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium">Total Planning Cost</span>
                            <span className="text-xs font-mono text-zinc-300 bg-zinc-800/50 px-2 py-1 rounded border border-zinc-700/50">${totalCost.toFixed(3)}</span>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                )}
              </div>
            </>
          )}

          <div className="w-px h-3 bg-zinc-800 hidden sm:block"></div>
          
          <div className="hidden sm:flex items-center gap-3 text-[10px] font-mono px-1">
            <Tooltip content="Total session cost">
              <div className="flex items-center gap-1.5 cursor-default">
                <span className="text-zinc-500">cost</span>
                <span className={sessionCostUsd > 0.5 ? "text-rose-400" : sessionCostUsd > 0.1 ? "text-amber-400" : "text-zinc-300"}>
                  ${sessionCostUsd.toFixed(4)}
                </span>
              </div>
            </Tooltip>
            <Tooltip content="Max prompt tokens used">
              <div className="flex items-center gap-1.5 cursor-default">
                <span className="text-zinc-500">tkns</span>
                <span className="text-zinc-300">
                  {maxPromptTokens >= 1000 ? `${(maxPromptTokens / 1000).toFixed(1)}k` : maxPromptTokens}
                </span>
              </div>
            </Tooltip>
          </div>
        </div>
      </div>

      {/* Right: Actions */}
      <div className="flex items-center justify-end gap-3 flex-1">
        {/* Stats */}
        <div className="hidden xl:flex items-center gap-4 text-[10px] font-mono text-zinc-500 mr-2">
          {contextCompactionEnabled && (
            <Tooltip content="Context compaction is active">
              <span className="text-indigo-400 bg-indigo-500/10 px-1.5 py-0.5 rounded border border-indigo-500/20">compacted</span>
            </Tooltip>
          )}
          {routingDecisionCount > 0 && (
            <Tooltip
              content={(
                <div className="space-y-1 text-[11px]">
                  <div>Routing decisions: {routingDecisionCount}</div>
                  <div>Heuristic: {routingHeuristicCount}</div>
                  <div>Model: {routingModelCount}</div>
                  <div>Fallback: {routingFallbackCount}</div>
                  <div>Author chat: {routeAuthorChatCount}</div>
                  <div>Lightweight refine: {routeLightweightCount}</div>
                  <div>Full refine: {routeFullRefineCount}</div>
                  <div>Existing-feature routes: {routeExistingFeatureCount}</div>
                </div>
              )}
            >
              <span className="text-cyan-400 bg-cyan-500/10 px-1.5 py-0.5 rounded border border-cyan-500/20">
                routing {routingDecisionCount}
              </span>
            </Tooltip>
          )}
        </div>

        <div className="flex items-center gap-1.5 ml-1 border-l border-zinc-800 pl-3">
          <ThemeToggle />
          {onDelete != null && (
            <div className={clsx("relative", moreOpen && "z-[9999]")} ref={moreRef}>
              <Tooltip content="More actions">
                <button
                  type="button"
                  onClick={() => setMoreOpen((v) => !v)}
                  className={clsx(
                    "p-2 rounded-md transition-all duration-200",
                    moreOpen ? "bg-zinc-800 text-zinc-200" : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/80"
                  )}
                  aria-expanded={moreOpen}
                  aria-haspopup="true"
                >
                  <ChevronDown className="w-4 h-4" />
                </button>
              </Tooltip>
              {moreOpen && (
                <div className="absolute right-0 top-full mt-1.5 p-1 min-w-[160px] rounded-lg border border-zinc-800 bg-zinc-900 shadow-xl z-[100]">
                  <button
                    type="button"
                    onClick={() => {
                      setMoreOpen(false);
                      onDelete();
                    }}
                    className="w-full flex items-center gap-2.5 px-2.5 py-2 text-left text-sm font-medium text-rose-400/90 hover:bg-rose-500/10 hover:text-rose-400 rounded-md transition-colors"
                  >
                    <Trash2 className="w-4 h-4 shrink-0" />
                    Delete plan
                  </button>
                </div>
              )}
            </div>
          )}
        </div>

      </div>
    </div>
  );
}
