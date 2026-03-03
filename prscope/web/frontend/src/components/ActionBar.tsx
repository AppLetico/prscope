import type { SessionStatus } from "../types";
import { Download, CheckCircle2, FileDiff, RefreshCw, Sparkles, BrainCircuit, ChevronDown, Trash2 } from "lucide-react";
import { Link } from "react-router-dom";
import { clsx } from "clsx";
import { useRef, useEffect, useState } from "react";
import { Tooltip } from "./ui/Tooltip";
import { ModelSelector } from "./ModelSelector";

interface ActionBarProps {
  repoName: string;
  title: string;
  round: number;
  status: SessionStatus;
  convergenceScore?: number;
  onCritique: () => void;
  onApprove: () => void;
  onExport: () => void;
  onToggleDiff: () => void;
  canCritique: boolean;
  canApprove: boolean;
  canExport: boolean;
  isDiffMode: boolean;
  authorModel: string;
  criticModel: string;
  modelOptions: Array<{
    model_id: string;
    provider: string;
    available: boolean;
    unavailable_reason?: string | null;
  }>;
  onAuthorModelChange: (modelId: string) => void;
  onCriticModelChange: (modelId: string) => void;
  sessionCostUsd?: number;
  maxPromptTokens?: number;
  contextWindowTokens?: number | null;
  contextPercent?: number | null;
  contextCompactionEnabled?: boolean;
  onDelete?: () => void;
}

export function ActionBar({
  repoName,
  title,
  round,
  status,
  convergenceScore = 0,
  onCritique,
  onApprove,
  onExport,
  onToggleDiff,
  canCritique,
  canApprove,
  canExport,
  isDiffMode,
  authorModel,
  criticModel,
  modelOptions,
  onAuthorModelChange,
  onCriticModelChange,
  sessionCostUsd = 0,
  maxPromptTokens = 0,
  contextWindowTokens = null,
  contextPercent = null,
  contextCompactionEnabled = false,
  onDelete,
}: ActionBarProps) {
  const [moreOpen, setMoreOpen] = useState(false);
  const moreRef = useRef<HTMLDivElement>(null);

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
  const isConverged = status === "converged" || status === "approved" || status === "exported";
  const isRefining = status === "refining" || status === "discovery";
  const statusTooltip = {
    created: "Session created.",
    preparing: "Preparing codebase memory.",
    discovering: "Discovery mode: gathering requirements.",
    discovery: "Discovery mode: gathering requirements.",
    drafting: "Drafting initial plan.",
    refining: "Refining plan with critique rounds.",
    converged: "Plan has converged and is ready for approval.",
    approved: "Plan approved.",
    exported: "Plan exported.",
  }[status];
  
  return (
    <div className="h-14 flex items-center justify-between px-4 border-b border-zinc-800/50 bg-zinc-950/80 backdrop-blur-md sticky top-0 z-50">
      
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
          <span className="font-medium text-zinc-200 truncate max-w-[250px]">{title}</span>
        </div>
      </div>

      {/* Center: Status Pill */}
      <div className="hidden md:flex items-center justify-center flex-1">
        <div className="flex items-center gap-3 bg-zinc-900/40 border border-zinc-800/40 rounded-full px-3 py-1.5 shadow-sm">
          <div className="flex items-center gap-2">
            <Tooltip content={statusTooltip}>
              <div className="flex items-center gap-2 cursor-default">
                <span className="text-[10px] font-mono text-zinc-500 font-medium bg-zinc-800/40 px-1.5 py-0.5 rounded border border-zinc-700/30">R{round}</span>
                <div className="w-1 h-1 rounded-full bg-zinc-700/50"></div>
                <div className="flex items-center gap-1.5">
                  {isConverged ? (
                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.6)]"></div>
                  ) : isRefining ? (
                    <div className="w-1.5 h-1.5 rounded-full bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.6)] animate-pulse"></div>
                  ) : (
                    <div className="w-1.5 h-1.5 rounded-full bg-zinc-500"></div>
                  )}
                  <span className={clsx(
                    "text-[10px] font-bold uppercase tracking-wider",
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
              <div className="w-px h-3 bg-zinc-800/60"></div>
              <Tooltip content={`Agreement between author and critic on current plan: ${Math.round(convergenceScore * 100)}%`}>
                <div className="flex items-center gap-2 cursor-default">
                  <span className="text-[10px] font-mono text-zinc-500">conv</span>
                  <div className="w-16 h-1 bg-zinc-800/60 rounded-full overflow-hidden">
                    <div 
                      className={clsx(
                        "h-full transition-all duration-1000 ease-out",
                        isConverged ? "bg-emerald-500" : "bg-gradient-to-r from-amber-500 to-amber-400"
                      )}
                      style={{ width: `${Math.max(10, Math.min(100, convergenceScore * 100))}%` }}
                    ></div>
                  </div>
                </div>
              </Tooltip>
            </>
          )}
        </div>
      </div>

      {/* Right: Actions */}
      <div className="flex items-center justify-end gap-3 flex-1">
        {/* Stats */}
        <div className="hidden xl:flex items-center gap-4 text-[10px] font-mono text-zinc-500 mr-2">
          <Tooltip content="Total session cost">
            <div className="flex items-center gap-1.5">
              <span className="text-zinc-600">cost</span>
              <span className={sessionCostUsd > 0.5 ? "text-rose-400" : sessionCostUsd > 0.1 ? "text-amber-400" : "text-zinc-300"}>
                ${sessionCostUsd.toFixed(4)}
              </span>
            </div>
          </Tooltip>
          <Tooltip content="Max prompt tokens used">
            <div className="flex items-center gap-1.5">
              <span className="text-zinc-600">tkns</span>
              <span className="text-zinc-300">
                {maxPromptTokens >= 1000 ? `${(maxPromptTokens / 1000).toFixed(1)}k` : maxPromptTokens}
              </span>
            </div>
          </Tooltip>
          {contextPercent !== null && (
            <Tooltip content={`Context window usage${contextWindowTokens ? ` (${contextWindowTokens} max)` : ''}`}>
              <div className="flex items-center gap-1.5">
                <span className="text-zinc-600">ctx</span>
                <span className={contextPercent >= 85 ? "text-rose-400" : contextPercent >= 70 ? "text-amber-400" : "text-zinc-300"}>
                  {contextPercent.toFixed(1)}%
                </span>
              </div>
            </Tooltip>
          )}
          {contextCompactionEnabled && (
            <Tooltip content="Context compaction is active">
              <span className="text-indigo-400 bg-indigo-500/10 px-1.5 py-0.5 rounded border border-indigo-500/20">compacted</span>
            </Tooltip>
          )}
        </div>

        {/* Models Pill */}
        <div className="hidden lg:flex items-center bg-zinc-900/60 border border-zinc-800/80 rounded-lg p-0.5 shadow-sm">
          <ModelSelector
            label="Author"
            value={authorModel}
            onChange={onAuthorModelChange}
            options={modelOptions}
            icon={<Sparkles className="w-3 h-3" />}
            className="hover:bg-zinc-800/80"
          />
          <div className="w-px h-4 bg-zinc-800 mx-0.5"></div>
          <ModelSelector
            label="Critic"
            value={criticModel}
            onChange={onCriticModelChange}
            options={modelOptions}
            icon={<BrainCircuit className="w-3 h-3" />}
            className="hover:bg-zinc-800/80"
          />
        </div>

        <div className="flex items-center gap-1 ml-1 border-l border-zinc-800/50 pl-3">
          {onDelete != null && (
            <div className="relative" ref={moreRef}>
              <Tooltip content="More actions">
                <button
                  type="button"
                  onClick={() => setMoreOpen((v) => !v)}
                  className={clsx(
                    "p-1.5 rounded-md transition-all duration-200",
                    moreOpen ? "bg-zinc-800 text-zinc-200" : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/80"
                  )}
                  aria-expanded={moreOpen}
                  aria-haspopup="true"
                >
                  <ChevronDown className="w-3.5 h-3.5" />
                </button>
              </Tooltip>
              {moreOpen && (
                <div className="absolute right-0 top-full mt-1.5 p-1 min-w-[160px] rounded-lg border border-zinc-800 bg-zinc-950 shadow-xl z-[100]">
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
          <Tooltip content="Toggle diff view">
            <button 
              onClick={onToggleDiff}
              className={clsx(
                "p-1.5 rounded-md transition-all duration-200",
                isDiffMode 
                  ? "bg-indigo-500/20 text-indigo-400" 
                  : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/80"
              )}
            >
              <FileDiff className="w-3.5 h-3.5" />
            </button>
          </Tooltip>
          
          <Tooltip content="Export plan">
            <button 
              onClick={onExport}
              disabled={!canExport}
              className={clsx(
                "p-1.5 rounded-md transition-all duration-200",
                canExport
                  ? "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/80"
                  : "text-zinc-600 cursor-not-allowed"
              )}
            >
              <Download className="w-3.5 h-3.5" />
            </button>
          </Tooltip>
        </div>

        <div className="flex items-center gap-2 ml-1">
          {canCritique && (
            <button
              onClick={onCritique}
              className="flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-medium rounded-md bg-zinc-800/60 text-zinc-300 border border-zinc-700/60 hover:bg-zinc-700 hover:text-zinc-100 transition-all shadow-sm active:scale-95"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Critique
            </button>
          )}
          
          {canApprove && (
            <button
              onClick={onApprove}
              className="flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-medium rounded-md bg-zinc-100 text-zinc-950 hover:bg-white transition-all shadow-sm active:scale-95"
            >
              <CheckCircle2 className="w-3.5 h-3.5" />
              Approve
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
