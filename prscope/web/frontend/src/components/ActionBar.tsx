import type { SessionStatus } from "../types";
import { Download, CheckCircle2, FileDiff, RefreshCw } from "lucide-react";
import { Link } from "react-router-dom";
import { clsx } from "clsx";
import { Tooltip } from "./ui/Tooltip";

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
}: ActionBarProps) {
  
  const isConverged = status === "converged" || status === "approved" || status === "exported";
  const isRefining = status === "refining" || status === "discovery";
  
  return (
    <div className="h-14 flex items-center justify-between px-6 border-b border-zinc-800/50 bg-zinc-950/80 backdrop-blur-md sticky top-0 z-50">
      
      {/* Left: Context */}
      <div className="flex items-center gap-3">
        <Link
          to="/"
          className="flex items-center justify-center p-1 -ml-2 text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800/50 rounded-md transition-colors"
          title="Back to sessions"
        >
          <img src="/logo.svg" alt="prscope" className="w-6 h-6 rounded-md" />
        </Link>
        <div className="flex items-center gap-2 text-sm">
          <span className="font-mono text-zinc-500">{repoName}</span>
          <span className="text-zinc-700">/</span>
          <span className="font-medium text-zinc-200 truncate max-w-[200px]">{title}</span>
        </div>
        <div className="h-4 w-px bg-zinc-800 mx-1"></div>
        <div className="flex items-center gap-2">
          <Tooltip content="Current planning round">
            <span className="text-xs font-mono text-zinc-500 bg-zinc-900 px-1.5 py-0.5 rounded border border-zinc-800 cursor-default">
              R{round}
            </span>
          </Tooltip>
          <Tooltip content={`Status: ${status}`}>
            <div className="flex items-center gap-1.5 cursor-default">
              {isConverged ? (
                <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.6)]"></div>
              ) : isRefining ? (
                <div className="w-2 h-2 rounded-full bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.6)] animate-pulse"></div>
              ) : (
                <div className="w-2 h-2 rounded-full bg-zinc-500"></div>
              )}
              <span className={clsx(
                "text-xs font-medium capitalize",
                isConverged ? "text-emerald-400" : isRefining ? "text-amber-400" : "text-zinc-400"
              )}>
                {status}
              </span>
            </div>
          </Tooltip>
        </div>
      </div>

      {/* Center: Convergence Pulse (Only show during refining/converged) */}
      {(status === "refining" || status === "converged") && (
        <Tooltip content={`Convergence Score: ${Math.round(convergenceScore * 100)}%`}>
          <div className="hidden md:flex items-center gap-3 cursor-default">
            <span className="text-[10px] uppercase tracking-widest text-zinc-500 font-semibold">Convergence</span>
            <div className="w-32 h-1.5 bg-zinc-900 rounded-full overflow-hidden border border-zinc-800/50">
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
      )}

      {/* Right: Actions */}
      <div className="flex items-center gap-2">
        <Tooltip content="Toggle diff view">
          <button 
            onClick={onToggleDiff}
            className={clsx(
              "p-1.5 rounded-md transition-colors",
              isDiffMode 
                ? "bg-indigo-500/20 text-indigo-400 hover:bg-indigo-500/30" 
                : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800"
            )}
          >
            <FileDiff className="w-4 h-4" />
          </button>
        </Tooltip>
        
        <Tooltip content="Export plan">
          <button 
            onClick={onExport}
            disabled={!canExport}
            className={clsx(
              "p-1.5 rounded-md transition-colors",
              canExport
                ? "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800"
                : "text-zinc-600 cursor-not-allowed"
            )}
          >
            <Download className="w-4 h-4" />
          </button>
        </Tooltip>

        <div className="h-4 w-px bg-zinc-800 mx-1"></div>

        {canCritique && (
          <Tooltip content="Force adversarial critique round">
            <button
              onClick={onCritique}
              className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded-md bg-zinc-800 text-zinc-200 border border-zinc-700 hover:bg-zinc-700 hover:border-zinc-600 transition-all shadow-sm active:scale-95"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Critique
            </button>
          </Tooltip>
        )}
        
        {canApprove && (
          <Tooltip content="Approve current plan">
            <button
              onClick={onApprove}
              className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded-md bg-white text-zinc-950 hover:bg-zinc-200 transition-all shadow-sm active:scale-95"
            >
              <CheckCircle2 className="w-4 h-4" />
              Approve
            </button>
          </Tooltip>
        )}
      </div>
    </div>
  );
}
