import { BrainCircuit, CheckCircle2, RefreshCw } from "lucide-react";
import { clsx } from "clsx";
import { ModelSelector } from "./ModelSelector";
import type { SessionStatus } from "../types";

interface ModelOption {
  model_id: string;
  provider: string;
  available: boolean;
  unavailable_reason?: string | null;
}

interface CritiqueCardProps {
  round: number;
  status: SessionStatus;
  convergenceScore?: number;
  canCritique: boolean;
  canApprove: boolean;
  critiquePending?: boolean;
  criticModel: string;
  modelOptions: ModelOption[];
  onCriticModelChange: (modelId: string) => void;
  onCritique: () => void;
  onApprove: () => void;
}

function promptText(status: SessionStatus, round: number, convergenceScore?: number): string {
  const conv = convergenceScore !== undefined ? ` · ${Math.round(convergenceScore * 100)}% agreement` : "";
  if (status === "converged") return `Plan converged${conv}. Approve or run another pass.`;
  if (round === 0) return "Initial draft ready. Run a critique pass to start refining.";
  return `Round ${round} complete${conv}. Run a critique pass to continue refining.`;
}

export function CritiqueCard({
  round,
  status,
  convergenceScore,
  canCritique,
  canApprove,
  critiquePending = false,
  criticModel,
  modelOptions,
  onCriticModelChange,
  onCritique,
  onApprove,
}: CritiqueCardProps) {
  if (!canCritique && !canApprove) return null;

  return (
    <div className="absolute bottom-6 left-6 right-6 z-20">
      <div className="bg-zinc-900/90 backdrop-blur-xl border border-zinc-800 rounded-2xl shadow-2xl">
        {/* Prompt text — same padding/height as the chat textarea single row */}
        <p className="px-4 pt-3 pb-2 text-sm text-zinc-400 min-h-[44px] flex items-center">
          {promptText(status, round, convergenceScore)}
        </p>

        {/* Footer row — identical to the chat input footer */}
        <div className="flex items-center justify-between gap-2 px-3 pb-2.5 pt-1 border-t border-zinc-700/40">
          <div className="flex items-center bg-zinc-900/50 border border-zinc-700/60 rounded-lg p-0.5">
            <ModelSelector
              label="Critic"
              value={criticModel}
              onChange={onCriticModelChange}
              options={modelOptions}
              icon={<BrainCircuit className="w-3 h-3" />}
              className="hover:bg-zinc-900/90"
              dropUp
            />
          </div>

          <div className="flex items-center gap-2">
            {canCritique && (
              <button
                type="button"
                onClick={onCritique}
                disabled={critiquePending}
                className={clsx(
                  "flex items-center gap-1.5 px-4 py-2 text-xs font-semibold rounded-xl transition-all duration-200 shadow-lg active:scale-95",
                  critiquePending
                    ? "bg-indigo-600/70 text-indigo-200 cursor-not-allowed"
                    : "bg-indigo-500 text-white hover:bg-indigo-400 shadow-indigo-500/25 hover:shadow-indigo-400/40"
                )}
              >
                <RefreshCw className={clsx("w-3.5 h-3.5", critiquePending && "animate-spin")} />
                {critiquePending ? "Critiquing…" : "Critique"}
              </button>
            )}
            {canApprove && (
              <button
                type="button"
                onClick={onApprove}
                className="flex items-center gap-1.5 px-4 py-2 text-xs font-semibold rounded-xl bg-zinc-100 text-zinc-950 hover:bg-zinc-50 transition-all duration-200 shadow-lg active:scale-95"
              >
                <CheckCircle2 className="w-3.5 h-3.5" />
                Approve
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
