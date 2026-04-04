import { AlertTriangle, ChevronDown, ChevronUp, X } from "lucide-react";
import { useEffect, useId, useState } from "react";

import { humanizePlanValidationError } from "../utils/planValidationCopy";

type PlanValidationToastProps = {
  rawMessage: string;
  /** True when the failure happened while applying a critique to the plan (user-initiated). */
  fromApplyRevision?: boolean;
  onDismiss: () => void;
};

/**
 * Center-screen overlay (modal-style): persistent until dismissed. Humanized copy
 * with optional raw detail; does not auto-dismiss.
 */
export function PlanValidationToast({
  rawMessage,
  fromApplyRevision = false,
  onDismiss,
}: PlanValidationToastProps) {
  const [detailsOpen, setDetailsOpen] = useState(false);
  const friendly = humanizePlanValidationError(rawMessage);
  const detailsId = useId();
  const titleId = useId();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onDismiss();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onDismiss]);

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center p-4 sm:p-8 bg-zinc-950/65 backdrop-blur-[3px]"
      role="alertdialog"
      aria-modal="true"
      aria-labelledby={titleId}
      aria-describedby={`${titleId}-desc`}
    >
      <div
        className="relative w-full max-w-lg rounded-2xl border border-amber-500/35 bg-zinc-900 shadow-2xl shadow-black/50 ring-1 ring-amber-500/15 animate-in fade-in zoom-in-95 duration-200"
        role="document"
      >
        <div className="flex items-start justify-between gap-3 border-b border-zinc-800/90 px-5 pt-5 pb-3">
          <div className="flex min-w-0 gap-3">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-amber-500/15 text-amber-400">
              <AlertTriangle className="h-5 w-5" aria-hidden />
            </div>
            <h2
              id={titleId}
              className="text-base font-semibold leading-snug text-amber-50 tracking-tight pr-2"
            >
              {friendly.title}
            </h2>
          </div>
          <button
            type="button"
            onClick={onDismiss}
            className="shrink-0 rounded-lg p-2 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-100"
            aria-label="Dismiss"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div id={`${titleId}-desc`} className="space-y-4 px-5 pb-5 pt-4">
          <p className="text-sm leading-relaxed text-zinc-300">{friendly.summary}</p>

          <div className="rounded-xl border border-amber-500/25 bg-amber-500/[0.08] px-4 py-3">
            <p className="text-xs font-semibold text-amber-200/90">What to do</p>
            <p className="mt-2 text-sm leading-relaxed text-amber-50/95">{friendly.whatToDo}</p>
          </div>

          <p className="text-xs leading-relaxed text-zinc-500">
            {fromApplyRevision
              ? "The critique in the thread is unchanged—only updating the plan draft was blocked. Fix the issue above, then try Apply revision again."
              : "The last critic review in the thread is unchanged—only this draft update was blocked."}
          </p>

          <div className="border-t border-zinc-800/80 pt-3">
            <button
              type="button"
              onClick={() => setDetailsOpen((v) => !v)}
              className="flex w-full items-center justify-between gap-2 rounded-lg px-0 py-1 text-left text-xs font-medium text-zinc-400 transition-colors hover:text-zinc-200"
              aria-expanded={detailsOpen}
              aria-controls={detailsId}
            >
              <span>Technical details</span>
              {detailsOpen ? (
                <ChevronUp className="h-4 w-4 shrink-0 opacity-70" aria-hidden />
              ) : (
                <ChevronDown className="h-4 w-4 shrink-0 opacity-70" aria-hidden />
              )}
            </button>
            {detailsOpen && (
              <pre
                id={detailsId}
                className="mt-2 max-h-36 overflow-y-auto rounded-lg border border-zinc-800 bg-zinc-950/90 p-3 font-mono text-[11px] leading-relaxed text-zinc-400 [overflow-wrap:anywhere]"
              >
                {rawMessage}
              </pre>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
