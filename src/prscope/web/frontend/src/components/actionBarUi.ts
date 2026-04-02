import type { SessionStatus } from "../types";

/** True when the session is post-convergence (export / handoff copy applies). */
export function isConvergedOrApproved(status: SessionStatus): boolean {
  return status === "converged" || status === "approved";
}

export function scoreColor(score: number | null | undefined): string {
  if (score == null) return "text-zinc-500";
  if (score >= 0.85) return "text-emerald-400";
  if (score >= 0.65) return "text-amber-400";
  return "text-rose-400";
}
