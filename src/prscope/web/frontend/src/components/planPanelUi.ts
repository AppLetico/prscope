import type { SessionStatus } from "../types";

/** Copy for the empty plan panel (no markdown content yet). */
export function getPlanPanelEmptyCopy(
  isProcessing: boolean,
  status?: SessionStatus,
): { title: string; subtitle: string } {
  const isInProgress = isProcessing && (status === "draft" || status === "refining");
  if (isInProgress) {
    return {
      title: "Generating plan...",
      subtitle: "Drafting and validation are in progress.",
    };
  }
  return {
    title: "No plan generated yet.",
    subtitle: "Add requirements or run discovery-style chat to begin.",
  };
}
