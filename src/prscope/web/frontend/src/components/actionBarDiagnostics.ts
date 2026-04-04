import type { DraftTimingDiagnostics } from "../types";

/** Plain-text snapshot of the Trace Diagnostics panel for clipboard sharing. */
export function formatTraceDiagnosticsForCopy(
  diagnostics: DraftTimingDiagnostics | null,
  sourceLabel: string,
  sourceDetail: string,
): string {
  const intro = "Internal counters for request tracing and refinement heuristics.";
  const header = [`Trace Diagnostics`, `Source: ${sourceLabel}`, "", intro, sourceDetail, ""];

  if (diagnostics == null) {
    return [...header, "No trace metrics recorded.", ""].join("\n");
  }

  const n = (v: unknown) => Number(v ?? 0);
  const heuristic = n(diagnostics.routing_heuristic_decisions);
  const model = n(diagnostics.routing_model_decisions);
  const fallback = n(diagnostics.routing_fallback_decisions);
  const authorChat = n(diagnostics.route_author_chat_total);
  const lightweight = n(diagnostics.route_lightweight_refine_total);
  const fullRefine = n(diagnostics.route_full_refine_total);
  const existingFeature = n(diagnostics.route_existing_feature_total);
  const turns = n(diagnostics.refinement_turns_total);
  const avgTokens = Math.round(n(diagnostics.average_refinement_turn_tokens));
  const invTriggered = n(diagnostics.investigation_trigger_total);
  const invRate = formatInvestigationDensity(
    diagnostics.investigation_trigger_rate != null
      ? Number(diagnostics.investigation_trigger_rate)
      : NaN,
  );
  const lastReason = String(diagnostics.investigation_trigger_reason_last ?? "").trim();

  const body = [
    "Decision source",
    `  Heuristic: ${heuristic}`,
    `  Model: ${model}`,
    `  Fallback: ${fallback}`,
    "",
    "Routes",
    `  Author chat: ${authorChat}`,
    `  Lightweight refine: ${lightweight}`,
    `  Full refine: ${fullRefine}`,
    `  Existing feature: ${existingFeature}`,
    "",
    "Refinement loop",
    `  Turns observed: ${turns}`,
    `  Avg tokens/turn: ${avgTokens}`,
    "",
    "Investigations",
    `  Triggered: ${invTriggered}`,
    `  Triggers / turn: ${invRate}`,
  ];

  if (lastReason) {
    body.push("", `Last trigger: ${lastReason}`);
  }

  body.push(
    "",
    "These diagnostics describe internal trace behavior. They do not represent critique rounds or review-note counts.",
  );

  return [...header, ...body].join("\n");
}

export function formatDiagnosticsSource(source: string | null | undefined): {
  label: string;
  detail: string;
} {
  if (source === "live_memory") {
    return {
      label: "Live",
      detail: "Updated from the active runtime for this session.",
    };
  }
  if (source === "persisted_session") {
    return {
      label: "Saved",
      detail: "Loaded from the persisted session record after refresh.",
    };
  }
  return {
    label: "Empty",
    detail: "No trace diagnostics were recorded for this session.",
  };
}

export function formatInvestigationDensity(rate: number | null | undefined): string {
  if (rate == null || !Number.isFinite(rate) || rate <= 0) {
    return "0.00x";
  }
  return `${rate.toFixed(2)}x`;
}
