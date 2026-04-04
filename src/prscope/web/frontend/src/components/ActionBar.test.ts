import { describe, expect, it } from "vitest";

import { isConvergedOrApproved, scoreColor } from "./actionBarUi";
import type { DraftTimingDiagnostics } from "../types";
import {
  formatDiagnosticsSource,
  formatInvestigationDensity,
  formatTraceDiagnosticsForCopy,
} from "./actionBarDiagnostics";

describe("formatDiagnosticsSource", () => {
  it("labels live runtime diagnostics", () => {
    expect(formatDiagnosticsSource("live_memory")).toEqual({
      label: "Live",
      detail: "Updated from the active runtime for this session.",
    });
  });

  it("labels persisted diagnostics after refresh", () => {
    expect(formatDiagnosticsSource("persisted_session")).toEqual({
      label: "Saved",
      detail: "Loaded from the persisted session record after refresh.",
    });
  });
});

describe("isConvergedOrApproved", () => {
  it("is true for converged and approved", () => {
    expect(isConvergedOrApproved("converged")).toBe(true);
    expect(isConvergedOrApproved("approved")).toBe(true);
  });

  it("is false for draft and refining", () => {
    expect(isConvergedOrApproved("draft")).toBe(false);
    expect(isConvergedOrApproved("refining")).toBe(false);
  });
});

describe("scoreColor", () => {
  it("maps score bands to tailwind color classes", () => {
    expect(scoreColor(undefined)).toBe("text-zinc-500");
    expect(scoreColor(0.9)).toBe("text-emerald-400");
    expect(scoreColor(0.7)).toBe("text-amber-400");
    expect(scoreColor(0.5)).toBe("text-rose-400");
  });
});

describe("formatInvestigationDensity", () => {
  it("formats investigation density as triggers per turn", () => {
    expect(formatInvestigationDensity(2)).toBe("2.00x");
    expect(formatInvestigationDensity(0.5)).toBe("0.50x");
  });

  it("falls back safely for empty values", () => {
    expect(formatInvestigationDensity(undefined)).toBe("0.00x");
    expect(formatInvestigationDensity(-1)).toBe("0.00x");
  });
});

describe("formatTraceDiagnosticsForCopy", () => {
  it("formats a full diagnostics snapshot for clipboard", () => {
    const d: DraftTimingDiagnostics = {
      routing_heuristic_decisions: 4,
      routing_model_decisions: 0,
      routing_fallback_decisions: 0,
      route_author_chat_total: 0,
      route_lightweight_refine_total: 3,
      route_full_refine_total: 1,
      route_existing_feature_total: 0,
      refinement_turns_total: 4,
      average_refinement_turn_tokens: 9368.2,
      investigation_trigger_total: 2,
      investigation_trigger_rate: 0.5,
      investigation_trigger_reason_last: "architecture_tradeoff",
    };
    const text = formatTraceDiagnosticsForCopy(d, "Live", "Updated from the active runtime for this session.");
    expect(text).toContain("Trace Diagnostics");
    expect(text).toContain("Source: Live");
    expect(text).toContain("Heuristic: 4");
    expect(text).toContain("Lightweight refine: 3");
    expect(text).toContain("Avg tokens/turn: 9368");
    expect(text).toContain("Triggers / turn: 0.50x");
    expect(text).toContain("Last trigger: architecture_tradeoff");
    expect(text).toContain("critique rounds or review-note counts");
  });

  it("omits last trigger when empty", () => {
    const d: DraftTimingDiagnostics = {
      investigation_trigger_reason_last: "",
    };
    const text = formatTraceDiagnosticsForCopy(d, "Empty", "No trace diagnostics were recorded for this session.");
    expect(text).not.toContain("Last trigger:");
  });
});
