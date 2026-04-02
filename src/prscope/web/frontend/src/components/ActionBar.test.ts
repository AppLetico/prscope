import { describe, expect, it } from "vitest";

import { isConvergedOrApproved, scoreColor } from "./actionBarUi";
import { formatDiagnosticsSource, formatInvestigationDensity } from "./actionBarDiagnostics";

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
