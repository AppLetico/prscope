import { describe, expect, it } from "vitest";

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
