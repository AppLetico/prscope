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
    detail: "No routing diagnostics were recorded for this session.",
  };
}

export function formatInvestigationDensity(rate: number | null | undefined): string {
  if (rate == null || !Number.isFinite(rate) || rate <= 0) {
    return "0.00x";
  }
  return `${rate.toFixed(2)}x`;
}
