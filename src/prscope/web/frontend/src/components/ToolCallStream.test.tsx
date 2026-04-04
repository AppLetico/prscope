import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { ToolCallStream } from "./ToolCallStream";
import type { ToolCallEntry } from "../types";

function tool(partial: Partial<ToolCallEntry> & Pick<ToolCallEntry, "id" | "name" | "status">): ToolCallEntry {
  return { ...partial };
}

describe("ToolCallStream", () => {
  it("renders nothing when toolCalls is empty", () => {
    const { container } = render(<ToolCallStream toolCalls={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("shows plan-phase label when that phase is running", () => {
    render(<ToolCallStream toolCalls={[tool({ id: 1, name: "draft_plan", status: "running" })]} />);
    expect(screen.getByRole("button", { name: /Drafting plan/i })).toBeInTheDocument();
  });

  it("shows Planning complete when only plan-phase tools and all done", () => {
    render(
      <ToolCallStream
        toolCalls={[
          tool({ id: 1, name: "draft_plan", status: "done", durationMs: 10 }),
          tool({ id: 2, name: "design_review", status: "done", durationMs: 20 }),
        ]}
      />,
    );
    expect(screen.getByRole("button", { name: /Planning complete/i })).toBeInTheDocument();
  });

  it("shows Running tools when plan-phase and repo tools mix with a running tool", () => {
    render(
      <ToolCallStream
        toolCalls={[
          tool({ id: 1, name: "draft_plan", status: "done" }),
          tool({ id: 2, name: "grep_code", status: "running" }),
        ]}
      />,
    );
    expect(screen.getByRole("button", { name: /Running tools \(2\)/i })).toBeInTheDocument();
  });

  it("shows Tools completed for discovery-only tools", () => {
    render(<ToolCallStream toolCalls={[tool({ id: 1, name: "grep_code", status: "done", durationMs: 5 })]} />);
    expect(screen.getByRole("button", { name: /Tools completed \(1\)/i })).toBeInTheDocument();
  });

  it("uses forceRunning for Working when rows are all done", () => {
    render(
      <ToolCallStream
        forceRunning
        toolCalls={[tool({ id: 1, name: "grep_code", status: "done" })]}
      />,
    );
    expect(screen.getByRole("button", { name: /Working/i })).toBeInTheDocument();
  });

  it("lists plan-phase friendly labels in the expanded panel", () => {
    const { container } = render(
      <ToolCallStream
        defaultOpen
        toolCalls={[
          tool({ id: 1, name: "design_review", status: "done", durationMs: 1 }),
        ]}
      />,
    );
    const root = container.querySelector(".mt-2.mb-4");
    expect(root).toBeTruthy();
    expect(within(root as HTMLElement).getByText("Design review")).toBeInTheDocument();
  });
});
