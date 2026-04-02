import { describe, expect, it } from "vitest";

import { getPlanPanelEmptyCopy } from "./planPanelUi";

describe("getPlanPanelEmptyCopy", () => {
  it("shows generating copy when processing in draft", () => {
    const copy = getPlanPanelEmptyCopy(true, "draft");
    expect(copy.title).toBe("Generating plan...");
    expect(copy.subtitle).toContain("Drafting");
  });

  it("shows idle copy when not processing", () => {
    const copy = getPlanPanelEmptyCopy(false, "draft");
    expect(copy.title).toBe("No plan generated yet.");
  });

  it("treats refining with processing as in progress", () => {
    const copy = getPlanPanelEmptyCopy(true, "refining");
    expect(copy.title).toBe("Generating plan...");
  });
});
