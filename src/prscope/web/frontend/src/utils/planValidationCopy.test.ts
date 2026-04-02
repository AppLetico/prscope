import { describe, expect, it } from "vitest";

import { humanizePlanValidationError } from "./planValidationCopy";

describe("humanizePlanValidationError", () => {
  it("maps localized API path failure to friendly copy", () => {
    const raw =
      "localized backend payload/response change must reference the existing API path; mention `src/prscope/web/api.py`";
    const h = humanizePlanValidationError(raw);
    expect(h.title).toContain("API file");
    expect(h.summary).toMatch(/payload|response/i);
    expect(h.whatToDo).toContain("src/prscope/web/api.py");
  });

  it("maps API model test failure", () => {
    const raw =
      "localized backend payload/response change must reference the API model regression target; mention `tests/test_web_api_models.py`";
    const h = humanizePlanValidationError(raw);
    expect(h.whatToDo).toContain("test_web_api_models");
  });

  it("falls back for unknown messages", () => {
    const h = humanizePlanValidationError("some unknown validation error");
    expect(h.title).toMatch(/didn’t pass|draft/i);
    expect(h.summary.length).toBeGreaterThan(20);
  });
});
