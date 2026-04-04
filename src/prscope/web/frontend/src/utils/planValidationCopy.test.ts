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
    expect(h.title).toMatch(/Couldn’t apply|revision/i);
    expect(h.summary.length).toBeGreaterThan(20);
    expect(h.whatToDo).toContain("Technical details");
  });

  it("maps author JSON contract failures", () => {
    expect(humanizePlanValidationError("No JSON block found in author response").title).toContain("JSON");
    expect(humanizePlanValidationError("Unterminated JSON object in author response").title).toContain("cut off");
    expect(humanizePlanValidationError("Expected JSON object payload").title).toContain("shape");
    expect(humanizePlanValidationError("Missing required PlanDocument fields: goals").summary).toContain("missing");
  });

  it("maps missing test target before compound required-section noise", () => {
    const raw =
      "required section is empty: Test Strategy; missing test target reference; reference one of: tests/test_a.py, tests/test_b.py";
    const h = humanizePlanValidationError(raw);
    expect(h.title).toBe("Name a regression test target");
    expect(h.whatToDo).toContain("test_a.py");
    expect(h.summary).not.toContain("tests/test_a.py section is empty");
  });

  it("parses required section name only up to semicolon", () => {
    const h = humanizePlanValidationError("required section is empty: Rollback Plan");
    expect(h.summary).toContain("Rollback Plan");
    expect(h.summary).not.toContain(";");
  });

  it("strips legacy autofix exhaustion prefix and still classifies the inner failure", () => {
    const raw =
      "Plan draft could not be auto-repaired to pass validation checks. grounding ratio 0.12 below required 0.45";
    const h = humanizePlanValidationError(raw);
    expect(h.title).toBe("Improve plan grounding");
  });

  it("classifies grounding ratio in a compound message", () => {
    const raw =
      "grounding ratio 0.20 below required 0.45; unknown file references: foo.py";
    const h = humanizePlanValidationError(raw);
    expect(h.title).toBe("Improve plan grounding");
  });
});
