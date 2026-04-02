/**
 * Maps backend validation messages (command_failed) to user-facing copy.
 * Raw strings are still available for the "Technical details" fold-out.
 */

export type PlanValidationFriendly = {
  title: string;
  summary: string;
  whatToDo: string;
};

function extractMentionPath(text: string): string | null {
  const m = text.match(/mention `([^`]+)`/);
  return m?.[1]?.trim() ?? null;
}

export function humanizePlanValidationError(raw: string): PlanValidationFriendly {
  const t = raw.trim();

  if (t.includes("localized backend payload/response change must reference the existing API path")) {
    const path = extractMentionPath(t) ?? "src/prscope/web/api.py";
    return {
      title: "Name the API file in your plan",
      summary:
        "The draft talks about changing request or response payloads, but it doesn’t cite the FastAPI module where those routes and models live. The quality gate blocks saving until that path is explicit.",
      whatToDo: `Add \`${path}\` to the Files Changed section and mention it in Implementation Steps so it’s clear which backend surface you’re editing.`,
    };
  }

  if (t.includes("localized backend payload/response change must reference the API model regression target")) {
    const path = extractMentionPath(t) ?? "tests/test_web_api_models.py";
    return {
      title: "Point to the API contract test",
      summary:
        "When backend response shapes change, the plan should name the regression test that locks the API models and response schema.",
      whatToDo: `Include \`${path}\` in Files Changed or Test Strategy so testing lines up with the contract change.`,
    };
  }

  if (t.includes("localized frontend UI change should reference a frontend regression target")) {
    const path = extractMentionPath(t);
    return {
      title: "Name the frontend test file",
      summary:
        "UI-only changes should reference the existing component or page test you’ll extend, so the plan stays grounded in the repo.",
      whatToDo: path
        ? `Mention \`${path}\` in Files Changed or Test Strategy.`
        : "Add the relevant *.test.ts / *.test.tsx file under Files Changed or Test Strategy.",
    };
  }

  if (t.startsWith("missing explicit helper reuse reference for ")) {
    return {
      title: "Name the helper you’re reusing",
      summary:
        "This task expects a concrete reference to an existing helper (for example snapshot or export helpers) instead of inventing new plumbing.",
      whatToDo: "Update Implementation Steps to name the exact helper or module from the codebase you’ll call.",
    };
  }

  if (t.startsWith("Files Changed entries missing from Implementation Steps:")) {
    return {
      title: "Match files to implementation steps",
      summary:
        "Every path listed under Files Changed should also appear in Implementation Steps so the plan reads as one coherent checklist.",
      whatToDo: "Add numbered steps that reference each file you listed, or trim Files Changed to match what you actually describe.",
    };
  }

  if (t.startsWith("required section is empty:")) {
    const section = t.replace(/^required section is empty:\s*/i, "").trim();
    return {
      title: "Fill in a required section",
      summary: `The ${section} section is empty. The planner won’t save the draft until every required heading has real content.`,
      whatToDo: `Write content for ${section}, then try Review or send another revision message.`,
    };
  }

  if (t.includes("under-scoped draft")) {
    return {
      title: "Files Changed is too narrow",
      summary:
        "The plan references more files in the body than it lists under **Files Changed**, or the opposite mismatch flagged by the checker.",
      whatToDo: "Align **Files Changed** with every path you reference in architecture and implementation steps.",
    };
  }

  return {
    title: "Plan draft didn’t pass checks",
    summary:
      "The author produced a draft update, but a quality gate rejected it before it could replace your saved plan. The critic review in chat is unchanged—this is only about the draft.",
    whatToDo:
      "Use the technical detail below to fix the plan, then run Review or ask the assistant to revise again. If it’s unclear, paste the error into chat and ask for a concrete edit.",
  };
}
