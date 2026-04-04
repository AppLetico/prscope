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

/** Try to classify a single failure fragment (one gate message, not compound). */
function humanizeOneFragment(t: string): PlanValidationFriendly | null {
  const s = t.trim();
  if (!s) return null;

  if (s.includes("missing test target reference")) {
    const candidates = s.match(/reference one of:\s*([^;]+)/i)?.[1]?.trim();
    const paths = candidates
      ? candidates
          .split(",")
          .map((x) => x.trim())
          .filter(Boolean)
          .slice(0, 3)
      : [];
    return {
      title: "Name a regression test target",
      summary:
        "When your requirements mention tests or coverage, the plan should name at least one verified test file (from the repo) in the Test Strategy or Files Changed.",
      whatToDo:
        paths.length > 0
          ? `Add a bullet under Test Strategy that cites one of: ${paths.map((p) => `\`${p}\``).join(", ")}.`
          : "Add a bullet under Test Strategy that cites one of the suggested test paths from the technical detail (in backticks).",
    };
  }

  if (s.includes("localized backend payload/response change must reference the existing API path")) {
    const path = extractMentionPath(s) ?? "src/prscope/web/api.py";
    return {
      title: "Name the API file in your plan",
      summary:
        "The draft talks about changing request or response payloads, but it doesn’t cite the FastAPI module where those routes and models live. The quality gate blocks saving until that path is explicit.",
      whatToDo: `Add \`${path}\` to the Files Changed section and mention it in Implementation Steps so it’s clear which backend surface you’re editing.`,
    };
  }

  if (s.includes("localized backend payload/response change must reference the API model regression target")) {
    const path = extractMentionPath(s) ?? "tests/test_web_api_models.py";
    return {
      title: "Point to the API contract test",
      summary:
        "When backend response shapes change, the plan should name the regression test that locks the API models and response schema.",
      whatToDo: `Include \`${path}\` in Files Changed or Test Strategy so testing lines up with the contract change.`,
    };
  }

  if (s.includes("localized frontend UI change should reference a frontend regression target")) {
    const path = extractMentionPath(s);
    return {
      title: "Name the frontend test file",
      summary:
        "UI-only changes should reference the existing component or page test you’ll extend, so the plan stays grounded in the repo.",
      whatToDo: path
        ? `Mention \`${path}\` in Files Changed or Test Strategy.`
        : "Add the relevant *.test.ts / *.test.tsx file under Files Changed or Test Strategy.",
    };
  }

  if (s.startsWith("missing explicit helper reuse reference for ")) {
    return {
      title: "Name the helper you’re reusing",
      summary:
        "This task expects a concrete reference to an existing helper (for example snapshot or export helpers) instead of inventing new plumbing.",
      whatToDo: "Update Implementation Steps to name the exact helper or module from the codebase you’ll call.",
    };
  }

  if (s.startsWith("Files Changed entries missing from Implementation Steps:")) {
    return {
      title: "Match files to implementation steps",
      summary:
        "Every path listed under Files Changed should also appear in Implementation Steps so the plan reads as one coherent checklist.",
      whatToDo: "Add numbered steps that reference each file you listed, or trim Files Changed to match what you actually describe.",
    };
  }

  {
    const m = s.match(/^required section is empty:\s*([^;]+)/i);
    if (m) {
      const section = m[1].trim();
      return {
        title: "Fill in a required section",
        summary: `The ${section} section is empty. The planner won’t save the draft until every required heading has real content.`,
        whatToDo: `Write content for ${section}, then try Review or send another revision message.`,
      };
    }
  }

  if (s.includes("under-scoped draft")) {
    return {
      title: "Files Changed is too narrow",
      summary:
        "The plan references more files in the body than it lists under **Files Changed**, or the opposite mismatch flagged by the checker.",
      whatToDo: "Align **Files Changed** with every path you reference in architecture and implementation steps.",
    };
  }

  if (s.includes("grounding ratio")) {
    return {
      title: "Improve plan grounding",
      summary:
        "The draft doesn’t cite enough verified repo paths (backticks) relative to how many paths it mentions. The checker needs stronger ties to files we can confirm exist.",
      whatToDo:
        "Add or correct backtick file paths so they match the repo, and align **Files Changed** with what you describe in Architecture and Implementation Steps.",
    };
  }

  if (s.includes("files changed paths outside evidence allowlist") || s.includes("outside evidence allowlist")) {
    return {
      title: "Stay within evidence scope",
      summary:
        "A path under **Files Changed** isn’t in the evidence allowlist for this refinement (what we’ve read or anchored on for this session).",
      whatToDo:
        "Narrow **Files Changed** to paths from the technical detail, or run a refinement that reads the files you need first.",
    };
  }

  if (s.startsWith("unknown file references:") || s.startsWith("replace unverified path")) {
    return {
      title: "Fix file path references",
      summary:
        "The plan mentions file paths that aren’t verified against the repo snapshot for this session.",
      whatToDo:
        "Replace unknown paths with verified paths from the technical detail (suggested replacements may appear there).",
    };
  }

  if (s.includes("revision introduced unverified file references")) {
    return {
      title: "Verify new file references",
      summary:
        "This revision added references to files that aren’t in the verified set for this round.",
      whatToDo: "Use paths that appear in the repo evidence or trim references until they match verified files.",
    };
  }

  if (s.includes("No JSON block found in author response")) {
    return {
      title: "Author reply wasn’t valid JSON",
      summary:
        "The planner expected a single JSON object from the author model (your plan update), but the response didn’t contain a parseable object.",
      whatToDo: "Try Review or send the revision again. If it keeps happening, shorten the request or switch the author model in settings.",
    };
  }

  if (s.includes("Unterminated JSON object in author response")) {
    return {
      title: "Author JSON was cut off",
      summary: "The author model started a JSON object but didn’t finish it—often a token or timeout limit.",
      whatToDo: "Retry the revision with a smaller change, or raise the author output cap if your setup allows it.",
    };
  }

  if (s.includes("Expected JSON object payload")) {
    return {
      title: "Author JSON shape was wrong",
      summary: "The author response wasn’t a JSON object at the top level (array or plain text instead of `{...}`).",
      whatToDo: "Run Review again or ask for a smaller incremental edit so the model stays in contract.",
    };
  }

  if (s.startsWith("Missing required PlanDocument fields:")) {
    return {
      title: "Plan update was incomplete",
      summary:
        "The author JSON was missing required fields for the plan document. The server can’t merge a partial payload safely.",
      whatToDo: "Try applying the critique again. If the error repeats, paste the technical detail into chat and ask for a full plan JSON.",
    };
  }

  return null;
}

export function humanizePlanValidationError(raw: string): PlanValidationFriendly {
  let t = raw.trim();
  // Legacy / backend autofix exhaustion prefix (no longer preferred server-side).
  t = t.replace(/^Plan draft could not be auto-repaired to pass validation checks\.\s*/i, "").trim();

  const full = humanizeOneFragment(t);
  if (full) return full;

  const segments = t
    .split("; ")
    .map((x) => x.trim())
    .filter(Boolean);
  for (const seg of segments) {
    const h = humanizeOneFragment(seg);
    if (h) return h;
  }

  return {
    title: "Couldn’t apply this revision",
    summary:
      "The author produced a draft update, but a quality gate rejected it before it could replace your saved plan. The critic review in chat is unchanged—this is only about the draft.",
    whatToDo:
      "Open **Technical details** for the exact message, fix the plan if you can, then run Review or ask for another revise. You can also paste the error into chat for a concrete edit.",
  };
}
