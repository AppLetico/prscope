/**
 * Preprocess LLM plan markdown before passing to ReactMarkdown.
 *
 * The LLM frequently wraps code in single backticks across multiple lines,
 * including across blank lines:
 *
 *   `from fastapi import ...
 *
 *   def handler():
 *       return {"status": "healthy"}`
 *
 * This is not valid CommonMark (backtick spans cannot cross paragraph breaks).
 * The result is literal backtick characters rendering inside the code block.
 *
 * This function uses a line-by-line state machine to convert those blocks into
 * proper fenced code blocks before the markdown parser ever sees the content.
 * It does NOT touch single-line inline code (e.g. `/api/health`) or existing
 * triple-backtick fenced blocks.
 */
export function preprocessPlanMarkdown(content: string): string {
  const lines = content.split("\n");
  const output: string[] = [];

  // Whether we are inside an existing triple-backtick or tilde fence.
  let inFence = false;
  // Lines accumulated for an open single-backtick block.
  let collected: string[] | null = null;

  for (const line of lines) {
    // ── Track existing fenced blocks ──────────────────────────────────────
    if (/^[ \t]*(```|~~~)/.test(line)) {
      if (collected !== null) {
        // A new fence while collecting — flush the pending block as raw text
        // (malformed input) and start fresh.
        output.push("`" + collected.join("\n"));
        collected = null;
      }
      inFence = !inFence;
      output.push(line);
      continue;
    }

    if (inFence) {
      output.push(line);
      continue;
    }

    // ── Detect start of a single-backtick block ───────────────────────────
    // A line that starts with exactly one backtick (not `` or ```) and does
    // NOT also end with a backtick on the same character (which would make it
    // a complete single-line inline span that we leave alone).
    if (
      collected === null &&
      /^`(?!`)/.test(line) &&
      !(line.length > 1 && line.endsWith("`") && !line.endsWith("``"))
    ) {
      // Strip the leading backtick (and any space the LLM puts after it).
      collected = [line.slice(1).trimStart()];
      continue;
    }

    // ── Accumulate / close an open block ─────────────────────────────────
    if (collected !== null) {
      if (line.endsWith("`") && !line.endsWith("``")) {
        // Closing line — strip the trailing backtick (and any space before it).
        collected.push(line.slice(0, -1).trimEnd());
        // Trim blank lines at top and bottom of the collected content.
        while (collected.length > 0 && collected[0].trim() === "") collected.shift();
        while (collected.length > 0 && collected[collected.length - 1].trim() === "") collected.pop();
        // Use explicit "text" language so the code component can identify
        // this as block code (react-markdown v10 uses className, not inline prop).
        output.push("```text");
        output.push(...collected);
        output.push("```");
        collected = null;
      } else {
        // Still inside the block — accumulate (blank lines are preserved).
        collected.push(line);
      }
      continue;
    }

    output.push(line);
  }

  // Unclosed block at end-of-input — emit raw to avoid data loss.
  if (collected !== null) {
    output.push("`" + collected.join("\n"));
  }

  return output.join("\n");
}
