/**
 * Lexical overlap for critic issue lines (aligns with IssueSimilarityService._find_lexical_duplicate).
 * Merges paraphrases such as "lack of rollback detail" vs "expand the rollback plan with steps".
 */

const STOPWORDS = new Set([
  "the",
  "a",
  "an",
  "and",
  "or",
  "to",
  "of",
  "in",
  "for",
  "with",
  "is",
  "are",
  "be",
  "that",
  "this",
  "it",
  "as",
  "on",
]);

function tokenSet(text: string): Set<string> {
  const parts = text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .map((p) => p.trim())
    .filter((p) => p.length > 2 && !STOPWORDS.has(p));
  return new Set(parts);
}

/** Jaccard similarity on token sets (same formula as backend issue similarity). */
export function jaccardIssueSimilarity(a: string, b: string): number {
  const A = tokenSet(a);
  const B = tokenSet(b);
  if (A.size === 0 || B.size === 0) return 0;
  let inter = 0;
  for (const t of A) {
    if (B.has(t)) inter += 1;
  }
  const union = A.size + B.size - inter;
  return union > 0 ? inter / union : 0;
}

/**
 * Combined score for duplicate detection: max(Jaccard, recall onto smaller set).
 * Catches paraphrases where one line is a subset of the other (e.g. "vague monitoring strategy"
 * vs "lack of specific … monitoring strategy for middleware") where raw Jaccard stays low.
 */
export function issueSimilarityScore(a: string, b: string): number {
  const jac = jaccardIssueSimilarity(a, b);
  const A = tokenSet(a);
  const B = tokenSet(b);
  if (A.size === 0 || B.size === 0) return jac;
  let inter = 0;
  for (const t of A) {
    if (B.has(t)) inter += 1;
  }
  const smaller = Math.min(A.size, B.size);
  const recall = smaller > 0 ? inter / smaller : 0;
  return Math.max(jac, recall);
}

export type IssueSeverity = "major" | "minor";

/**
 * Drop or merge items that describe the same underlying gap (problem vs suggested fix).
 * Keeps earlier items when both are major; upgrades minor to major when a duplicate major appears.
 */
export function dedupeSimilarCriticIssues(
  items: Array<{ text: string; severity: IssueSeverity }>,
  threshold = 0.5,
): Array<{ text: string; severity: IssueSeverity }> {
  const out: Array<{ text: string; severity: IssueSeverity }> = [];
  for (const item of items) {
    const t = item.text.trim();
    if (!t) continue;
    const dupIdx = out.findIndex((existing) => issueSimilarityScore(existing.text, t) >= threshold);
    if (dupIdx < 0) {
      out.push({ text: t, severity: item.severity });
      continue;
    }
    const existing = out[dupIdx];
    if (item.severity === "major" && existing.severity === "minor") {
      out[dupIdx] = { text: t, severity: "major" };
    }
  }
  return out;
}

/** Dedupe open issue nodes for display (keeps first in sort order when similar). */
export function dedupeOpenIssueNodesByDescription<T extends { description: string }>(
  issues: T[],
  threshold = 0.5,
): T[] {
  const out: T[] = [];
  for (const issue of issues) {
    const d = issue.description.trim();
    if (!d) continue;
    const isDup = out.some((o) => issueSimilarityScore(o.description, d) >= threshold);
    if (!isDup) out.push(issue);
  }
  return out;
}
