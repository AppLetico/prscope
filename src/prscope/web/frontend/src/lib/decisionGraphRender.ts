import type { DecisionGraph, DecisionNode } from "../types";

function hasSection(content: string, heading: string): boolean {
  const pattern = new RegExp(`^##\\s+${heading}\\b`, "im");
  return pattern.test(content);
}

function insertIntoSection(content: string, heading: string, addition: string): string {
  const pattern = new RegExp(`(^##\\s+${heading}\\b[\\s\\S]*?)(?=^##\\s+|$)`, "im");
  const match = content.match(pattern);
  if (!match || !match[1]) return content;
  const sectionBody = match[1].trimEnd();
  return content.replace(pattern, `${sectionBody}\n\n${addition}\n`);
}

function appendSection(content: string, heading: string, body: string): string {
  const trimmed = content.trimEnd();
  return `${trimmed}\n\n## ${heading}\n${body}\n`;
}

function normalizedDecisionText(text: string): string {
  return text.trim().toLowerCase();
}

function resolvedArchitectureNodes(graph: DecisionGraph): DecisionNode[] {
  return Object.values(graph.nodes)
    .filter((node) => node.section === "architecture" && Boolean(node.value?.trim()))
    .sort((left, right) => left.description.localeCompare(right.description));
}

function unresolvedNodes(graph: DecisionGraph): DecisionNode[] {
  return Object.values(graph.nodes)
    .filter((node) => (node.required ?? true) && !node.value?.trim())
    .sort((left, right) => left.description.localeCompare(right.description));
}

function renderDecisionStateSubsection(nodes: DecisionNode[]): string {
  const lines = ["### Decision State"];
  for (const node of nodes) {
    lines.push(`- ${node.description}: ${node.value}`);
  }
  return lines.join("\n");
}

function renderOpenQuestions(nodes: DecisionNode[]): string {
  return nodes.map((node) => `- ${node.description}`).join("\n");
}

export function augmentPlanMarkdownWithDecisionGraph(
  content: string,
  decisionGraph: DecisionGraph | null | undefined,
): string {
  if (!decisionGraph || Object.keys(decisionGraph.nodes ?? {}).length === 0) {
    return content;
  }

  let next = content;

  const resolvedArchitecture = resolvedArchitectureNodes(decisionGraph);
  if (resolvedArchitecture.length > 0 && !/^\s*###\s+Decision State\b/im.test(next)) {
    const addition = renderDecisionStateSubsection(resolvedArchitecture);
    next = hasSection(next, "Architecture")
      ? insertIntoSection(next, "Architecture", addition)
      : appendSection(next, "Architecture", addition);
  }

  const unresolved = unresolvedNodes(decisionGraph);
  if (unresolved.length === 0) {
    return next;
  }

  const missingQuestions = unresolved.filter((node) => {
    const normalized = normalizedDecisionText(node.description);
    return !next.toLowerCase().includes(normalized);
  });
  if (missingQuestions.length === 0) {
    return next;
  }

  const renderedQuestions = renderOpenQuestions(missingQuestions);
  if (hasSection(next, "Open Questions")) {
    return insertIntoSection(next, "Open Questions", renderedQuestions);
  }
  return appendSection(next, "Open Questions", renderedQuestions);
}
