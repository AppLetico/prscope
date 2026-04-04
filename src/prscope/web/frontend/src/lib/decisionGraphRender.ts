import type { DecisionGraph, DecisionNode } from "../types";

/** Safe Mermaid node id (alphanumeric + underscore; leading digit prefixed). */
export function sanitizeMermaidNodeId(id: string): string {
  const s = id.replace(/[^a-zA-Z0-9_]/g, "_");
  if (!s || /^[0-9]/.test(s)) {
    return `n_${s || "id"}`;
  }
  return s;
}

function labelForMermaid(s: string, max = 100): string {
  return s.replace(/\r?\n/g, " ").replace(/"/g, "'").replace(/[\[\]]/g, "").slice(0, max);
}

/**
 * Build a small flowchart from structured decision graph data (deterministic; no LLM).
 * Returns null when there is nothing to draw.
 */
export function decisionGraphToMermaid(graph: DecisionGraph): string | null {
  const nodes = Object.values(graph.nodes ?? {});
  if (nodes.length === 0) return null;

  const idMap = new Map<string, string>();
  for (const n of nodes) {
    idMap.set(n.id, sanitizeMermaidNodeId(n.id));
  }

  const edges = graph.edges ?? [];
  const lines: string[] = [];

  if (edges.length > 0) {
    lines.push("flowchart LR");
    for (const n of nodes) {
      const mid = idMap.get(n.id)!;
      const lab = labelForMermaid(n.description || n.id);
      lines.push(`    ${mid}["${lab}"]`);
    }
    for (const e of edges) {
      const s = idMap.get(e.source);
      const t = idMap.get(e.target);
      if (!s || !t) continue;
      const rel = labelForMermaid((e.relation || "·").replace(/\|/g, " "), 40);
      lines.push(`    ${s} -->|${rel}| ${t}`);
    }
  } else {
    lines.push("flowchart TB");
    const bySection = new Map<string, DecisionNode[]>();
    for (const n of nodes) {
      const sec = n.section || "other";
      if (!bySection.has(sec)) bySection.set(sec, []);
      bySection.get(sec)!.push(n);
    }
    for (const [section, list] of bySection) {
      const subId = sanitizeMermaidNodeId(`sub_${section}`);
      lines.push(`    subgraph ${subId} ["${labelForMermaid(section, 48)}"]`);
      for (const n of list) {
        const mid = idMap.get(n.id)!;
        lines.push(`      ${mid}["${labelForMermaid(n.description || n.id)}"]`);
      }
      lines.push(`    end`);
    }
  }

  return lines.join("\n");
}

/** True when Architecture already contains a mermaid fence (skip auto map to avoid noise). */
export function hasMermaidInArchitectureSection(md: string): boolean {
  return /^##\s+Architecture\b[\s\S]*?```mermaid/mi.test(md);
}

function shouldShowAutoDecisionMap(graph: DecisionGraph): boolean {
  const nodes = Object.values(graph.nodes ?? {});
  if (nodes.length === 0) return false;
  const edges = graph.edges ?? [];
  if (edges.length > 0) return true;
  return nodes.length >= 2;
}

/** Injects ### Decision map (auto) with fenced mermaid under Architecture when useful. */
function maybeInsertDecisionMapMermaid(content: string, graph: DecisionGraph): string {
  if (!shouldShowAutoDecisionMap(graph)) return content;
  if (hasMermaidInArchitectureSection(content)) return content;
  const diagram = decisionGraphToMermaid(graph);
  if (!diagram) return content;
  const addition = `### Decision map (auto)\n\n\`\`\`mermaid\n${diagram}\n\`\`\``;
  if (hasSection(content, "Architecture")) {
    return insertIntoSection(content, "Architecture", addition);
  }
  return appendSection(content, "Architecture", addition);
}

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

/**
 * Injects decision state and open questions into plan markdown for the left panel.
 * UX follow-up: consider folding long pressure blocks (e.g. collapsible details) or
 * surfacing unresolved decisions primarily via chat follow-ups instead of plan body.
 */
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

  next = maybeInsertDecisionMapMermaid(next, decisionGraph);

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
