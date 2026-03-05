import { useState, useEffect, useRef, type JSX } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { FileText, Copy, Check, Download } from "lucide-react";
import { clsx } from "clsx";
import mermaid from "mermaid";
import { Tooltip } from "./ui/Tooltip";
import { preprocessPlanMarkdown } from "../lib/markdown";
import { planMarkdownComponents } from "../lib/markdownComponents";
import type { IssueGraphSnapshot, SessionStatus } from "../types";

interface PlanPanelProps {
  content: string;
  status?: SessionStatus;
  canExport?: boolean;
  onExport?: () => void;
  health?: {
    snapshotUpdatedAt?: string;
    openIssuesCount?: number;
    constraintViolationsCount?: number;
    issueGraph?: IssueGraphSnapshot | null;
  };
}

function renderIssueGraph(graph: IssueGraphSnapshot): JSX.Element | null {
  const nodes = Array.isArray(graph.nodes) ? graph.nodes : [];
  const edges = Array.isArray(graph.edges) ? graph.edges : [];
  if (!nodes.length) return null;
  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  const causesChildren = new Map<string, string[]>();
  const causesParents = new Map<string, string[]>();
  const dependencies = new Map<string, string[]>();
  for (const edge of edges) {
    if (edge.relation === "causes") {
      const children = causesChildren.get(edge.source) ?? [];
      children.push(edge.target);
      causesChildren.set(edge.source, children);
      const parents = causesParents.get(edge.target) ?? [];
      parents.push(edge.source);
      causesParents.set(edge.target, parents);
    } else if (edge.relation === "depends_on") {
      const deps = dependencies.get(edge.source) ?? [];
      deps.push(edge.target);
      dependencies.set(edge.source, deps);
    }
  }
  const duplicateAlias = graph.duplicate_alias ?? {};
  const rootIds = nodes
    .filter((node) => node.status === "open" && !(causesParents.get(node.id)?.length))
    .map((node) => node.id);

  const renderNode = (id: string, depth: number, visited: Set<string>): JSX.Element | null => {
    if (visited.has(id)) return null;
    const node = nodeById.get(id);
    if (!node || node.status !== "open") return null;
    const nextVisited = new Set(visited);
    nextVisited.add(id);
    const children = (causesChildren.get(id) ?? []).sort();
    const deps = (dependencies.get(id) ?? [])
      .map((depId) => nodeById.get(depId)?.description || depId)
      .filter(Boolean);
    const aliases = Object.entries(duplicateAlias)
      .filter(([, canonical]) => canonical === id)
      .map(([alias]) => alias)
      .sort();
    return (
      <li key={`${id}-${depth}`} className="text-xs text-zinc-300">
        <div className="flex flex-col gap-1">
          <span className="font-medium">{node.description}</span>
          {deps.length > 0 ? (
            <span className="text-[11px] text-zinc-500">depends on: {deps.join(", ")}</span>
          ) : null}
          {aliases.length > 0 ? (
            <span className="text-[11px] text-zinc-500">aliases: {aliases.join(", ")}</span>
          ) : null}
        </div>
        {children.length > 0 ? (
          <ul className="ml-4 mt-1 border-l border-zinc-800 pl-3 space-y-1">
            {children.map((childId) => renderNode(childId, depth + 1, nextVisited))}
          </ul>
        ) : null}
      </li>
    );
  };

  if (!rootIds.length) return null;
  return (
    <div className="rounded border border-zinc-800 bg-zinc-900/80 px-3 py-2">
      <div className="text-[11px] uppercase tracking-wide text-zinc-500 mb-2">Issue Graph</div>
      <ul className="space-y-2">
        {rootIds.sort().map((id) => renderNode(id, 0, new Set<string>()))}
      </ul>
    </div>
  );
}

export function PlanPanel({
  content,
  status,
  canExport: _canExport,
  onExport: _onExport,
  health,
}: PlanPanelProps) {
  const [copied, setCopied] = useState(false);
  const mermaidRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    mermaid.initialize({
      startOnLoad: false,
      theme: "dark",
      securityLevel: "loose",
      fontFamily: "inherit",
    });
  }, []);

  useEffect(() => {
    if (content && mermaidRef.current) {
      void mermaid.run({
        nodes: mermaidRef.current.querySelectorAll(".mermaid"),
        suppressErrors: true,
      });
    }
  }, [content]);

  if (!content) {
    const isInProgress = status === "draft" || status === "refining";
    return (
      <div className="h-full flex flex-col items-center justify-center text-zinc-500 bg-zinc-900">
        <FileText className="w-12 h-12 mb-4 opacity-20" />
        {isInProgress ? (
          <>
            <p className="text-sm">Generating plan...</p>
            <p className="text-xs opacity-60 mt-1">
              Drafting and validation are in progress.
            </p>
          </>
        ) : (
          <>
            <p className="text-sm">No plan generated yet.</p>
            <p className="text-xs opacity-60 mt-1">
              Add requirements or run discovery-style chat to begin.
            </p>
          </>
        )}
      </div>
    );
  }

  const copyPlan = async () => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard write may be unavailable in restricted contexts.
    }
  };

  return (
    <div className="h-full flex flex-col bg-zinc-900 relative">
      <div className="flex-1 overflow-y-auto scroll-smooth">
        <div className="sticky top-4 z-10 flex items-center justify-end gap-2 px-6">
          {health ? (
            <div className="mr-auto flex items-center gap-2 text-[11px] text-zinc-400 flex-wrap">
              <span className="rounded border border-zinc-800 bg-zinc-900/80 px-2 py-0.5">
                Open issues: {health.openIssuesCount ?? 0}
              </span>
              <span className="rounded border border-zinc-800 bg-zinc-900/80 px-2 py-0.5">
                Constraint violations: {health.constraintViolationsCount ?? 0}
              </span>
              {health.snapshotUpdatedAt ? (
                <span className="rounded border border-zinc-800 bg-zinc-900/80 px-2 py-0.5">
                  Snapshot: {new Date(health.snapshotUpdatedAt).toLocaleTimeString()}
                </span>
              ) : null}
              {health.issueGraph?.summary?.root_open != null ? (
                <span className="rounded border border-zinc-800 bg-zinc-900/80 px-2 py-0.5">
                  Root open: {health.issueGraph.summary.root_open}
                </span>
              ) : null}
            </div>
          ) : null}
          {_onExport && (
            <Tooltip content="Export plan">
              <button
                type="button"
                onClick={_onExport}
                disabled={!_canExport}
                className={clsx(
                  "p-1.5 rounded-md transition-colors border border-zinc-800/80 bg-zinc-900/80 backdrop-blur",
                  _canExport
                    ? "text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800/70"
                    : "text-zinc-700 cursor-not-allowed"
                )}
                aria-label="Export plan"
              >
                <Download className="w-4 h-4" />
              </button>
            </Tooltip>
          )}
          <Tooltip content={copied ? "Copied" : "Copy plan"}>
            <button
              type="button"
              onClick={() => void copyPlan()}
              className="p-1.5 rounded-md text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800/70 transition-colors border border-zinc-800/80 bg-zinc-900/80 backdrop-blur"
              aria-label="Copy plan"
            >
              {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
            </button>
          </Tooltip>
        </div>
        <div className="pb-12">
          <div className="max-w-3xl mx-auto pt-6 py-12 px-4 md:px-8" ref={mermaidRef}>
            {health?.issueGraph ? (
              <div className="mb-6">{renderIssueGraph(health.issueGraph)}</div>
            ) : null}
            <article className="prose prose-zinc max-w-none">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={planMarkdownComponents}
              >
                {preprocessPlanMarkdown(content)}
              </ReactMarkdown>
            </article>
          </div>
        </div>
      </div>
    </div>
  );
}
