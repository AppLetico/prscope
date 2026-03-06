import { useState, useEffect, useRef, type JSX } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { FileText, Copy, Check, Download, AlertCircle, ShieldAlert, Clock, Target } from "lucide-react";
import { clsx } from "clsx";
import mermaid from "mermaid";
import { Tooltip } from "./ui/Tooltip";
import { IssuePanel } from "./IssuePanel";
import { preprocessPlanMarkdown } from "../lib/markdown";
import { planMarkdownComponents } from "../lib/markdownComponents";
import type { IssueGraphNode, IssueGraphSnapshot, SessionStatus } from "../types";

interface PlanPanelProps {
  content: string;
  status?: SessionStatus;
  isProcessing?: boolean;
  canExport?: boolean;
  onExport?: () => void;
  onAppendIssuePrompt?: (text: string) => void;
  health?: {
    snapshotUpdatedAt?: string;
    openIssuesCount?: number;
    constraintViolationsCount?: number;
    constraintViolations?: string[];
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
  isProcessing = false,
  canExport: _canExport,
  onExport: _onExport,
  onAppendIssuePrompt,
  health,
}: PlanPanelProps) {
  const [copied, setCopied] = useState(false);
  const [showIssuesPopup, setShowIssuesPopup] = useState(false);
  const [activeIssueTab, setActiveIssueTab] = useState<"issues" | "violations" | "root-causes">("issues");
  const issuesPopupRef = useRef<HTMLDivElement>(null);
  const toolbarRef = useRef<HTMLDivElement>(null);
  const mermaidRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (toolbarRef.current && !toolbarRef.current.contains(event.target as Node)) {
        setShowIssuesPopup(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

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
    const isInProgress = isProcessing && (status === "draft" || status === "refining");
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

  const openIssues = (health?.issueGraph?.nodes?.filter((node) => node.status === "open") ?? []) as IssueGraphNode[];
  const appendIssuePrompt = (issue: IssueGraphNode) => {
    const severity = issue.severity ? issue.severity.toUpperCase() : "UNSPECIFIED";
    const prompt = [
      `Issue ${issue.id} (${severity}): ${issue.description}`,
      "",
      "Please fix this issue and update the plan accordingly.",
      "Include any required architectural/task changes, dependencies, and acceptance criteria updates.",
    ].join("\n");
    onAppendIssuePrompt?.(prompt);
  };
  const appendAllIssuesPrompt = () => {
    if (!openIssues.length) return;
    const issueLines = openIssues.map((issue, idx) => {
      const severity = issue.severity ? issue.severity.toUpperCase() : "UNSPECIFIED";
      return `${idx + 1}. ${issue.id} (${severity}): ${issue.description}`;
    });
    const prompt = [
      "Please fix the following open issues and update the plan accordingly:",
      "",
      ...issueLines,
      "",
      "For each issue, adjust architecture/tasks/dependencies/acceptance criteria as needed.",
    ].join("\n");
    onAppendIssuePrompt?.(prompt);
  };

  return (
    <div className="h-full flex flex-col bg-zinc-900 relative">
      <div className="flex-1 overflow-y-auto scroll-smooth">
        <div className="flex items-center justify-between gap-4 px-6 py-4 border-b border-zinc-800/40">
          {health ? (
            <div className="flex items-center gap-3 text-[11px] text-zinc-400 flex-wrap" ref={toolbarRef}>
              <div className="relative" ref={issuesPopupRef}>
                <Tooltip content="Total unresolved issues">
                  <button 
                    type="button"
                    onClick={() => {
                      setActiveIssueTab("issues");
                      setShowIssuesPopup(!showIssuesPopup);
                    }}
                    className={clsx("flex items-center gap-1.5 px-2.5 py-1 rounded-full border cursor-pointer hover:opacity-80 transition-opacity", (health.openIssuesCount ?? 0) > 0 ? "bg-amber-500/10 border-amber-500/20 text-amber-500" : "bg-zinc-800/30 border-zinc-800/50 text-zinc-500")}
                  >
                    <AlertCircle className="w-3.5 h-3.5" />
                    <span className="font-medium">{health.openIssuesCount ?? 0}</span>
                    <span className="opacity-80 hidden sm:inline">Issues</span>
                  </button>
                </Tooltip>

                {showIssuesPopup && (
                  <IssuePanel
                    openIssues={openIssues}
                    constraintViolations={health.constraintViolations ?? []}
                    rootCausesCount={health.issueGraph?.summary?.root_open ?? 0}
                    onAppendIssue={appendIssuePrompt}
                    onAppendAllIssues={appendAllIssuesPrompt}
                    onClose={() => setShowIssuesPopup(false)}
                    className="left-0 right-auto origin-top-left w-[400px]"
                    initialTab={activeIssueTab}
                  />
                )}
              </div>

              <Tooltip content="Violated constraints">
                <button
                  type="button"
                  onClick={() => {
                    setActiveIssueTab("violations");
                    setShowIssuesPopup(true);
                  }}
                  className={clsx("flex items-center gap-1.5 px-2.5 py-1 rounded-full border cursor-pointer hover:opacity-80 transition-opacity", (health.constraintViolationsCount ?? 0) > 0 ? "bg-red-500/10 border-red-500/20 text-red-500" : "bg-zinc-800/30 border-zinc-800/50 text-zinc-500")}
                >
                  <ShieldAlert className="w-3.5 h-3.5" />
                  <span className="font-medium">{health.constraintViolationsCount ?? 0}</span>
                  <span className="opacity-80 hidden sm:inline">Violations</span>
                </button>
              </Tooltip>

              {health.issueGraph?.summary?.root_open != null ? (
                <Tooltip content="Root causes: Open issues with no upstream causes">
                  <button
                    type="button"
                    onClick={() => {
                      setActiveIssueTab("root-causes");
                      setShowIssuesPopup(true);
                    }}
                    className={clsx("flex items-center gap-1.5 px-2.5 py-1 rounded-full border cursor-pointer hover:opacity-80 transition-opacity", health.issueGraph.summary.root_open > 0 ? "bg-blue-500/10 border-blue-500/20 text-blue-500" : "bg-zinc-800/30 border-zinc-800/50 text-zinc-500")}
                  >
                    <Target className="w-3.5 h-3.5" />
                    <span className="font-medium">{health.issueGraph.summary.root_open}</span>
                    <span className="opacity-80 hidden sm:inline">Root causes</span>
                  </button>
                </Tooltip>
              ) : null}

              {health.snapshotUpdatedAt ? (
                <Tooltip content={`Last updated: ${new Date(health.snapshotUpdatedAt).toLocaleString()}`}>
                  <div className="flex items-center gap-1.5 px-2 py-1 text-zinc-600">
                    <Clock className="w-3.5 h-3.5" />
                    <span>{new Date(health.snapshotUpdatedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                  </div>
                </Tooltip>
              ) : null}
            </div>
          ) : <div />}
          
          <div className="flex items-center gap-1">
            {_onExport && (
              <Tooltip content="Export plan">
                <button
                  type="button"
                  onClick={_onExport}
                  disabled={!_canExport}
                  className={clsx(
                    "p-1.5 rounded-md transition-colors",
                    _canExport
                      ? "text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800"
                      : "text-zinc-800 cursor-not-allowed"
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
                className="p-1.5 rounded-md text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800 transition-colors"
                aria-label="Copy plan"
              >
                {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
              </button>
            </Tooltip>
          </div>
        </div>
        <div className="pb-12">
          <div className="max-w-3xl mx-auto pt-6 py-12 px-4 md:px-8" ref={mermaidRef}>
            {health?.issueGraph ? (
              <div className="mb-6">{renderIssueGraph(health.issueGraph)}</div>
            ) : null}
            <article className="prose prose-zinc prose-invert max-w-none">
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
