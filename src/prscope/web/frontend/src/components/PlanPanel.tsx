import { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { FileText, Copy, Check, Download, AlertCircle, Clock } from "lucide-react";
import { clsx } from "clsx";
import mermaid from "mermaid";
import { Tooltip } from "./ui/Tooltip";
import { IssuePanel } from "./IssuePanel";
import { getPressuredDecisions, getTopPressureSummary } from "../lib/impactView";
import { preprocessPlanMarkdown } from "../lib/markdown";
import { planMarkdownComponents } from "../lib/markdownComponents";
import { augmentPlanMarkdownWithDecisionGraph } from "../lib/decisionGraphRender";
import type { ArchitectureImpactView, DecisionGraph, IssueGraphNode, IssueGraphSnapshot, SessionStatus } from "../types";

interface PlanPanelProps {
  content: string;
  decisionGraph?: DecisionGraph | null;
  impactView?: ArchitectureImpactView | null;
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

export function PlanPanel({
  content,
  decisionGraph = null,
  impactView = null,
  status,
  isProcessing = false,
  canExport: _canExport,
  onExport: _onExport,
  onAppendIssuePrompt,
  health,
}: PlanPanelProps) {
  const [copied, setCopied] = useState(false);
  const [showIssuesPopup, setShowIssuesPopup] = useState(false);
  const [activeIssueTab, setActiveIssueTab] = useState<"issues" | "violations" | "root-causes" | "resolved">("root-causes");
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
  const resolvedIssues = (health?.issueGraph?.nodes?.filter((node) => node.status === "resolved") ?? [])
    .slice()
    .sort((a, b) => a.id.localeCompare(b.id)) as IssueGraphNode[];
  const rootIssues = (() => {
    const nodes = health?.issueGraph?.nodes ?? [];
    const edges = health?.issueGraph?.edges ?? [];
    const parents = new Map<string, Set<string>>();
    for (const edge of edges) {
      if (edge.relation !== "causes") continue;
      const current = parents.get(edge.target) ?? new Set<string>();
      current.add(edge.source);
      parents.set(edge.target, current);
    }
    return nodes
      .filter((node) => node.status === "open" && !(parents.get(node.id)?.size))
      .sort((a, b) => a.id.localeCompare(b.id));
  })();
  const rootCausesCount = health?.issueGraph?.summary?.root_open ?? 0;
  const constraintViolationsCount = health?.constraintViolationsCount ?? 0;
  const reviewItemsCount = (health?.openIssuesCount ?? 0) + constraintViolationsCount + rootCausesCount;
  const reviewLabel = reviewItemsCount > 0 ? `${reviewItemsCount} review notes` : "Review notes";
  const pressuredDecisions = getPressuredDecisions(impactView);
  const pressuredDecisionCount = pressuredDecisions.length;
  const topPressureSummary = getTopPressureSummary(impactView, decisionGraph);
  const planCharacterCount = content.trim().length;
  const planWordCount = content.trim() ? content.trim().split(/\s+/).length : 0;
  const renderedContent = preprocessPlanMarkdown(augmentPlanMarkdownWithDecisionGraph(content, decisionGraph));
  const planSizeLabel =
    planCharacterCount >= 1000
      ? `${(planCharacterCount / 1000).toFixed(planCharacterCount >= 10000 ? 0 : 1)}k chars`
      : `${planCharacterCount} chars`;
  const appendIssuePrompt = (issue: IssueGraphNode) => {
    const prompt = [
      `Please update the plan to address ${issue.description}.`,
      "Adjust the approach, tasks, dependencies, and success checks if needed.",
    ].join("\n");
    onAppendIssuePrompt?.(prompt);
  };
  const appendAllIssuesPrompt = () => {
    if (!openIssues.length) return;
    const issueLines = openIssues.map((issue, idx) => `${idx + 1}. ${issue.description}`);
    const prompt = [
      "Please update the plan to address these review notes:",
      "",
      ...issueLines,
      "",
      "Adjust the approach, tasks, dependencies, and success checks where needed.",
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
                <Tooltip content="Open review notes, issues, and violations">
                  <button 
                    type="button"
                    onClick={() => {
                      setActiveIssueTab("root-causes");
                      setShowIssuesPopup(!showIssuesPopup);
                    }}
                    className={clsx(
                      "flex items-center gap-2 px-3 py-1.5 rounded-full border cursor-pointer transition-colors",
                      reviewItemsCount > 0
                        ? "bg-amber-500/10 border-amber-500/20 text-amber-400 hover:bg-amber-500/15"
                        : "bg-zinc-800/30 border-zinc-800/50 text-zinc-400 hover:bg-zinc-800/50",
                    )}
                  >
                    <AlertCircle className="w-3.5 h-3.5" />
                    <span className="font-medium">{reviewLabel}</span>
                  </button>
                </Tooltip>

                {showIssuesPopup && (
                  <IssuePanel
                    openIssues={openIssues}
                    rootIssues={rootIssues}
                    resolvedIssues={resolvedIssues}
                    constraintViolations={health.constraintViolations ?? []}
                    decisionGraph={decisionGraph}
                    impactView={impactView}
                    rootCausesCount={rootCausesCount}
                    onAppendIssue={appendIssuePrompt}
                    onAppendAllIssues={appendAllIssuesPrompt}
                    onClose={() => setShowIssuesPopup(false)}
                    className="left-0 right-auto origin-top-left w-[560px] max-w-[calc(100vw-2rem)]"
                    initialTab={activeIssueTab}
                  />
                )}
              </div>

              {pressuredDecisionCount > 0 ? (
                <Tooltip content="Architectural decisions currently under pressure from review findings">
                  <div className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-indigo-500/20 bg-indigo-500/10 text-indigo-300">
                    <AlertCircle className="w-3.5 h-3.5" />
                    <span className="font-medium">
                      {pressuredDecisionCount} decision{pressuredDecisionCount === 1 ? "" : "s"} under pressure
                    </span>
                  </div>
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
              <Tooltip content={`Plan length: ${planWordCount.toLocaleString()} words, ${planCharacterCount.toLocaleString()} characters`}>
                <div className="flex items-center gap-1.5 px-2 py-1 text-zinc-600">
                  <FileText className="w-3.5 h-3.5" />
                  <span>{planSizeLabel}</span>
                </div>
              </Tooltip>
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
        {topPressureSummary ? (
          <div className="border-b border-zinc-800/30 bg-indigo-500/5 px-6 py-3">
            <div className="flex flex-wrap items-center gap-2 text-xs text-indigo-200">
              <span className="font-semibold uppercase tracking-wider text-indigo-300">Top pressure</span>
              <span className="rounded-full border border-indigo-500/20 bg-indigo-500/10 px-2 py-0.5">
                {topPressureSummary.label}
              </span>
              {topPressureSummary.dominantCluster ? (
                <>
                  <span className="text-zinc-400">Root cause: {topPressureSummary.dominantCluster.rootIssue}</span>
                  <span className="text-zinc-500">Action: {topPressureSummary.dominantCluster.suggestedAction}</span>
                </>
              ) : null}
            </div>
          </div>
        ) : null}
        <div className="pb-12">
          <div className="max-w-3xl mx-auto pt-6 py-12 px-4 md:px-8" ref={mermaidRef}>
            <article className="prose prose-zinc prose-invert max-w-none">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={planMarkdownComponents}
              >
                {renderedContent}
              </ReactMarkdown>
            </article>
          </div>
        </div>
      </div>
    </div>
  );
}
