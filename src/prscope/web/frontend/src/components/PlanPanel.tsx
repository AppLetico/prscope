import { useState, useEffect, useRef, useLayoutEffect, useCallback, type CSSProperties } from "react";
import { createPortal } from "react-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { FileText, Copy, Check, Download, AlertCircle, Clock } from "lucide-react";
import { clsx } from "clsx";
import { Tooltip } from "./ui/Tooltip";
import { IssuePanel } from "./IssuePanel";
import { dedupeOpenIssueNodesByDescription } from "../lib/issueDedupe";
import { getRelatedDecisionSummaries } from "../lib/impactView";
import { preprocessPlanMarkdown } from "../lib/markdown";
import { planMarkdownComponents } from "../lib/markdownComponents";
import { augmentPlanMarkdownWithDecisionGraph } from "../lib/decisionGraphRender";
import type { ArchitectureImpactView, DecisionGraph, IssueGraphNode, IssueGraphSnapshot, SessionStatus } from "../types";
import { getPlanPanelEmptyCopy } from "./planPanelUi";

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
  const [activeIssueTab, setActiveIssueTab] = useState<"issues" | "violations" | "resolved">("issues");
  /** Anchor for fixed positioning of the portaled IssuePanel (must match the review-notes control). */
  const issuesTriggerRef = useRef<HTMLDivElement>(null);
  const issuePanelRef = useRef<HTMLDivElement>(null);
  const [issuePanelFloatStyle, setIssuePanelFloatStyle] = useState<CSSProperties>({});
  const toolbarRef = useRef<HTMLDivElement>(null);

  const positionIssuePanel = useCallback(() => {
    const el = issuesTriggerRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const margin = 12;
    const maxW = Math.min(560, window.innerWidth - margin * 2);
    const left = Math.min(Math.max(margin, r.left), window.innerWidth - maxW - margin);
    setIssuePanelFloatStyle({
      position: "fixed",
      top: r.bottom + margin,
      left,
      width: maxW,
      maxHeight: "min(600px, calc(100vh - 24px))",
      zIndex: 500,
    });
  }, []);

  useLayoutEffect(() => {
    if (!showIssuesPopup) return;
    positionIssuePanel();
    const onScrollOrResize = () => positionIssuePanel();
    window.addEventListener("resize", onScrollOrResize);
    document.addEventListener("scroll", onScrollOrResize, true);
    return () => {
      window.removeEventListener("resize", onScrollOrResize);
      document.removeEventListener("scroll", onScrollOrResize, true);
    };
  }, [showIssuesPopup, positionIssuePanel]);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      const target = event.target as Node;
      if (toolbarRef.current?.contains(target)) return;
      if (issuePanelRef.current?.contains(target)) return;
      setShowIssuesPopup(false);
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  if (!content) {
    const emptyCopy = getPlanPanelEmptyCopy(isProcessing, status);
    return (
      <div className="h-full flex flex-col items-center justify-center text-zinc-500 bg-zinc-900">
        <FileText className="w-12 h-12 mb-4 opacity-20" />
        <p className="text-sm">{emptyCopy.title}</p>
        <p className="text-xs opacity-60 mt-1">{emptyCopy.subtitle}</p>
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

  const openIssuesRaw = (health?.issueGraph?.nodes?.filter((node) => node.status === "open") ?? []) as IssueGraphNode[];
  const openIssues = dedupeOpenIssueNodesByDescription(openIssuesRaw, 0.5);
  const resolvedIssues = (health?.issueGraph?.nodes?.filter((node) => node.status === "resolved") ?? [])
    .slice()
    .sort((a, b) => a.id.localeCompare(b.id)) as IssueGraphNode[];
  const rootIssueIds = (() => {
    const issueGraph = health?.issueGraph;
    if (!issueGraph) return [] as string[];
    const causedIssueIds = new Set(
      issueGraph.edges
        .filter((edge) => edge.relation === "causes")
        .map((edge) => edge.target),
    );
    return openIssues
      .filter((issue) => !causedIssueIds.has(issue.id))
      .map((issue) => issue.id);
  })();
  const constraintViolationsCount = health?.constraintViolationsCount ?? 0;
  const reviewItemsCount = (health?.openIssuesCount ?? 0) + constraintViolationsCount;
  const reviewLabel = reviewItemsCount > 0 ? `${reviewItemsCount} review notes` : "Review notes";
  const planCharacterCount = content.trim().length;
  const planWordCount = content.trim() ? content.trim().split(/\s+/).length : 0;
  const renderedContent = preprocessPlanMarkdown(augmentPlanMarkdownWithDecisionGraph(content, decisionGraph));
  const planSizeLabel =
    planCharacterCount >= 1000
      ? `${(planCharacterCount / 1000).toFixed(planCharacterCount >= 10000 ? 0 : 1)}k chars`
      : `${planCharacterCount} chars`;
  const appendIssuePrompt = (issue: IssueGraphNode) => {
    const decisions = getRelatedDecisionSummaries(issue, impactView, decisionGraph);
    const decisionLines = decisions
      .map((d) => {
        const text = (d.label || d.decisionId).trim();
        return text ? `- ${text}` : "";
      })
      .filter(Boolean);
    const decisionBlock =
      decisionLines.length > 0
        ? `\n\nRelated architectural decisions (context only—continue in chat to resolve or update the plan):\n${decisionLines.join("\n")}`
        : "";
    const prompt = [
      `Please update the plan to address ${issue.description}.`,
      `Tracked issue id: \`${issue.id}\` — include this id in resolved_issues when validation confirms the fix.`,
      "Adjust the approach, tasks, dependencies, and success checks if needed.",
      decisionBlock,
    ].join("\n");
    onAppendIssuePrompt?.(prompt);
  };
  const appendAllIssuesPrompt = () => {
    if (!openIssues.length) return;
    const decisionLabels = new Set<string>();
    for (const issue of openIssues) {
      for (const d of getRelatedDecisionSummaries(issue, impactView, decisionGraph)) {
        decisionLabels.add(d.label);
      }
    }
    const decisionLines = [...decisionLabels].map((l) => l.trim()).filter(Boolean).map((l) => `- ${l}`);
    const decisionFoot =
      decisionLines.length > 0
        ? `\n\nArchitectural decision context (reply in chat when you need to commit or change direction):\n${decisionLines.join("\n")}`
        : "";
    const issueLines = openIssues.map((issue, idx) => `${idx + 1}. ${issue.description}`);
    const idLine =
      openIssues.length > 0
        ? `Tracked issue ids: ${openIssues.map((i) => `\`${i.id}\``).join(", ")} — include fixed ids in resolved_issues.`
        : "";
    const prompt = [
      "Please update the plan to address these review notes:",
      "",
      ...issueLines,
      "",
      idLine,
      "Adjust the approach, tasks, dependencies, and success checks where needed.",
      decisionFoot,
    ]
      .filter((line) => line !== "")
      .join("\n");
    onAppendIssuePrompt?.(prompt);
  };

  return (
    <div className="h-full flex flex-col bg-zinc-900 relative">
      <div className="flex-1 overflow-y-auto scroll-smooth">
        <div className="flex items-center justify-between gap-4 px-6 py-4 border-b border-zinc-800/40">
          {health ? (
            <div className="flex items-center gap-3 text-[11px] text-zinc-400 flex-wrap" ref={toolbarRef}>
              <div className="relative" ref={issuesTriggerRef}>
                <Tooltip content="Open review notes, issues, and violations">
                  <button 
                    type="button"
                    onClick={() => {
                      const nextTab = openIssues.length > 0
                        ? "issues"
                        : constraintViolationsCount > 0
                          ? "violations"
                          : resolvedIssues.length > 0
                            ? "resolved"
                            : "issues";
                      setActiveIssueTab(nextTab);
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
              </div>

              {showIssuesPopup
                ? createPortal(
                    <IssuePanel
                      ref={issuePanelRef}
                      portaled
                      floatingStyle={issuePanelFloatStyle}
                      openIssues={openIssues}
                      rootIssueIds={rootIssueIds}
                      resolvedIssues={resolvedIssues}
                      constraintViolations={health.constraintViolations ?? []}
                      decisionGraph={decisionGraph}
                      impactView={impactView}
                      onAppendIssue={appendIssuePrompt}
                      onAppendAllIssues={appendAllIssuesPrompt}
                      onClose={() => setShowIssuesPopup(false)}
                      initialTab={activeIssueTab}
                    />,
                    document.body,
                  )
                : null}

              <Tooltip
                content={
                  health.snapshotUpdatedAt
                    ? `Last updated: ${new Date(health.snapshotUpdatedAt).toLocaleString()} · ${planWordCount.toLocaleString()} words`
                    : `Plan length: ${planWordCount.toLocaleString()} words, ${planCharacterCount.toLocaleString()} characters`
                }
              >
                <div className="flex items-center gap-1.5 px-2 py-1 text-zinc-600">
                  {health.snapshotUpdatedAt ? (
                    <>
                      <Clock className="w-3.5 h-3.5 shrink-0" />
                      <span>
                        {new Date(health.snapshotUpdatedAt).toLocaleTimeString([], {
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </span>
                      <span className="text-zinc-700" aria-hidden>
                        ·
                      </span>
                    </>
                  ) : null}
                  <FileText className="w-3.5 h-3.5 shrink-0" />
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
        <div className="pb-12">
          <div className="max-w-3xl mx-auto pt-6 py-12 px-4 md:px-8">
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
