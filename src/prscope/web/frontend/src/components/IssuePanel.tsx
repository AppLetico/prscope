import { useState, useRef, useEffect, type ComponentType } from "react";
import { AlertCircle, ShieldAlert, Target, X, MessageSquarePlus, CheckCircle2, Info } from "lucide-react";
import { clsx } from "clsx";
import { getRelatedDecisionSummaries } from "../lib/impactView";
import type { ArchitectureImpactView, DecisionGraph, IssueGraphNode } from "../types";

interface IssuePanelProps {
  openIssues: IssueGraphNode[];
  rootIssues: IssueGraphNode[];
  resolvedIssues: IssueGraphNode[];
  constraintViolations: string[];
  decisionGraph?: DecisionGraph | null;
  impactView?: ArchitectureImpactView | null;
  rootCausesCount: number;
  onAppendIssue: (issue: IssueGraphNode) => void;
  onAppendAllIssues: () => void;
  onClose: () => void;
  className?: string;
  initialTab?: Tab;
}

type Tab = "issues" | "violations" | "root-causes" | "resolved";

export function IssuePanel({
  openIssues,
  rootIssues,
  resolvedIssues,
  constraintViolations,
  decisionGraph = null,
  impactView = null,
  rootCausesCount,
  onAppendIssue,
  onAppendAllIssues,
  onClose,
  className,
  initialTab = "root-causes",
}: IssuePanelProps) {
  const [activeTab, setActiveTab] = useState<Tab>(initialTab);
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setActiveTab(initialTab);
  }, [initialTab]);

  return (
    <div 
      ref={panelRef}
      className={clsx(
        "absolute top-full mt-3 right-0 z-50 w-[480px] max-h-[600px] flex flex-col",
        "bg-zinc-900/95 backdrop-blur-xl border border-zinc-800",
        "rounded-xl shadow-2xl shadow-black/50 ring-1 ring-white/10",
        "origin-top-right animate-in fade-in zoom-in-95 duration-200",
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-800/50">
        <h3 className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.4)]" />
          Review Notes
        </h3>
        <button 
          onClick={onClose}
          className="text-zinc-500 hover:text-zinc-300 transition-colors p-1 hover:bg-zinc-800 rounded-md"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Tabs */}
      <div className="flex flex-wrap items-stretch gap-1 px-2 pt-2 border-b border-zinc-800/50 bg-zinc-900/50">
        <TabButton 
          active={activeTab === "root-causes"} 
          onClick={() => setActiveTab("root-causes")}
          icon={Target}
          label="Start Here"
          count={rootCausesCount}
          color="blue"
        />
        <TabButton 
          active={activeTab === "issues"} 
          onClick={() => setActiveTab("issues")}
          icon={AlertCircle}
          label="Issues"
          count={openIssues.length}
          color="amber"
        />
        <TabButton 
          active={activeTab === "violations"} 
          onClick={() => setActiveTab("violations")}
          icon={ShieldAlert}
          label="Violations"
          count={constraintViolations.length}
          color="red"
        />
        <TabButton
          active={activeTab === "resolved"}
          onClick={() => setActiveTab("resolved")}
          icon={CheckCircle2}
          label="Resolved"
          count={resolvedIssues.length}
          color="emerald"
        />
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 min-h-[300px]">
        {activeTab === "issues" && (
          <div className="space-y-4">
            {openIssues.length > 0 ? (
              <>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-zinc-500 font-medium uppercase tracking-wider">
                    {openIssues.length} Open issues
                  </span>
                  <button
                    onClick={onAppendAllIssues}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 hover:bg-indigo-500/20 hover:text-indigo-300 transition-all text-xs font-medium group"
                  >
                    <MessageSquarePlus className="w-3.5 h-3.5 group-hover:scale-110 transition-transform" />
                    Add all to chat
                  </button>
                </div>
                <div className="space-y-3">
                  {openIssues.map((issue) => (
                    <IssueCard 
                      key={issue.id} 
                      issue={issue} 
                      decisionGraph={decisionGraph}
                      impactView={impactView}
                      onAdd={() => onAppendIssue(issue)} 
                    />
                  ))}
                </div>
              </>
            ) : (
              <EmptyState 
                icon={CheckCircle2} 
                title="All clear" 
                description="No open issues found in the current plan." 
              />
            )}
          </div>
        )}

        {activeTab === "violations" && (
          <div className="space-y-4">
            {constraintViolations.length > 0 ? (
              <>
                 <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-zinc-500 font-medium uppercase tracking-wider">
                    {constraintViolations.length} Constraint violations
                  </span>
                </div>
                <div className="space-y-3">
                  {constraintViolations.map((violation, idx) => (
                    <div key={idx} className="p-4 rounded-lg bg-red-500/5 border border-red-500/10 hover:border-red-500/20 transition-colors">
                      <div className="flex gap-3">
                        <ShieldAlert className="w-5 h-5 text-red-500 shrink-0 mt-0.5" />
                        <p className="text-sm text-zinc-300 leading-relaxed">{violation}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <EmptyState 
                icon={CheckCircle2} 
                title="No violations" 
                description="The plan adheres to all defined constraints." 
              />
            )}
          </div>
        )}

        {activeTab === "root-causes" && (
          <div className="space-y-4">
            {rootCausesCount > 0 ? (
              <>
                <div className="p-4 rounded-lg bg-blue-500/5 border border-blue-500/10 mb-4">
                  <div className="flex gap-3">
                    <Info className="w-5 h-5 text-blue-400 shrink-0 mt-0.5" />
                    <div>
                      <h4 className="text-sm font-medium text-blue-400 mb-1">Best place to start</h4>
                      <p className="text-xs text-blue-300/80 leading-relaxed">
                        These are the issues that do not appear to be caused by another issue in the current graph.
                        Fixing one of these often clears several related follow-on issues.
                      </p>
                    </div>
                  </div>
                </div>
                <div className="space-y-3">
                  {rootIssues.map((issue) => (
                    <IssueCard
                      key={issue.id}
                      issue={issue}
                      decisionGraph={decisionGraph}
                      impactView={impactView}
                      onAdd={() => onAppendIssue(issue)}
                    />
                  ))}
                </div>
              </>
            ) : (
              <EmptyState 
                icon={CheckCircle2} 
                title="No starting issues" 
                description="No standalone starting issues were found in the current graph." 
              />
            )}
          </div>
        )}

        {activeTab === "resolved" && (
          <div className="space-y-4">
            {resolvedIssues.length > 0 ? (
              <>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-zinc-500 font-medium uppercase tracking-wider">
                    {resolvedIssues.length} Resolved issues
                  </span>
                </div>
                <div className="space-y-3">
                  {resolvedIssues.map((issue) => (
                    <IssueCard
                      key={issue.id}
                      issue={issue}
                      decisionGraph={decisionGraph}
                      impactView={impactView}
                      onAdd={() => onAppendIssue(issue)}
                      canAdd={false}
                      statusLabel={`Resolved${issue.resolved_round ? ` in round ${issue.resolved_round}` : ""}`}
                      statusTone="resolved"
                      provenanceLabel={getResolvedProvenanceLabel(issue)}
                      provenanceTone={issue.resolution_source === "lightweight" ? "lightweight" : "review"}
                    />
                  ))}
                </div>
              </>
            ) : (
              <EmptyState
                icon={CheckCircle2}
                title="No resolved issues"
                description="Resolved review notes will appear here when issues are closed."
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function TabButton({ 
  active, 
  onClick, 
  icon: Icon, 
  label, 
  count, 
  color 
}: { 
  active: boolean; 
  onClick: () => void; 
  icon: ComponentType<{ className?: string }>; 
  label: string; 
  count: number;
  color: "amber" | "red" | "blue" | "emerald";
}) {
  const colorStyles = {
    amber: active ? "text-amber-400 border-amber-500/50 bg-amber-500/10" : "text-zinc-400 hover:text-amber-300 hover:bg-zinc-800/50",
    red: active ? "text-red-400 border-red-500/50 bg-red-500/10" : "text-zinc-400 hover:text-red-300 hover:bg-zinc-800/50",
    blue: active ? "text-blue-400 border-blue-500/50 bg-blue-500/10" : "text-zinc-400 hover:text-blue-300 hover:bg-zinc-800/50",
    emerald: active ? "text-emerald-400 border-emerald-500/50 bg-emerald-500/10" : "text-zinc-400 hover:text-emerald-300 hover:bg-zinc-800/50",
  };

  return (
    <button
      onClick={onClick}
      className={clsx(
        "flex h-11 shrink-0 items-center gap-2 px-4 py-2.5 text-sm font-medium border-b-2 transition-all justify-center whitespace-nowrap",
        active ? "border-b-2" : "border-transparent",
        colorStyles[color]
      )}
    >
      <Icon className={clsx("w-4 h-4", active && "animate-pulse")} />
      <span>{label}</span>
      {count > 0 && (
        <span className={clsx(
          "ml-1 text-[10px] px-1.5 py-0.5 rounded-full font-bold",
          active ? "bg-white/10" : "bg-zinc-800 text-zinc-500"
        )}>
          {count}
        </span>
      )}
    </button>
  );
}

function IssueCard({
  issue,
  decisionGraph,
  impactView,
  onAdd,
  canAdd = true,
  statusLabel,
  statusTone = "default",
  provenanceLabel,
  provenanceTone = "review",
}: {
  issue: IssueGraphNode;
  decisionGraph?: DecisionGraph | null;
  impactView?: ArchitectureImpactView | null;
  onAdd: () => void;
  canAdd?: boolean;
  statusLabel?: string;
  statusTone?: "default" | "resolved";
  provenanceLabel?: string;
  provenanceTone?: "review" | "lightweight";
}) {
  const displaySeverity = getDisplaySeverity(issue);
  const relatedDecisions = getRelatedDecisionSummaries(issue, impactView, decisionGraph);
  const topRelatedDecision = relatedDecisions[0];
  const severityColors = {
    must_fix: "bg-red-500/10 text-red-400 border-red-500/20",
    needs_attention: "bg-amber-500/10 text-amber-400 border-amber-500/20",
    note: "bg-blue-500/10 text-blue-400 border-blue-500/20",
    unspecified: "bg-zinc-800 text-zinc-400 border-zinc-700",
  };
  const severityLabel = {
    must_fix: "Must fix",
    needs_attention: "Needs attention",
    note: "Note",
    unspecified: "Unspecified",
  } as const;
  const severity = displaySeverity as keyof typeof severityColors;

  return (
    <div className="group relative p-4 rounded-lg bg-zinc-900 border border-zinc-800 hover:border-zinc-700 hover:bg-zinc-800/30 transition-all duration-200">
      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="flex items-center gap-2">
          <span className="font-mono text-xs text-zinc-500 bg-zinc-800/50 px-1.5 py-0.5 rounded">
            {issue.id}
          </span>
          <span className={clsx(
            "text-[10px] px-2 py-0.5 rounded-full uppercase tracking-wider font-semibold border",
            severityColors[severity]
          )}>
            {severityLabel[severity]}
          </span>
          {statusLabel ? (
            <span
              className={clsx(
                "text-[10px] px-2 py-0.5 rounded-full uppercase tracking-wider font-semibold border",
                statusTone === "resolved"
                  ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                  : "bg-zinc-800 text-zinc-400 border-zinc-700",
              )}
            >
              {statusLabel}
            </span>
          ) : null}
          {provenanceLabel ? (
            <span
              className={clsx(
                "text-[10px] px-2 py-0.5 rounded-full uppercase tracking-wider font-semibold border",
                provenanceTone === "lightweight"
                  ? "bg-amber-500/10 text-amber-400 border-amber-500/20"
                  : "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
              )}
            >
              {provenanceLabel}
            </span>
          ) : null}
        </div>
      </div>
      
      <p className="text-sm text-zinc-300 leading-relaxed mb-3">
        {issue.description}
      </p>

      {relatedDecisions.length > 0 ? (
        <div className="mb-3 space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[10px] uppercase tracking-wider text-zinc-500">Related decisions</span>
            {relatedDecisions.map((decision) => (
              <span
                key={decision.decisionId}
                className={clsx(
                  "rounded-full border px-2 py-0.5 text-[10px] font-medium",
                  decision.riskLevel === "high"
                    ? "border-red-500/20 bg-red-500/10 text-red-300"
                    : decision.riskLevel === "medium"
                    ? "border-amber-500/20 bg-amber-500/10 text-amber-300"
                    : "border-zinc-700 bg-zinc-800/70 text-zinc-300",
                )}
              >
                {decision.label}
              </span>
            ))}
          </div>
          {topRelatedDecision?.dominantCluster ? (
            <div className="rounded-md border border-zinc-800/70 bg-zinc-950/40 px-3 py-2 text-xs text-zinc-400">
              <span className="font-medium text-zinc-300">Dominant pressure:</span>{" "}
              {topRelatedDecision.dominantCluster.rootIssue}
              <span className="text-zinc-500"> · Action: {topRelatedDecision.dominantCluster.suggestedAction}</span>
            </div>
          ) : null}
        </div>
      ) : null}

      {canAdd ? (
        <div className="flex items-center justify-end pt-2 border-t border-zinc-800/50 mt-2">
          <button
            onClick={(e) => {
              e.stopPropagation();
              onAdd();
            }}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium text-zinc-400 hover:text-indigo-300 hover:bg-indigo-500/10 transition-colors"
          >
            <MessageSquarePlus className="w-3.5 h-3.5" />
            Add to chat
          </button>
        </div>
      ) : null}
    </div>
  );
}

function getDisplaySeverity(issue: IssueGraphNode): "must_fix" | "needs_attention" | "note" | "unspecified" {
  const raw = (issue.severity || "").toLowerCase();
  const description = issue.description.toLowerCase();
  if (raw === "info") return "note";
  if (raw === "minor") return "needs_attention";
  if (
    raw === "major"
    && /(potential|may lead|may hinder|could make|harder to maintain|consider)/.test(description)
  ) {
    return "needs_attention";
  }
  if (raw === "major") return "must_fix";
  return "unspecified";
}

function getResolvedProvenanceLabel(issue: IssueGraphNode): string | undefined {
  if (issue.status !== "resolved") return undefined;
  return issue.resolution_source === "lightweight" ? "Lightweight edit" : "Review flow";
}

function EmptyState({ icon: Icon, title, description }: { icon: ComponentType<{ className?: string }>; title: string; description: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <div className="w-12 h-12 rounded-full bg-zinc-800/50 flex items-center justify-center mb-3">
        <Icon className="w-6 h-6 text-zinc-600" />
      </div>
      <h4 className="text-sm font-medium text-zinc-300 mb-1">{title}</h4>
      <p className="text-xs text-zinc-500 max-w-[200px]">{description}</p>
    </div>
  );
}
