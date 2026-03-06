import { useState, useRef, useEffect, type ComponentType } from "react";
import { AlertCircle, ShieldAlert, Target, X, MessageSquarePlus, CheckCircle2, Info } from "lucide-react";
import { clsx } from "clsx";
import type { IssueGraphNode } from "../types";

interface IssuePanelProps {
  openIssues: IssueGraphNode[];
  constraintViolations: string[];
  rootCausesCount: number;
  onAppendIssue: (issue: IssueGraphNode) => void;
  onAppendAllIssues: () => void;
  onClose: () => void;
  className?: string;
  initialTab?: Tab;
}

type Tab = "issues" | "violations" | "root-causes";

export function IssuePanel({
  openIssues,
  constraintViolations,
  rootCausesCount,
  onAppendIssue,
  onAppendAllIssues,
  onClose,
  className,
  initialTab = "issues",
}: IssuePanelProps) {
  const [activeTab, setActiveTab] = useState<Tab>(initialTab);
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setActiveTab(initialTab);
  }, [initialTab]);

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
          System Health
        </h3>
        <button 
          onClick={onClose}
          className="text-zinc-500 hover:text-zinc-300 transition-colors p-1 hover:bg-zinc-800 rounded-md"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Tabs */}
      <div className="flex items-center px-2 pt-2 border-b border-zinc-800/50 bg-zinc-900/50">
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
          active={activeTab === "root-causes"} 
          onClick={() => setActiveTab("root-causes")}
          icon={Target}
          label="Root Causes"
          count={rootCausesCount}
          color="blue"
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
                    {openIssues.length} Active Issues
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
                      onAdd={() => onAppendIssue(issue)} 
                    />
                  ))}
                </div>
              </>
            ) : (
              <EmptyState 
                icon={CheckCircle2} 
                title="All Clear" 
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
                    {constraintViolations.length} Active Violations
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
                title="No Violations" 
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
                      <h4 className="text-sm font-medium text-blue-400 mb-1">Root Cause Analysis</h4>
                      <p className="text-xs text-blue-300/80 leading-relaxed">
                        Focusing on these issues often resolves downstream dependencies automatically.
                      </p>
                    </div>
                  </div>
                </div>
                {/* 
                  Since we don't have the graph structure passed down to filter exactly which are root causes,
                  we'll show a helpful message. In a real implementation, we'd filter `openIssues` by `causesParents.length === 0`.
                */}
                <div className="text-center py-8">
                  <Target className="w-12 h-12 text-zinc-700 mx-auto mb-3" />
                  <p className="text-sm text-zinc-400">
                    {rootCausesCount} root cause{rootCausesCount !== 1 ? 's' : ''} identified.
                  </p>
                  <p className="text-xs text-zinc-600 mt-1 max-w-[200px] mx-auto">
                    Check the Issue Graph visualization to identify and target these specific nodes.
                  </p>
                </div>
              </>
            ) : (
              <EmptyState 
                icon={CheckCircle2} 
                title="No Root Causes" 
                description="No isolated root causes found." 
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
  color: "amber" | "red" | "blue";
}) {
  const colorStyles = {
    amber: active ? "text-amber-400 border-amber-500/50 bg-amber-500/10" : "text-zinc-400 hover:text-amber-300 hover:bg-zinc-800/50",
    red: active ? "text-red-400 border-red-500/50 bg-red-500/10" : "text-zinc-400 hover:text-red-300 hover:bg-zinc-800/50",
    blue: active ? "text-blue-400 border-blue-500/50 bg-blue-500/10" : "text-zinc-400 hover:text-blue-300 hover:bg-zinc-800/50",
  };

  return (
    <button
      onClick={onClick}
      className={clsx(
        "flex items-center gap-2 px-4 py-2.5 text-sm font-medium border-b-2 transition-all flex-1 justify-center",
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

function IssueCard({ issue, onAdd }: { issue: IssueGraphNode; onAdd: () => void }) {
  const severityColors = {
    major: "bg-red-500/10 text-red-400 border-red-500/20",
    minor: "bg-amber-500/10 text-amber-400 border-amber-500/20",
    info: "bg-blue-500/10 text-blue-400 border-blue-500/20",
    unspecified: "bg-zinc-800 text-zinc-400 border-zinc-700",
  };

  const severity = (issue.severity || "unspecified").toLowerCase() as keyof typeof severityColors;

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
            {issue.severity || "Unspecified"}
          </span>
        </div>
      </div>
      
      <p className="text-sm text-zinc-300 leading-relaxed mb-3">
        {issue.description}
      </p>

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
    </div>
  );
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
