import { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { FileText, Copy, Check, FileDiff, Download } from "lucide-react";
import { clsx } from "clsx";
import mermaid from "mermaid";
import { Tooltip } from "./ui/Tooltip";
import { preprocessPlanMarkdown } from "../lib/markdown";
import { planMarkdownComponents } from "../lib/markdownComponents";
import type { SessionStatus } from "../types";

interface PlanPanelProps {
  content: string;
  isDiffMode?: boolean;
  status?: SessionStatus;
  canExport?: boolean;
  onToggleDiff?: () => void;
  onExport?: () => void;
}

export function PlanPanel({
  content,
  isDiffMode = false,
  status,
  canExport: _canExport,
  onToggleDiff: _onToggleDiff,
  onExport: _onExport,
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
    if (!isDiffMode && content && mermaidRef.current) {
      void mermaid.run({
        nodes: mermaidRef.current.querySelectorAll(".mermaid"),
        suppressErrors: true,
      });
    }
  }, [content, isDiffMode]);

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

  // Naive diff highlighting for unified diff format
  const renderDiff = () =>
    content
      .split("\n")
      .map((line) => {
        if (line.startsWith("+") && !line.startsWith("+++"))
          return `<span class="bg-emerald-500/20 text-emerald-300 px-1 rounded-sm block">${line}</span>`;
        if (line.startsWith("-") && !line.startsWith("---"))
          return `<span class="bg-rose-500/20 text-rose-300 line-through px-1 rounded-sm block">${line}</span>`;
        if (line.startsWith("@@"))
          return `<span class="text-indigo-400 font-mono text-xs block mt-4 mb-1">${line}</span>`;
        return `${line}\n`;
      })
      .join("");

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
          {_onToggleDiff && (
            <Tooltip content="Toggle diff view">
              <button
                type="button"
                onClick={_onToggleDiff}
                className={clsx(
                  "p-1.5 rounded-md transition-colors border border-zinc-800/80 bg-zinc-900/80 backdrop-blur",
                  isDiffMode 
                    ? "text-indigo-400 bg-indigo-500/10 border-indigo-500/30" 
                    : "text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800/70"
                )}
                aria-label="Toggle diff view"
              >
                <FileDiff className="w-4 h-4" />
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
          <div className="max-w-3xl mx-auto pt-0 py-12 px-4 md:px-8" ref={mermaidRef}>
            <article className="prose prose-zinc max-w-none">
              {isDiffMode ? (
                <div
                  className="font-mono text-sm whitespace-pre-wrap"
                  dangerouslySetInnerHTML={{ __html: renderDiff() }}
                />
              ) : (
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={planMarkdownComponents}
                >
                  {preprocessPlanMarkdown(content)}
                </ReactMarkdown>
              )}
            </article>
          </div>
        </div>
      </div>
    </div>
  );
}
