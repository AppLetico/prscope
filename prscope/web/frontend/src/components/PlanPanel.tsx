import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { FileText, Copy, Check } from "lucide-react";
import { Tooltip } from "./ui/Tooltip";

interface PlanPanelProps {
  content: string;
  isDiffMode?: boolean;
}

export function PlanPanel({ content, isDiffMode = false }: PlanPanelProps) {
  const [copied, setCopied] = useState(false);

  if (!content) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-zinc-500 bg-zinc-900/10">
        <FileText className="w-12 h-12 mb-4 opacity-20" />
        <p className="text-sm">No plan generated yet.</p>
        <p className="text-xs opacity-60 mt-1">Start discovery or provide requirements to begin.</p>
      </div>
    );
  }

  // Very basic diff rendering if in diff mode
  const renderContent = () => {
    if (!isDiffMode) return content;
    
    // Naive diff highlighting for unified diff format
    return content.split('\n').map((line) => {
      if (line.startsWith('+') && !line.startsWith('+++')) {
        return `<span class="bg-emerald-500/20 text-emerald-300 px-1 rounded-sm block">${line}</span>`;
      }
      if (line.startsWith('-') && !line.startsWith('---')) {
        return `<span class="bg-rose-500/20 text-rose-300 line-through px-1 rounded-sm block">${line}</span>`;
      }
      if (line.startsWith('@@')) {
        return `<span class="text-indigo-400 font-mono text-xs block mt-4 mb-1">${line}</span>`;
      }
      return `${line}\n`;
    }).join('');
  };

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
    <div className="h-full overflow-y-auto bg-zinc-900/10 scroll-smooth relative">
      <div className="sticky top-4 z-10 flex justify-end px-6">
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
      <div className="max-w-3xl mx-auto py-12 px-8">
        <article className="prose prose-invert prose-zinc max-w-none">
          {isDiffMode ? (
            <div 
              className="font-mono text-sm whitespace-pre-wrap"
              dangerouslySetInnerHTML={{ __html: renderContent() }} 
            />
          ) : (
            <ReactMarkdown>{content}</ReactMarkdown>
          )}
        </article>
      </div>
    </div>
  );
}
