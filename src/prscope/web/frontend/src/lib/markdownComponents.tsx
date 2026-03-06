/**
 * Shared react-markdown component overrides used by both PlanPanel and ChatPanel.
 *
 * react-markdown v10 removed the `inline` prop from the code component.
 * We distinguish inline vs block code using the `className` prop: block code
 * always has a `language-X` class. Our preprocessor adds `text` as the
 * language for untagged fenced blocks so they are still treated as block code.
 */
import type { ReactNode } from "react";
import type { Components } from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";

// Mermaid diagram detection (plan panel only — chat can ignore these)
const MERMAID_PREFIXES = [
  "graph ",
  "flowchart ",
  "sequenceDiagram",
  "classDiagram",
  "stateDiagram",
  "erDiagram",
  "journey",
  "gantt",
  "pie ",
  "mindmap",
  "timeline",
];

function looksLikeMermaidBlock(raw: string): boolean {
  const trimmed = raw.trimStart();
  const firstLine = (trimmed.split("\n")[0] ?? "").trim();
  return MERMAID_PREFIXES.some((prefix) => firstLine.startsWith(prefix));
}

interface BuildOptions {
  /** "plan" wraps pre in a styled border box. "chat" uses a lighter container. */
  variant?: "plan" | "chat";
  /** Enable mermaid diagram detection (plan panel only). */
  mermaid?: boolean;
}

export function buildMarkdownComponents({ variant = "chat", mermaid: enableMermaid = false }: BuildOptions = {}): Components {
  return {
    pre: ({ children }: { children?: ReactNode }) => {
      if (variant === "plan") {
        return (
          <div className="overflow-x-auto rounded-lg bg-zinc-950/50 border border-zinc-800 my-4">
            {children}
          </div>
        );
      }
      // Chat variant: let .chat-message-prose pre CSS handle the container
      return (
        <div className="overflow-x-auto rounded-lg bg-zinc-900/80 border border-zinc-800/60 my-3">
          {children}
        </div>
      );
    },

    // Cast needed: react-markdown's Components["code"] has a strict intersection
    // type that doesn't allow an index signature, so we cast via unknown.
    code: (({ className, children }: { className?: string; children?: ReactNode }) => {
      const match = /language-(\w+)/.exec(className || "");
      const codeString = String(children).replace(/\n$/, "");
      const lang = match?.[1];
      const isMermaidBlock =
        enableMermaid && (lang === "mermaid" || looksLikeMermaidBlock(codeString));

      // No language class → inline code span
      if (!match) {
        return (
          <code className="rounded bg-zinc-800/60 px-1.5 py-0.5 text-[13px] text-indigo-200 font-mono">
            {children}
          </code>
        );
      }

      if (isMermaidBlock) {
        return (
          <div className="mermaid flex justify-center py-4">
            {codeString}
          </div>
        );
      }

      const fontSize = variant === "plan" ? "13px" : "12.5px";

      return (
        <SyntaxHighlighter
          style={vscDarkPlus}
          language={lang === "text" ? undefined : lang}
          PreTag="pre"
          showLineNumbers={false}
          customStyle={{
            margin: 0,
            padding: "0.875rem 1rem",
            background: "transparent",
            fontSize,
            lineHeight: "1.6",
          }}
        >
          {codeString}
        </SyntaxHighlighter>
      );
    }) as Components["code"],
  };
}

/** Pre-built components for PlanPanel (styled border box, mermaid enabled). */
export const planMarkdownComponents = buildMarkdownComponents({
  variant: "plan",
  mermaid: true,
});

/** Pre-built components for ChatPanel (lighter container, no mermaid). */
export const chatMarkdownComponents = buildMarkdownComponents({
  variant: "chat",
  mermaid: false,
});
