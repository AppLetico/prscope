import { useEffect, useRef, useState } from "react";
import { runMermaidDiagram } from "./mermaidRender";

const DEBOUNCE_MS = 150;

interface MermaidBlockProps {
  codeString: string;
}

/**
 * Per-block Mermaid render with debounced run, error fallback, and raw source recovery.
 */
export function MermaidBlock({ codeString }: MermaidBlockProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    const timer = window.setTimeout(() => {
      const el = ref.current;
      if (!el) return;
      void (async () => {
        try {
          await runMermaidDiagram(el, codeString);
          if (!cancelled) setError(null);
        } catch (e) {
          if (!cancelled) {
            setError(e instanceof Error ? e.message : String(e));
          }
        }
      })();
    }, DEBOUNCE_MS);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [codeString]);

  return (
    <div className="my-4 space-y-2">
      {error ? (
        <div
          className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 text-sm"
          role="alert"
        >
          <p className="text-amber-200 font-medium">Diagram could not render</p>
          <p className="text-amber-200/70 text-xs mt-1">{error}</p>
          <details className="mt-2 text-xs text-zinc-400">
            <summary className="cursor-pointer">Mermaid source</summary>
            <pre className="mt-2 overflow-x-auto rounded bg-zinc-950/80 p-2 text-zinc-300 whitespace-pre-wrap">
              {codeString}
            </pre>
          </details>
        </div>
      ) : null}
      {/* Keep mounted so ref survives failed renders and retries when content updates */}
      <div
        ref={ref}
        className={error ? "hidden" : "mermaid flex justify-center py-4"}
        aria-hidden={error ? true : undefined}
        aria-label={error ? undefined : "Diagram"}
      />
    </div>
  );
}
