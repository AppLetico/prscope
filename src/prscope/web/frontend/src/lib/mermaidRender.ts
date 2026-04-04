/**
 * Lazy-loaded Mermaid and one-shot diagram rendering for the plan panel.
 * securityLevel "loose" is intentional: plan markdown is trusted local/session content.
 */
let mermaidPromise: Promise<typeof import("mermaid").default> | null = null;

function loadMermaid(): Promise<typeof import("mermaid").default> {
  if (!mermaidPromise) {
    mermaidPromise = import("mermaid").then((m) => m.default);
  }
  return mermaidPromise;
}

let initialized = false;

export async function initMermaid(): Promise<void> {
  const mermaid = await loadMermaid();
  if (initialized) return;
  mermaid.initialize({
    startOnLoad: false,
    theme: "dark",
    securityLevel: "loose",
    fontFamily: "inherit",
  });
  initialized = true;
}

/**
 * Renders Mermaid source into a container element. Clears previous SVG/state first.
 */
export async function runMermaidDiagram(el: HTMLElement, source: string): Promise<void> {
  await initMermaid();
  const mermaid = await loadMermaid();
  el.removeAttribute("data-processed");
  el.innerHTML = "";
  el.textContent = source;
  await mermaid.run({ nodes: [el], suppressErrors: false });
}
