import { Group, Panel, Separator } from "react-resizable-panels";
import { useMemo } from "react";
import type { ReactNode } from "react";
import { useMediaQuery } from "../hooks/useMediaQuery";

interface ResizableLayoutProps {
  left: ReactNode;
  right: ReactNode;
}

const STORAGE_KEY = "planning_layout_v1";

export function ResizableLayout({ left, right }: ResizableLayoutProps) {
  const isMobile = useMediaQuery("(max-width: 768px)");

  const defaultLayout = useMemo(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return undefined;
      const parsed = JSON.parse(raw) as Record<string, number>;
      if (typeof parsed.left === "number" && typeof parsed.right === "number") {
        return { left: parsed.left, right: parsed.right };
      }
    } catch {
      // Ignore broken layout cache
    }
    return undefined;
  }, []);

  return (
    <Group
      orientation={isMobile ? "vertical" : "horizontal"}
      id={STORAGE_KEY}
      defaultLayout={defaultLayout}
      onLayoutChange={(next) => {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      }}
      className="h-full w-full"
    >
      <Panel id="left" defaultSize={isMobile ? 50 : 60} minSize={20} className="h-full">
        {left}
      </Panel>
      
      <Separator className={`bg-transparent hover:bg-indigo-500/50 active:bg-indigo-500 transition-colors duration-150 z-10 relative ${isMobile ? "h-1 cursor-row-resize border-t border-zinc-800" : "w-1 cursor-col-resize border-l border-zinc-800"}`} />
      
      <Panel id="right" defaultSize={isMobile ? 50 : 40} minSize={20} className="h-full">
        {right}
      </Panel>
    </Group>
  );
}
