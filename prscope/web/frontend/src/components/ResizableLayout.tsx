import { Group, Panel, Separator } from "react-resizable-panels";
import { useMemo } from "react";
import type { ReactNode } from "react";

interface ResizableLayoutProps {
  left: ReactNode;
  right: ReactNode;
}

const STORAGE_KEY = "planning_layout_v1";

export function ResizableLayout({ left, right }: ResizableLayoutProps) {
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
      orientation="horizontal"
      id={STORAGE_KEY}
      defaultLayout={defaultLayout}
      onLayoutChange={(next) => {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      }}
      className="h-full w-full"
    >
      <Panel id="left" defaultSize={60} minSize={30} className="h-full">
        {left}
      </Panel>
      
      <Separator className="w-1 bg-transparent hover:bg-indigo-500/50 active:bg-indigo-500 transition-colors duration-150 cursor-col-resize z-10 relative border-l border-zinc-800/50" />
      
      <Panel id="right" defaultSize={40} minSize={25} className="h-full">
        {right}
      </Panel>
    </Group>
  );
}
