import { useMemo } from "react";
import { useSyncExternalStore } from "react";

function getServerSnapshot(): boolean {
  return false;
}

export function useMediaQuery(query: string): boolean {
  const media = useMemo(
    () => (typeof window !== "undefined" ? window.matchMedia(query) : null),
    [query],
  );
  const subscribe = useMemo(
    () => (onStoreChange: () => void) => {
      if (!media) return () => {};
      media.addEventListener("change", onStoreChange);
      return () => media.removeEventListener("change", onStoreChange);
    },
    [media],
  );
  const getSnapshot = useMemo(() => () => media?.matches ?? false, [media]);
  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
}
