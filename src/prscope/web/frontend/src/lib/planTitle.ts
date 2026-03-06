const VERSION_SUFFIX = /\s*\(version\s+\d+\)\s*$/i;

export function cleanPlanTitle(title: string): string {
  const baseTitle = title.trim();
  if (!baseTitle) return title;
  return baseTitle.replace(VERSION_SUFFIX, "");
}
