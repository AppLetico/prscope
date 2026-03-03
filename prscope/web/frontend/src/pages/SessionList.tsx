import { useQuery } from "@tanstack/react-query";
import { Link, useSearchParams } from "react-router-dom";
import { listSessions, setRepoContext, getActiveRepoContext } from "../lib/api";
import { Plus, GitMerge, CircleDashed, FileText } from "lucide-react";
import { Tooltip } from "../components/ui/Tooltip";
import { useMemo, useState, useEffect } from "react";

function formatSessionDate(createdAt: string, updatedAt: string): { label: string; text: string } {
  const created = createdAt ? new Date(createdAt).getTime() : 0;
  const updated = updatedAt ? new Date(updatedAt).getTime() : 0;
  if (!created && !updated) return { label: "", text: "" };
  const ts = Math.max(created, updated);
  const now = Date.now();
  const diffMs = now - ts;
  const diffMins = Math.floor(diffMs / 60_000);
  const diffHours = Math.floor(diffMs / 3600_000);
  const diffDays = Math.floor(diffMs / 86400_000);
  let text: string;
  if (diffMins < 1) text = "Just now";
  else if (diffMins < 60) text = `${diffMins}m ago`;
  else if (diffHours < 24) text = `${diffHours}h ago`;
  else if (diffDays === 1) text = "Yesterday";
  else if (diffDays < 7) text = `${diffDays} days ago`;
  else
    text = new Date(ts).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: ts < now - 365 * 86400_000 ? "numeric" : undefined,
    });
  const label = updated > created ? "Updated" : "Created";
  return { label, text };
}

function StatusBadge({ status }: { status: string }) {
  if (status === "converged" || status === "approved" || status === "exported") {
    return (
      <Tooltip content="Plan has stabilized and is ready for review">
        <div className="flex items-center gap-1.5 cursor-default">
          <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.6)]"></div>
          <span className="text-xs font-medium text-emerald-400 capitalize">{status}</span>
        </div>
      </Tooltip>
    );
  }
  if (status === "refining" || status === "discovery") {
    return (
      <Tooltip content="Actively working on the plan">
        <div className="flex items-center gap-1.5 cursor-default">
          <div className="w-2 h-2 rounded-full bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.6)] animate-pulse"></div>
          <span className="text-xs font-medium text-amber-400 capitalize">{status}</span>
        </div>
      </Tooltip>
    );
  }
  return (
    <div className="flex items-center gap-1.5">
      <div className="w-2 h-2 rounded-full bg-zinc-500"></div>
      <span className="text-xs font-medium text-zinc-400 capitalize">{status}</span>
    </div>
  );
}

export function SessionListPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const urlRepo = searchParams.get("repo");
  const storedRepo = getActiveRepoContext();
  const initialRepo = urlRepo ?? storedRepo ?? null;
  const [selectedRepo, setSelectedRepo] = useState<string | null>(initialRepo);

  useEffect(() => {
    const r = searchParams.get("repo");
    setSelectedRepo(r || null);
  }, [searchParams]);

  const { data, isLoading, error } = useQuery({
    queryKey: ["sessions", null],
    queryFn: () => listSessions(null),
  });

  const sessions = data?.items ?? [];
  const repoOptions = useMemo(() => {
    const repos = [...new Set(sessions.map((s) => s.repo_name))].sort();
    return repos;
  }, [sessions]);

  const displaySessions =
    selectedRepo == null ? sessions : sessions.filter((s) => s.repo_name === selectedRepo);

  const handleRepoChange = (repo: string | null) => {
    setSelectedRepo(repo);
    setRepoContext(repo);
    if (repo) {
      setSearchParams({ repo });
    } else {
      setSearchParams({});
    }
  };

  return (
    <main className="max-w-4xl mx-auto mt-16 px-6 pb-24">
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-4">
          <img src="/logo.svg" alt="prscope" className="w-10 h-10 rounded-xl shadow-sm" />
          <div>
            <h1 className="text-2xl font-semibold tracking-tight text-zinc-100">Plans</h1>
            <p className="text-sm text-zinc-500 mt-1">Manage and review your implementation plans.</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <label className="text-sm text-zinc-500 whitespace-nowrap">Repo</label>
          <select
            value={selectedRepo ?? ""}
            onChange={(e) => handleRepoChange(e.target.value || null)}
            className="bg-zinc-800 border border-zinc-700 text-zinc-200 text-sm rounded-md pl-3 pr-8 py-2 appearance-none cursor-pointer hover:border-zinc-600 focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500"
            style={{ backgroundImage: `url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3e%3cpath stroke='%239ca3af' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='M6 8l4 4 4-4'/%3e%3c/svg%3e")`, backgroundPosition: "right 0.5rem center", backgroundRepeat: "no-repeat", backgroundSize: "1.5em 1.5em" }}
          >
            <option value="">All repos</option>
            {repoOptions.map((repo) => (
              <option key={repo} value={repo}>
                {repo}
              </option>
            ))}
          </select>
          <Link
            to={selectedRepo ? `/new?repo=${encodeURIComponent(selectedRepo)}` : "/new"}
            className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-500 hover:bg-indigo-400 text-white rounded-md text-sm font-medium transition-colors shadow-sm"
          >
            <Plus className="w-4 h-4" />
            New Plan
          </Link>
        </div>
      </div>

      {isLoading ? (
        <div className="flex flex-col items-center justify-center py-24 text-zinc-500">
          <CircleDashed className="w-8 h-8 animate-spin mb-4" />
          <p className="text-sm">Loading sessions...</p>
        </div>
      ) : null}

      {error ? (
        <div className="p-4 bg-rose-500/10 border border-rose-500/20 rounded-lg text-rose-400 text-sm">
          Failed to load sessions: {String(error)}
        </div>
      ) : null}

      {!isLoading && !error && sessions.length === 0 ? (
        <div className="border border-dashed border-zinc-800 rounded-xl py-24 flex flex-col items-center justify-center text-center">
          <div className="w-12 h-12 bg-zinc-900 rounded-full flex items-center justify-center mb-4 border border-zinc-800">
            <FileText className="w-6 h-6 text-zinc-500" />
          </div>
          <h3 className="text-zinc-200 font-medium mb-1">No plans yet</h3>
          <p className="text-zinc-500 text-sm max-w-sm mb-6">
            Start a new planning session to scope out your next feature or refactor.
          </p>
          <Link
            to={selectedRepo ? `/new?repo=${encodeURIComponent(selectedRepo)}` : "/new"}
            className="inline-flex items-center gap-2 px-4 py-2 bg-zinc-100 hover:bg-white text-zinc-900 rounded-md text-sm font-medium transition-colors shadow-sm"
          >
            Create your first plan
          </Link>
        </div>
      ) : null}

      {!isLoading && !error && sessions.length > 0 ? (
        <div className="border border-zinc-800/50 rounded-xl overflow-hidden bg-zinc-900/20">
          {displaySessions.length === 0 ? (
            <div className="py-12 text-center text-zinc-500 text-sm">
              No plans for this repo. Select &quot;All repos&quot; or create a new plan.
            </div>
          ) : (
            displaySessions.map((session) => (
            <Link
              key={session.id}
              to={`/sessions/${session.id}${session.repo_name ? `?repo=${encodeURIComponent(session.repo_name)}` : ""}`}
              className="group flex items-center justify-between py-4 px-5 border-b border-zinc-800/50 last:border-0 hover:bg-zinc-800/40 transition-colors duration-150"
            >
              <div className="flex items-center gap-4">
                <div className="w-8 h-8 rounded-md bg-zinc-800 flex items-center justify-center border border-zinc-700/50 group-hover:border-zinc-600 transition-colors">
                  <GitMerge className="w-4 h-4 text-zinc-400 group-hover:text-zinc-300" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-zinc-100 group-hover:text-white transition-colors">
                      {session.title}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-xs font-mono text-zinc-500">{session.repo_name}</span>
                    <span className="text-zinc-700 text-xs">•</span>
                    <span className="text-xs text-zinc-500">Round {session.current_round}</span>
                    {(session.created_at || session.updated_at) && (() => {
                      const { label, text } = formatSessionDate(session.created_at ?? "", session.updated_at ?? "");
                      if (!text) return null;
                      return (
                        <>
                          <span className="text-zinc-700 text-xs">•</span>
                          <span className="text-xs text-zinc-500" title={label === "Updated" ? "Last updated" : "Created"}>
                            {label} {text}
                          </span>
                        </>
                      );
                    })()}
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-6">
                <StatusBadge status={session.status} />
              </div>
            </Link>
          ))
          )}
        </div>
      ) : null}
    </main>
  );
}
