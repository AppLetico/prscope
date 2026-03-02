import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { listSessions } from "../lib/api";
import { Plus, GitMerge, CircleDashed, FileText } from "lucide-react";
import { Tooltip } from "../components/ui/Tooltip";

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
  const { data, isLoading, error } = useQuery({
    queryKey: ["sessions"],
    queryFn: listSessions,
  });

  const sessions = data?.items ?? [];

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
        <Link
          to="/new"
          className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-500 hover:bg-indigo-400 text-white rounded-md text-sm font-medium transition-colors shadow-sm"
        >
          <Plus className="w-4 h-4" />
          New Plan
        </Link>
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
            to="/new"
            className="inline-flex items-center gap-2 px-4 py-2 bg-zinc-100 hover:bg-white text-zinc-900 rounded-md text-sm font-medium transition-colors shadow-sm"
          >
            Create your first plan
          </Link>
        </div>
      ) : null}

      {!isLoading && !error && sessions.length > 0 ? (
        <div className="border border-zinc-800/50 rounded-xl overflow-hidden bg-zinc-900/20">
          {sessions.map((session) => (
            <Link
              key={session.id}
              to={`/sessions/${session.id}`}
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
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-6">
                <StatusBadge status={session.status} />
              </div>
            </Link>
          ))}
        </div>
      ) : null}
    </main>
  );
}
