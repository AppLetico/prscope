import { useState, useRef, useEffect } from "react";
import { useNavigate, Link } from "react-router-dom";
import { createChatSession, createRequirementsSession, getActiveRepoContext } from "../lib/api";
import { ArrowLeft, MessageSquare, FileText, Loader2 } from "lucide-react";
import { clsx } from "clsx";

export function NewSessionPage() {
  const navigate = useNavigate();
  const activeRepo = getActiveRepoContext();
  const [mode, setMode] = useState<"chat" | "requirements">("chat");
  const [requirements, setRequirements] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (mode === "requirements" && inputRef.current) {
      inputRef.current.focus();
    }
  }, [mode]);

  const submit = async () => {
    if (mode === "requirements" && !requirements.trim()) return;
    
    try {
      setError(null);
      setSubmitting(true);
      if (mode === "chat") {
        const result = await createChatSession();
        navigate(`/sessions/${result.session.id}`);
        return;
      }
      const result = await createRequirementsSession(requirements.trim());
      navigate(`/sessions/${result.session.id}`);
    } catch (err) {
      setError(String(err));
    } finally {
      setSubmitting(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      void submit();
    }
  };

  return (
    <main className="max-w-2xl mx-auto mt-16 px-6 pb-24">
      <div className="mb-8">
        <Link to="/" className="inline-flex items-center text-sm text-zinc-500 hover:text-zinc-300 transition-colors mb-6">
          <ArrowLeft className="w-4 h-4 mr-1" />
          Back to sessions
        </Link>
        <div className="flex items-center gap-4">
          <img src="/logo.svg" alt="prscope" className="w-10 h-10 rounded-xl shadow-sm" />
          <div>
            <h1 className="text-3xl font-semibold tracking-tight text-zinc-100">New Plan</h1>
            <p className="text-zinc-500 mt-2">Start a new planning session to scope out your next feature.</p>
            {!activeRepo ? (
              <p className="text-zinc-600 text-sm mt-2">
                No explicit repo selected in the UI. Backend auto-detection is enabled; use `?repo=&lt;name&gt;` only to pin a repo.
              </p>
            ) : (
              <p className="text-zinc-600 text-sm mt-2">
                Repo: <span className="font-mono">{activeRepo}</span>
              </p>
            )}
          </div>
        </div>
      </div>

      <div className="space-y-8">
        {/* Mode Picker - Vercel Style Segmented Control */}
        <div>
          <div className="inline-flex p-1 bg-zinc-900/80 rounded-lg border border-zinc-800/80">
            <button
              onClick={() => setMode("chat")}
              className={clsx(
                "flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md transition-all duration-200",
                mode === "chat"
                  ? "bg-zinc-800 text-zinc-100 shadow-sm"
                  : "text-zinc-400 hover:text-zinc-200"
              )}
            >
              <MessageSquare className="w-4 h-4" />
              Chat Discovery
            </button>
            <button
              onClick={() => setMode("requirements")}
              className={clsx(
                "flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md transition-all duration-200",
                mode === "requirements"
                  ? "bg-zinc-800 text-zinc-100 shadow-sm"
                  : "text-zinc-400 hover:text-zinc-200"
              )}
            >
              <FileText className="w-4 h-4" />
              From Requirements
            </button>
          </div>
        </div>

        <div className="bg-zinc-900/30 border border-zinc-800/50 rounded-xl p-6 shadow-sm">
          {mode === "chat" ? (
            <div className="py-8 text-center flex flex-col items-center">
              <div className="w-12 h-12 bg-indigo-500/10 rounded-full flex items-center justify-center mb-4">
                <MessageSquare className="w-6 h-6 text-indigo-400" />
              </div>
              <h3 className="text-zinc-200 font-medium mb-2">Interactive Discovery</h3>
              <p className="text-zinc-500 text-sm max-w-sm mx-auto">
                Start with a blank slate. The AI will scan your codebase and ask targeted questions to help you define the requirements.
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="flex justify-between items-end">
                <label className="block text-sm font-medium text-zinc-300">
                  Initial Requirements
                </label>
                <span className="text-xs text-zinc-600 font-mono">Markdown supported</span>
              </div>
              <textarea
                ref={inputRef}
                className="w-full min-h-[200px] p-4 rounded-lg border border-zinc-800 bg-zinc-950 text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-transparent transition-all resize-y font-mono text-sm leading-relaxed"
                placeholder="Describe the feature, bug fix, or refactor you want to plan..."
                value={requirements}
                onChange={(e) => setRequirements(e.target.value)}
                onKeyDown={handleKeyDown}
              />
            </div>
          )}

          {error ? (
            <div className="mt-6 p-4 bg-rose-500/10 border border-rose-500/20 rounded-lg text-rose-400 text-sm">
              {error}
            </div>
          ) : null}

          <div className="mt-8 flex items-center justify-between border-t border-zinc-800/50 pt-6">
            <div className="text-xs text-zinc-500 flex items-center gap-1">
              {mode === "requirements" && (
                <>Press <kbd className="px-1.5 py-0.5 rounded bg-zinc-800 border border-zinc-700 font-mono text-[10px]">⌘</kbd> <kbd className="px-1.5 py-0.5 rounded bg-zinc-800 border border-zinc-700 font-mono text-[10px]">↵</kbd> to start</>
              )}
            </div>
            <button
              type="button"
              disabled={submitting || (mode === "requirements" && !requirements.trim())}
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-indigo-500 hover:bg-indigo-400 text-white rounded-md text-sm font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-sm active:scale-95"
              onClick={() => void submit()}
            >
              {submitting ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Starting...
                </>
              ) : (
                "Start Planning"
              )}
            </button>
          </div>
        </div>
      </div>
    </main>
  );
}
