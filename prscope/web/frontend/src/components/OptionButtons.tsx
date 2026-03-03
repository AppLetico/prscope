import type { DiscoveryQuestion } from "../types";
import { HelpCircle } from "lucide-react";

interface OptionButtonsProps {
  questions: DiscoveryQuestion[];
  selectedAnswers: Record<number, string>;
  selectedIsOther: Record<number, boolean>;
  otherInputs: Record<number, string>;
  onSelect: (questionIndex: number, optionText: string, isOther: boolean) => void;
  onOtherInputChange: (questionIndex: number, value: string) => void;
}

export function OptionButtons({
  questions,
  selectedAnswers,
  selectedIsOther,
  otherInputs,
  onSelect,
  onOtherInputChange,
}: OptionButtonsProps) {
  const toDisplayText = (value: string): string => value
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/__([^_]+)__/g, "$1")
    .replace(/\*([^*]+)\*/g, "$1")
    .replace(/_([^_]+)_/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/\s+/g, " ")
    .trim();

  if (questions.length === 0) return null;

  return (
    <div className="mt-6 space-y-6">
      {questions.map((q) => (
        <div key={q.index} className="animate-in slide-in-from-bottom-2 fade-in duration-300 ease-out">
          <div className="flex gap-3 mb-3">
            <div className="mt-0.5">
              <div className="w-6 h-6 rounded-full bg-indigo-500/10 flex items-center justify-center border border-indigo-500/20">
                <HelpCircle className="w-3.5 h-3.5 text-indigo-400" />
              </div>
            </div>
            <p className="text-sm font-medium text-zinc-200 leading-relaxed">
              {toDisplayText(q.text)}
            </p>
          </div>
          
          <div className="flex flex-col gap-2 pl-9">
            {q.options.map((opt) => (
              <button
                key={`${q.index}-${opt.letter}`}
                className={`w-full text-left p-3 rounded-lg border transition-all group shadow-sm ${
                  selectedAnswers[q.index] === opt.text
                    ? "border-indigo-500/50 bg-indigo-500/10"
                    : "border-zinc-800 bg-zinc-900/50 hover:bg-zinc-800 hover:border-zinc-700"
                }`}
                type="button"
                onClick={() => onSelect(q.index, opt.text, opt.is_other)}
              >
                <div className="flex items-start">
                  <span
                    className={`inline-flex items-center justify-center w-6 h-6 rounded mr-3 text-xs font-mono shrink-0 transition-colors ${
                      selectedAnswers[q.index] === opt.text
                        ? "bg-indigo-500/30 text-indigo-200"
                        : "bg-zinc-800 text-zinc-400 group-hover:text-zinc-200 group-hover:bg-zinc-700"
                    }`}
                  >
                    {opt.letter}
                  </span>
                  <span
                    className={`text-sm transition-colors mt-0.5 ${
                      selectedAnswers[q.index] === opt.text
                        ? "text-indigo-100"
                        : "text-zinc-300 group-hover:text-zinc-100"
                    }`}
                  >
                    {opt.is_other ? "Other (Type your answer below...)" : toDisplayText(opt.text)}
                  </span>
                </div>
              </button>
            ))}
            {selectedIsOther[q.index] && (
              <input
                type="text"
                value={otherInputs[q.index] ?? ""}
                onChange={(event) => onOtherInputChange(q.index, event.target.value)}
                placeholder="Type your answer..."
                className="w-full rounded-lg border border-indigo-500/30 bg-zinc-900/70 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:ring-1 focus:ring-indigo-400"
              />
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
