import type { DiscoveryQuestion } from "../types";
import { HelpCircle } from "lucide-react";

interface OptionButtonsProps {
  questions: DiscoveryQuestion[];
  onSelect: (questionIndex: number, optionText: string, isOther: boolean) => void;
}

export function OptionButtons({ questions, onSelect }: OptionButtonsProps) {
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
              {q.text}
            </p>
          </div>
          
          <div className="flex flex-col gap-2 pl-9">
            {q.options.map((opt) => (
              <button
                key={`${q.index}-${opt.letter}`}
                className="w-full text-left p-3 rounded-lg border border-zinc-800 bg-zinc-900/50 hover:bg-zinc-800 hover:border-zinc-700 transition-all group shadow-sm"
                type="button"
                onClick={() => onSelect(q.index, opt.text, opt.is_other)}
              >
                <div className="flex items-start">
                  <span className="inline-flex items-center justify-center w-6 h-6 rounded bg-zinc-800 text-zinc-400 group-hover:text-zinc-200 group-hover:bg-zinc-700 mr-3 text-xs font-mono shrink-0 transition-colors">
                    {opt.letter}
                  </span>
                  <span className="text-sm text-zinc-300 group-hover:text-zinc-100 transition-colors mt-0.5">
                    {opt.is_other ? "Other (Type your answer below...)" : opt.text}
                  </span>
                </div>
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
