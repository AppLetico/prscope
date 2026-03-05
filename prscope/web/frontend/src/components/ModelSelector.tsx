import { useState, useRef, useEffect } from "react";
import { ChevronDown, Check } from "lucide-react";
import { clsx } from "clsx";
import { Tooltip } from "./ui/Tooltip";

interface ModelOption {
  model_id: string;
  provider: string;
  available: boolean;
  unavailable_reason?: string | null;
}

interface ModelSelectorProps {
  value: string;
  onChange: (modelId: string) => void;
  options: ModelOption[];
  label: string;
  icon?: React.ReactNode;
  className?: string;
  dropUp?: boolean;
}

export function ModelSelector({ value, onChange, options, label, icon, className, dropUp = false }: ModelSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const selectedOption = options.find((o) => o.model_id === value);
  
  // Group by provider
  const groupedOptions = options.reduce((acc, option) => {
    if (!acc[option.provider]) {
      acc[option.provider] = [];
    }
    acc[option.provider].push(option);
    return acc;
  }, {} as Record<string, ModelOption[]>);

  return (
    <div className="relative min-w-0 shrink" ref={containerRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={clsx(
          "flex items-center gap-2 h-8 px-2.5 rounded-md transition-all duration-200",
          "text-xs font-medium focus:outline-none min-w-0",
          isOpen 
            ? "bg-zinc-800 text-zinc-100 shadow-inner" 
            : "text-zinc-400 hover:text-zinc-200",
          className
        )}
        title={`${label} model`}
      >
        {icon && <span className={clsx("shrink-0", isOpen ? "text-indigo-400" : "text-zinc-500")}>{icon}</span>}
        <div className="flex flex-col items-start leading-none text-left min-w-0">
          <span className="text-[8px] text-zinc-500 uppercase tracking-wider font-semibold mb-0.5">{label}</span>
          <span className="truncate max-w-[80px] sm:max-w-[110px] text-[10px] font-mono">
            {selectedOption ? selectedOption.model_id : value || "Select model"}
          </span>
        </div>
        <ChevronDown className={clsx("w-3 h-3 text-zinc-500 transition-transform duration-200 ml-0.5 shrink-0", isOpen && "rotate-180")} />
      </button>

      {isOpen && (
        <div className={clsx(
          "absolute left-0 w-72 max-h-[400px] overflow-y-auto bg-zinc-900/95 backdrop-blur-xl border border-zinc-800 rounded-lg shadow-2xl shadow-black/50 z-50 py-1.5 animate-in fade-in zoom-in-95 duration-100",
          dropUp
            ? "bottom-full mb-2 origin-bottom-left"
            : "top-full mt-2 origin-top-right"
        )}>
          {Object.entries(groupedOptions).map(([provider, providerOptions]) => (
            <div key={provider} className="mb-2 last:mb-0">
              <div className="px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider text-zinc-500 bg-zinc-900/90 sticky top-0 backdrop-blur-md z-10 border-b border-zinc-800">
                {provider}
              </div>
              <div className="flex flex-col py-1">
                {providerOptions.map((option) => {
                  const isSelected = option.model_id === value;
                  
                  const buttonContent = (
                    <button
                      disabled={!option.available}
                      onClick={() => {
                        onChange(option.model_id);
                        setIsOpen(false);
                      }}
                      className={clsx(
                        "flex items-center justify-between w-full px-3 py-2 text-xs text-left transition-colors",
                        !option.available 
                          ? "opacity-50 cursor-not-allowed text-zinc-500" 
                          : isSelected
                            ? "bg-indigo-500/10 text-indigo-300"
                            : "text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100"
                      )}
                    >
                      <span className="truncate pr-4">{option.model_id}</span>
                      {isSelected && <Check className="w-3.5 h-3.5 text-indigo-400 shrink-0" />}
                    </button>
                  );

                  if (!option.available && option.unavailable_reason) {
                    return (
                      <Tooltip 
                        key={option.model_id} 
                        content={option.unavailable_reason}
                        side="left"
                      >
                        <div className="w-full">
                          {buttonContent}
                        </div>
                      </Tooltip>
                    );
                  }

                  return <div key={option.model_id}>{buttonContent}</div>;
                })}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
