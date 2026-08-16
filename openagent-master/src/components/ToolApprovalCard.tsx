import React from "react";
import { Check, X, ShieldAlert, FileCode, Terminal, AlertTriangle } from "lucide-react";

interface ToolApprovalCardProps {
  toolName: string;
  args: Record<string, any>;
  riskLevel?: "read_only" | "mutation_safe" | "mutation_dangerous";
  reason?: string;
  onApprove: () => void;
  onReject: (reason?: string) => void;
}

export function ToolApprovalCard({
  toolName,
  args,
  riskLevel = "mutation_safe",
  reason,
  onApprove,
  onReject,
}: ToolApprovalCardProps) {
  const isDangerous = riskLevel === "mutation_dangerous";

  return (
    <div
      className={`my-3 p-4 rounded-xl border font-mono text-xs ${
        isDangerous
          ? "bg-red-950/40 border-red-500/50 shadow-[0_0_15px_rgba(255,51,102,0.2)]"
          : "bg-slate-900/90 border-cyan-500/40 shadow-[0_0_15px_rgba(0,240,255,0.15)]"
      }`}
    >
      <div className="flex items-center justify-between border-b border-slate-800 pb-2 mb-3">
        <div className="flex items-center gap-2">
          {isDangerous ? (
            <ShieldAlert className="size-4 text-red-400" />
          ) : (
            <FileCode className="size-4 text-cyan-400" />
          )}
          <span className="font-bold text-white uppercase tracking-wider">
            Tool Approval Required: {toolName}
          </span>
        </div>
        <span
          className={`px-2 py-0.5 rounded text-[10px] font-bold ${
            isDangerous
              ? "bg-red-500/20 text-red-300 border border-red-500/40"
              : "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
          }`}
        >
          {riskLevel.toUpperCase()}
        </span>
      </div>

      {reason && (
        <div className="text-[11px] text-amber-300 mb-2 flex items-center gap-1.5">
          <AlertTriangle className="size-3.5 shrink-0" />
          <span>{reason}</span>
        </div>
      )}

      {/* Target Args & Code Snippet */}
      <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800 mb-3 space-y-1 text-slate-300 overflow-x-auto text-[11px]">
        {args.filePath && <div><strong className="text-slate-500">File:</strong> {args.filePath}</div>}
        {args.command && <div><strong className="text-slate-500">Command:</strong> <code className="text-cyan-300">{args.command}</code></div>}
        {args.target && (
          <div className="mt-1">
            <span className="text-red-400">- {args.target}</span>
            <br />
            <span className="text-emerald-400">+ {args.replacement}</span>
          </div>
        )}
      </div>

      {/* Action Buttons */}
      <div className="flex justify-end gap-2 pt-1">
        <button
          onClick={() => onReject("User rejected tool execution")}
          className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white flex items-center gap-1 font-bold text-xs"
        >
          <X className="size-3.5" /> Reject
        </button>
        <button
          onClick={onApprove}
          className="px-3 py-1.5 rounded-lg bg-cyan-400 hover:bg-cyan-300 text-slate-950 font-black flex items-center gap-1 text-xs shadow-[0_0_10px_rgba(0,240,255,0.4)]"
        >
          <Check className="size-3.5 stroke-[3]" /> Approve & Execute
        </button>
      </div>
    </div>
  );
}
