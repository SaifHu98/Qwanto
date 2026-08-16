import React, { useState } from "react"
import { 
  Folder, Shield, Terminal, Play, CheckCircle, XCircle, 
  FileCode, Layers, Zap, AlertTriangle, RefreshCw, Send, Check
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

export interface StepItem {
  id: string
  type: "plan" | "tool_call" | "tool_result" | "user" | "assistant"
  title: string
  content: string
  toolName?: string
  toolArgs?: any
  status?: "pending" | "approved" | "rejected" | "executed"
  diff?: string
}

export function CodingAgentView() {
  const [workspaceRoot, setWorkspaceRoot] = useState<string>("D:/EcoUni/qwanto")
  const [mode, setMode] = useState<"plan" | "agent">("plan")
  const [activeModel, setActiveModel] = useState<string>("DeepSeek-V4-Pro-4B-twla.qwn")
  const [prompt, setPrompt] = useState<string>("")
  const [steps, setSteps] = useState<StepItem[]>([
    {
      id: "step-1",
      type: "user",
      title: "User Prompt",
      content: "Refactor error handling in `c/openai_server.py` to prevent stack trace leaks and redact secrets in local-only mode.",
    },
    {
      id: "step-2",
      type: "plan",
      title: "🛡️ Execution Plan Formulated (Plan Mode)",
      content: "1. Inspect `c/openai_server.py` error handling\n2. Add `redact_secrets` filter on all client responses\n3. Run test suite `python -m pytest c/tests/ -q` to verify zero regressions.",
    },
    {
      id: "step-3",
      type: "tool_call",
      title: "Proposed Tool Action: edit_file",
      toolName: "edit_file",
      toolArgs: { path: "c/openai_server.py", old_str: "traceback.format_exc()", new_str: "'[REDACTED]' " },
      status: "pending",
      diff: "- traceback.format_exc()\n+ '[REDACTED]'",
      content: "Target: `c/openai_server.py`",
    }
  ])

  const handleApprove = (id: string) => {
    setSteps(prev => prev.map(s => s.id === id ? { ...s, status: "approved" } : s))
  }

  const handleReject = (id: string) => {
    setSteps(prev => prev.map(s => s.id === id ? { ...s, status: "rejected" } : s))
  }

  return (
    <div className="space-y-6 font-mono text-xs">
      {/* Top Configuration Bar */}
      <div className="p-4 rounded-2xl glass-panel flex flex-wrap items-center justify-between gap-4 border-cyan-500/30">
        <div className="flex items-center gap-3">
          <Folder className="size-5 text-cyan-400" />
          <div>
            <span className="text-[10px] text-slate-500 block uppercase">Active Workspace Root</span>
            <span className="text-white font-bold text-sm">{workspaceRoot}</span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Mode Switch */}
          <div className="flex items-center gap-1 p-1 bg-slate-950 rounded-xl border border-slate-800">
            <button
              onClick={() => setMode("plan")}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                mode === "plan" ? "bg-purple-600 text-white shadow-lg shadow-purple-500/20" : "text-slate-400 hover:text-white"
              }`}
            >
              🛡️ Plan Mode
            </button>
            <button
              onClick={() => setMode("agent")}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                mode === "agent" ? "bg-cyan-600 text-white shadow-lg shadow-cyan-500/20" : "text-slate-400 hover:text-white"
              }`}
            >
              ⚡ Agent Mode
            </button>
          </div>

          {/* Model Selector */}
          <div className="p-2 rounded-xl bg-slate-950 border border-slate-800 text-left">
            <span className="text-[9px] text-slate-500 block">QWANTO MODEL</span>
            <span className="text-cyan-300 font-bold">{activeModel}</span>
          </div>
        </div>
      </div>

      {/* Conversation & Step Timeline */}
      <div className="space-y-4">
        {steps.map(step => (
          <div key={step.id} className="p-4 rounded-xl glass-panel space-y-2 border-slate-800">
            <div className="flex items-center justify-between">
              <span className="font-bold text-white text-sm">{step.title}</span>
              {step.status && (
                <Badge className={
                  step.status === "approved" ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40" :
                  step.status === "rejected" ? "bg-red-500/20 text-red-300 border-red-500/40" :
                  "bg-amber-500/20 text-amber-300 border-amber-500/40"
                }>
                  {step.status.toUpperCase()}
                </Badge>
              )}
            </div>

            <p className="text-slate-300 whitespace-pre-wrap">{step.content}</p>

            {/* Diff Preview if present */}
            {step.diff && (
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-900 font-mono text-[11px] space-y-1">
                <span className="text-slate-500 block text-[9px]">TARGET MODIFICATION PREVIEW</span>
                <pre className="text-cyan-300 overflow-x-auto">{step.diff}</pre>
              </div>
            )}

            {/* Approval Controls */}
            {step.status === "pending" && (
              <div className="flex items-center gap-2 pt-2 border-t border-slate-800/80">
                <Button 
                  onClick={() => handleApprove(step.id)}
                  className="bg-emerald-600 hover:bg-emerald-500 text-white text-xs h-8"
                >
                  <Check className="size-3.5 mr-1" /> Approve & Execute
                </Button>
                <Button 
                  onClick={() => handleReject(step.id)}
                  variant="destructive" 
                  className="bg-red-950 border border-red-500/40 text-red-400 hover:bg-red-900/50 text-xs h-8"
                >
                  <XCircle className="size-3.5 mr-1" /> Reject
                </Button>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Input Prompt Box */}
      <div className="p-3 rounded-2xl glass-panel flex items-center gap-3 border-cyan-500/40">
        <input 
          type="text" 
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          placeholder={`Describe your coding task for Qwanto (${mode === "plan" ? "Plan Mode active: read-only until approved" : "Agent Mode active"})...`}
          className="flex-1 bg-transparent text-white placeholder-slate-500 focus:outline-none text-xs"
        />
        <Button className="bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold h-9">
          <Send className="size-4 mr-1.5" /> Execute
        </Button>
      </div>
    </div>
  )
}
