import { ShieldCheck, Terminal, Folder } from "lucide-react"
import type { ReactNode } from "react"
import { Badge } from "@/components/ui/badge"

export interface StepItem {
  id: string
  type: "plan" | "tool_call" | "tool_result" | "user" | "assistant"
  title: string
  content: string
  toolName?: string
  toolArgs?: unknown
  status?: "pending" | "approved" | "rejected" | "executed"
  diff?: string
}

/**
 * Browser-safe boundary notice. Native approvals are implemented by the Tauri
 * host; this component must not pretend that a browser can execute them.
 */
export function CodingAgentView() {
  return (
    <div className="p-6 max-w-3xl mx-auto space-y-5">
      <div className="rounded-2xl glass-panel p-5 space-y-2">
        <div className="flex items-center gap-2"><ShieldCheck className="size-5 text-primary" /><h1 className="text-xl font-semibold">Desktop coding agent</h1><Badge>Tauri only</Badge></div>
        <p className="text-sm text-muted-foreground">The browser dashboard cannot inspect files, run commands, or approve edits. Open Qwanto Desktop to use the native agent boundary.</p>
      </div>
      <div className="grid gap-3 sm:grid-cols-3 text-sm">
        <BoundaryCard icon={<Folder className="size-4" />} title="Workspace" text="Canonical workspace required" />
        <BoundaryCard icon={<ShieldCheck className="size-4" />} title="Approvals" text="Writes and commands are token-gated" />
        <BoundaryCard icon={<Terminal className="size-4" />} title="Commands" text="Explicit argv; no shell interpolation" />
      </div>
    </div>
  )
}

function BoundaryCard({ icon, title, text }: { icon: ReactNode; title: string; text: string }) {
  return <div className="rounded-xl glass-panel p-4 space-y-2"><div className="flex items-center gap-2 text-primary">{icon}<span className="font-semibold">{title}</span></div><p className="text-xs text-muted-foreground">{text}</p></div>
}
