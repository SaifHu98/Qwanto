import { useEffect, useMemo, useState, type ReactNode } from "react"
import {
  Check,
  ChevronLeft,
  ChevronRight,
  Command,
  FileCode2,
  FolderOpen,
  GitCompare,
  MessageSquare,
  PanelRightClose,
  PanelRightOpen,
  Play,
  Plus,
  Search,
  Settings2,
  ShieldCheck,
  Square,
  X,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import type { ChatMessage, DiscoveredModel } from "@/lib/api"
import type { AgentSession, DesktopGatewayStatus, DesktopToolResult } from "@/lib/desktop"
import { desktopInvoke } from "@/lib/desktop"
import type { GatewayConnectionState } from "@/lib/gateway"
import { cn } from "@/lib/utils"
import { modelIsSelectable } from "@/lib/gateway"
import type { AgentProfile } from "@/lib/agent"
import type { SessionUsage } from "./DesktopSettingsView"

type DesktopSection = "project" | "chats" | "files" | "changes" | "settings"
type InspectorTab = "diff" | "approvals" | "output" | "file"

interface DesktopAgentViewProps {
  gateway: DesktopGatewayStatus | null
  gatewayState: GatewayConnectionState
  gatewayMessage: string
  connected: boolean
  onProbe: () => void
  model: string
  models: string[]
  discoveredModels: DiscoveredModel[]
  onSelectModel: (model: string) => void
  onLoadModel: (model: string) => void
  onStopModel: () => void
  loadingModel: boolean
  mode: "plan" | "agent"
  onModeChange: (mode: "plan" | "agent") => void
  profile?: AgentProfile
  onProfileChange?: (profile: AgentProfile) => void
  messages: ChatMessage[]
  draft: string
  onDraftChange: (value: string) => void
  onSend: () => void
  onStopGeneration: () => void
  onClear: () => void
  usage?: SessionUsage
  loading: boolean
  error: string
  onRestartGateway?: () => void
  onResumeSession?: (session: AgentSession) => void
  settingsContent?: ReactNode
}

export function DesktopAgentView(props: DesktopAgentViewProps) {
  const [section, setSection] = useState<DesktopSection>("chats")
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>("file")
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [inspectorCollapsed, setInspectorCollapsed] = useState(false)
  const [workspace, setWorkspace] = useState("")
  const [workspaceInput, setWorkspaceInput] = useState("")
  const [fileOutput, setFileOutput] = useState("")
  const [filePath, setFilePath] = useState("")
  const [fileTree, setFileTree] = useState<string[]>([])
  const [diffOutput, setDiffOutput] = useState("")
  const [commandOutput, setCommandOutput] = useState("")
  const [approval, setApproval] = useState<DesktopToolResult | null>(null)
  const [toolError, setToolError] = useState("")
  const [sessionId] = useState(() => `desktop-${crypto.randomUUID?.() || Date.now()}`)
  const [savedSessions, setSavedSessions] = useState<AgentSession[]>([])

  const qwnModels = useMemo(
    () => props.discoveredModels.filter((candidate) => modelIsSelectable(candidate)),
    [props.discoveredModels],
  )

  const invokeTool = async (toolName: string, args: Record<string, unknown>, approvalToken?: string) => {
    setToolError("")
    try {
      const result = await desktopInvoke<DesktopToolResult>("execute_agent_tool", {
        sessionId,
        toolName,
        args,
        approvalToken,
      })
      if (result.output) setCommandOutput(result.output)
      return result
    } catch (error) {
      setToolError(error instanceof Error ? error.message : "Desktop tool failed.")
      return null
    }
  }

  const refreshFiles = async (root = workspace) => {
    const result = await invokeTool("list_directory", root ? { path: root } : {})
    if (result?.success) setFileTree(result.output.split("\n").filter(Boolean))
    if (result?.error) setToolError(result.error)
  }

  const selectFile = async (entry: string) => {
    const name = entry.replace(/^\[(?:DIR|FILE)\]\s+/, "")
    if (entry.startsWith("[DIR]")) return
    setFilePath(name)
    const result = await invokeTool("read_file", { path: name })
    if (result?.success) {
      setFileOutput(result.output)
      setInspectorTab("file")
    } else if (result?.error) setToolError(result.error)
  }

  const setWorkspaceRoot = async () => {
    if (!workspaceInput.trim()) return
    try {
      const canonical = await desktopInvoke<string>("set_workspace_root", { rootPath: workspaceInput.trim() })
      setWorkspace(canonical)
      setWorkspaceInput(canonical)
      await refreshFiles(canonical)
      setSection("project")
    } catch (error) {
      setToolError(error instanceof Error ? error.message : "Workspace could not be opened.")
    }
  }

  const inspectDiff = async () => {
    const result = await invokeTool("execute_command", {
      program: "git",
      args: ["diff", "--", "."],
      cwd: workspace || undefined,
    })
    if (!result) return
    if (result.outcome === "needs_approval") {
      setApproval(result)
      setInspectorTab("approvals")
      return
    }
    setDiffOutput(result.output)
    setInspectorTab("diff")
  }

  const approveCommand = async () => {
    if (!approval?.approval_token) return
    const result = await invokeTool(
      "execute_command",
      { program: "git", args: ["diff", "--", "."], cwd: workspace || undefined },
      approval.approval_token,
    )
    if (result?.success) {
      setDiffOutput(result.output || "No working-tree changes.")
      setCommandOutput(result.output)
      setApproval(null)
      setInspectorTab("diff")
    } else if (result?.error) setToolError(result.error)
  }

  useEffect(() => {
    if (section === "files" && workspace) void refreshFiles()
  }, [section, workspace])

  useEffect(() => {
    if (!workspace) return
    void desktopInvoke<AgentSession[]>("list_agent_sessions")
      .then((sessions) => setSavedSessions(sessions.filter((session) => session.workspace_root === workspace).slice(0, 5)))
      .catch(() => setSavedSessions([]))
  }, [workspace, props.messages.length])

  useEffect(() => {
    if (!workspace || !props.messages.length) return
    const now = new Date().toISOString()
    const session: AgentSession = {
      id: sessionId,
      title: "Local agent session",
      created_at: now,
      updated_at: now,
      workspace_root: workspace,
      active_model: props.model,
      mode: props.mode,
      steps: props.messages.map((message) => ({ id: message.id, timestamp: now, step_type: message.role, content: message.content })),
    }
    void desktopInvoke("save_agent_session", { session }).catch(() => undefined)
  }, [workspace, props.messages, props.model, props.mode, sessionId])

  return (
    <div className="desktop-agent-shell" data-testid="desktop-agent-shell">
      <aside className={cn("desktop-sidebar", sidebarCollapsed && "is-collapsed")}>
        <div className="desktop-sidebar-head">
          {!sidebarCollapsed && <div className="desktop-wordmark"><img className="brand-icon" src="/qwanto-icon.png" alt="Qwanto" /><span>Qwanto Desktop</span></div>}
          <button className="icon-button" aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"} onClick={() => setSidebarCollapsed((value) => !value)}>
            {sidebarCollapsed ? <ChevronRight className="size-4" /> : <ChevronLeft className="size-4" />}
          </button>
        </div>
        {!sidebarCollapsed && (
          <>
            <section className="desktop-project-switcher">
              <span className="desktop-label">PROJECT</span>
              <button className="project-switcher-button" onClick={() => setSection("project")}>
                <FolderOpen className="size-4" /><span>{workspace ? workspace.split(/[\\/]/).pop() : "Choose a project"}</span>
              </button>
            </section>
            <nav className="desktop-primary-nav" aria-label="Desktop Agent">
              {([
                ["project", "Project", FolderOpen],
                ["chats", "Chats", MessageSquare],
                ["files", "Files", FileCode2],
                ["changes", "Changes", GitCompare],
                ["settings", "Settings", Settings2],
              ] as const).map(([id, label, Icon]) => (
                <button key={id} className={cn("desktop-nav-item", section === id && "active")} onClick={() => setSection(id)}>
                  <Icon className="size-4" /><span>{label}</span>
                </button>
              ))}
            </nav>
            <section className="desktop-conversation-list">
              <div className="desktop-section-heading"><span>CONVERSATIONS</span><button className="icon-button" aria-label="New conversation" onClick={props.onClear}><Plus className="size-3.5" /></button></div>
              <button className="desktop-conversation active"><MessageSquare className="size-3.5" /><span>New local session</span></button>
            </section>
            <section className="desktop-file-tree">
              <div className="desktop-section-heading"><span>FILES</span><button className="icon-button" aria-label="Refresh files" onClick={() => void refreshFiles()}><Search className="size-3.5" /></button></div>
              {fileTree.slice(0, 30).map((entry) => <button key={entry} className="desktop-tree-item" onClick={() => void selectFile(entry)}>{entry}</button>)}
              {!fileTree.length && <span className="desktop-muted">Open a project to inspect files.</span>}
            </section>
          </>
        )}
      </aside>

      <main className="desktop-main-column">
        <header className="desktop-topbar">
          <div className="desktop-topbar-model"><span className="desktop-label">MODEL</span><select value={props.model} onChange={(event) => props.onSelectModel(event.target.value)} disabled={!props.connected}>
            <option value="">No model selected</option>
            {qwnModels.map((candidate) => <option key={candidate.path} value={candidate.path}>{candidate.name}</option>)}
          </select></div>
          <label className="desktop-profile-picker"><span className="desktop-label">PROFILE</span><select value={props.profile || "balanced"} onChange={(event) => props.onProfileChange?.(event.target.value as AgentProfile)}><option value="fast">Fast</option><option value="balanced">Balanced</option><option value="deep">Deep</option></select></label>
          <div className="desktop-mode-toggle" role="group" aria-label="Agent mode">
            <button className={props.mode === "plan" ? "active" : ""} onClick={() => props.onModeChange("plan")}>Plan</button>
            <button className={props.mode === "agent" ? "active" : ""} onClick={() => props.onModeChange("agent")}>Agent</button>
          </div>
          <div className={cn("desktop-gateway-state", props.connected && "connected")} role="status"><span />{props.connected ? "Gateway ready" : props.gateway?.state === "starting" ? "Starting gateway" : "Gateway unavailable"}</div>
          {props.gatewayState === "connected" && props.model ? <Button size="sm" variant="destructive" onClick={props.onStopModel} disabled={!props.connected}><Square className="size-3" /> Stop</Button> : <Button size="sm" onClick={() => props.onLoadModel(props.model)} disabled={!props.connected || !props.model || props.loadingModel}><Play className="size-3" /> Start</Button>}
          <button className="icon-button" aria-label="Open settings" onClick={() => setSection("settings")}><Settings2 className="size-4" /></button>
        </header>

        {!props.connected && <div className="desktop-gateway-banner"><strong>{props.gateway?.state === "starting" ? "Starting local gateway…" : "Local gateway unavailable"}</strong><span>{props.gatewayMessage || props.gateway?.error || "Qwanto Desktop is preparing its private loopback service."}</span><Button size="sm" variant="secondary" onClick={() => { setInspectorCollapsed(false); setInspectorTab("output"); setCommandOutput(props.gateway?.error || props.gatewayMessage || "No gateway error recorded.") }}>Open logs</Button><Button size="sm" variant="secondary" onClick={props.onRestartGateway}>Restart gateway</Button><Button size="sm" variant="secondary" onClick={props.onProbe}>Retry</Button></div>}
        {props.error && <div className="desktop-error" role="alert">{props.error}</div>}
        {toolError && <div className="desktop-error" role="alert">{toolError}<button onClick={() => setToolError("")}><X className="size-3.5" /></button></div>}

        <div className="desktop-workspace">
          <section className="desktop-center-panel">
            {section === "project" && <ProjectPanel workspace={workspace} input={workspaceInput} onInput={setWorkspaceInput} onOpen={() => void setWorkspaceRoot()} sessions={savedSessions} onResume={props.onResumeSession} />}
            {section === "settings" && <div className="desktop-settings-host">{props.settingsContent}</div>}
            {section === "files" && <FilesPanel entries={fileTree} onSelect={selectFile} onRefresh={() => void refreshFiles()} />}
            {section === "changes" && <ChangesPanel diff={diffOutput} onInspect={() => void inspectDiff()} />}
            {section === "chats" && <ChatPanel {...props} />}
          </section>

          {!inspectorCollapsed && <aside className="desktop-inspector">
            <div className="desktop-inspector-tabs">
              {(["diff", "approvals", "output", "file"] as InspectorTab[]).map((tab) => <button key={tab} className={inspectorTab === tab ? "active" : ""} onClick={() => setInspectorTab(tab)}>{tab === "diff" ? "Diff" : tab === "approvals" ? "Approvals" : tab === "output" ? "Output" : "Selected file"}</button>)}<button className="icon-button inspector-collapse-button" aria-label="Hide inspector" onClick={() => setInspectorCollapsed(true)}><PanelRightClose className="size-3.5" /></button>
            </div>
            {inspectorTab === "diff" && <pre className="desktop-code-preview">{diffOutput || "No diff loaded. Open Changes to inspect the working tree."}</pre>}
            {inspectorTab === "approvals" && <ApprovalPanel approval={approval} onApprove={() => void approveCommand()} onReject={() => setApproval(null)} />}
            {inspectorTab === "output" && <pre className="desktop-code-preview">{commandOutput || "Command output will appear here."}</pre>}
            {inspectorTab === "file" && <><div className="desktop-file-title">{filePath || "Selected file"}</div><pre className="desktop-code-preview">{fileOutput || "Select a file from the project tree."}</pre></>}
          </aside>}
          {inspectorCollapsed && <button className="inspector-show-button" aria-label="Show inspector" onClick={() => setInspectorCollapsed(false)}><PanelRightOpen className="size-4" /></button>}
        </div>
      </main>
    </div>
  )
}

function ProjectPanel({ workspace, input, onInput, onOpen, sessions, onResume }: { workspace: string; input: string; onInput: (value: string) => void; onOpen: () => void; sessions: AgentSession[]; onResume?: (session: AgentSession) => void }) {
  return <div className="desktop-panel-content"><div className="desktop-eyebrow">PROJECT</div><h1>Choose a local project</h1><p className="desktop-muted">The desktop agent can inspect and change files only inside this canonical workspace.</p><div className="desktop-project-form"><Input value={input} placeholder="C:\\Projects\\my-app" onChange={(event) => onInput(event.target.value)} /><Button onClick={onOpen}><FolderOpen className="size-4" /> Open project</Button></div>{workspace && <div className="desktop-success"><Check className="size-4" /> Workspace: {workspace}</div>}{workspace && sessions.length > 0 && onResume && <section className="desktop-resume-card"><div><strong>Resume local checkpoint</strong><p className="desktop-muted">Saved locally in the desktop session store. No source is uploaded.</p></div><Button size="sm" variant="secondary" onClick={() => onResume(sessions[0])}>Resume latest</Button></section>}</div>
}

function FilesPanel({ entries, onSelect, onRefresh }: { entries: string[]; onSelect: (entry: string) => void; onRefresh: () => void }) {
  return <div className="desktop-panel-content"><div className="desktop-panel-heading"><div><div className="desktop-eyebrow">FILES</div><h1>Project files</h1></div><Button size="sm" variant="secondary" onClick={onRefresh}><Search className="size-3.5" /> Refresh</Button></div><div className="desktop-file-list">{entries.map((entry) => <button key={entry} onClick={() => onSelect(entry)}>{entry}</button>)}{!entries.length && <p className="desktop-muted">Open a project to load its file tree.</p>}</div></div>
}

function ChangesPanel({ diff, onInspect }: { diff: string; onInspect: () => void }) {
  return <div className="desktop-panel-content"><div className="desktop-panel-heading"><div><div className="desktop-eyebrow">CHANGES</div><h1>Working tree</h1></div><Button size="sm" variant="secondary" onClick={onInspect}><GitCompare className="size-3.5" /> Inspect diff</Button></div><pre className="desktop-code-preview large">{diff || "No diff loaded."}</pre></div>
}

function ApprovalPanel({ approval, onApprove, onReject }: { approval: DesktopToolResult | null; onApprove: () => void; onReject: () => void }) {
  if (!approval) return <div className="desktop-empty-inspector"><ShieldCheck className="size-5" /><p>No pending approvals.</p><span>Writes and commands pause here until you approve them.</span></div>
  return <div className="desktop-approval-card"><ShieldCheck className="size-5" /><h2>Approval required</h2><p>{approval.action_details?.description || "The desktop agent requested a privileged action."}</p>{approval.action_details?.command && <code>{approval.action_details.command}</code>}<div className="desktop-approval-actions"><Button size="sm" onClick={onApprove}>Approve</Button><Button size="sm" variant="secondary" onClick={onReject}>Reject</Button></div></div>
}

function ChatPanel(props: DesktopAgentViewProps) {
  const usage = props.usage || { promptTokens: null, completionTokens: null, totalTokens: null, elapsedMs: null, ttftMs: null, tokensPerSecond: null, contextUse: null, toolCalls: null, queueState: "idle" }
  return <div className="desktop-chat-panel"><div className="desktop-chat-heading"><div><div className="desktop-eyebrow">LOCAL AGENT</div><h1>Build with your machine</h1><p className="desktop-muted">Plan first, then execute only through the approval-gated desktop boundary.</p></div><Button size="sm" variant="ghost" onClick={props.onClear} disabled={!props.messages.length}><Command className="size-3.5" /> Clear</Button></div><div className="desktop-timeline"><div className="desktop-timeline-step complete"><span>1</span><div><strong>Plan</strong><p>{props.mode === "plan" ? "Plan Mode is read-only." : "Agent Mode can request approved tools."}</p></div></div><div className="desktop-timeline-step"><span>2</span><div><strong>Execute</strong><p>File edits and commands appear in the inspector before they run.</p></div></div>{props.messages.map((message) => <article key={message.id} className={cn("desktop-message", message.role)}><span>{message.role === "user" ? "You" : "Q"}</span><p>{message.content}</p></article>)}</div><section className="session-usage-panel" aria-label="Session usage"><div><span>Prompt</span><strong>{usage.promptTokens ?? "Unavailable"}</strong></div><div><span>Completion</span><strong>{usage.completionTokens ?? "Unavailable"}</strong></div><div><span>Total</span><strong>{usage.totalTokens ?? "Unavailable"}</strong></div><div><span>Elapsed</span><strong>{usage.elapsedMs != null ? `${(usage.elapsedMs / 1000).toFixed(1)}s` : "Unavailable"}</strong></div><div><span>TTFT</span><strong>{usage.ttftMs != null ? `${usage.ttftMs.toFixed(0)}ms` : "Unavailable"}</strong></div><div><span>Tokens/s</span><strong>{usage.tokensPerSecond?.toFixed(2) || "Unavailable"}</strong></div><div><span>Context</span><strong>{usage.contextUse != null ? `${usage.contextUse}%` : "Unavailable"}</strong></div><div><span>Tools</span><strong>{usage.toolCalls ?? "Unavailable"}</strong></div><div><span>Queue</span><strong>{usage.queueState}</strong></div></section><div className="desktop-composer"><Textarea value={props.draft} onChange={(event) => props.onDraftChange(event.target.value)} placeholder="Ask the local agent to inspect or explain your project…" onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); props.onSend() } }} /><div className="desktop-composer-footer"><span>{props.mode === "plan" ? "Plan Mode · read-only" : "Agent Mode · approvals required"}</span>{props.loading ? <Button variant="destructive" size="sm" onClick={props.onStopGeneration}><Square className="size-3" /> Stop</Button> : <Button size="sm" onClick={props.onSend} disabled={!props.draft.trim() || !props.connected || !props.model}><Play className="size-3" /> Send</Button>}</div></div></div>
}
