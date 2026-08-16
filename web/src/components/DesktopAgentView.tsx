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
  Play,
  Plus,
  Search,
  Settings2,
  ShieldCheck,
  Square,
  Terminal,
  X,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import type { ChatMessage, DiscoveredModel } from "@/lib/api"
import type { DesktopGatewayStatus, DesktopToolResult } from "@/lib/desktop"
import { desktopInvoke } from "@/lib/desktop"
import type { GatewayConnectionState } from "@/lib/gateway"
import { cn } from "@/lib/utils"

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
  messages: ChatMessage[]
  draft: string
  onDraftChange: (value: string) => void
  onSend: () => void
  onStopGeneration: () => void
  onClear: () => void
  loading: boolean
  error: string
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

  const qwnModels = useMemo(
    () => props.discoveredModels.filter((candidate) => candidate.type === "qwn" && candidate.compatibility_state === "compatible"),
    [props.discoveredModels],
  )

  const invokeTool = async (toolName: string, args: Record<string, unknown>, approvalToken?: string) => {
    setToolError("")
    try {
      return await desktopInvoke<DesktopToolResult>("execute_agent_tool", {
        sessionId,
        toolName,
        args,
        approvalToken,
      })
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

  return (
    <div className="desktop-agent-shell" data-testid="desktop-agent-shell">
      <aside className={cn("desktop-sidebar", sidebarCollapsed && "is-collapsed")}>
        <div className="desktop-sidebar-head">
          {!sidebarCollapsed && <div className="desktop-wordmark"><span className="desktop-mark">Q</span><span>Qwanto Desktop</span></div>}
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
            {props.models.filter(Boolean).map((candidate) => <option key={candidate} value={candidate}>{candidate}</option>)}
          </select></div>
          <div className="desktop-mode-toggle" role="group" aria-label="Agent mode">
            <button className={props.mode === "plan" ? "active" : ""} onClick={() => props.onModeChange("plan")}>Plan</button>
            <button className={props.mode === "agent" ? "active" : ""} onClick={() => props.onModeChange("agent")}>Agent</button>
          </div>
          <div className={cn("desktop-gateway-state", props.connected && "connected")} role="status"><span />{props.connected ? "Gateway ready" : props.gateway?.state === "starting" ? "Starting gateway" : "Gateway unavailable"}</div>
          {props.model ? <Button size="sm" variant="destructive" onClick={props.onStopModel} disabled={!props.connected}><Square className="size-3" /> Stop</Button> : <Button size="sm" onClick={() => props.onLoadModel(props.model)} disabled={!props.connected || !props.model || props.loadingModel}><Play className="size-3" /> Start</Button>}
          <button className="icon-button" aria-label={inspectorCollapsed ? "Show inspector" : "Hide inspector"} onClick={() => setInspectorCollapsed((value) => !value)}><Settings2 className="size-4" /></button>
        </header>

        {!props.connected && <div className="desktop-gateway-banner"><strong>{props.gateway?.state === "starting" ? "Starting local gateway…" : "Local gateway unavailable"}</strong><span>{props.gatewayMessage || props.gateway?.error || "Qwanto Desktop is preparing its private loopback service."}</span><Button size="sm" variant="secondary" onClick={props.onProbe}>Retry</Button></div>}
        {props.error && <div className="desktop-error" role="alert">{props.error}</div>}
        {toolError && <div className="desktop-error" role="alert">{toolError}<button onClick={() => setToolError("")}><X className="size-3.5" /></button></div>}

        <div className="desktop-workspace">
          <section className="desktop-center-panel">
            {section === "project" && <ProjectPanel workspace={workspace} input={workspaceInput} onInput={setWorkspaceInput} onOpen={() => void setWorkspaceRoot()} />}
            {section === "settings" && <SettingsPanel model={props.model} models={qwnModels} onSelectModel={props.onSelectModel} gateway={props.gateway}>{props.settingsContent}</SettingsPanel>}
            {section === "files" && <FilesPanel entries={fileTree} onSelect={selectFile} onRefresh={() => void refreshFiles()} />}
            {section === "changes" && <ChangesPanel diff={diffOutput} onInspect={() => void inspectDiff()} />}
            {section === "chats" && <ChatPanel {...props} />}
          </section>

          {!inspectorCollapsed && <aside className="desktop-inspector">
            <div className="desktop-inspector-tabs">
              {(["diff", "approvals", "output", "file"] as InspectorTab[]).map((tab) => <button key={tab} className={inspectorTab === tab ? "active" : ""} onClick={() => setInspectorTab(tab)}>{tab === "diff" ? "Diff" : tab === "approvals" ? "Approvals" : tab === "output" ? "Output" : "File"}</button>)}
            </div>
            {inspectorTab === "diff" && <pre className="desktop-code-preview">{diffOutput || "No diff loaded. Open Changes to inspect the working tree."}</pre>}
            {inspectorTab === "approvals" && <ApprovalPanel approval={approval} onApprove={() => void approveCommand()} onReject={() => setApproval(null)} />}
            {inspectorTab === "output" && <pre className="desktop-code-preview">{commandOutput || "Command output will appear here."}</pre>}
            {inspectorTab === "file" && <><div className="desktop-file-title">{filePath || "Selected file"}</div><pre className="desktop-code-preview">{fileOutput || "Select a file from the project tree."}</pre></>}
          </aside>}
        </div>
      </main>
    </div>
  )
}

function ProjectPanel({ workspace, input, onInput, onOpen }: { workspace: string; input: string; onInput: (value: string) => void; onOpen: () => void }) {
  return <div className="desktop-panel-content"><div className="desktop-eyebrow">PROJECT</div><h1>Choose a local project</h1><p className="desktop-muted">The desktop agent can inspect and change files only inside this canonical workspace.</p><div className="desktop-project-form"><Input value={input} placeholder="C:\\Projects\\my-app" onChange={(event) => onInput(event.target.value)} /><Button onClick={onOpen}><FolderOpen className="size-4" /> Open project</Button></div>{workspace && <div className="desktop-success"><Check className="size-4" /> Workspace: {workspace}</div>}</div>
}

function SettingsPanel({ model, models, onSelectModel, gateway, children }: { model: string; models: DiscoveredModel[]; onSelectModel: (model: string) => void; gateway: DesktopGatewayStatus | null; children?: ReactNode }) {
  return <div className="desktop-panel-content"><div className="desktop-eyebrow">SETTINGS</div><h1>Local runtime settings</h1><div className="desktop-settings-grid"><section><h2>Models</h2><p className="desktop-muted">Import, convert, and download models through the supervised local gateway. Model weights are never bundled.</p><label className="desktop-field">Active model<select value={model} onChange={(event) => onSelectModel(event.target.value)}><option value="">Choose a validated QWN model</option>{models.map((candidate) => <option key={candidate.path} value={candidate.path}>{candidate.name}</option>)}</select></label></section><section><h2>Runtime diagnostics</h2><p className="desktop-muted">Advanced telemetry, benchmark evidence, security details, and logs stay here instead of competing with the agent workspace.</p><div className="desktop-diagnostic"><Terminal className="size-4" /> Gateway {gateway?.api_url || "starting"}</div></section></div>{children}</div>
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
  return <div className="desktop-chat-panel"><div className="desktop-chat-heading"><div><div className="desktop-eyebrow">LOCAL AGENT</div><h1>Build with your machine</h1><p className="desktop-muted">Plan first, then execute only through the approval-gated desktop boundary.</p></div><Button size="sm" variant="ghost" onClick={props.onClear} disabled={!props.messages.length}><Command className="size-3.5" /> Clear</Button></div><div className="desktop-timeline"><div className="desktop-timeline-step complete"><span>1</span><div><strong>Plan</strong><p>{props.mode === "plan" ? "Plan Mode is read-only." : "Agent Mode can request approved tools."}</p></div></div><div className="desktop-timeline-step"><span>2</span><div><strong>Execute</strong><p>File edits and commands appear in the inspector before they run.</p></div></div>{props.messages.map((message) => <article key={message.id} className={cn("desktop-message", message.role)}><span>{message.role === "user" ? "You" : "Q"}</span><p>{message.content}</p></article>)}</div><div className="desktop-composer"><Textarea value={props.draft} onChange={(event) => props.onDraftChange(event.target.value)} placeholder="Ask the local agent to inspect or explain your project…" onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); props.onSend() } }} /><div className="desktop-composer-footer"><span>{props.mode === "plan" ? "Plan Mode · read-only" : "Agent Mode · approvals required"}</span>{props.loading ? <Button variant="destructive" size="sm" onClick={props.onStopGeneration}><Square className="size-3" /> Stop</Button> : <Button size="sm" onClick={props.onSend} disabled={!props.draft.trim() || !props.connected || !props.model}><Play className="size-3" /> Send</Button>}</div></div></div>
}
