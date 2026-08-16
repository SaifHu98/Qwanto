import { useState } from "react"
import { ArrowUp, CircleStop, Feather, LoaderCircle, RefreshCw, Settings2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import type { ChatMessage, DiscoveredModel } from "@/lib/api"
import type { GatewayConnectionState } from "@/lib/gateway"
import { GatewayStatusBanner } from "@/components/GatewayStatusBanner"

interface BrowserChatViewProps {
  baseUrl: string
  apiKey: string
  onBaseUrlChange: (value: string) => void
  onApiKeyChange: (value: string) => void
  model: string
  models: string[]
  discoveredModels: DiscoveredModel[]
  onModelChange: (value: string) => void
  temperature: number
  onTemperatureChange: (value: number) => void
  maxTokens: number
  onMaxTokensChange: (value: number) => void
  connected: boolean
  gatewayState: GatewayConnectionState
  gatewayMessage: string
  onProbe: () => void
  probing: boolean
  messages: ChatMessage[]
  draft: string
  onDraftChange: (value: string) => void
  onSend: () => void
  onStop: () => void
  onClear: () => void
  loading: boolean
  error: string
}

export function BrowserChatView(props: BrowserChatViewProps) {
  const [settingsOpen, setSettingsOpen] = useState(false)
  const selectableModels = props.discoveredModels.filter((candidate) => candidate.type === "qwn" && candidate.compatibility_state === "compatible")

  return (
    <div className="browser-console-shell" data-testid="browser-chat-shell">
      <header className="browser-console-header">
        <div className="browser-brand"><img className="brand-icon" src="/qwanto-icon.png" alt="Qwanto Code" /><div><strong>Qwanto Code</strong><span>Local chat</span></div></div>
        <div className="browser-header-actions"><div className="browser-connection-pill"><span className={props.connected ? "connected" : ""} />{props.connected ? "Connected" : "Not connected"}</div><button className="icon-button" aria-label="Basic settings" onClick={() => setSettingsOpen((value) => !value)}><Settings2 className="size-4" /></button></div>
      </header>
      {settingsOpen && <section className="browser-settings" aria-label="Basic settings"><label>Gateway endpoint<Input value={props.baseUrl} onChange={(event) => props.onBaseUrlChange(event.target.value)} /></label><label>API key<Input type="password" value={props.apiKey} placeholder="optional" onChange={(event) => props.onApiKeyChange(event.target.value)} /></label><label>Model<select value={props.model} onChange={(event) => props.onModelChange(event.target.value)} disabled={!props.connected}><option value="">Choose a local model</option>{selectableModels.map((candidate) => <option key={candidate.path} value={candidate.path}>{candidate.name}</option>)}{props.models.filter(Boolean).map((candidate) => <option key={candidate} value={candidate}>{candidate}</option>)}</select></label><label>Temperature<input type="range" min="0" max="2" step="0.1" value={props.temperature} onChange={(event) => props.onTemperatureChange(Number(event.target.value))} /></label><label>Max output tokens<Input type="number" min="1" max="4096" value={props.maxTokens} onChange={(event) => props.onMaxTokensChange(Number(event.target.value))} /></label></section>}
      <GatewayStatusBanner state={props.gatewayState} message={props.gatewayMessage || "The browser connects only to an already-running local gateway."} onProbe={props.onProbe} probing={props.probing} />
      <main className="browser-chat-main">
        <section className="browser-chat-card">
          {!props.connected && <div className="browser-local-note"><strong>Browser console</strong><span>This page is a safe local chat client. It cannot read projects, edit files, run commands, or use desktop agent tools. Open Qwanto Code to start the local gateway automatically.</span></div>}
          <div className="browser-message-list">{props.messages.length ? props.messages.map((message) => <article key={message.id} className={`browser-message ${message.role}`}><span>{message.role === "user" ? "You" : <Feather className="size-4" />}</span><p>{message.content || (props.loading ? <LoaderCircle className="size-4 animate-spin" /> : "")}</p></article>) : <div className="browser-empty"><Feather className="size-7" /><h1>Private conversation, local machine</h1><p>Connect to an existing Qwanto gateway and chat without granting this browser page project or terminal access.</p></div>}</div>
          {props.error && <div className="browser-error" role="alert">{props.error}</div>}
          <div className="browser-composer"><Textarea value={props.draft} onChange={(event) => props.onDraftChange(event.target.value)} placeholder={props.connected ? "Message the local model…" : "Connect to a local gateway first…"} disabled={!props.connected} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); props.onSend() } }} /><div className="browser-composer-footer"><button onClick={props.onClear} disabled={!props.messages.length || props.loading}>Clear</button><span>Chat only · no desktop permissions</span>{props.loading ? <Button variant="destructive" size="icon" aria-label="Stop generation" onClick={props.onStop}><CircleStop className="size-4" /></Button> : <Button size="icon" aria-label="Send message" disabled={!props.connected || !props.model || !props.draft.trim()} onClick={props.onSend}><ArrowUp className="size-4" /></Button>}</div></div>
        </section>
      </main>
      {!props.connected && <Button className="browser-probe-button" onClick={props.onProbe} disabled={props.probing}><RefreshCw className={`size-4 ${props.probing ? "animate-spin" : ""}`} /> Probe existing gateway</Button>}
    </div>
  )
}
