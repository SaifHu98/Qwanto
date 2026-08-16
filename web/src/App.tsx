import { useEffect, useRef, useState } from "react"
import { BrowserChatView } from "@/components/BrowserChatView"
import { DesktopAgentView } from "@/components/DesktopAgentView"
import { DesktopSettingsView } from "@/components/DesktopSettingsView"
import type { ChatAttachment, ChatMessage, DiscoveredModel, HealthResponse } from "@/lib/api"
import { getHealth, listDiscoveredModels, listModels, loadModel, streamChat, unloadModel } from "@/lib/api"
import { chooseRecommendedModel, classifyGatewayFailure, gatewayStateFromHealth, type GatewayConnectionState } from "@/lib/gateway"
import { desktopInvoke, type AgentSession, type DesktopGatewayStatus } from "@/lib/desktop"
import { stored } from "@/lib/storage"
import { profileConfig, type AgentProfile } from "@/lib/agent"
import { resolveSkillInvocation } from "@/lib/extensions"
import type { SessionUsage } from "@/components/DesktopSettingsView"

const makeMessage = (role: ChatMessage["role"], content: string, attachments?: ChatAttachment[], skill?: ChatMessage["skill"]): ChatMessage => ({
  id: globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`,
  role,
  content,
  ...(attachments?.length ? { attachments } : {}),
  ...(skill ? { skill } : {}),
})

export function isDesktopShell(): boolean {
  if (typeof window === "undefined") return false
  return window.location.protocol === "tauri:" || window.location.protocol === "asset:" || "__TAURI_INTERNALS__" in window
}

export default function App() {
  const desktopShell = isDesktopShell()
  const servedByEngine = typeof window !== "undefined" && window.location.port !== "5173" && window.location.protocol.startsWith("http")
  const defaultBase = servedByEngine ? `${window.location.origin}/v1` : "http://127.0.0.1:8000/v1"
  const [baseUrl, setBaseUrl] = useState(() => stored(localStorage, "qwanto.baseUrl", defaultBase))
  const [apiKey, setApiKey] = useState("")
  const [model, setModel] = useState(() => stored(localStorage, "qwanto.model", ""))
  const [models, setModels] = useState<string[]>([])
  const [discoveredModels, setDiscoveredModels] = useState<DiscoveredModel[]>([])
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [connected, setConnected] = useState(false)
  const [connecting, setConnecting] = useState(false)
  const [gatewayState, setGatewayState] = useState<GatewayConnectionState>("not-running")
  const [gatewayMessage, setGatewayMessage] = useState("Connect to an already-running local gateway.")
  const [desktopGateway, setDesktopGateway] = useState<DesktopGatewayStatus | null>(null)
  const [agentMode, setAgentMode] = useState<"plan" | "agent">("plan")
  const [agentProfile, setAgentProfile] = useState<AgentProfile>("balanced")
  const [temperature, setTemperature] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(512)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [draft, setDraft] = useState("")
  const [loading, setLoading] = useState(false)
  const [loadingModel, setLoadingModel] = useState(false)
  const [error, setError] = useState("")
  const [logs, setLogs] = useState<Array<{ time: string; type: "error" | "warn" | "info"; message: string }>>([])
  const [sessionUsage, setSessionUsage] = useState<SessionUsage>({ promptTokens: null, completionTokens: null, totalTokens: null, elapsedMs: null, ttftMs: null, tokensPerSecond: null, contextUse: null, toolCalls: 0, queueState: "idle" })
  const [searchContext, setSearchContext] = useState("")
  const abortRef = useRef<AbortController | null>(null)
  const desktopProbeRef = useRef(false)

  const addLog = (type: "error" | "warn" | "info", message: string) => {
    setLogs((current) => [...current.slice(-200), { time: new Date().toLocaleTimeString(), type, message }])
  }

  useEffect(() => {
    try {
      localStorage.setItem("qwanto.baseUrl", baseUrl)
      localStorage.setItem("qwanto.model", model)
    } catch { /* local persistence is optional */ }
  }, [baseUrl, model])

  const connect = async (scanModels = false) => {
    setConnecting(true)
    setError("")
    try {
      const healthResult = await getHealth(baseUrl, apiKey)
      const state = gatewayStateFromHealth(healthResult)
      if (state !== "connected" && state !== "model-required") throw new Error("Incompatible Qwanto gateway version.")
      setHealth(healthResult)
      if (scanModels) {
        const [availableModels, inventory] = await Promise.all([listModels(baseUrl, apiKey), listDiscoveredModels(baseUrl, apiKey)])
        setModels(availableModels)
        setDiscoveredModels(inventory.models || [])
        const choice = chooseRecommendedModel(inventory.models || [], model)
        if (choice.model && choice.model.path !== model) setModel(choice.model.path)
        if (!choice.model && model) setModel("")
      }
      setConnected(true)
      setGatewayState(state)
      setGatewayMessage(state === "model-required" ? "Gateway ready. Choose a validated local QWN model to start inference." : "Local gateway ready.")
    } catch (cause) {
      setConnected(false)
      setHealth(null)
      const state = cause instanceof Error && cause.message.includes("Incompatible") ? "incompatible-version" : classifyGatewayFailure(cause)
      setGatewayState(state)
      const message = state === "wrong-server"
        ? "This is a static web server, not the Qwanto gateway."
        : state === "incompatible-version"
          ? "The gateway API version is incompatible."
          : desktopShell
            ? "The local gateway sidecar is not ready yet."
            : "Start Qwanto Code or an existing local gateway, then probe again."
      setGatewayMessage(message)
      setError(cause instanceof Error ? cause.message : "Could not reach the local gateway.")
      addLog("error", `Connect: ${cause instanceof Error ? cause.message : cause}`)
    } finally {
      setConnecting(false)
    }
  }

  useEffect(() => {
    if (!desktopShell) return
    let disposed = false
    const sync = async () => {
      try {
        const status = await desktopInvoke<DesktopGatewayStatus>("get_gateway_status")
        if (disposed) return
        setDesktopGateway(status)
        if (status.api_url) setBaseUrl(`${status.api_url}/v1`)
        if (status.state === "failed") {
          setGatewayState("failed")
          setGatewayMessage(status.error || "The gateway sidecar failed to start.")
        }
      } catch (cause) {
        if (!disposed) {
          setDesktopGateway({ state: "failed", api_url: null, port: null, error: cause instanceof Error ? cause.message : "Desktop bridge unavailable.", sidecar_packaged: false })
          setGatewayState("failed")
          setGatewayMessage("Open this page inside Qwanto Code to start the gateway automatically.")
        }
      }
    }
    void sync()
    const timer = window.setInterval(() => void sync(), 1500)
    return () => { disposed = true; window.clearInterval(timer) }
  }, [desktopShell])

  useEffect(() => {
    if (!desktopShell || !desktopGateway?.api_url || desktopProbeRef.current || !baseUrl.startsWith(desktopGateway.api_url)) return
    desktopProbeRef.current = true
    void connect(false)
  }, [desktopShell, desktopGateway?.api_url, baseUrl])

  useEffect(() => {
    if (servedByEngine && !desktopShell) void connect(true)
  }, [servedByEngine, desktopShell])

  const handleLoadModel = async (path: string) => {
    if (!connected || !path) return
    setLoadingModel(true)
    setError("")
    try {
      const profile = profileConfig(agentProfile)
      const result = await loadModel(baseUrl, path, "auto", undefined, apiKey, profile.contextSize)
      setModel(result.model_id)
      await connect()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not start the selected model.")
      addLog("error", `Model start: ${cause instanceof Error ? cause.message : cause}`)
    } finally { setLoadingModel(false) }
  }

  const handleStopModel = async () => {
    try {
      await unloadModel(baseUrl, apiKey)
      setModel("")
      setHealth(null)
      setGatewayState("model-required")
      setGatewayMessage("Gateway ready. Choose a validated local QWN model to start inference.")
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Could not stop the model.") }
  }

  const handleMode = async (mode: "plan" | "agent") => {
    setAgentMode(mode)
    if (!desktopShell) return
    try { await desktopInvoke("set_execution_mode", { mode }) }
    catch (cause) { setError(cause instanceof Error ? cause.message : "Could not change agent mode.") }
  }

  const send = async (attachments: ChatAttachment[] = []) => {
    const content = draft.trim()
    if (!content || !connected || !model || loading) return
    const profile = profileConfig(agentProfile)
    const invocation = resolveSkillInvocation(content)
    const prompt = invocation?.prompt || content
    const contextualContent = searchContext ? `${prompt}\n\n[Approved web sources]\n${searchContext}` : prompt
    const user = makeMessage("user", contextualContent, attachments, invocation?.skill)
    const assistant = makeMessage("assistant", "")
    const history = [...messages, user]
    setMessages([...history, assistant])
    setDraft("")
    setLoading(true)
    const startedAt = performance.now()
    setSessionUsage((current) => ({ ...current, queueState: "queued", toolCalls: 0 }))
    setError("")
    const controller = new AbortController()
    abortRef.current = controller
    try {
      const result = await streamChat({
        baseUrl,
        apiKey,
        model,
        messages: history,
        temperature: desktopShell ? profile.temperature : temperature,
        maxTokens: desktopShell ? profile.maxTokens : maxTokens,
        enableThinking: false,
        signal: controller.signal,
        onDelta: (delta) => setMessages((current) => current.map((item) => item.id === assistant.id ? { ...item, content: item.content + delta } : item)),
      })
      const elapsedMs = performance.now() - startedAt
      setSessionUsage({
        promptTokens: result.usage?.prompt_tokens ?? null,
        completionTokens: result.usage?.completion_tokens ?? null,
        totalTokens: result.usage?.total_tokens ?? null,
        elapsedMs,
        ttftMs: result.ttftMs,
        tokensPerSecond: result.tokensPerSecond,
        contextUse: null,
        toolCalls: 0,
        queueState: result.queueWaitMs != null && result.queueWaitMs > 0 ? `waited ${Math.round(result.queueWaitMs)} ms` : "complete",
      })
      setSearchContext("")
    } catch (cause) {
      if (!controller.signal.aborted) {
        setError(cause instanceof Error ? cause.message : "Generation failed.")
        addLog("error", `Generation: ${cause instanceof Error ? cause.message : cause}`)
        setMessages((current) => current.filter((item) => item.id !== assistant.id || item.content))
      }
    } finally {
      abortRef.current = null
      setLoading(false)
    }
  }

  const clear = () => setMessages([])

  const resumeSession = (session: AgentSession) => {
    setMessages(session.steps.filter((step) => step.step_type === "user" || step.step_type === "assistant").map((step) => ({ id: step.id, role: step.step_type as ChatMessage["role"], content: step.content })))
    setModel(session.active_model)
    setAgentMode(session.mode === "agent" ? "agent" : "plan")
  }

  if (desktopShell) return <DesktopAgentView
    gateway={desktopGateway}
    gatewayState={gatewayState}
    gatewayMessage={gatewayMessage}
    connected={connected}
    onProbe={() => void connect(false)}
    model={model}
    models={models}
    discoveredModels={discoveredModels}
    onSelectModel={setModel}
    onLoadModel={(path) => void handleLoadModel(path)}
    onStopModel={() => void handleStopModel()}
    loadingModel={loadingModel}
    mode={agentMode}
    onModeChange={(mode) => void handleMode(mode)}
    profile={agentProfile}
    onProfileChange={setAgentProfile}
    messages={messages}
    draft={draft}
    onDraftChange={setDraft}
    onSend={() => void send()}
    onStopGeneration={() => abortRef.current?.abort()}
    onClear={clear}
    onResumeSession={resumeSession}
    usage={sessionUsage}
    loading={loading}
    error={error}
    onRestartGateway={async () => {
      try { await desktopInvoke("restart_gateway"); await connect(false) }
      catch (cause) { setError(cause instanceof Error ? cause.message : "Gateway restart failed."); addLog("error", `Gateway restart: ${cause instanceof Error ? cause.message : cause}`) }
    }}
    settingsContent={<DesktopSettingsView baseUrl={baseUrl} apiKey={apiKey} gatewayReady={connected} gatewayReadyElapsedMs={desktopGateway?.ready_elapsed_ms} logs={logs} model={model} models={discoveredModels} onInventoryLoaded={(inventory) => { setDiscoveredModels(inventory); setModels(inventory.map((candidate) => candidate.path)) }} onSelectModel={setModel} onActivateModel={(path) => void handleLoadModel(path)} loadingModel={loadingModel} profile={agentProfile} onProfileChange={setAgentProfile} usage={sessionUsage} onIncludeSearchContext={(sources) => setSearchContext(sources.map((source) => `${source.title}\n${source.url}\n${source.snippet}`).join("\n\n"))} />}
  />

  return <BrowserChatView
    baseUrl={baseUrl}
    apiKey={apiKey}
    onBaseUrlChange={setBaseUrl}
    onApiKeyChange={setApiKey}
    model={model}
    models={models}
    discoveredModels={discoveredModels}
    onModelChange={setModel}
    temperature={temperature}
    onTemperatureChange={setTemperature}
    maxTokens={maxTokens}
    onMaxTokensChange={setMaxTokens}
    connected={connected}
    gatewayState={gatewayState}
    gatewayMessage={gatewayMessage}
    onProbe={() => void connect(true)}
    probing={connecting}
    messages={messages}
    draft={draft}
    onDraftChange={setDraft}
    onSend={() => void send()}
    onStop={() => abortRef.current?.abort()}
    onClear={clear}
    loading={loading}
    error={error}
  />
}

export function QwantoBrand() {
  return <img className="brand-icon" src="/qwanto-icon.png" alt="Qwanto Code" />
}
