import { useEffect, useRef, useState } from "react"
import { Feather } from "lucide-react"
import { BrowserChatView } from "@/components/BrowserChatView"
import { DesktopAgentView } from "@/components/DesktopAgentView"
import { DesktopSettingsView } from "@/components/DesktopSettingsView"
import type { ChatMessage, DiscoveredModel, HealthResponse } from "@/lib/api"
import { getHealth, listDiscoveredModels, listModels, loadModel, streamChat, unloadModel } from "@/lib/api"
import { chooseRecommendedModel, classifyGatewayFailure, gatewayStateFromHealth, type GatewayConnectionState } from "@/lib/gateway"
import { desktopInvoke, type DesktopGatewayStatus } from "@/lib/desktop"
import { stored } from "@/lib/storage"

const makeMessage = (role: ChatMessage["role"], content: string): ChatMessage => ({
  id: globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`,
  role,
  content,
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
  const [temperature, setTemperature] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(512)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [draft, setDraft] = useState("")
  const [loading, setLoading] = useState(false)
  const [loadingModel, setLoadingModel] = useState(false)
  const [error, setError] = useState("")
  const [logs, setLogs] = useState<Array<{ time: string; type: "error" | "warn" | "info"; message: string }>>([])
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

  const connect = async () => {
    setConnecting(true)
    setError("")
    try {
      const healthResult = await getHealth(baseUrl, apiKey)
      const state = gatewayStateFromHealth(healthResult)
      if (state !== "connected" && state !== "model-required") throw new Error("Incompatible Qwanto gateway version.")
      const [availableModels, inventory] = await Promise.all([listModels(baseUrl, apiKey), listDiscoveredModels(baseUrl, apiKey)])
      setHealth(healthResult)
      setModels(availableModels)
      setDiscoveredModels(inventory.models || [])
      setConnected(true)
      setGatewayState(state)
      const choice = chooseRecommendedModel(inventory.models || [], model)
      if (choice.model && !model) setModel(choice.model.path)
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
            : "Start Qwanto Desktop or an existing local gateway, then probe again."
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
          setGatewayMessage("Open this page inside Qwanto Desktop to start the gateway automatically.")
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
    void connect()
  }, [desktopShell, desktopGateway?.api_url, baseUrl])

  useEffect(() => {
    if (servedByEngine && !desktopShell) void connect()
  }, [servedByEngine, desktopShell])

  const handleLoadModel = async (path: string) => {
    if (!connected || !path) return
    setLoadingModel(true)
    setError("")
    try {
      const result = await loadModel(baseUrl, path, "auto", undefined, apiKey)
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

  const send = async () => {
    const content = draft.trim()
    if (!content || !connected || !model || loading) return
    const user = makeMessage("user", content)
    const assistant = makeMessage("assistant", "")
    const history = [...messages, user]
    setMessages([...history, assistant])
    setDraft("")
    setLoading(true)
    setError("")
    const controller = new AbortController()
    abortRef.current = controller
    try {
      await streamChat({
        baseUrl,
        apiKey,
        model,
        messages: history,
        temperature,
        maxTokens,
        enableThinking: false,
        signal: controller.signal,
        onDelta: (delta) => setMessages((current) => current.map((item) => item.id === assistant.id ? { ...item, content: item.content + delta } : item)),
      })
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

  if (desktopShell) return <DesktopAgentView
    gateway={desktopGateway}
    gatewayState={gatewayState}
    gatewayMessage={gatewayMessage}
    connected={connected}
    onProbe={() => void connect()}
    model={model}
    models={models}
    discoveredModels={discoveredModels}
    onSelectModel={setModel}
    onLoadModel={(path) => void handleLoadModel(path)}
    onStopModel={() => void handleStopModel()}
    loadingModel={loadingModel}
    mode={agentMode}
    onModeChange={(mode) => void handleMode(mode)}
    messages={messages}
    draft={draft}
    onDraftChange={setDraft}
    onSend={() => void send()}
    onStopGeneration={() => abortRef.current?.abort()}
    onClear={clear}
    loading={loading}
    error={error}
    settingsContent={<DesktopSettingsView baseUrl={baseUrl} apiKey={apiKey} gatewayReady={connected} logs={logs} onSelectModel={setModel} />}
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
    onProbe={() => void connect()}
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
  return <span className="desktop-mark" aria-label="Qwanto"><Feather className="size-4" /></span>
}
