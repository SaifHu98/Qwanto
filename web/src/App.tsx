import { useEffect, useMemo, useRef, useState } from "react"
import {
  Activity,
  ArrowUp,
  BrainCircuit,
  CircleStop,
  Clock,
  Cpu,
  Database,
  Feather,
  Gauge,
  HardDrive,
  KeyRound,
  Layers,
  Link2,
  LoaderCircle,
  MemoryStick,
  MessageSquareText,
  MonitorDot,
  RefreshCw,
  SlidersHorizontal,
  Timer,
  Trash2,
  Zap,
  Download,
  Server,
  Globe,
  FolderSync,
  Waypoints,
  Copy,
  Check,
  Mic,
  MicOff,
  Paperclip,
  Search,
  RotateCcw,
  Image,
  File,
  X,
  Sparkles,
  ShieldCheck,
  Code2,
  BarChart3,
  Lock,
} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { PresetsView } from "./components/PresetsView"
import { TelemetryView } from "./components/TelemetryView"
import { DoctorView } from "./components/DoctorView"
import { WorkbenchView } from "./components/WorkbenchView"
import { BenchmarksView } from "./components/BenchmarksView"
import { SecurityView } from "./components/SecurityView"
import { ConverterView } from "./components/ConverterView"
import type { SystemPreset } from "@/lib/api"
import {
  getHealth,
  listModels,
  streamChat,
  type ChatMessage,
  type HealthResponse,
  type StreamChatResult,
  getQwantoConfig,
  listDiscoveredModels,
  loadModel,
  downloadModel,
  getDownloadStatus,
  cancelDownloadModel,
  pauseDownloadModel,
  resumeDownloadModel,
  deleteModel,
  configDownload,
  getModelPaths,
  addModelPath,
  removeModelPath,
  setResourceLimits,
  type QwantoConfig,
  type DiscoveredModel,
  type DownloadStatus,
} from "@/lib/api"
import { activeRequests, supportsCacheSlots } from "@/lib/runtime"
import { Brain } from "./Brain"
import { persistPublicSettings, stored } from "@/lib/storage"
import { cn } from "@/lib/utils"

const message = (role: ChatMessage["role"], content: string): ChatMessage => {
  let id: string
  try { id = crypto.randomUUID() } catch { id = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => { const r = Math.random() * 16 | 0; return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16) }) }
  return { id, role, content }
}

export default function App() {
  // When the page is served by the engine itself (coli web), same-origin is the
  // right default: no CORS, no manual endpoint editing. The Vite dev server
  // (port 5173) keeps the classic default.
  const servedByEngine = typeof window !== "undefined" && window.location.port !== "5173" && window.location.protocol.startsWith("http")
  const defaultBase = servedByEngine ? `${window.location.origin}/v1` : "http://127.0.0.1:8000/v1"
  const [baseUrl, setBaseUrl] = useState(() => {
    const saved = stored(localStorage, "qwanto.baseUrl", defaultBase)
    // migrate: a stored FACTORY default pointing at another origin would trip CORS
    // when the page is engine-served — upgrade it to same-origin once.
    if (servedByEngine && saved === "http://127.0.0.1:8000/v1" && defaultBase !== saved) return defaultBase
    return saved
  })
  const [apiKey, setApiKey] = useState("")
  const [models, setModels] = useState<string[]>([])
  const [model, setModel] = useState(() => stored(localStorage, "qwanto.model", "glm-5.2-qwanto"))
  const [temperature, setTemperature] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(512)
  const [ctxSize, setCtxSize] = useState(() => { try { return Number(localStorage.getItem("qwanto.ctxSize")) || 16384 } catch { return 16384 } })
  const [flashAttention, setFlashAttention] = useState(() => { try { return localStorage.getItem("qwanto.flashAttention") !== "false" } catch { return true } })
  const [kvCacheQuant, setKvCacheQuant] = useState(() => stored(localStorage, "qwanto.kvCacheQuant", "q4_0"))
  const [specDecoding, setSpecDecoding] = useState(() => { try { return localStorage.getItem("qwanto.specDecoding") === "true" } catch { return false } })
  const [draftModelPath, setDraftModelPath] = useState(() => stored(localStorage, "qwanto.draftModelPath", ""))
  const [thinking, setThinking] = useState(false)
  const [cacheSlot, setCacheSlot] = useState(0)
  const [conversations, setConversations] = useState<Record<number, ChatMessage[]>>(() => {
    try {
      const saved = localStorage.getItem("qwanto.conversations")
      if (saved) {
        const parsed = JSON.parse(saved)
        if (typeof parsed === "object" && parsed !== null) return parsed
      }
    } catch {}
    return { 0: [] }
  })
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthError, setHealthError] = useState("")
  const [lastRun, setLastRun] = useState<StreamChatResult | null>(null)
  const [draft, setDraft] = useState("")
  const [loading, setLoading] = useState(false)
  const [streamStart, setStreamStart] = useState<number | null>(null)
  const [tokenCount, setTokenCount] = useState(0)
  const [tokPerSec, setTokPerSec] = useState<number | null>(null)
  const [ttft, setTtft] = useState<number | null>(null)
  const [totalTokens, setTotalTokens] = useState({ prompt: 0, completion: 0 })
  const [connecting, setConnecting] = useState(false)
  const [logs, setLogs] = useState<Array<{time: string, type: "error" | "warn" | "info", message: string}>>([])
  const logRef = useRef<HTMLDivElement>(null)
  const [connected, setConnected] = useState(false)
  const [systemInstruction, setSystemInstruction] = useState("")
  const [view, setView] = useState<"chat" | "brain" | "models" | "converter" | "logs" | "presets" | "telemetry" | "doctor" | "workbench" | "benchmarks" | "security">(() => {
    const saved = stored(localStorage, "qwanto.view", "chat")
    return (["chat", "brain", "models", "converter", "logs", "presets", "telemetry", "doctor", "workbench", "benchmarks", "security"].includes(saved) ? saved : "chat") as any
  })

  const addLog = (type: "error" | "warn" | "info", message: string) => {
    const time = new Date().toLocaleTimeString()
    setLogs(prev => [...prev.slice(-200), { time, type, message }])
  }
  const [error, setError] = useState("")

  // Chat features: copy, regenerate, voice, attachments, web search
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const [isRecording, setIsRecording] = useState(false)
  const [attachments, setAttachments] = useState<Array<{name: string, type: string, data: string}>>([])
  const [webSearchEnabled, setWebSearchEnabled] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Model Manager states
  const [qwantoConfig, setQwantoConfig] = useState<QwantoConfig | null>(null)
  const [discoveredModels, setDiscoveredModels] = useState<DiscoveredModel[]>([])
  const [searchPaths, setSearchPaths] = useState<string[]>([])
  const [customModelPaths, setCustomModelPaths] = useState<string[]>([])
  const [newModelPath, setNewModelPath] = useState("")
  const [downloadStatus, setDownloadStatus] = useState<DownloadStatus | null>(null)
  const [customPath, setCustomPath] = useState("")
  const [downloadUrl, setDownloadUrl] = useState("")
  const [downloadPath, setDownloadPath] = useState(() => stored(localStorage, "qwanto.downloadPath", "D:\\Models"))
  const [downloadFilename, setDownloadFilename] = useState("")
  const [switchingModel, setSwitchingModel] = useState(false)
  const [modelError, setModelError] = useState("")
  const [dlError, setDlError] = useState("")
  const [deletingModel, setDeletingModel] = useState<string | null>(null)
  const [dlConnections, setDlConnections] = useState(8)
  const [dlSpeedLimit, setDlSpeedLimit] = useState(0)
  const [resourceCpu, setResourceCpu] = useState(() => Number(stored(localStorage, "qwanto.resCpu", "100")))
  const [resourceRam, setResourceRam] = useState(() => Number(stored(localStorage, "qwanto.resRam", "100")))
  const [resourceVram, setResourceVram] = useState(() => Number(stored(localStorage, "qwanto.resVram", "100")))
  const [resourceDisk, setResourceDisk] = useState(() => Number(stored(localStorage, "qwanto.resDisk", "100")))
  const autoConnected = useRef(false)
  const abortRef = useRef<AbortController | null>(null)
  const probeRef = useRef<AbortController | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const messages = conversations[cacheSlot] || []
  const kvSlots = Math.max(1, health?.kv_slots || 1)
  const active = activeRequests(health)
  const capacity = health?.scheduler?.capacity || kvSlots
  const failures = health?.scheduler ? health.scheduler.rejected + health.scheduler.timed_out + health.scheduler.cancelled : 0

  const updateMessages = (next: ChatMessage[] | ((current: ChatMessage[]) => ChatMessage[])) =>
    setConversations((current) => ({
      ...current,
      [cacheSlot]: typeof next === "function" ? next(current[cacheSlot] || []) : next,
    }))

  // EFFECT #1
  useEffect(() => {
    persistPublicSettings(localStorage, baseUrl, model)
  }, [baseUrl, model])

  // EFFECT: Save download path to localStorage
  useEffect(() => {
    try { localStorage.setItem("qwanto.downloadPath", downloadPath) } catch {}
  }, [downloadPath])

  // EFFECT: Save ctxSize to localStorage
  useEffect(() => {
    try { localStorage.setItem("qwanto.ctxSize", String(ctxSize)) } catch {}
  }, [ctxSize])

  // EFFECT: Persist acceleration settings
  useEffect(() => {
    try {
      localStorage.setItem("qwanto.flashAttention", String(flashAttention))
      localStorage.setItem("qwanto.kvCacheQuant", kvCacheQuant)
      localStorage.setItem("qwanto.specDecoding", String(specDecoding))
      localStorage.setItem("qwanto.draftModelPath", draftModelPath)
    } catch {}
  }, [flashAttention, kvCacheQuant, specDecoding, draftModelPath])

  const pickFolder = async () => {
    try {
      // Modern browsers: File System Access API
      if ("showDirectoryPicker" in window) {
        const dirHandle = await (window as any).showDirectoryPicker({ mode: "readwrite" })
        // Get the directory name - we can't get the full path for security, but we can use the name
        // For local usage, we'll prompt the user to confirm or type the path
        const name = dirHandle.name
        setDownloadPath(name)
        return
      }
    } catch (err) {
      // User cancelled or API not supported - fallback to text input
    }
  }

  // EFFECT #2
  useEffect(() => {
    setConnected(false)
    setHealth(null)
    setHealthError("")
  }, [baseUrl, apiKey])

  // EFFECT #3
  useEffect(() => () => {
    probeRef.current?.abort()
    abortRef.current?.abort()
  }, [])

  // EFFECT #4
  useEffect(() => {
    if (!connected) return
    let disposed = false
    const poll = async () => {
      if (document.visibilityState === "hidden") return
      try {
        const result = await getHealth(baseUrl, apiKey)
        if (!disposed) { setHealth(result); setHealthError("") }
      } catch (cause) {
        if (!disposed) setHealthError(cause instanceof Error ? cause.message : "Runtime metrics unavailable")
      }
    }
    const timer = window.setInterval(() => void poll(), 10000)
    return () => { disposed = true; window.clearInterval(timer) }
  }, [apiKey, baseUrl, connected])

  // EFFECT #5
  useEffect(() => {
    if (cacheSlot >= kvSlots) setCacheSlot(0)
  }, [cacheSlot, kvSlots])

  // EFFECT #6
  useEffect(() => { setLastRun(null) }, [cacheSlot])

  // EFFECT #6b: Persist active tab
  useEffect(() => { try { localStorage.setItem("qwanto.view", view) } catch {} }, [view])

  // EFFECT #6c: Persist conversations
  useEffect(() => { try { localStorage.setItem("qwanto.conversations", JSON.stringify(conversations)) } catch {} }, [conversations])

  // EFFECT #6d: Capture console errors
  useEffect(() => {
    const origError = console.error
    const origWarn = console.warn
    console.error = (...args: any[]) => {
      origError.apply(console, args)
      addLog("error", args.map(a => typeof a === "string" ? a : JSON.stringify(a)).join(" "))
    }
    console.warn = (...args: any[]) => {
      origWarn.apply(console, args)
      addLog("warn", args.map(a => typeof a === "string" ? a : JSON.stringify(a)).join(" "))
    }
    const onUnhandled = (e: PromiseRejectionEvent) => {
      addLog("error", `Unhandled: ${e.reason?.message || e.reason || "Unknown error"}`)
    }
    window.addEventListener("unhandledrejection", onUnhandled)
    return () => { console.error = origError; console.warn = origWarn; window.removeEventListener("unhandledrejection", onUnhandled) }
  }, [])

  // EFFECT #6c: Persist resource settings
  useEffect(() => {
    try {
      localStorage.setItem("qwanto.resCpu", String(resourceCpu))
      localStorage.setItem("qwanto.resRam", String(resourceRam))
      localStorage.setItem("qwanto.resVram", String(resourceVram))
      localStorage.setItem("qwanto.resDisk", String(resourceDisk))
    } catch {}
  }, [resourceCpu, resourceRam, resourceVram, resourceDisk])

  // EFFECT #7
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }) }, [messages])

  // EFFECT #8: Poll Qwanto configuration, discovered models, and download status
  useEffect(() => {
    if (!connected) return
    let disposed = false
    const pollModels = async () => {
      try {
        const result = await listDiscoveredModels(baseUrl, apiKey)
        if (!disposed) {
          setDiscoveredModels(result.models || [])
          setSearchPaths(result.search_paths || [])
        }
      } catch (err) { /* ignore */ }
    }
    const pollConfig = async () => {
      try {
        const config = await getQwantoConfig(baseUrl, apiKey)
        if (!disposed) setQwantoConfig(config)
      } catch (err) { /* ignore */ }
    }
    const pollPaths = async () => {
      try {
        const paths = await getModelPaths(baseUrl, apiKey)
        if (!disposed) setCustomModelPaths(paths)
      } catch (err) { /* ignore */ }
    }
    const pollDl = async () => {
      try {
        const status = await getDownloadStatus(baseUrl, apiKey)
        if (!disposed) {
          setDownloadStatus(status)
          if (status.status === "downloading" || status.status === "paused") {
            setTimeout(pollDl, 1000)
          }
        }
      } catch (err) { /* ignore */ }
    }
    
    pollModels()
    pollConfig()
    pollPaths()
    pollDl()
    
    const timer = setInterval(() => {
      pollModels()
      pollConfig()
      pollPaths()
    }, 15000)
    
    return () => {
      disposed = true
      clearInterval(timer)
    }
  }, [baseUrl, apiKey, connected])

  const connect = async () => {
    probeRef.current?.abort()
    const controller = new AbortController()
    probeRef.current = controller
    setConnecting(true)
    setError("")
    try {
      const found = await listModels(baseUrl, apiKey, controller.signal)
      setModels(found)
      if (found.length && !found.includes(model)) setModel(found[0])
      setConnected(true)
      try {
        setHealth(await getHealth(baseUrl, apiKey, controller.signal))
        setHealthError("")
      } catch (cause) {
        if (!controller.signal.aborted) {
          setHealth(null)
          setHealthError(cause instanceof Error ? cause.message : "Runtime metrics unavailable")
        }
      }
    } catch (cause) {
      if (controller.signal.aborted) return
      // If the server is reachable (loadModel succeeded) but listModels fails
      // (e.g. llama-cpp backend not ready), stay connected with a warning
      try {
        const health = await getHealth(baseUrl, apiKey, controller.signal)
        setConnected(true)
        setHealth(health)
        setError("")
        setHealthError("Model list unavailable — backend may still be starting")
      } catch {
        setConnected(false)
        setError(cause instanceof Error ? cause.message : "Could not reach the server.")
        addLog("error", `Connect: ${cause instanceof Error ? cause.message : cause}`)
      }
    } finally {
      if (probeRef.current === controller) { probeRef.current = null; setConnecting(false) }
    }
  }

  const handleLoadModel = async (modelPath: string, backend = "auto", backendUrl?: string) => {
    setSwitchingModel(true)
    setModelError("")
    try {
      const res = await loadModel(baseUrl, modelPath, backend, backendUrl, apiKey, ctxSize, {
        flashAttention,
        kvCacheQuant,
        speculativeDecoding: specDecoding,
        draftModelPath: specDecoding ? draftModelPath.trim() : "",
      })
      setModel(res.model_id)
      // Immediately refresh config and discovered models
      try {
        const [config, result] = await Promise.all([
          getQwantoConfig(baseUrl, apiKey),
          listDiscoveredModels(baseUrl, apiKey)
        ])
        setQwantoConfig(config)
        setDiscoveredModels(result.models || [])
        setSearchPaths(result.search_paths || [])
      } catch { /* polling will catch up */ }
      await connect()
    } catch (err) {
      setModelError(err instanceof Error ? err.message : "Failed to load model")
      addLog("error", `Load model: ${err instanceof Error ? err.message : err}`)
    } finally {
      setSwitchingModel(false)
    }
  }

  const handleStartDownload = async (url: string, filename?: string, destPath?: string) => {
    setDlError("")
    try {
      await downloadModel(baseUrl, url, filename, destPath, apiKey)
      const status = await getDownloadStatus(baseUrl, apiKey)
      setDownloadStatus(status)
    } catch (err) {
      setDlError(err instanceof Error ? err.message : "Failed to start download")
      addLog("error", `Download: ${err instanceof Error ? err.message : err}`)
    }
  }

  const handleCancelDownload = async () => {
    try {
      await cancelDownloadModel(baseUrl, apiKey)
      const status = await getDownloadStatus(baseUrl, apiKey)
      setDownloadStatus(status)
    } catch (err) {
      setDlError(err instanceof Error ? err.message : "Failed to cancel download")
    }
  }

  const handlePauseDownload = async () => {
    try {
      await pauseDownloadModel(baseUrl, apiKey)
      const status = await getDownloadStatus(baseUrl, apiKey)
      setDownloadStatus(status)
    } catch (err) {
      setDlError(err instanceof Error ? err.message : "Failed to pause download")
    }
  }

  const handleResumeDownload = async () => {
    try {
      await resumeDownloadModel(baseUrl, apiKey)
      const status = await getDownloadStatus(baseUrl, apiKey)
      setDownloadStatus(status)
    } catch (err) {
      setDlError(err instanceof Error ? err.message : "Failed to resume download")
    }
  }

  const handleDeleteModel = async (path: string, name: string) => {
    if (!confirm(`Delete "${name}"? This cannot be undone.`)) return
    setDeletingModel(path)
    try {
      await deleteModel(baseUrl, path, apiKey)
      const result = await listDiscoveredModels(baseUrl, apiKey)
      setDiscoveredModels(result.models || [])
      setSearchPaths(result.search_paths || [])
    } catch (err) {
      setModelError(err instanceof Error ? err.message : "Failed to delete model")
    } finally {
      setDeletingModel(null)
    }
  }

  const handleConfigDownload = async (connections?: number, speedLimit?: number) => {
    try {
      await configDownload(baseUrl, connections, speedLimit, apiKey)
    } catch { /* ignore */ }
  }

  const handleAddModelPath = async () => {
    if (!newModelPath.trim()) return
    try {
      await addModelPath(baseUrl, newModelPath.trim(), apiKey)
      setNewModelPath("")
      const paths = await getModelPaths(baseUrl, apiKey)
      setCustomModelPaths(paths)
      const result = await listDiscoveredModels(baseUrl, apiKey)
      setDiscoveredModels(result.models || [])
      setSearchPaths(result.search_paths || [])
    } catch (err) {
      setModelError(err instanceof Error ? err.message : "Failed to add path")
    }
  }

  const handleRemoveModelPath = async (path: string) => {
    try {
      await removeModelPath(baseUrl, path, apiKey)
      const paths = await getModelPaths(baseUrl, apiKey)
      setCustomModelPaths(paths)
      const result = await listDiscoveredModels(baseUrl, apiKey)
      setDiscoveredModels(result.models || [])
      setSearchPaths(result.search_paths || [])
    } catch (err) {
      setModelError(err instanceof Error ? err.message : "Failed to remove path")
    }
  }

  if (servedByEngine && !autoConnected.current && !connected) {
    autoConnected.current = true
    setTimeout(() => connect(), 0)
  }

  const canSend = useMemo(() => draft.trim() && model && !loading, [draft, loading, model])

  const send = async () => {
    const content = draft.trim()
    if (!content || loading) return
    let fullContent = content
    // Prepend web search results if enabled
    if (webSearchEnabled && searchQuery.trim()) {
      fullContent = `[Web search query: "${searchQuery.trim()}"]\n${fullContent}`
    }
    // Add attachments as text references
    if (attachments.length > 0) {
      const attachText = attachments.map(a => `[Attached: ${a.name} (${a.type})]`).join("\n")
      fullContent = `${attachText}\n${fullContent}`
    }
    const user = message("user", fullContent)
    const assistant = message("assistant", "")
    const history = [...messages, user]
    setDraft("")
    setSearchQuery("")
    setAttachments([])
    setError("")
    updateMessages([...history, assistant])
    setLoading(true)
    setStreamStart(null)
    setTokenCount(0)
    setTokPerSec(null)
    setTtft(null)
    const t0 = performance.now()
    let firstToken = true
    let count = 0
    const controller = new AbortController()
    abortRef.current = controller
    try {
      const result = await streamChat({
        baseUrl,
        apiKey,
        model,
        messages: history,
        temperature,
        maxTokens,
        enableThinking: thinking,
        cacheSlot: supportsCacheSlots(health) ? cacheSlot : undefined,
        signal: controller.signal,
        onDelta: (delta) => {
          if (firstToken) { setTtft(performance.now() - t0); setStreamStart(performance.now()); firstToken = false }
          count++
          setTokenCount(count)
          const elapsed = (performance.now() - (firstToken ? t0 : t0)) / 1000
          if (elapsed > 0.3) setTokPerSec(count / ((performance.now() - t0) / 1000))
          updateMessages((current) => current.map((item) =>
            item.id === assistant.id ? { ...item, content: item.content + delta } : item,
          ))
        },
      })
      const finalElapsed = (performance.now() - t0) / 1000
      if (count > 0 && finalElapsed > 0) setTokPerSec(count / finalElapsed)
      if (result.usage) setTotalTokens(prev => ({
        prompt: prev.prompt + (result.usage?.prompt_tokens || 0),
        completion: prev.completion + (result.usage?.completion_tokens || 0),
      }))
      setLastRun(result)
      setConnected(true)
    } catch (cause) {
      if (controller.signal.aborted) {
        updateMessages((current) => current.filter((item) => item.id !== assistant.id || item.content))
      } else {
        setError(cause instanceof Error ? cause.message : "Generation failed.")
        addLog("error", `Generation: ${cause instanceof Error ? cause.message : cause}`)
        updateMessages((current) => current.filter((item) => item.id !== assistant.id || item.content))
      }
    } finally {
      abortRef.current = null
      setLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-row">
          <div className="brand-mark"><Feather className="size-5" /></div>
          <div><h1>Qwanto</h1><p>local giant · powered by Colibrì</p></div>
        </div>

        <section className="side-section">
          <div className="section-title"><Link2 className="size-3.5" /> Connection</div>
          <label>API endpoint<Input value={baseUrl} onChange={(event) => setBaseUrl(event.target.value)} /></label>
          <label>API key<div className="relative"><KeyRound className="field-icon" /><Input className="pl-9" type="password" value={apiKey} placeholder="optional" onChange={(event) => setApiKey(event.target.value)} /></div><span className="field-help">Kept in memory only · sent to this endpoint</span></label>
          <Button type="button" variant="secondary" onClick={connect} disabled={connecting}>
            {connecting ? <LoaderCircle className="size-4 animate-spin" /> : <RefreshCw className="size-4" />}
            Probe server
          </Button>
          <div className={cn("connection-state", connected && "connected")} aria-live="polite"><span />{connected ? "Engine reachable" : "Not connected"}</div>
        </section>

        <section className="side-section runtime-section" aria-live="polite">
          <div className="section-title"><Activity className="size-3.5" /> Runtime</div>
          {health?.hwinfo ? <div className="hw-panel">
            {health.hwinfo.cpu ? <div className="hw-row"><Cpu className="size-3.5" /><span>{health.hwinfo.cpu}</span></div> : null}
            {health.hwinfo.gpus > 0 ? <div className="hw-row"><MonitorDot className="size-3.5" /><span>{health.hwinfo.gpus}× GPU<small>{health.hwinfo.vram_total_gb.toFixed(0)} GB VRAM</small></span></div> : null}
            <div className="hw-row"><MemoryStick className="size-3.5" /><span>{health.hwinfo.ram_total_gb.toFixed(0)} GB RAM<small>{health.hwinfo.ram_avail_gb.toFixed(0)} GB free</small></span></div>
            <div className="hw-row"><HardDrive className="size-3.5" /><span>{health.hwinfo.cores} cores</span></div>
          </div> : null}
          {health?.scheduler ? <>
            <div className="runtime-grid">
              <div><span>Active</span><strong>{active}<small> / {capacity}</small></strong></div>
              <div><span>Queued</span><strong>{health.scheduler.queued}<small> / {health.scheduler.max_queue}</small></strong></div>
              <div><span>Completed</span><strong>{health.scheduler.completed}</strong></div>
              <div><span>Failures</span><strong>{failures}</strong></div>
            </div>
            {health.tiers ? (() => {
              const t = health.tiers
              const total = Math.max(t.vram + t.ram + t.disk, 1)
              return <div className="tier-panel">
                <div className="tier-bar" role="img" aria-label={`Experts: ${t.vram} VRAM, ${t.ram} RAM, ${t.disk} disk`}>
                  <span className="tier-vram" style={{ width: `${(100 * t.vram) / total}%` }} />
                  <span className="tier-ram" style={{ width: `${(100 * t.ram) / total}%` }} />
                  <span className="tier-disk" style={{ width: `${(100 * t.disk) / total}%` }} />
                </div>
                <div className="tier-legend">
                  <span><i className="tier-vram" />VRAM <strong>{t.vram.toLocaleString()}</strong><small>{t.vram_gb.toFixed(1)} GB</small></span>
                  <span><i className="tier-ram" />RAM <strong>{t.ram.toLocaleString()}</strong><small>{t.ram_gb.toFixed(1)} GB</small></span>
                  <span><i className="tier-disk" />Disk <strong>{t.disk.toLocaleString()}</strong></span>
                </div>
              </div>
            })() : null}
            {totalTokens.prompt + totalTokens.completion > 0 ? <div className="session-stats">
              <span><Database className="size-3" /> Session: <strong>{totalTokens.prompt.toLocaleString()}</strong> prompt + <strong>{totalTokens.completion.toLocaleString()}</strong> completion</span>
            </div> : null}
            <div className="runtime-foot"><span className="runtime-dot" /> Scheduler online <code>{kvSlots} KV</code></div>
          </> : <p className="runtime-unavailable">{connected ? (healthError || "Runtime metrics unavailable") : "Probe the server to inspect runtime state."}</p>}
        </section>

        <section className="side-section">
          <div className="section-title"><SlidersHorizontal className="size-3.5" /> Inference</div>
          <label>Model<select value={model} onChange={(event) => setModel(event.target.value)}>{models.length ? models.map((id) => <option key={id}>{id}</option>) : <option>{model}</option>}</select></label>
          {health?.kv_slots && health.kv_slots > 1 ? <label>KV session<select value={cacheSlot} onChange={(event) => setCacheSlot(Number(event.target.value))} disabled={loading}>
            {Array.from({ length: kvSlots }, (_, slot) => <option key={slot} value={slot}>Session {slot + 1}</option>)}
          </select><span className="field-help">Isolated context · conversation follows the selected slot</span></label> : null}
          <label><span className="label-line"><span>Temperature</span><code>{temperature.toFixed(1)}</code></span><input className="range" type="range" min="0" max="2" step="0.1" value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} /></label>
          <label>Max output tokens<Input type="number" min={1} max={4096} value={maxTokens} onChange={(event) => { const value = Number(event.target.value); if (Number.isFinite(value)) setMaxTokens(Math.min(4096, Math.max(1, Math.round(value)))) }} /></label>
          <label>Context size<Input type="number" min={512} max={2000000} step={1024} value={ctxSize} onChange={(event) => { const value = Number(event.target.value); if (Number.isFinite(value)) setCtxSize(Math.min(2000000, Math.max(512, Math.round(value)))) }} /><span className="field-help">Restart server to apply</span></label>
          <button type="button" className={cn("toggle-row", thinking && "active")} aria-pressed={thinking} onClick={() => setThinking((value) => !value)}>
            <span><BrainCircuit className="size-4" /> Reasoning</span><i><b /></i>
          </button>

          <div className="section-title"><Zap className="size-3.5" /> Acceleration</div>
          <button type="button" className={cn("toggle-row", flashAttention && "active")} aria-pressed={flashAttention} onClick={() => setFlashAttention((value) => !value)}>
            <span><Zap className="size-4" /> Flash Attention</span><i><b /></i>
          </button>
          <label>KV cache quantization<select value={kvCacheQuant} onChange={(event) => setKvCacheQuant(event.target.value)}>
            <option value="f16">f16 (off — max precision)</option>
            <option value="q8_0">q8_0 (½ memory)</option>
            <option value="q4_0">q4_0 (¼ memory)</option>
          </select><span className="field-help">Smaller KV cache → more layers fit in VRAM</span></label>
          <button type="button" className={cn("toggle-row", specDecoding && "active")} aria-pressed={specDecoding} onClick={() => setSpecDecoding((value) => !value)}>
            <span><Waypoints className="size-4" /> Speculative decoding</span><i><b /></i>
          </button>
          {specDecoding && <label>Draft model (GGUF)<Input type="text" placeholder="path\to\small-draft.gguf" value={draftModelPath} onChange={(event) => setDraftModelPath(event.target.value)} /><span className="field-help">Small same-family model · 2–3× faster generation</span></label>}
          <span className="field-help">Applied on model load / reload</span>
        </section>

        <div className="sidebar-foot"><Cpu className="size-3.5" /><span>OpenAI-compatible transport</span></div>
      </aside>

      <main className="chat-panel">
        <header className="topbar">
          <div className="flex items-center gap-2">
            <div><span className="eyebrow">ACTIVE MODEL</span><strong>{model}</strong></div>
            {model.toLowerCase().endsWith(".qwn") && (
              <Badge className="bg-emerald-950/80 text-emerald-300 border-emerald-800/60 font-mono text-[10px] gap-1">
                <Zap className="size-3 text-emerald-400" /> QWN NATIVE
              </Badge>
            )}
          </div>
          <div className="view-tabs">
            <button className={view === "chat" ? "active" : ""} onClick={() => setView("chat")}><MessageSquareText className="size-3.5" /> Chat</button>
            <button className={view === "converter" ? "active" : ""} onClick={() => setView("converter")}><Zap className="size-3.5" /> Converter</button>
            <button className={view === "presets" ? "active" : ""} onClick={() => setView("presets")}><Sparkles className="size-3.5" /> Studio</button>
            <button className={view === "telemetry" ? "active" : ""} onClick={() => setView("telemetry")}><Activity className="size-3.5" /> Telemetry</button>
            <button className={view === "benchmarks" ? "active" : ""} onClick={() => setView("benchmarks")}><BarChart3 className="size-3.5" /> Benchmarks</button>
            <button className={view === "security" ? "active" : ""} onClick={() => setView("security")}><Lock className="size-3.5" /> Security</button>
            <button className={view === "workbench" ? "active" : ""} onClick={() => setView("workbench")}><Code2 className="size-3.5" /> Workbench</button>
            <button className={view === "doctor" ? "active" : ""} onClick={() => setView("doctor")}><ShieldCheck className="size-3.5" /> Doctor</button>
            <button className={view === "brain" ? "active" : ""} onClick={() => setView("brain")}><BrainCircuit className="size-3.5" /> Brain</button>
            <button className={view === "models" ? "active" : ""} onClick={() => setView("models")}><Server className="size-3.5" /> Models</button>
            <button className={view === "logs" ? "active" : ""} onClick={() => setView("logs")}><Database className="size-3.5" /> Logs {logs.length > 0 && <span className="logs-badge">{logs.filter(l => l.type === "error").length || logs.length}</span>}</button>
          </div>
          <div className="top-actions">
              {model.toLowerCase().endsWith(".qwn") && (
                <Badge className="border border-emerald-800/50 bg-emerald-950/30 text-emerald-300 font-mono text-[10px]">
                  4KiB NVMe Paged · AVX2
                </Badge>
              )}
              {loading && tokenCount === 0 ? <Badge className="badge-loading"><LoaderCircle className="size-3 animate-spin" /> Generating...</Badge> : null}
              {loading && tokenCount > 0 ? <Badge className="badge-live"><Zap className="size-3 flash" /> {tokenCount} tokens · {tokPerSec ? `${tokPerSec.toFixed(1)} tok/s` : "..."}</Badge> : null}
              {!loading && tokPerSec != null ? <Badge className="badge-speed"><Gauge className="size-3" /> {tokPerSec.toFixed(1)} tok/s</Badge> : null}
              {!loading && ttft != null ? <Badge><Timer className="size-3" /> TTFT {(ttft/1000).toFixed(1)}s</Badge> : null}
              {!loading && lastRun?.usage ? <Badge><Layers className="size-3" /> {lastRun.usage.prompt_tokens}→{lastRun.usage.completion_tokens} ({lastRun.usage.total_tokens} total)</Badge> : null}
              {lastRun?.queueWaitMs != null ? <Badge><Clock className="size-3" /> queue {Math.round(lastRun.queueWaitMs)}ms</Badge> : null}
              <Badge><MonitorDot className="size-3" /> slot {cacheSlot + 1}</Badge>
              <Button variant="ghost" size="sm" onClick={() => { updateMessages([]); setTokPerSec(null); setTtft(null); setTokenCount(0); setTotalTokens({prompt:0,completion:0}) }} disabled={!messages.length || loading}><Trash2 className="size-3.5" /> Clear</Button>
            </div>
            {loading && <div className="loading-progress-bar"><div className="loading-progress-fill" /></div>}
        </header>

        {view === "models" ? (
          <div className="models-page">
            <div className="models-grid">
              {/* Active Model Configuration Card */}
              <div className="models-card active-card">
                <h3 className="card-title"><Server className="size-4" /> Active Model Status</h3>
                {switchingModel && <div className="loading-overlay"><LoaderCircle className="size-6 animate-spin text-primary" /> <span>Loading Model...</span></div>}
                
                <div className="status-item">
                  <span className="label">Active Model ID:</span>
                  <span className="value font-mono text-primary">{model}</span>
                </div>
                <div className="status-item">
                  <span className="label">Model Path:</span>
                  <span className="value font-mono">{qwantoConfig?.model_path || "N/A"}</span>
                </div>
                <div className="status-item">
                  <span className="label">Backend Engine:</span>
                  <span className="value font-bold uppercase text-blue-400">{qwantoConfig?.backend || (health?.scheduler ? "native" : "N/A")}</span>
                </div>
                {qwantoConfig?.proxy_url && (
                  <div className="status-item">
                    <span className="label">Proxy URL:</span>
                    <span className="value font-mono">{qwantoConfig.proxy_url}</span>
                  </div>
                )}
                
                {modelError && <div className="error-banner mt-3">{modelError}</div>}
                
                {/* Custom Model Path Form */}
                <div className="custom-path-box mt-4">
                  <span className="label font-bold text-xs text-muted-foreground uppercase mb-2 block">Load Custom Model Path</span>
                  <div className="flex gap-2">
                    <Input placeholder="e.g. D:\models\llama-3-8b.gguf" value={customPath} onChange={e => setCustomPath(e.target.value)} />
                    <Button onClick={() => handleLoadModel(customPath)} disabled={!customPath || switchingModel}>
                      <FolderSync className="size-4 mr-2" /> Load
                    </Button>
                  </div>
                </div>
              </div>

              {/* Resource Control Card */}
              <div className="models-card">
                <h3 className="card-title"><Gauge className="size-4" /> Resource Limits</h3>
                <div className="text-xs text-muted-foreground mb-3">Control hardware usage. Higher = faster inference.</div>
                
                {resourceVram === 0 && (
                  <div className="text-[10px] p-2 mb-3 rounded border border-yellow-800 bg-yellow-950/30 text-yellow-400">
                    ⚠ VRAM at 0% = GPU disabled. Model runs on CPU only (much slower). Set VRAM ≥ 25% to use your RTX 5070 Ti.
                  </div>
                )}
                
                <div className="resource-slider mb-3">
                  <div className="flex justify-between text-xs mb-1">
                    <span className="flex items-center gap-1"><Cpu className="size-3" /> CPU Threads</span>
                    <span className="font-mono text-primary">{resourceCpu}%</span>
                  </div>
                  <input type="range" min="10" max="100" step="5" value={resourceCpu} onChange={e => { const v = Number(e.target.value); setResourceCpu(v); setResourceLimits(baseUrl, { cpu: v, ram: resourceRam, vram: resourceVram, disk: resourceDisk }, apiKey).catch(() => {}) }} className="w-full" />
                  <div className="text-[9px] text-muted-foreground mt-0.5">{Math.ceil(32 * resourceCpu / 100)} of 32 threads</div>
                </div>

                <div className="resource-slider mb-3">
                  <div className="flex justify-between text-xs mb-1">
                    <span className="flex items-center gap-1"><MemoryStick className="size-3" /> RAM Cache</span>
                    <span className="font-mono text-primary">{resourceRam}%</span>
                  </div>
                  <input type="range" min="5" max="100" step="5" value={resourceRam} onChange={e => { const v = Number(e.target.value); setResourceRam(v); setResourceLimits(baseUrl, { cpu: resourceCpu, ram: v, vram: resourceVram, disk: resourceDisk }, apiKey).catch(() => {}) }} className="w-full" />
                  <div className="text-[9px] text-muted-foreground mt-0.5">~{Math.round(31.1 * resourceRam / 100)} GB of 31.1 GB</div>
                </div>

                <div className="resource-slider mb-3">
                  <div className="flex justify-between text-xs mb-1">
                    <span className="flex items-center gap-1"><Waypoints className="size-3" /> GPU (VRAM)</span>
                    <span className={cn("font-mono", resourceVram === 0 ? "text-yellow-400" : "text-primary")}>{resourceVram}%</span>
                  </div>
                  <input type="range" min="0" max="100" step="5" value={resourceVram} onChange={e => { const v = Number(e.target.value); setResourceVram(v); setResourceLimits(baseUrl, { cpu: resourceCpu, ram: resourceRam, vram: v, disk: resourceDisk }, apiKey).catch(() => {}) }} className="w-full" />
                  <div className="text-[9px] text-muted-foreground mt-0.5">{resourceVram === 0 ? "GPU disabled" : `~${Math.round(12 * resourceVram / 100)} GB of 12 GB VRAM`}</div>
                </div>

                <div className="resource-slider mb-3">
                  <div className="flex justify-between text-xs mb-1">
                    <span className="flex items-center gap-1"><HardDrive className="size-3" /> Disk I/O</span>
                    <span className="font-mono text-primary">{resourceDisk}%</span>
                  </div>
                  <input type="range" min="10" max="100" step="5" value={resourceDisk} onChange={e => { const v = Number(e.target.value); setResourceDisk(v); setResourceLimits(baseUrl, { cpu: resourceCpu, ram: resourceRam, vram: resourceVram, disk: v }, apiKey).catch(() => {}) }} className="w-full" />
                </div>

                <div className="text-[10px] text-muted-foreground mt-2 p-2 bg-[#0a0f12] rounded border border-border">
                  <div className="font-bold mb-1">Performance: ~{((resourceCpu + resourceRam + (resourceVram * 2) + resourceDisk) / 5).toFixed(0)}%</div>
                  <div>GPU counts 2× for speed. Changes apply on next model load.</div>
                </div>
              </div>

              {/* Local Discovered Models List */}
              <div className="models-card">
                <h3 className="card-title"><HardDrive className="size-4" /> Discovered Local Models</h3>
                
                {/* Search Paths */}
                {searchPaths.length > 0 && (
                  <div className="mb-3 p-2 bg-[#0a0f12] rounded border border-border">
                    <div className="text-xs text-muted-foreground mb-1">Searching in:</div>
                    <div className="flex flex-wrap gap-1">
                      {searchPaths.map((p, i) => (
                        <span key={i} className="text-[10px] font-mono bg-[#151c20] px-1.5 py-0.5 rounded">{p}</span>
                      ))}
                    </div>
                  </div>
                )}
                
                <div className="models-list-scroll">
                  {!discoveredModels.length ? (
                    <div className="text-muted-foreground text-xs p-4 text-center">No local models found. Add a search path below or check your download directory.</div>
                  ) : (
                    discoveredModels.map((m, idx) => (
                      <div key={idx} className={cn("model-list-item", (qwantoConfig?.model_path === m.path) && "active")}>
                        <div className="model-info">
                          <span className="model-name font-mono">{m.name}</span>
                          <span className="model-path font-mono text-xs text-muted-foreground">{m.path}</span>
                        </div>
                        <div className="model-actions flex gap-2 items-center">
                          <Badge className="uppercase font-mono text-[9px] bg-[#151c20] text-muted-foreground">{m.type}</Badge>
                          <Button 
                            size="sm" 
                            variant={(qwantoConfig?.model_path === m.path) ? "ghost" : "secondary"}
                            disabled={qwantoConfig?.model_path === m.path || switchingModel}
                            onClick={() => handleLoadModel(m.path, m.type === "gguf" ? "llama-cpp" : "native")}
                          >
                            {(qwantoConfig?.model_path === m.path) ? "✓ Active" : "Switch"}
                          </Button>
                          <Button 
                            size="sm" 
                            variant="ghost"
                            className="text-red-400 hover:text-red-300 hover:bg-red-950"
                            disabled={qwantoConfig?.model_path === m.path || deletingModel === m.path}
                            onClick={() => handleDeleteModel(m.path, m.name)}
                          >
                            <Trash2 className="size-3" />
                          </Button>
                        </div>
                      </div>
                    ))
                  )}
                </div>
                
                {/* Custom Search Paths */}
                <div className="mt-3 border-t border-border pt-3">
                  <div className="text-xs text-muted-foreground mb-2">Custom Search Paths:</div>
                  {customModelPaths.length > 0 && (
                    <div className="flex flex-wrap gap-1 mb-2">
                      {customModelPaths.map((p, i) => (
                        <span key={i} className="flex items-center gap-1 text-[10px] font-mono bg-[#151c20] px-1.5 py-0.5 rounded">
                          {p}
                          <button onClick={() => handleRemoveModelPath(p)} className="text-red-400 hover:text-red-300 ml-1">×</button>
                        </span>
                      ))}
                    </div>
                  )}
                  <div className="flex gap-2">
                    <Input 
                      placeholder="Add folder path (e.g. D:\MyModels)" 
                      value={newModelPath} 
                      onChange={e => setNewModelPath(e.target.value)}
                      className="flex-1"
                    />
                    <Button variant="secondary" size="sm" onClick={handleAddModelPath} disabled={!newModelPath.trim()}>
                      Add
                    </Button>
                  </div>
                  <div className="text-[10px] text-muted-foreground mt-1">Also scans: download destination, active model's folder, QWANTO_MODEL_PATHS env var</div>
                </div>
              </div>

              {/* Background Model Downloader Card */}
              <div className="models-card md:col-span-2">
                <h3 className="card-title"><Download className="size-4" /> Model Downloader</h3>
                {downloadStatus && (downloadStatus.status === "downloading" || downloadStatus.status === "paused") ? (
                  <div className="download-progress-box">
                    <div className="progress-details mb-2">
                      <span className="filename font-bold block mb-1 text-primary">{downloadStatus.filename}</span>
                      <div className="flex justify-between text-xs text-muted-foreground mb-1">
                        <span>{downloadStatus.status === "paused" ? "⏸ Paused" : "⬇ Downloading"}</span>
                        <span className="font-mono text-primary">{downloadStatus.speed.toFixed(2)} MB/s</span>
                        <span>{(downloadStatus.downloaded / (1024 * 1024 * 1024)).toFixed(2)} / {(downloadStatus.total / (1024 * 1024 * 1024)).toFixed(2)} GB</span>
                      </div>
                      {downloadStatus.chunks_total > 0 && (
                        <div className="flex justify-between text-xs text-muted-foreground mb-1">
                          <span>Chunks: {downloadStatus.chunks_done}/{downloadStatus.chunks_total}</span>
                          <span>Connections: {downloadStatus.connections}</span>
                          {downloadStatus.speed_limit > 0 && <span>Limit: {(downloadStatus.speed_limit / (1024 * 1024)).toFixed(1)} MB/s</span>}
                        </div>
                      )}
                    </div>
                    <div className="progress-bar-container mb-3">
                      <div className={cn("progress-bar-fill", downloadStatus.status === "paused" && "paused")} style={{ width: `${downloadStatus.progress}%` }}></div>
                    </div>
                    {/* Download Controls */}
                    <div className="flex flex-wrap gap-2 mb-3">
                      {downloadStatus.status === "downloading" ? (
                        <Button variant="secondary" size="sm" onClick={handlePauseDownload}>⏸ Pause</Button>
                      ) : (
                        <Button variant="secondary" size="sm" onClick={handleResumeDownload}>▶ Resume</Button>
                      )}
                      <Button variant="destructive" size="sm" onClick={handleCancelDownload}>✕ Cancel</Button>
                    </div>
                    {/* Speed Controls */}
                    <div className="flex flex-wrap gap-3 items-center text-xs text-muted-foreground">
                      <div className="flex items-center gap-1">
                        <span>Connections:</span>
                        <select 
                          className="bg-input border border-border rounded px-1 py-0.5 text-xs"
                          value={downloadStatus.connections || 8}
                          onChange={e => { setDlConnections(Number(e.target.value)); handleConfigDownload(Number(e.target.value), undefined) }}
                        >
                          {[1,2,4,8,16,32].map(n => <option key={n} value={n}>{n}</option>)}
                        </select>
                      </div>
                      <div className="flex items-center gap-1">
                        <span>Speed limit:</span>
                        <select 
                          className="bg-input border border-border rounded px-1 py-0.5 text-xs"
                          value={downloadStatus.speed_limit || 0}
                          onChange={e => { setDlSpeedLimit(Number(e.target.value)); handleConfigDownload(undefined, Number(e.target.value)) }}
                        >
                          <option value={0}>Unlimited</option>
                          <option value={5242880}>5 MB/s</option>
                          <option value={10485760}>10 MB/s</option>
                          <option value={20971520}>20 MB/s</option>
                          <option value={52428800}>50 MB/s</option>
                          <option value={104857600}>100 MB/s</option>
                        </select>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="download-form flex flex-col gap-3">
                    <div className="flex gap-2">
                      <Input 
                        placeholder="Model Download URL (HuggingFace direct link or standard GGUF link)" 
                        value={downloadUrl} 
                        onChange={e => {
                          setDownloadUrl(e.target.value)
                          // Auto-detect filename from URL
                          if (e.target.value) {
                            try {
                              const url = new URL(e.target.value)
                              const pathParts = url.pathname.split("/")
                              const lastPart = pathParts[pathParts.length - 1]
                              if (lastPart && lastPart.includes(".")) {
                                const decoded = decodeURIComponent(lastPart).split("?")[0]
                                setDownloadFilename(decoded)
                              }
                            } catch {}
                          }
                        }} 
                      />
                      <Button onClick={() => handleStartDownload(downloadUrl, downloadFilename, downloadPath || undefined)} disabled={!downloadUrl}>
                        Download
                      </Button>
                    </div>
                    <div className="flex gap-2 items-center">
                      <Input 
                        placeholder="Filename (auto-detected from URL)" 
                        value={downloadFilename} 
                        onChange={e => setDownloadFilename(e.target.value)}
                        className="flex-1"
                      />
                      <Input 
                        placeholder="Save to folder (e.g. D:\Models)" 
                        value={downloadPath} 
                        onChange={e => setDownloadPath(e.target.value)} 
                        className="flex-1"
                      />
                      <Button 
                        variant="secondary" 
                        size="sm" 
                        onClick={pickFolder}
                        title="Browse for folder"
                      >
                        <FolderSync className="size-4" />
                      </Button>
                    </div>
                    {downloadStatus && downloadStatus.status === "completed" && (
                      <div className="success-banner">Download Completed: {downloadStatus.filename} was successfully downloaded!</div>
                    )}
                    {downloadStatus && downloadStatus.status === "error" && (
                      <div className="error-banner">Download Error: {downloadStatus.error}</div>
                    )}
                    {dlError && <div className="error-banner">{dlError}</div>}
                    
                    <div className="presets-section mt-2">
                      <span className="font-bold text-xs text-muted-foreground uppercase mb-2 block">Popular Presets</span>
                      <div className="presets-grid flex flex-wrap gap-2">
                        <button 
                          className="preset-btn"
                          onClick={() => {
                            setDownloadUrl("https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf")
                            setDownloadFilename("qwen2.5-7b-instruct-q4_k_m.gguf")
                          }}
                        >
                          Qwen 2.5 7B Q4_K_M GGUF
                        </button>
                        <button 
                          className="preset-btn"
                          onClick={() => {
                            setDownloadUrl("https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf")
                            setDownloadFilename("Meta-Llama-3-8B-Instruct-Q4_K_M.gguf")
                          }}
                        >
                          Llama 3 8B Q4_K_M GGUF
                        </button>
                        <button 
                          className="preset-btn"
                          onClick={() => {
                            setDownloadUrl("https://huggingface.co/microsoft/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf")
                            setDownloadFilename("Phi-3.5-mini-instruct-Q4_K_M.gguf")
                          }}
                        >
                          Phi 3.5 Mini Q4_K_M GGUF
                        </button>
                        <button 
                          className="preset-btn"
                          onClick={() => {
                            setDownloadUrl("https://huggingface.co/giladgd/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M-GGUF/resolve/main/qwen3-coder-30b-a3b-instruct-q4_k_m.gguf")
                            setDownloadFilename("qwen3-coder-30b-a3b-instruct-q4_k_m.gguf")
                          }}
                        >
                          Qwen3 Coder 30B Q4_K_M
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : view === "presets" ? (
          <PresetsView
            baseUrl={baseUrl}
            apiKey={apiKey}
            onApplyPreset={(preset: SystemPreset) => {
              if (preset.system_prompt) setSystemInstruction(preset.system_prompt)
              setTemperature(preset.temperature)
              addLog("info", `Applied preset: ${preset.name} (temp=${preset.temperature}, top_p=${preset.top_p})`)
            }}
          />
        ) : view === "telemetry" ? (
          <TelemetryView baseUrl={baseUrl} apiKey={apiKey} />
        ) : view === "benchmarks" ? (
          <BenchmarksView baseUrl={baseUrl} apiKey={apiKey} />
        ) : view === "security" ? (
          <SecurityView baseUrl={baseUrl} apiKey={apiKey} />
        ) : view === "workbench" ? (
          <WorkbenchView
            baseUrl={baseUrl}
            model={model}
            apiKey={apiKey}
            temperature={temperature}
            maxTokens={maxTokens}
          />
        ) : view === "converter" ? (
          <ConverterView
            baseUrl={baseUrl}
            apiKey={apiKey}
            onModelLoaded={(loadedPath) => {
              setModel(loadedPath)
              getHealth(baseUrl, apiKey).then(h => setHealth(h)).catch(() => {})
            }}
            onNavigateToChat={() => setView("chat")}
          />
        ) : view === "doctor" ? (
          <DoctorView baseUrl={baseUrl} apiKey={apiKey} />
        ) : view === "brain" ? (
          <Brain baseUrl={baseUrl} apiKey={apiKey} connected={connected} />
        ) : view === "logs" ? (
          <div className="logs-page">
            <div className="logs-header">
              <h3 className="card-title"><Database className="size-4" /> Live Logs ({logs.length})</h3>
              <div className="flex gap-2">
                <Button variant="secondary" size="sm" onClick={() => { navigator.clipboard.writeText(logs.map(l => `[${l.time}] [${l.type.toUpperCase()}] ${l.message}`).join("\n")) }}>
                  Copy All
                </Button>
                <Button variant="ghost" size="sm" onClick={() => setLogs([])}>
                  <Trash2 className="size-3.5" /> Clear
                </Button>
              </div>
            </div>
            <div className="logs-list" ref={logRef}>
              {logs.length === 0 ? (
                <div className="text-muted-foreground text-sm p-8 text-center">No logs yet. Errors and warnings will appear here.</div>
              ) : (
                logs.map((log, i) => (
                  <div key={i} className={`log-entry log-${log.type}`}>
                    <span className="log-time">{log.time}</span>
                    <span className={`log-type log-type-${log.type}`}>{log.type.toUpperCase()}</span>
                    <span className="log-msg">{log.message}</span>
                  </div>
                ))
              )}
              <div ref={bottomRef} />
            </div>
          </div>
        ) : (
          <>
            <div className="conversation">
              {!messages.length ? (
                <div className="empty-state">
                  <div className="orb"><Feather /></div>
                  <span className="eyebrow">QWANTO ENGINE</span>
                  <h2>Ask the giant.<br /><em>Keep the machine yours.</em></h2>
                  <p>Connect to a local Qwanto server and stream responses directly from your hardware. Nothing leaves the endpoint you choose.</p>
                  <div className="suggestions">
                    {["Explain how expert routing works", "Write a small C benchmark", "Compare RAM and VRAM caching"].map((item) => <button key={item} onClick={() => setDraft(item)}>{item}<ArrowUp className="size-3.5 rotate-45" /></button>)}
                  </div>
                </div>
              ) : (
                <div className="message-list">
                  {messages.map((item, idx) => (
                    <article key={item.id} className={cn("message", item.role)}>
                      <div className="avatar">{item.role === "user" ? "Y" : <Feather className="size-4" />}</div>
                      <div>
                        <div className="message-meta">{item.role === "user" ? "You" : "Qwanto"}</div>
                        <div className="message-body">{item.content || (loading ? <span className="typing-indicator"><span /><span /><span /></span> : "")}</div>
                        {item.role === "assistant" && item.content && !loading && (
                          <div className="message-actions">
                            <button className="msg-action-btn" title="Copy" onClick={() => { navigator.clipboard.writeText(item.content); setCopiedId(item.id); setTimeout(() => setCopiedId(null), 1500) }}>
                              {copiedId === item.id ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
                            </button>
                            <button className="msg-action-btn" title="Regenerate" onClick={() => { if (loading) return; const userMsg = messages.slice(0, idx).reverse().find(m => m.role === "user"); if (userMsg) { updateMessages(current => current.slice(0, idx)); setDraft(userMsg.content.replace(/^\[Attached:.*\]\n?/gm, "").replace(/^\[Web search query:.*\]\n?/gm, "").trim()); } }}>
                              <RotateCcw className="size-3.5" />
                            </button>
                          </div>
                        )}
                      </div>
                    </article>
                  ))}
                  <div ref={bottomRef} />
                </div>
              )}
            </div>

            <div className="composer-wrap">
              {error && <div className="error-banner" role="alert">{error}</div>}
              <div className="composer">
                {attachments.length > 0 && (
                  <div className="attachments-bar">
                    {attachments.map((a, i) => (
                      <div key={i} className="attachment-chip">
                        {a.type.startsWith("image/") ? <Image className="size-3.5" /> : <File className="size-3.5" />}
                        <span>{a.name}</span>
                        <button onClick={() => setAttachments(prev => prev.filter((_, idx) => idx !== i))}><X className="size-3" /></button>
                      </div>
                    ))}
                  </div>
                )}
                {webSearchEnabled && (
                  <div className="search-bar">
                    <Search className="size-3.5" />
                    <input value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} placeholder="Search query..." />
                    <button onClick={() => { setWebSearchEnabled(false); setSearchQuery("") }}><X className="size-3" /></button>
                  </div>
                )}
                <Textarea value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Message Qwanto…" onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void send() } }} />
                <div className="composer-foot">
                  <div className="composer-tools">
                    <button className="tool-btn" title="Attach file" onClick={() => fileInputRef.current?.click()}><Paperclip className="size-3.5" /></button>
                    <button className={cn("tool-btn", webSearchEnabled && "active")} title="Web search" onClick={() => setWebSearchEnabled(!webSearchEnabled)}><Globe className="size-3.5" /></button>
                    <button className={cn("tool-btn", isRecording && "recording")} title={isRecording ? "Stop recording" : "Voice input"} onClick={() => {
                      if (isRecording) { window._speechRec?.stop(); setIsRecording(false); return }
                      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
                      if (!SpeechRecognition) { setError("Speech recognition not supported in this browser."); return }
                      const rec = new SpeechRecognition()
                      rec.continuous = false
                      rec.interimResults = false
                      rec.lang = "en-US"
                      rec.onresult = (e: any) => { const transcript = e.results[0][0].transcript; setDraft(prev => prev ? prev + " " + transcript : transcript); setIsRecording(false) }
                      rec.onerror = () => setIsRecording(false)
                      rec.onend = () => setIsRecording(false)
                      window._speechRec = rec
                      rec.start()
                      setIsRecording(true)
                    }}>{isRecording ? <MicOff className="size-3.5" /> : <Mic className="size-3.5" />}</button>
                    <input ref={fileInputRef} type="file" multiple accept="image/*,.pdf,.txt,.md,.csv,.json,.py,.js,.ts,.c,.cpp,.h" className="hidden" onChange={(e) => {
                      const files = Array.from(e.target.files || [])
                      files.forEach(f => {
                        const reader = new FileReader()
                        reader.onload = () => {
                          const data = reader.result as string
                          setAttachments(prev => [...prev, { name: f.name, type: f.type, data }])
                        }
                        reader.readAsDataURL(f)
                      })
                      e.target.value = ""
                    }} />
                  </div>
                  <span><MessageSquareText className="size-3.5" /> Enter to send · Shift+Enter for newline</span>
                  {loading ? <Button variant="destructive" size="icon" aria-label="Stop generation" onClick={() => abortRef.current?.abort()}><CircleStop className="size-4" /></Button> : <Button size="icon" aria-label="Send message" disabled={!canSend} onClick={() => void send()}><ArrowUp className="size-4" /></Button>}
                </div>
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  )
}
