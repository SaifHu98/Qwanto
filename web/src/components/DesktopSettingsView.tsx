import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react"
import {
  Activity,
  Check,
  ChevronDown,
  Download,
  FileInput,
  Gauge,
  Globe2,
  HardDrive,
  MemoryStick,
  RefreshCw,
  Search,
  ShieldCheck,
  Trash2,
  Upload,
  X,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import type { BenchmarkReport, ConversionStatus, DiscoveredModel, DoctorReport, DownloadStatus, SecurityReport, TelemetryData } from "@/lib/api"
import { cancelConversion, cancelDownloadModel, deleteModel, downloadModel, getBenchmarks, getConversionStatus, getDoctorReport, getDownloadStatus, getSecurityReport, getTelemetry, importLocalModel, listDiscoveredModels, startConversion } from "@/lib/api"
import { modelIsSelectable } from "@/lib/gateway"
import type { AgentProfile } from "@/lib/agent"
import { AGENT_PROFILES, profileConfig } from "@/lib/agent"
import { desktopInvoke, type DesktopToolResult, type ProjectMemory } from "@/lib/desktop"

type SettingsSection = "models" | "runtime" | "agent" | "memory" | "privacy" | "diagnostics"
type ModelDialog = "import" | "download" | "convert" | null
type Report = TelemetryData | DoctorReport | BenchmarkReport | SecurityReport

export interface SessionUsage {
  promptTokens: number | null
  completionTokens: number | null
  totalTokens: number | null
  elapsedMs: number | null
  ttftMs: number | null
  tokensPerSecond: number | null
  contextUse: number | null
  toolCalls: number | null
  queueState: string
}

interface SearchSource { title: string; url: string; snippet: string; timestamp?: string }

interface DesktopSettingsViewProps {
  baseUrl: string
  apiKey: string
  gatewayReady: boolean
  logs: Array<{ time: string; type: string; message: string }>
  model: string
  models: DiscoveredModel[]
  onSelectModel: (path: string) => void
  onActivateModel: (path: string) => void
  loadingModel: boolean
  profile: AgentProfile
  onProfileChange: (profile: AgentProfile) => void
  usage: SessionUsage
  onIncludeSearchContext?: (sources: SearchSource[]) => void
}

const sections: Array<{ id: SettingsSection; label: string; icon: typeof Gauge }> = [
  { id: "models", label: "Models", icon: HardDrive },
  { id: "runtime", label: "Runtime", icon: Gauge },
  { id: "agent", label: "Agent", icon: Activity },
  { id: "memory", label: "Project Memory", icon: MemoryStick },
  { id: "privacy", label: "Privacy & Internet", icon: Globe2 },
  { id: "diagnostics", label: "Diagnostics", icon: ShieldCheck },
]

function formatBytes(value?: number) {
  if (!value || value < 1) return "Unavailable"
  const units = ["B", "KB", "MB", "GB", "TB"]
  let amount = value
  let unit = 0
  while (amount >= 1024 && unit < units.length - 1) { amount /= 1024; unit += 1 }
  return `${amount.toFixed(amount >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`
}

function listFromText(value: string) { return value.split("\n").map((item) => item.trim()).filter(Boolean) }

function SettingsDialog({ mode, onClose, onImport, onDownload, onConvert }: {
  mode: Exclude<ModelDialog, null>
  onClose: () => void
  onImport: (path: string, destination: string) => void
  onDownload: (url: string, filename: string, sha256: string, consent: boolean) => void
  onConvert: (source: string, output: string, quant: string) => void
}) {
  const [path, setPath] = useState("")
  const [destination, setDestination] = useState("")
  const [url, setUrl] = useState("")
  const [filename, setFilename] = useState("")
  const [sha256, setSha256] = useState("")
  const [consent, setConsent] = useState(false)
  const [source, setSource] = useState("")
  const [output, setOutput] = useState("")
  const [quant, setQuant] = useState("q4_0")
  return <div className="settings-dialog-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
    <section className="settings-dialog" role="dialog" aria-modal="true" aria-labelledby="settings-dialog-title">
      <div className="settings-dialog-header"><div><span className="desktop-eyebrow">MODELS</span><h2 id="settings-dialog-title">{mode === "import" ? "Import local model" : mode === "download" ? "Download model" : "Convert model"}</h2></div><button className="icon-button" aria-label="Close dialog" onClick={onClose}><X className="size-4" /></button></div>
      {mode === "import" && <div className="settings-dialog-form"><p className="desktop-muted">Choose an existing local checkpoint. The gateway copies it into the managed model library and validates it before activation.</p><label className="desktop-field">Source path<Input autoFocus value={path} placeholder="C:\\Models\\model.gguf" onChange={(event) => setPath(event.target.value)} /></label><label className="desktop-field">Destination directory or filename <span className="settings-optional">Optional</span><Input value={destination} placeholder="Managed library default" onChange={(event) => setDestination(event.target.value)} /></label><Button onClick={() => onImport(path.trim(), destination.trim())} disabled={!path.trim()}><Upload className="size-4" /> Import and validate</Button></div>}
      {mode === "download" && <div className="settings-dialog-form"><p className="desktop-muted">Downloads are external network actions. Confirm the source, license, and checksum before the local gateway starts a resumable download.</p><label className="desktop-field">HTTPS source URL<Input autoFocus value={url} placeholder="https://huggingface.co/..." onChange={(event) => setUrl(event.target.value)} /></label><label className="desktop-field">Filename<Input value={filename} placeholder="model.gguf" onChange={(event) => setFilename(event.target.value)} /></label><label className="desktop-field">Expected SHA-256 <span className="settings-optional">Recommended</span><Input value={sha256} placeholder="64 hexadecimal characters" onChange={(event) => setSha256(event.target.value)} /></label><label className="desktop-consent"><input type="checkbox" checked={consent} onChange={(event) => setConsent(event.target.checked)} /> I approve this source and applicable license. The artifact stays on this machine.</label><Button onClick={() => onDownload(url.trim(), filename.trim(), sha256.trim(), consent)} disabled={!url.trim() || !consent}><Download className="size-4" /> Start consented download</Button></div>}
      {mode === "convert" && <div className="settings-dialog-form"><p className="desktop-muted">Conversion stays local. The resulting QWN container is structurally validated and smoke-tested before it can be activated.</p><label className="desktop-field">Source checkpoint path<Input autoFocus value={source} placeholder="C:\\Models\\model.safetensors" onChange={(event) => setSource(event.target.value)} /></label><label className="desktop-field">Output .qwn path <span className="settings-optional">Optional</span><Input value={output} placeholder="Managed model library default" onChange={(event) => setOutput(event.target.value)} /></label><label className="desktop-field">Quantization<select value={quant} onChange={(event) => setQuant(event.target.value)}><option value="q4_0">Q4_0</option><option value="hyper_vsq2">HyperVSQ-2</option><option value="vsq">VSQ</option><option value="none">None</option></select></label><Button onClick={() => onConvert(source.trim(), output.trim(), quant)} disabled={!source.trim()}><FileInput className="size-4" /> Start local conversion</Button></div>}
    </section>
  </div>
}

function ModelFacts({ model }: { model: DiscoveredModel }) {
  return <div className="model-facts"><div><span>Format</span><strong>{model.format || (model.type === "qwn" ? ".qwn container" : model.type)}</strong></div><div><span>Quantization</span><strong>{model.quantization || "Unavailable"}</strong></div><div><span>Size</span><strong>{model.size_formatted || formatBytes(model.size_bytes)}</strong></div><div><span>Compatibility</span><strong>{model.compatibility_state || "Unavailable"}</strong></div><div><span>Hardware fit</span><strong>{model.hardware_fit?.status || "Unavailable"}</strong></div></div>
}

function SettingsModels({ models, model, recommendationReason, onSelectModel, onActivateModel, loadingModel, onOpenDialog, downloadStatus, conversionStatus, onCancelDownload, onCancelConversion, onDelete }: {
  models: DiscoveredModel[]
  model: string
  recommendationReason: string
  onSelectModel: (path: string) => void
  onActivateModel: (path: string) => void
  loadingModel: boolean
  onOpenDialog: (mode: Exclude<ModelDialog, null>) => void
  downloadStatus: DownloadStatus | null
  conversionStatus: ConversionStatus | null
  onCancelDownload: () => void
  onCancelConversion: () => void
  onDelete: (model: DiscoveredModel) => void
}) {
  const [query, setQuery] = useState("")
  const [sort, setSort] = useState<"name" | "size" | "state">("name")
  const active = models.find((candidate) => candidate.path === model || candidate.name === model)
  const filtered = useMemo(() => models.filter((candidate) => `${candidate.name} ${candidate.format || ""} ${candidate.quantization || ""}`.toLowerCase().includes(query.toLowerCase())).sort((left, right) => sort === "size" ? (right.size_bytes || 0) - (left.size_bytes || 0) : sort === "state" ? (left.compatibility_state || "").localeCompare(right.compatibility_state || "") : left.name.localeCompare(right.name)), [models, query, sort])
  const conversionEta = conversionStatus?.progress && conversionStatus.elapsed ? Math.max(0, Math.round((conversionStatus.elapsed * (100 - conversionStatus.progress)) / conversionStatus.progress)) : null
  return <div className="settings-section-content" data-testid="settings-models"><div className="settings-section-heading"><div><span className="desktop-eyebrow">MODEL LIBRARY</span><h1>Models</h1><p className="desktop-muted">Only validated QWN models that fit this machine can be activated. Filename alone is never trusted.</p></div><div className="settings-action-row"><Button size="sm" onClick={() => onOpenDialog("import")}><Upload className="size-3.5" /> Import local model</Button><Button size="sm" variant="secondary" onClick={() => onOpenDialog("download")}><Download className="size-3.5" /> Download model</Button><Button size="sm" variant="secondary" onClick={() => onOpenDialog("convert")}><FileInput className="size-3.5" /> Convert model</Button></div></div>
    <section className="settings-card active-model-card"><div className="settings-card-header"><div><span className="desktop-eyebrow">ACTIVE MODEL</span><h2>{active?.name || "No validated model active"}</h2></div>{active?.recommended && <span className="settings-badge recommended">Recommended</span>}</div>{active ? <><ModelFacts model={active} /><p className="settings-card-note">{active.recommendation_reason || recommendationReason || "Selected by the user after local validation."}</p><div className="settings-card-meta">Disk location: <code>{active.disk_location || active.path}</code></div><Button size="sm" onClick={() => onActivateModel(active.path)} disabled={!modelIsSelectable(active) || loadingModel}>{loadingModel ? "Activating…" : "Activate validated model"}</Button></> : <p className="desktop-muted">Choose a compatible QWN model from the library, then activate it after validation and hardware-fit checks pass.</p>}</section>
    <section className="settings-card"><div className="settings-card-header"><div><h2>Local model library</h2><span className="settings-count">{filtered.length} of {models.length} discovered</span></div><div className="settings-library-tools"><label className="settings-search"><Search className="size-3.5" /><input aria-label="Search models" value={query} placeholder="Search models" onChange={(event) => setQuery(event.target.value)} /></label><label className="settings-sort"><span>Sort</span><select value={sort} onChange={(event) => setSort(event.target.value as typeof sort)}><option value="name">Name</option><option value="size">Size</option><option value="state">Validation state</option></select></label></div></div><div className="settings-model-list">{filtered.map((candidate) => { const selectable = modelIsSelectable(candidate); return <article className="settings-model-row" key={candidate.path}><div className="settings-model-main"><div className="settings-model-title"><strong>{candidate.name}</strong><span className={`settings-state state-${candidate.compatibility_state || "unknown"}`}>{candidate.qwn_validation?.status || candidate.compatibility_state || "unknown"}</span></div><div className="settings-model-tags"><span>{candidate.format || candidate.type}</span><span>{candidate.quantization || "Quantization unavailable"}</span><span>{candidate.size_formatted || formatBytes(candidate.size_bytes)}</span></div><div className="settings-model-location">{candidate.disk_location || candidate.path}</div></div><div className="settings-model-actions"><Button size="sm" variant="ghost" onClick={() => onSelectModel(candidate.path)}>{candidate.path === model ? "Selected" : "Select"}</Button><Button size="sm" variant="secondary" onClick={() => onActivateModel(candidate.path)} disabled={!selectable || loadingModel}>{candidate.path === model ? "Activate" : "Use"}</Button><button className="settings-delete-button" aria-label={`Delete ${candidate.name}`} onClick={() => onDelete(candidate)}><Trash2 className="size-3.5" /></button></div></article> })}{!filtered.length && <div className="settings-empty">No models match this search.</div>}</div></section>
    {downloadStatus && downloadStatus.status !== "idle" && <section className="settings-card queue-card"><div className="settings-card-header"><div><span className="desktop-eyebrow">DOWNLOAD QUEUE</span><h2>{downloadStatus.filename || "Model download"}</h2></div><span className={`settings-state state-${downloadStatus.status}`}>{downloadStatus.status}</span></div><div className="settings-progress"><span style={{ width: `${Math.max(0, Math.min(100, downloadStatus.progress || 0))}%` }} /></div><div className="queue-facts"><span>{downloadStatus.progress != null ? `${downloadStatus.progress.toFixed(1)}%` : "Progress unavailable"}</span><span>{formatBytes(downloadStatus.downloaded)} / {formatBytes(downloadStatus.total)}</span><span>ETA {downloadStatus.eta_seconds != null ? `${Math.ceil(downloadStatus.eta_seconds)}s` : "Unavailable"}</span><span>{downloadStatus.verification || "Unverified"}</span></div><div className="settings-card-meta">Output path: <code>{downloadStatus.dest_path || "Unavailable"}</code></div>{downloadStatus.status === "downloading" || downloadStatus.status === "paused" ? <Button size="sm" variant="secondary" onClick={onCancelDownload}>Cancel download</Button> : null}{downloadStatus.error && <p className="settings-error-text">{downloadStatus.error}</p>}</section>}
    {conversionStatus && conversionStatus.status !== "idle" && <section className="settings-card queue-card"><div className="settings-card-header"><div><span className="desktop-eyebrow">CONVERSION QUEUE</span><h2>{conversionStatus.output || "QWN conversion"}</h2></div><span className={`settings-state state-${conversionStatus.status}`}>{conversionStatus.status}</span></div><div className="settings-progress"><span style={{ width: `${Math.max(0, Math.min(100, conversionStatus.progress || 0))}%` }} /></div><div className="queue-facts"><span>{conversionStatus.progress != null ? `${conversionStatus.progress}%` : "Progress unavailable"}</span><span>ETA {conversionEta != null ? `${conversionEta}s` : "Unavailable"}</span><span>{conversionStatus.stage || "Stage unavailable"}</span><span>{conversionStatus.status === "done" ? "Validated" : "Validation pending"}</span></div><div className="settings-card-meta">Output path: <code>{conversionStatus.output || "Unavailable"}</code></div>{conversionStatus.status === "converting" ? <Button size="sm" variant="secondary" onClick={onCancelConversion}>Cancel conversion</Button> : null}{conversionStatus.status === "done" && conversionStatus.output && <Button size="sm" onClick={() => { const result = models.find((candidate) => candidate.path === conversionStatus.output); if (result && modelIsSelectable(result)) onActivateModel(result.path) }} disabled={!models.some((candidate) => candidate.path === conversionStatus.output && modelIsSelectable(candidate))}>Activate validated output</Button>}{conversionStatus.message && <p className="settings-card-note">{conversionStatus.message}</p>}</section>}
  </div>
}

export function DesktopSettingsView(props: DesktopSettingsViewProps) {
  const [section, setSection] = useState<SettingsSection>("models")
  const [models, setModels] = useState(props.models)
  const [recommendationReason, setRecommendationReason] = useState("")
  const [dialog, setDialog] = useState<ModelDialog>(null)
  const [message, setMessage] = useState("")
  const [downloadStatus, setDownloadStatus] = useState<DownloadStatus | null>(null)
  const [conversionStatus, setConversionStatus] = useState<ConversionStatus | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<DiscoveredModel | null>(null)
  const [report, setReport] = useState<Report | null>(null)
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([])
  const [memoryLoading, setMemoryLoading] = useState(false)
  const [memoryDraft, setMemoryDraft] = useState<ProjectMemory | null>(null)
  const [internetEnabled, setInternetEnabled] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")
  const [searchResults, setSearchResults] = useState<SearchSource[]>([])
  const [selectedSources, setSelectedSources] = useState<string[]>([])
  const [searchApproval, setSearchApproval] = useState<DesktopToolResult | null>(null)

  useEffect(() => setModels(props.models), [props.models])
  const refreshModels = async () => { if (!props.gatewayReady) return; try { const result = await listDiscoveredModels(props.baseUrl, props.apiKey); setModels(result.models || []); setRecommendationReason(result.recommendation?.reason || "") } catch (error) { setMessage(error instanceof Error ? error.message : "Model inventory unavailable.") } }
  useEffect(() => { if (!props.gatewayReady) return; void refreshModels(); const timer = window.setInterval(() => void refreshModels(), 2500); return () => window.clearInterval(timer) }, [props.baseUrl, props.apiKey, props.gatewayReady])
  useEffect(() => { if (!props.gatewayReady) return; const poll = async () => { try { setDownloadStatus(await getDownloadStatus(props.baseUrl, props.apiKey)) } catch { /* gateway banner carries connection state */ } try { setConversionStatus(await getConversionStatus(props.baseUrl, props.apiKey)) } catch { /* gateway banner carries connection state */ } }; void poll(); const timer = window.setInterval(() => void poll(), 1000); return () => window.clearInterval(timer) }, [props.baseUrl, props.apiKey, props.gatewayReady])
  useEffect(() => { if (section !== "memory" || !props.gatewayReady) return; setMemoryLoading(true); void desktopInvoke<ProjectMemory>("get_project_memory").then(setMemoryDraft).catch((error) => setMessage(error instanceof Error ? error.message : "Project memory is unavailable.")).finally(() => setMemoryLoading(false)) }, [section, props.gatewayReady])

  const selectSection = (id: SettingsSection) => { setSection(id); setMessage("") }
  const handleTabKey = (event: KeyboardEvent<HTMLButtonElement>, index: number) => { if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return; event.preventDefault(); const offset = event.key === "ArrowDown" ? 1 : event.key === "ArrowUp" ? -1 : 0; const nextIndex = event.key === "Home" ? 0 : event.key === "End" ? sections.length - 1 : (index + offset + sections.length) % sections.length; selectSection(sections[nextIndex].id); tabRefs.current[nextIndex]?.focus() }
  const runReport = async (name: "telemetry" | "doctor" | "benchmarks" | "security") => { if (!props.gatewayReady) return; try { setReport(name === "telemetry" ? await getTelemetry(props.baseUrl, props.apiKey) : name === "doctor" ? await getDoctorReport(props.baseUrl, props.apiKey) : name === "benchmarks" ? await getBenchmarks(props.baseUrl, props.apiKey) : await getSecurityReport(props.baseUrl, props.apiKey)) } catch (error) { setMessage(error instanceof Error ? error.message : "The local report is unavailable.") } }
  const updateMemory = (patch: Partial<ProjectMemory>) => setMemoryDraft((current) => current ? { ...current, ...patch } : current)
  const saveMemory = async () => { if (!memoryDraft) return; try { const saved = await desktopInvoke<ProjectMemory>("save_project_memory", { memory: memoryDraft }); setMemoryDraft(saved); setMessage("Project memory saved locally in this workspace.") } catch (error) { setMessage(error instanceof Error ? error.message : "Project memory could not be saved.") } }
  const exportMemory = async () => { try { const value = await desktopInvoke<string>("export_project_memory"); const blob = new Blob([value], { type: "application/json" }); const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = "qwanto-project-memory.json"; link.click(); URL.revokeObjectURL(link.href); setMessage("Project memory exported locally.") } catch (error) { setMessage(error instanceof Error ? error.message : "Project memory could not be exported.") } }
  const startImport = async (path: string, destination: string) => { try { await importLocalModel(props.baseUrl, path, props.apiKey, destination || undefined); setDialog(null); setMessage("Local model import started; validation will appear in the queue."); void refreshModels() } catch (error) { setMessage(error instanceof Error ? error.message : "Model import failed.") } }
  const startDownload = async (url: string, filename: string, sha256: string, consent: boolean) => { try { const host = new URL(url).hostname; await downloadModel(props.baseUrl, url, filename || undefined, undefined, props.apiKey, { approvedHost: host, sha256: sha256 || undefined, overwrite: false, provider: "direct_https", licenseConfirmed: consent }); setDialog(null); setMessage("Consented download started.") } catch (error) { setMessage(error instanceof Error ? error.message : "Model download failed.") } }
  const startConvert = async (source: string, output: string, quant: string) => { try { await startConversion(props.baseUrl, props.apiKey, source, output || undefined, quant); setDialog(null); setMessage("Local conversion started; validation will appear in the queue.") } catch (error) { setMessage(error instanceof Error ? error.message : "Model conversion failed.") } }
  const confirmDelete = async (target: DiscoveredModel) => { try { await deleteModel(props.baseUrl, target.path, props.apiKey); setDeleteTarget(null); setMessage(`Deleted ${target.name}.`); void refreshModels() } catch (error) { setMessage(error instanceof Error ? error.message : "Model could not be deleted.") } }
  const runSearch = async (approvalToken?: string) => { if (!searchQuery.trim() || !internetEnabled) return; try { const result = await desktopInvoke<DesktopToolResult>("web_search", { query: searchQuery.trim(), approvalToken }); if (result.outcome === "needs_approval") { setSearchApproval(result); return } if (result.success) { const parsed = JSON.parse(result.output) as { results?: SearchSource[] }; setSearchResults(parsed.results || []); setSearchApproval(null) } else setMessage(result.error || "Search was not completed.") } catch (error) { setMessage(error instanceof Error ? error.message : "Search is available only in Qwanto Desktop.") } }
  const currentSection = sections.find((item) => item.id === section)

  return <div className="desktop-settings-layout" data-testid="desktop-settings-layout"><aside className="settings-section-nav" aria-label="Settings sections"><div className="settings-nav-title">SETTINGS</div><div role="tablist" aria-orientation="vertical">{sections.map(({ id, label, icon: Icon }, index) => <button key={id} ref={(element) => { tabRefs.current[index] = element }} role="tab" aria-selected={section === id} aria-current={section === id ? "page" : undefined} tabIndex={section === id ? 0 : -1} data-settings-tab={id} className={section === id ? "active" : ""} onClick={() => selectSection(id)} onKeyDown={(event) => handleTabKey(event, index)}><Icon className="size-4" /><span>{label}</span><ChevronDown className="settings-nav-chevron size-3.5" /></button>)}</div><p className="settings-nav-note">Local settings stay on this machine. Project memory is stored inside the selected workspace.</p></aside><section className="desktop-settings-panel" role="tabpanel" aria-label={currentSection?.label}>{section === "models" && <SettingsModels models={models} model={props.model} recommendationReason={recommendationReason} onSelectModel={props.onSelectModel} onActivateModel={props.onActivateModel} loadingModel={props.loadingModel} onOpenDialog={setDialog} downloadStatus={downloadStatus} conversionStatus={conversionStatus} onCancelDownload={() => void cancelDownloadModel(props.baseUrl, props.apiKey)} onCancelConversion={() => void cancelConversion(props.baseUrl, props.apiKey)} onDelete={setDeleteTarget} />}
      {section === "runtime" && <div className="settings-section-content"><div className="settings-section-heading"><div><span className="desktop-eyebrow">LOCAL RUNTIME</span><h1>Runtime</h1><p className="desktop-muted">Only parameters reported by the gateway are exposed. Hardware and performance controls without a runtime contract remain unavailable.</p></div></div><div className="settings-runtime-grid"><section className="settings-card"><h2>Active profile mapping</h2><div className="settings-runtime-facts"><span>Profile<strong>{profileConfig(props.profile).label}</strong></span><span>Context<strong>{profileConfig(props.profile).contextSize.toLocaleString()} tokens</strong></span><span>Max output<strong>{profileConfig(props.profile).maxTokens} tokens</strong></span><span>Temperature<strong>{profileConfig(props.profile).temperature}</strong></span><span>Top-p<strong>{profileConfig(props.profile).topP}</strong></span></div></section><section className="settings-card"><h2>Runtime-reported metrics</h2><div className="settings-runtime-facts"><span>Prompt tokens<strong>{props.usage.promptTokens ?? "Unavailable"}</strong></span><span>Completion tokens<strong>{props.usage.completionTokens ?? "Unavailable"}</strong></span><span>TTFT<strong>{props.usage.ttftMs != null ? `${props.usage.ttftMs.toFixed(0)} ms` : "Unavailable"}</strong></span><span>Tokens/s<strong>{props.usage.tokensPerSecond?.toFixed(2) || "Unavailable"}</strong></span><span>Queue<strong>{props.usage.queueState}</strong></span></div></section><section className="settings-card settings-disabled-card"><h2>Not reported by this runtime</h2><p className="desktop-muted">CPU thread count, GPU offload percentage, KV-cache quantization, batching, speculative decoding, and seed controls are disabled until qwnrun reports a stable contract for them.</p></section></div></div>}
      {section === "agent" && <div className="settings-section-content"><div className="settings-section-heading"><div><span className="desktop-eyebrow">AGENT PROFILES</span><h1>Agent</h1><p className="desktop-muted">Profiles map only to real gateway parameters: context size, maximum output tokens, temperature, and top-p.</p></div></div><div className="agent-profile-list">{AGENT_PROFILES.map((candidate) => <button key={candidate.id} className={`agent-profile-card ${props.profile === candidate.id ? "active" : ""}`} onClick={() => props.onProfileChange(candidate.id)}><div className="agent-profile-header"><strong>{candidate.label}</strong>{props.profile === candidate.id && <Check className="size-4" />}</div><p>{candidate.description}</p><span>{candidate.contextSize.toLocaleString()} context · {candidate.maxTokens} output · temp {candidate.temperature} · top-p {candidate.topP}</span></button>)}</div><section className="settings-card settings-disabled-card"><h2>Unsupported controls stay disabled</h2><p className="desktop-muted">Qwanto does not display fake performance sliders. CPU threads, GPU offload, KV-cache quantization, batching, speculative decoding, and seed are shown only when the runtime reports support.</p></section></div>}
      {section === "memory" && <div className="settings-section-content"><div className="settings-section-heading"><div><span className="desktop-eyebrow">WORKSPACE MEMORY</span><h1>Project Memory</h1><p className="desktop-muted">Reviewable, editable memory stored at <code>.qwanto/project-memory.json</code> inside the selected workspace. It is never uploaded silently.</p></div></div>{memoryLoading && <div className="settings-loading"><RefreshCw className="size-4 animate-spin" /> Loading local memory…</div>}{memoryDraft && <div className="settings-memory-form"><label className="desktop-field">Enabled<input type="checkbox" checked={memoryDraft.enabled} onChange={(event) => { updateMemory({ enabled: event.target.checked }); void desktopInvoke<ProjectMemory>("set_project_memory_enabled", { enabled: event.target.checked }).then(setMemoryDraft).catch((error) => setMessage(error instanceof Error ? error.message : "Memory setting could not be saved.")) }} /></label><label className="desktop-field">Project summary<Textarea value={memoryDraft.summary} onChange={(event) => updateMemory({ summary: event.target.value })} /></label><label className="desktop-field">Architecture notes<Textarea value={memoryDraft.architecture_notes} onChange={(event) => updateMemory({ architecture_notes: event.target.value })} /></label><label className="desktop-field">User conventions<Textarea value={memoryDraft.user_conventions} onChange={(event) => updateMemory({ user_conventions: event.target.value })} /></label><label className="desktop-field">Accepted decisions <span className="settings-optional">One per line</span><Textarea value={memoryDraft.accepted_decisions.join("\n")} onChange={(event) => updateMemory({ accepted_decisions: listFromText(event.target.value) })} /></label><label className="desktop-field">Task checkpoints <span className="settings-optional">One per line · used to resume safely</span><Textarea value={memoryDraft.task_checkpoints.join("\n")} onChange={(event) => updateMemory({ task_checkpoints: listFromText(event.target.value) })} /></label><div className="settings-action-row"><Button onClick={() => void saveMemory()}>Save memory</Button><Button variant="secondary" onClick={() => void exportMemory()}>Export JSON</Button><Button variant="ghost" onClick={() => void desktopInvoke<ProjectMemory>("clear_project_memory").then((value) => { setMemoryDraft(value); setMessage("Project memory cleared locally.") }).catch((error) => setMessage(error instanceof Error ? error.message : "Memory could not be cleared."))}>Clear memory</Button></div></div>}</div>}
      {section === "privacy" && <div className="settings-section-content"><div className="settings-section-heading"><div><span className="desktop-eyebrow">LOCAL-FIRST BOUNDARY</span><h1>Privacy &amp; Internet</h1><p className="desktop-muted">Inference remains local. Internet search is an external tool, disabled by default, and each search requires a fresh desktop approval.</p></div></div><section className="settings-card"><label className="settings-toggle"><input type="checkbox" checked={internetEnabled} onChange={(event) => setInternetEnabled(event.target.checked)} /><span><strong>Enable optional web search</strong><small>No browser search path or cloud inference fallback is enabled.</small></span></label><div className="settings-search-form"><Input aria-label="Search query" value={searchQuery} placeholder="Search the public web with approval…" onChange={(event) => setSearchQuery(event.target.value)} /><Button onClick={() => void runSearch()} disabled={!internetEnabled || !searchQuery.trim()}><Search className="size-4" /> Search with approval</Button></div>{searchApproval && <div className="settings-approval-card"><ShieldCheck className="size-4" /><div><strong>Approval required</strong><p>{searchApproval.action_details?.description || "Allow this external search?"}</p></div><div className="settings-action-row"><Button size="sm" onClick={() => void runSearch(searchApproval.approval_token || undefined)}>Approve once</Button><Button size="sm" variant="ghost" onClick={() => setSearchApproval(null)}>Reject</Button></div></div>}{searchResults.length > 0 && <div className="settings-search-results"><div className="settings-card-header"><h2>Sources</h2><Button size="sm" variant="secondary" onClick={() => props.onIncludeSearchContext?.(searchResults.filter((source) => selectedSources.includes(source.url)))} disabled={!selectedSources.length}>Include selected in next agent prompt</Button></div>{searchResults.map((source) => <label className="settings-source-row" key={source.url}><input type="checkbox" checked={selectedSources.includes(source.url)} onChange={(event) => setSelectedSources((current) => event.target.checked ? [...current, source.url] : current.filter((url) => url !== source.url))} /><span><a href={source.url} target="_blank" rel="noreferrer">{source.title || source.url}</a><small>{source.snippet || "No snippet reported."} · {source.timestamp || "Timestamp unavailable"}</small></span></label>)}</div>}</section><p className="settings-card-note"><strong>Trusted sessions are not enabled.</strong> Every search remains individually approval-gated.</p></div>}
      {section === "diagnostics" && <div className="settings-section-content"><div className="settings-section-heading"><div><span className="desktop-eyebrow">LOCAL EVIDENCE</span><h1>Diagnostics</h1><p className="desktop-muted">Reports are read from the local gateway and retain honest unavailable classifications.</p></div><Button size="sm" variant="secondary" onClick={() => void runReport("telemetry")} disabled={!props.gatewayReady}><RefreshCw className="size-3.5" /> Refresh</Button></div><div className="diagnostic-action-grid"><Button variant="secondary" onClick={() => void runReport("telemetry")} disabled={!props.gatewayReady}>Telemetry</Button><Button variant="secondary" onClick={() => void runReport("doctor")} disabled={!props.gatewayReady}>Runtime doctor</Button><Button variant="secondary" onClick={() => void runReport("benchmarks")} disabled={!props.gatewayReady}>Benchmark evidence</Button><Button variant="secondary" onClick={() => void runReport("security")} disabled={!props.gatewayReady}>Security boundary</Button></div><pre className="desktop-code-preview large">{report ? JSON.stringify(report, null, 2) : props.logs.length ? props.logs.map((log) => `[${log.time}] ${log.type.toUpperCase()} ${log.message}`).join("\n") : "No diagnostic report loaded."}</pre></div>}
      {message && <div className="desktop-success settings-message" role="status">{message}</div>}</section>{dialog && <SettingsDialog mode={dialog} onClose={() => setDialog(null)} onImport={(path, destination) => void startImport(path, destination)} onDownload={(url, filename, sha256, consent) => void startDownload(url, filename, sha256, consent)} onConvert={(source, output, quant) => void startConvert(source, output, quant)} />}{deleteTarget && <div className="settings-dialog-backdrop" role="presentation"><section className="settings-dialog settings-confirm-dialog" role="dialog" aria-modal="true" aria-labelledby="delete-model-title"><div className="settings-dialog-header"><div><span className="desktop-eyebrow">MODEL LIBRARY</span><h2 id="delete-model-title">Delete {deleteTarget.name}?</h2></div><button className="icon-button" aria-label="Close confirmation" onClick={() => setDeleteTarget(null)}><X className="size-4" /></button></div><p className="desktop-muted">This removes the local file from the managed model library. It cannot be undone from Qwanto.</p><div className="settings-action-row"><Button variant="destructive" onClick={() => void confirmDelete(deleteTarget)}><Trash2 className="size-4" /> Delete model</Button><Button variant="secondary" onClick={() => setDeleteTarget(null)}>Cancel</Button></div></section></div>}</div>
}
