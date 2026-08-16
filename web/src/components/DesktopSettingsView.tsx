import { useEffect, useState } from "react"
import { BarChart3, Download, FileCog, Gauge, LockKeyhole, ScrollText, ShieldCheck } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import type { BenchmarkReport, DiscoveredModel, DoctorReport, SecurityReport, TelemetryData } from "@/lib/api"
import { downloadModel, getBenchmarks, getDoctorReport, getSecurityReport, getTelemetry, listDiscoveredModels, startConversion } from "@/lib/api"

type SettingsSection = "models" | "conversion" | "downloads" | "diagnostics" | "benchmarks" | "security" | "logs"

interface DesktopSettingsViewProps {
  baseUrl: string
  apiKey: string
  gatewayReady: boolean
  logs: Array<{ time: string; type: string; message: string }>
  onSelectModel: (path: string) => void
}

export function DesktopSettingsView({ baseUrl, apiKey, gatewayReady, logs, onSelectModel }: DesktopSettingsViewProps) {
  const [section, setSection] = useState<SettingsSection>("models")
  const [models, setModels] = useState<DiscoveredModel[]>([])
  const [source, setSource] = useState("")
  const [output, setOutput] = useState("")
  const [quant, setQuant] = useState("q4_0")
  const [downloadUrl, setDownloadUrl] = useState("")
  const [downloadFilename, setDownloadFilename] = useState("")
  const [downloadSha, setDownloadSha] = useState("")
  const [downloadConsent, setDownloadConsent] = useState(false)
  const [report, setReport] = useState<TelemetryData | DoctorReport | BenchmarkReport | SecurityReport | null>(null)
  const [message, setMessage] = useState("")

  useEffect(() => {
    if (!gatewayReady || section !== "models") return
    void listDiscoveredModels(baseUrl, apiKey).then((result) => setModels(result.models || [])).catch((error) => setMessage(error instanceof Error ? error.message : "Model inventory unavailable."))
  }, [baseUrl, apiKey, gatewayReady, section])

  const runReport = async (name: Exclude<SettingsSection, "models" | "conversion" | "downloads" | "logs">) => {
    if (!gatewayReady) return
    setMessage("")
    try {
      const value = name === "diagnostics" ? await getDoctorReport(baseUrl, apiKey) : name === "benchmarks" ? await getBenchmarks(baseUrl, apiKey) : name === "security" ? await getSecurityReport(baseUrl, apiKey) : await getTelemetry(baseUrl, apiKey)
      setReport(value)
    } catch (error) { setMessage(error instanceof Error ? error.message : "The local report is unavailable.") }
  }

  const nav: Array<[SettingsSection, string, typeof FileCog]> = [
    ["models", "Models", FileCog], ["conversion", "Conversion", FileCog], ["downloads", "Downloads", Download],
    ["diagnostics", "Diagnostics", Gauge], ["benchmarks", "Benchmarks", BarChart3], ["security", "Security", LockKeyhole], ["logs", "Logs", ScrollText],
  ]

  return <div className="desktop-advanced-settings" data-testid="desktop-advanced-settings"><div className="desktop-settings-nav">{nav.map(([id, label, Icon]) => <button key={id} className={section === id ? "active" : ""} onClick={() => { setSection(id); if (["diagnostics", "benchmarks", "security"].includes(id)) void runReport(id as Exclude<SettingsSection, "models" | "conversion" | "downloads" | "logs">) }}><Icon className="size-3.5" />{label}</button>)}</div><div className="desktop-settings-body">
    {section === "models" && <div className="desktop-advanced-card"><h2>Validated local models</h2><p className="desktop-muted">Only QWN files that pass container validation, qwnrun support, and hardware-fit checks can be activated.</p>{models.map((model) => <div className="desktop-model-row" key={model.path}><div><strong>{model.name}</strong><span>{model.quantization || "Unknown quantization"} · {model.compatibility_state}</span></div><Button size="sm" variant="secondary" onClick={() => onSelectModel(model.path)}>Select</Button></div>)}{!models.length && <p className="desktop-muted">No local model inventory is available yet.</p>}</div>}
    {section === "conversion" && <div className="desktop-advanced-card"><h2>Convert a local source</h2><p className="desktop-muted">Conversion stays local. The source, output, quantization, and resulting QWN validation are shown by the gateway.</p><Input placeholder="Source checkpoint path" value={source} onChange={(event) => setSource(event.target.value)} /><Input placeholder="Output .qwn path (optional)" value={output} onChange={(event) => setOutput(event.target.value)} /><select value={quant} onChange={(event) => setQuant(event.target.value)}><option value="q4_0">q4_0</option><option value="hyper_vsq2">hyper_vsq2</option><option value="vsq">vsq</option></select><Button onClick={() => void startConversion(baseUrl, apiKey, source, output || undefined, quant).then((result) => setMessage(result.message)).catch((error) => setMessage(error instanceof Error ? error.message : "Conversion failed."))} disabled={!gatewayReady || !source.trim()}>Start local conversion</Button></div>}
    {section === "downloads" && <div className="desktop-advanced-card"><h2>Optional model download</h2><p className="desktop-muted">Downloads require an explicit source, local disk destination, license confirmation, and checksum when supplied. The gateway handles resume and atomic publication.</p><Input placeholder="HTTPS source URL" value={downloadUrl} onChange={(event) => setDownloadUrl(event.target.value)} /><Input placeholder="Filename" value={downloadFilename} onChange={(event) => setDownloadFilename(event.target.value)} /><Input placeholder="Expected SHA-256 (recommended)" value={downloadSha} onChange={(event) => setDownloadSha(event.target.value)} /><label className="desktop-consent"><input type="checkbox" checked={downloadConsent} onChange={(event) => setDownloadConsent(event.target.checked)} /> I confirm the source and applicable license; the artifact stays on this machine.</label><Button onClick={() => void downloadModel(baseUrl, downloadUrl, downloadFilename || undefined, undefined, apiKey, { sha256: downloadSha || undefined }).then((result) => setMessage(result.message)).catch((error) => setMessage(error instanceof Error ? error.message : "Download failed."))} disabled={!gatewayReady || !downloadUrl.trim() || !downloadConsent}>Start consented download</Button></div>}
    {(["diagnostics", "benchmarks", "security"] as SettingsSection[]).includes(section) && <div className="desktop-advanced-card"><h2>{section[0].toUpperCase() + section.slice(1)}</h2><Button size="sm" variant="secondary" onClick={() => void runReport(section as Exclude<SettingsSection, "models" | "conversion" | "downloads" | "logs">)} disabled={!gatewayReady}><ShieldCheck className="size-3.5" /> Refresh report</Button><pre className="desktop-code-preview large">{report ? JSON.stringify(report, null, 2) : "No report loaded."}</pre></div>}
    {section === "logs" && <div className="desktop-advanced-card"><h2>Runtime logs</h2><pre className="desktop-code-preview large">{logs.length ? logs.map((log) => `[${log.time}] ${log.type.toUpperCase()} ${log.message}`).join("\n") : "No logs recorded."}</pre></div>}
    {message && <div className="desktop-success">{message}</div>}
  </div></div>
}
