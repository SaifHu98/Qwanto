import { useEffect, useState } from "react"
import type { ReactNode } from "react"
import { Activity, Cpu, HardDrive, RefreshCw, Server, ShieldCheck, Zap } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import type { HealthResponse, QwantoConfig, TelemetryData } from "@/lib/api"
import { getHealth, getQwantoConfig, getTelemetry } from "@/lib/api"

interface DashboardViewProps {
  baseUrl: string
  apiKey: string
  onNavigate: (view: any) => void
  activeModelName?: string
}

function valueOrUnavailable(value: unknown): string {
  return value === null || value === undefined || value === "" ? "Unavailable" : String(value)
}

export function DashboardView({ baseUrl, apiKey, onNavigate, activeModelName = "No model selected" }: DashboardViewProps) {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [telemetry, setTelemetry] = useState<TelemetryData | null>(null)
  const [config, setConfig] = useState<QwantoConfig | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")

  const refreshData = async () => {
    setLoading(true)
    setError("")
    const [healthResult, telemetryResult, configResult] = await Promise.allSettled([
      getHealth(baseUrl, apiKey),
      getTelemetry(baseUrl, apiKey),
      getQwantoConfig(baseUrl, apiKey),
    ])
    if (healthResult.status === "fulfilled") setHealth(healthResult.value)
    if (telemetryResult.status === "fulfilled") setTelemetry(telemetryResult.value)
    if (configResult.status === "fulfilled") setConfig(configResult.value)
    if (healthResult.status === "rejected" && telemetryResult.status === "rejected" && configResult.status === "rejected") {
      setError(healthResult.reason instanceof Error ? healthResult.reason.message : "Gateway is unavailable")
    }
    setLoading(false)
  }

  useEffect(() => {
    void refreshData()
  }, [baseUrl, apiKey])

  const scheduler = health?.scheduler
  const healthHardware = health?.hwinfo
  const telemetryHardware = telemetry?.hardware
  const modelName = config?.model_id || telemetry?.model_id || activeModelName
  const gpuNames = telemetryHardware?.gpu_names || (healthHardware?.gpu ? [healthHardware.gpu] : [])

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-2xl glass-panel p-5">
        <div>
          <div className="flex items-center gap-2">
            <Zap className="size-5 text-primary" />
            <h1 className="text-2xl font-black">Qwanto local dashboard</h1>
            <Badge>{health?.status || "Not connected"}</Badge>
          </div>
          <p className="text-sm text-muted-foreground mt-1">Live values are read from the selected local gateway. Unreported sensors remain unavailable.</p>
        </div>
        <Button size="sm" variant="secondary" onClick={() => void refreshData()} disabled={loading}>
          <RefreshCw className={`size-3.5 mr-1.5 ${loading ? "animate-spin" : ""}`} /> Refresh
        </Button>
      </div>

      {error && <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-3 text-sm" role="alert">{error}</div>}

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <FactCard icon={<Activity className="size-4" />} label="Requests" value={valueOrUnavailable(telemetry?.request_count)} detail="gateway counter" />
        <FactCard icon={<Cpu className="size-4" />} label="Generated tokens" value={valueOrUnavailable(telemetry?.total_tokens_generated)} detail="gateway counter" />
        <FactCard icon={<Server className="size-4" />} label="Active backend" value={valueOrUnavailable(config?.backend || telemetry?.active_backend)} detail={modelName} />
        <FactCard icon={<ShieldCheck className="size-4" />} label="Scheduler" value={scheduler ? `${scheduler.active} active` : "Unavailable"} detail={scheduler ? `${scheduler.queued} queued` : "health endpoint"} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <section className="rounded-2xl glass-panel p-5 space-y-4">
          <h2 className="font-semibold flex items-center gap-2"><Server className="size-4 text-primary" /> Active model</h2>
          <FactRow label="Model ID" value={modelName} />
          <FactRow label="Model path" value={config?.model_path || telemetry?.model_path} />
          <FactRow label="Context size" value={config?.ctx_size} />
          <FactRow label="KV slots" value={config?.kv_slots ?? health?.kv_slots} />
          <Button size="sm" variant="secondary" onClick={() => onNavigate("models")}>Manage local models</Button>
        </section>

        <section className="rounded-2xl glass-panel p-5 space-y-4">
          <h2 className="font-semibold flex items-center gap-2"><Cpu className="size-4 text-primary" /> Host telemetry</h2>
          <FactRow label="CPU cores" value={healthHardware?.cores ?? telemetryHardware?.cpu_cores} />
          <FactRow label="RAM available" value={healthHardware?.ram_avail_gb ?? telemetryHardware?.ram_available_gb ? `${healthHardware?.ram_avail_gb ?? telemetryHardware?.ram_available_gb} GB` : undefined} />
          <FactRow label="GPU count" value={healthHardware?.gpus ?? telemetryHardware?.gpus_detected} />
          <FactRow label="GPU names" value={gpuNames.length ? gpuNames.join(", ") : undefined} />
          <FactRow label="Uptime" value={telemetry?.uptime_formatted} />
        </section>
      </div>

      <section className="rounded-2xl glass-panel p-5 space-y-4">
        <h2 className="font-semibold flex items-center gap-2"><HardDrive className="size-4 text-primary" /> Honest runtime state</h2>
        <p className="text-sm text-muted-foreground">The dashboard does not synthesize throughput, temperatures, VRAM, NVMe bandwidth, or hardware identity. Open Benchmark evidence after running the local harness.</p>
        <div className="flex flex-wrap gap-2">
          <Button size="sm" onClick={() => onNavigate("benchmarks")}>Open benchmark evidence</Button>
          <Button size="sm" variant="secondary" onClick={() => onNavigate("doctor")}>Run local diagnostics</Button>
          <Button size="sm" variant="secondary" onClick={() => onNavigate("security")}>Review security boundary</Button>
        </div>
      </section>
    </div>
  )
}

function FactCard({ icon, label, value, detail }: { icon: ReactNode; label: string; value: string; detail: string }) {
  return (
    <div className="rounded-2xl glass-panel p-4 space-y-2">
      <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-muted-foreground">{icon}{label}</div>
      <div className="text-2xl font-semibold font-mono text-primary break-words">{value}</div>
      <div className="text-xs text-muted-foreground break-words">{detail}</div>
    </div>
  )
}

function FactRow({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="flex flex-wrap justify-between gap-3 border-b border-border/60 pb-2 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-mono text-right break-all">{valueOrUnavailable(value)}</span>
    </div>
  )
}
