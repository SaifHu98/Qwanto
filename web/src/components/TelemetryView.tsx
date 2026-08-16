import { useEffect, useState } from "react"
import { Activity, RefreshCw } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { getTelemetry, type TelemetryData } from "@/lib/api"

interface TelemetryViewProps {
  baseUrl: string
  apiKey: string
}

function value(value: unknown): string {
  return value === null || value === undefined || value === "" ? "Unavailable" : String(value)
}

export function TelemetryView({ baseUrl, apiKey }: TelemetryViewProps) {
  const [telemetry, setTelemetry] = useState<TelemetryData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")

  const refresh = async () => {
    setLoading(true)
    try {
      setTelemetry(await getTelemetry(baseUrl, apiKey))
      setError("")
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Telemetry is unavailable")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void refresh()
  }, [baseUrl, apiKey])

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-border pb-4">
        <div>
          <h1 className="text-2xl font-black flex items-center gap-2"><Activity className="size-6 text-primary" /> Gateway telemetry</h1>
          <p className="text-sm text-muted-foreground mt-1">Only counters and hardware fields returned by the connected gateway are shown.</p>
        </div>
        <Button size="sm" variant="secondary" onClick={() => void refresh()} disabled={loading}><RefreshCw className={`size-3.5 mr-1.5 ${loading ? "animate-spin" : ""}`} /> Refresh</Button>
      </div>
      {error && <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-3 text-sm" role="alert">{error}</div>}
      {telemetry && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <Metric label="Requests" value={value(telemetry.request_count)} />
            <Metric label="Generated tokens" value={value(telemetry.total_tokens_generated)} />
            <Metric label="Uptime" value={value(telemetry.uptime_formatted)} />
            <Metric label="Backend" value={value(telemetry.active_backend)} />
          </div>
          <section className="rounded-2xl glass-panel p-5 space-y-3">
            <h2 className="font-semibold">Host fields reported by gateway</h2>
            <Row label="Model" value={telemetry.model_id} />
            <Row label="Model path" value={telemetry.model_path} />
            <Row label="CPU cores" value={telemetry.hardware.cpu_cores} />
            <Row label="RAM available" value={`${value(telemetry.hardware.ram_available_gb)} GB`} />
            <Row label="GPU count" value={telemetry.hardware.gpus_detected} />
            <Row label="GPU names" value={telemetry.hardware.gpu_names.length ? telemetry.hardware.gpu_names.join(", ") : undefined} />
          </section>
          <section className="rounded-2xl glass-panel p-5 space-y-3">
            <div className="flex items-center gap-2"><h2 className="font-semibold">Recent requests</h2><Badge>{telemetry.recent_requests.length}</Badge></div>
            {telemetry.recent_requests.length ? <pre className="text-xs font-mono whitespace-pre-wrap break-all">{JSON.stringify(telemetry.recent_requests, null, 2)}</pre> : <p className="text-sm text-muted-foreground">No recent request records.</p>}
          </section>
        </>
      )}
      {!loading && !telemetry && <p className="text-sm text-muted-foreground">No telemetry is available.</p>}
    </div>
  )
}

function Metric({ label, value: metricValue }: { label: string; value: string }) {
  return <div className="rounded-2xl glass-panel p-4"><div className="text-xs uppercase tracking-wider text-muted-foreground">{label}</div><div className="mt-2 text-xl font-mono text-primary break-words">{metricValue}</div></div>
}

function Row({ label, value: rowValue }: { label: string; value: unknown }) {
  return <div className="flex flex-wrap justify-between gap-3 border-b border-border/60 pb-2 text-sm"><span className="text-muted-foreground">{label}</span><span className="font-mono text-right break-all">{value(rowValue)}</span></div>
}
