import { useEffect, useState } from "react"
import { BarChart3, RefreshCw } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { getBenchmarks, type BenchmarkEvidence, type BenchmarkReport } from "@/lib/api"

interface BenchmarksViewProps {
  baseUrl: string
  apiKey: string
}

function displayValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "Unavailable"
  if (typeof value === "object") return JSON.stringify(value)
  return String(value)
}

function EvidenceCard({ evidence }: { evidence: BenchmarkEvidence }) {
  const measured = evidence.measured_evidence
  const host = evidence.host_environment || {}
  const model = evidence.model_metadata || {}
  const runtime = evidence.runtime_metadata || {}

  return (
    <div className="rounded-2xl glass-panel p-5 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-wider text-muted-foreground">Evidence classification</p>
          <div className="flex items-center gap-2 mt-1">
            <Badge className="font-mono">{evidence.evidence_classification}</Badge>
            {evidence.schema_version && <span className="text-xs text-muted-foreground">schema {evidence.schema_version}</span>}
          </div>
        </div>
        <span className="text-xs font-mono text-muted-foreground">{displayValue(evidence.timestamp_utc)}</span>
      </div>

      {evidence.error_reason && (
        <div className="rounded-lg border border-amber-500/40 bg-amber-950/20 px-3 py-2 text-sm text-amber-200">
          {evidence.error_reason}
        </div>
      )}

      {measured ? (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Metric label="Generated tokens" value={displayValue(measured.generated_tokens)} />
          <Metric label="Wall seconds" value={displayValue(measured.wall_seconds)} />
          <Metric label="Throughput" value={`${displayValue(measured.tok_per_sec)} tok/s`} />
          <Metric label="TTFT" value={measured.ttft_ms == null ? "Unavailable" : `${measured.ttft_ms} ms`} />
        </div>
      ) : (
        <p className="text-sm text-muted-foreground">No measured throughput is available for this run.</p>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs font-mono">
        <FactGroup title="Runtime" values={{ path: runtime.executable_path, sha256: runtime.executable_sha256 }} />
        <FactGroup title="Model" values={{ path: model.path, bytes: model.file_size_bytes, sha256: model.sha256 }} />
        <FactGroup title="Host" values={{ os: host.os, cpu: host.cpu_model, threads: host.cpu_threads }} />
      </div>
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border bg-card/60 p-3">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className="mt-1 text-lg font-semibold font-mono text-primary">{value}</div>
    </div>
  )
}

function FactGroup({ title, values }: { title: string; values: Record<string, unknown> }) {
  return (
    <div className="rounded-lg border border-border bg-card/40 p-3 space-y-1">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{title}</div>
      {Object.entries(values).map(([key, value]) => (
        <div key={key} className="flex gap-2">
          <span className="text-muted-foreground">{key}:</span>
          <span className="break-all">{displayValue(value)}</span>
        </div>
      ))}
    </div>
  )
}

export function BenchmarksView({ baseUrl, apiKey }: BenchmarksViewProps) {
  const [data, setData] = useState<BenchmarkReport | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")

  const loadData = async () => {
    setLoading(true)
    setError("")
    try {
      setData(await getBenchmarks(baseUrl, apiKey))
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load benchmark evidence")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void loadData()
  }, [baseUrl, apiKey])

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-border pb-4">
        <div>
          <h1 className="text-2xl font-black flex items-center gap-2">
            <BarChart3 className="size-6 text-primary" /> Benchmark evidence
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Values appear only when the connected gateway exposes a real local qwnrun evidence artifact.
          </p>
        </div>
        <Button size="sm" variant="secondary" onClick={() => void loadData()} disabled={loading}>
          <RefreshCw className={`size-3.5 mr-1.5 ${loading ? "animate-spin" : ""}`} /> Refresh
        </Button>
      </div>

      {error && <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-3 text-sm" role="alert">{error}</div>}
      {!loading && data?.evidence && <EvidenceCard evidence={data.evidence} />}
      {!loading && !data?.evidence && (
        <div className="rounded-2xl glass-panel p-8 text-center space-y-2">
          <Badge>{data?.classification || "UNAVAILABLE"}</Badge>
          <h2 className="text-lg font-semibold">No benchmark evidence on this host</h2>
          <p className="text-sm text-muted-foreground max-w-xl mx-auto">
            Run the local benchmark harness with a real `.qwn` model and `qwnrun`, then reconnect this dashboard. Missing evidence is not replaced with a projection.
          </p>
          {data?.message && <p className="text-xs text-muted-foreground font-mono">{data.message}</p>}
        </div>
      )}
    </div>
  )
}
