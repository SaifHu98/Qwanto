import React, { useState, useEffect } from "react"
import { Gauge, Activity, Cpu, HardDrive, Zap, Clock, Layers, RefreshCw, CheckCircle, Server } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { getTelemetry, type TelemetryData } from "@/lib/api"

interface TelemetryViewProps {
  baseUrl: string
  apiKey: string
}

export function TelemetryView({ baseUrl, apiKey }: TelemetryViewProps) {
  const [telemetry, setTelemetry] = useState<TelemetryData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")
  const [autoRefresh, setAutoRefresh] = useState(true)

  const fetchTelemetry = async () => {
    try {
      const data = await getTelemetry(baseUrl, apiKey)
      setTelemetry(data)
      setError("")
    } catch (err: any) {
      setError(err?.message || "Failed to connect to telemetry service")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchTelemetry()
    if (!autoRefresh) return
    const interval = setInterval(fetchTelemetry, 3000)
    return () => clearInterval(interval)
  }, [baseUrl, apiKey, autoRefresh])

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <div className="flex items-center justify-between border-b border-border pb-4">
        <div>
          <h2 className="text-xl font-bold flex items-center gap-2">
            <Activity className="size-5 text-primary" /> Live Telemetry & System Performance
          </h2>
          <p className="text-xs text-muted-foreground mt-1">
            Real-time tracking of generation throughput, hardware allocation, and runtime statistics.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="secondary"
            size="sm"
            onClick={() => setAutoRefresh(!autoRefresh)}
            className="text-xs"
          >
            <RefreshCw className={`size-3.5 mr-1.5 ${autoRefresh ? "animate-spin text-primary" : ""}`} />
            {autoRefresh ? "Auto-refresh: ON" : "Auto-refresh: OFF"}
          </Button>
          <Button size="sm" onClick={fetchTelemetry}>
            Refresh Now
          </Button>
        </div>
      </div>

      {error && (
        <div className="p-3 bg-red-950/40 border border-red-800/50 rounded-lg text-xs text-red-300">
          {error}
        </div>
      )}

      {loading && !telemetry ? (
        <div className="text-center py-12 text-muted-foreground text-sm">Gathering system metrics...</div>
      ) : telemetry ? (
        <>
          {/* Key Metrics Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-4 border border-border bg-card rounded-xl space-y-1">
              <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider flex items-center gap-1.5">
                <Zap className="size-3.5 text-amber-400" /> Requests Served
              </span>
              <div className="text-2xl font-extrabold font-mono text-foreground">
                {telemetry.request_count}
              </div>
              <span className="text-[10px] text-muted-foreground">Total API requests</span>
            </div>

            <div className="p-4 border border-border bg-card rounded-xl space-y-1">
              <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider flex items-center gap-1.5">
                <Layers className="size-3.5 text-primary" /> Tokens Generated
              </span>
              <div className="text-2xl font-extrabold font-mono text-primary">
                {telemetry.total_tokens_generated.toLocaleString()}
              </div>
              <span className="text-[10px] text-muted-foreground">Total output tokens</span>
            </div>

            <div className="p-4 border border-border bg-card rounded-xl space-y-1">
              <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider flex items-center gap-1.5">
                <Clock className="size-3.5 text-blue-400" /> Engine Uptime
              </span>
              <div className="text-xl font-bold font-mono text-foreground">
                {telemetry.uptime_formatted}
              </div>
              <span className="text-[10px] text-muted-foreground">{telemetry.uptime_seconds}s active</span>
            </div>

            <div className="p-4 border border-border bg-card rounded-xl space-y-1">
              <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider flex items-center gap-1.5">
                <Server className="size-3.5 text-emerald-400" /> Active Backend
              </span>
              <div className="text-base font-bold font-mono text-emerald-400 uppercase truncate">
                {telemetry.active_backend}
              </div>
              <span className="text-[10px] text-muted-foreground truncate block">
                {telemetry.model_id || "No model loaded"}
              </span>
            </div>
          </div>

          {/* Hardware Specs Card */}
          <div className="p-5 border border-border bg-card/60 rounded-xl space-y-3">
            <h3 className="text-sm font-semibold flex items-center gap-2">
              <Cpu className="size-4 text-primary" /> Hardware & Accelerator Allocation
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-mono">
              <div className="p-3 bg-background/60 border border-border/50 rounded-lg flex items-center justify-between">
                <span className="text-muted-foreground">CPU Cores:</span>
                <span className="font-bold text-foreground">{telemetry.hardware.cpu_cores} Cores</span>
              </div>
              <div className="p-3 bg-background/60 border border-border/50 rounded-lg flex items-center justify-between">
                <span className="text-muted-foreground">Available RAM:</span>
                <span className="font-bold text-foreground">{telemetry.hardware.ram_available_gb} GB</span>
              </div>
              <div className="p-3 bg-background/60 border border-border/50 rounded-lg flex items-center justify-between">
                <span className="text-muted-foreground">GPU Acceleration:</span>
                <span className="font-bold text-primary">
                  {telemetry.hardware.gpus_detected > 0
                    ? `${telemetry.hardware.gpus_detected} GPU (${telemetry.hardware.gpu_names[0] || "Detected"})`
                    : "CPU Only"}
                </span>
              </div>
            </div>
          </div>
        </>
      ) : null}
    </div>
  )
}
