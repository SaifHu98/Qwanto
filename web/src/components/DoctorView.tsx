import React, { useState, useEffect } from "react"
import { ShieldCheck, CheckCircle2, AlertTriangle, XCircle, MinusCircle, RefreshCw, Terminal } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { getDoctorReport, type DoctorReport, type DoctorCheck } from "@/lib/api"
import { GatewayRequired } from "@/components/GatewayRequired"

interface DoctorViewProps {
  baseUrl: string
  apiKey: string
  gatewayReady: boolean
}

export function DoctorView({ baseUrl, apiKey, gatewayReady }: DoctorViewProps) {
  const [report, setReport] = useState<DoctorReport | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")

  const runDiagnostics = async () => {
    if (!gatewayReady) {
      setReport(null)
      setError("Connect to the Qwanto gateway before running diagnostics.")
      setLoading(false)
      return
    }
    setLoading(true)
    setError("")
    try {
      const data = await getDoctorReport(baseUrl, apiKey)
      setReport(data)
    } catch (err: any) {
      setError(err?.message || "Failed to execute diagnostic report")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    runDiagnostics()
  }, [baseUrl, apiKey, gatewayReady])

  const renderStatusBadge = (status: DoctorCheck["status"]) => {
    switch (status) {
      case "pass":
        return (
          <Badge className="bg-emerald-950/80 text-emerald-300 border-emerald-800/60 font-mono text-[10px] gap-1">
            <CheckCircle2 className="size-3" /> PASS
          </Badge>
        )
      case "warn":
        return (
          <Badge className="bg-amber-950/80 text-amber-300 border-amber-800/60 font-mono text-[10px] gap-1">
            <AlertTriangle className="size-3" /> WARN
          </Badge>
        )
      case "fail":
        return (
          <Badge className="bg-red-950/80 text-red-300 border-red-800/60 font-mono text-[10px] gap-1">
            <XCircle className="size-3" /> FAIL
          </Badge>
        )
      default:
        return (
          <Badge className="border border-border text-muted-foreground font-mono text-[10px] gap-1">
            <MinusCircle className="size-3" /> SKIP
          </Badge>
        )
    }
  }

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <div className="flex items-center justify-between border-b border-border pb-4">
        <div>
          <h2 className="text-xl font-bold flex items-center gap-2">
            <ShieldCheck className="size-5 text-primary" /> System Doctor & Health Diagnostics
          </h2>
          <p className="text-xs text-muted-foreground mt-1">
            Verifies engine installation integrity, CUDA linkage, storage permissions, and model configs.
          </p>
        </div>
        <Button size="sm" onClick={runDiagnostics} disabled={loading || !gatewayReady}>
          <RefreshCw className={`size-3.5 mr-1.5 ${loading ? "animate-spin" : ""}`} />
          Re-run Doctor
        </Button>
      </div>

      {error && (
        <div className="p-3 bg-red-950/40 border border-red-800/50 rounded-lg text-xs text-red-300">
          {error}
        </div>
      )}

      {!gatewayReady && <GatewayRequired message="Diagnostics are paused until the configured gateway answers its health check." />}

      {loading ? (
        <div className="text-center py-12 text-muted-foreground text-sm">Running diagnostic verification...</div>
      ) : report ? (
        <div className="space-y-3">
          {report.checks?.map((check) => (
            <div
              key={check.id}
              className="p-4 border border-border bg-card/80 rounded-xl flex items-start justify-between gap-4"
            >
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-xs font-bold text-foreground">{check.id}</span>
                  {renderStatusBadge(check.status)}
                </div>
                <p className="text-xs text-muted-foreground">{check.summary}</p>
                {check.details && (
                  <pre className="mt-2 p-2 bg-background/80 border border-border/40 rounded text-[11px] font-mono text-muted-foreground overflow-x-auto">
                    {JSON.stringify(check.details, null, 2)}
                  </pre>
                )}
              </div>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  )
}
