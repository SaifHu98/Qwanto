import React, { useState, useEffect } from "react"
import { ShieldCheck, Lock, ShieldAlert, KeyRound, CheckCircle2, AlertTriangle, RefreshCw, FileCode, Server } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { getSecurityReport, type SecurityReport } from "@/lib/api"

interface SecurityViewProps {
  baseUrl: string
  apiKey: string
}

export function SecurityView({ baseUrl, apiKey }: SecurityViewProps) {
  const [data, setData] = useState<SecurityReport | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")

  const loadSecurity = async () => {
    setLoading(true)
    setError("")
    try {
      const report = await getSecurityReport(baseUrl, apiKey)
      setData(report)
    } catch (err: any) {
      setError(err?.message || "Failed to load security audit report")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadSecurity()
  }, [baseUrl, apiKey])

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <div className="flex items-center justify-between border-b border-border pb-4">
        <div>
          <h2 className="text-xl font-bold flex items-center gap-2">
            <Lock className="size-5 text-primary" /> Security & Defense Audit
          </h2>
          <p className="text-xs text-muted-foreground mt-1">
            Proactive security posture verification, path traversal isolation, and API protection controls.
          </p>
        </div>
        <Button size="sm" variant="secondary" onClick={loadSecurity}>
          <RefreshCw className={`size-3.5 mr-1.5 ${loading ? "animate-spin" : ""}`} /> Run Audit
        </Button>
      </div>

      {error && (
        <div className="p-3 bg-red-950/40 border border-red-800/50 rounded-lg text-xs text-red-300">
          {error}
        </div>
      )}

      {loading ? (
        <div className="text-center py-12 text-muted-foreground text-sm">Auditing server security settings...</div>
      ) : data ? (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* API Authentication & Timing Security */}
            <div className="p-5 border border-border bg-card rounded-xl space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold flex items-center gap-2">
                  <KeyRound className="size-4 text-emerald-400" /> Authentication & Key Enforcement
                </h3>
                <Badge className={data.api_key_protected ? "bg-emerald-950 text-emerald-300 border-emerald-800" : "bg-amber-950 text-amber-300 border-amber-800"}>
                  {data.api_key_protected ? "API Key Enforced" : "Open Access"}
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground">
                {data.api_key_protected
                  ? "All API requests require a valid Bearer token authentication header."
                  : "Server is running without QWANTO_API_KEY set. Anyone with network access can query the endpoint."}
              </p>
              <div className="pt-2 flex items-center justify-between text-xs font-mono border-t border-border/40">
                <span className="text-muted-foreground">Timing Attack Protection:</span>
                <span className="text-emerald-400 font-semibold">Constant-time HMAC comparison</span>
              </div>
            </div>

            {/* Path Traversal & Isolation */}
            <div className="p-5 border border-border bg-card rounded-xl space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold flex items-center gap-2">
                  <ShieldCheck className="size-4 text-blue-400" /> Path Traversal Isolation
                </h3>
                <Badge className="bg-emerald-950 text-emerald-300 border-emerald-800">
                  Guarded
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground">
                Path resolution enforces canonical bounds. Arbitrary path deletion and directory traversal outside designated model search roots are strictly blocked.
              </p>
              <div className="pt-2 flex items-center justify-between text-xs font-mono border-t border-border/40">
                <span className="text-muted-foreground">Path Boundary Enforcement:</span>
                <span className="text-emerald-400 font-semibold">Project & Model Roots Only</span>
              </div>
            </div>
          </div>

          {/* Defense Headers & CORS Overview */}
          <div className="p-5 border border-border bg-card/60 rounded-xl space-y-3">
            <h3 className="text-sm font-semibold flex items-center gap-2">
              <ShieldAlert className="size-4 text-primary" /> HTTP Security Headers & Origin Validation
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs font-mono">
              <div className="p-3 bg-background/80 border border-border/50 rounded-lg space-y-1">
                <span className="text-muted-foreground block text-[10px] uppercase">X-Content-Type-Options</span>
                <span className="text-emerald-400 font-bold">nosniff</span>
              </div>
              <div className="p-3 bg-background/80 border border-border/50 rounded-lg space-y-1">
                <span className="text-muted-foreground block text-[10px] uppercase">X-Frame-Options</span>
                <span className="text-emerald-400 font-bold">DENY</span>
              </div>
              <div className="p-3 bg-background/80 border border-border/50 rounded-lg space-y-1">
                <span className="text-muted-foreground block text-[10px] uppercase">Max Body Limit</span>
                <span className="text-foreground font-bold">{(data.max_request_body_bytes / (1024 * 1024)).toFixed(0)} MB</span>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}
