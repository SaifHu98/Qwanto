import { Button } from "@/components/ui/button"
import type { GatewayConnectionState } from "@/lib/gateway"

const labels: Record<GatewayConnectionState, string> = {
  connected: "Gateway connected",
  starting: "Starting local gateway",
  "model-required": "Gateway ready · model required",
  failed: "Gateway failed",
  "wrong-server": "Wrong server selected",
  "incompatible-version": "Incompatible gateway version",
  "not-running": "Gateway not running",
}

export function GatewayStatusBanner({
  state,
  message,
  onProbe,
  probing = false,
}: {
  state: GatewayConnectionState
  message: string
  onProbe: () => void
  probing?: boolean
}) {
  return (
    <div className={`gateway-banner gateway-${state}`} role="status" aria-live="polite">
      <span className="gateway-banner-dot" />
      <div><strong>{labels[state]}</strong><p>{message}</p></div>
      {state !== "connected" && <Button size="sm" variant="secondary" onClick={onProbe} disabled={probing}>{probing ? "Probing..." : "Probe again"}</Button>}
    </div>
  )
}
