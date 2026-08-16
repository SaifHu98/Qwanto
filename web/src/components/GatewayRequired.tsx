import { PlugZap } from "lucide-react"

export function GatewayRequired({ message = "Connect to the Qwanto gateway before requesting local data." }: { message?: string }) {
  return (
    <div className="empty-state gateway-required" role="status">
      <PlugZap className="size-8 text-primary" />
      <h2>Gateway connection required</h2>
      <p>{message}</p>
    </div>
  )
}
