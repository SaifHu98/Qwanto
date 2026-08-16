import type { DiscoveredModel, HealthResponse } from "./api"

export type GatewayConnectionState =
  | "not-running"
  | "wrong-server"
  | "incompatible-version"
  | "connected"

export function classifyGatewayFailure(error: unknown): GatewayConnectionState {
  const message = error instanceof Error ? error.message.toLowerCase() : String(error).toLowerCase()
  if (message.includes("404") || message.includes("not found") || message.includes("static")) {
    return "wrong-server"
  }
  if (message.includes("incompatible") || message.includes("gateway version") || message.includes("api version")) {
    return "incompatible-version"
  }
  return "not-running"
}

export function gatewayStateFromHealth(health: HealthResponse): GatewayConnectionState {
  if (health.gateway && health.gateway !== "qwanto") return "incompatible-version"
  if (health.api_version && health.api_version !== "1") return "incompatible-version"
  return health.status === "ok" ? "connected" : "incompatible-version"
}

export function modelIsSelectable(model: DiscoveredModel | undefined): boolean {
  return Boolean(
    model &&
      model.type === "qwn" &&
      model.compatibility_state === "compatible" &&
      model.qwn_validation?.status === "passed" &&
      model.supported_by_qwnrun &&
      model.hardware_fit?.status === "fit",
  )
}

export function chooseRecommendedModel(
  models: DiscoveredModel[],
  explicitModel: string,
): { model: DiscoveredModel | null; reason: string } {
  const explicit = models.find((candidate) => candidate.path === explicitModel || candidate.name === explicitModel)
  if (modelIsSelectable(explicit)) {
    return { model: explicit!, reason: "Selected explicitly by the user; QWN validation and local fit passed." }
  }
  const recommended = models.find((candidate) => modelIsSelectable(candidate) && candidate.recommended)
  if (recommended) return { model: recommended, reason: recommended.recommendation_reason || "Verified local QWN recommendation." }
  return { model: null, reason: "No compatible, validated, hardware-fit QWN model is available." }
}

