import { describe, expect, it } from "vitest"
import { chooseRecommendedModel, classifyGatewayFailure, gatewayStateFromHealth, modelIsSelectable } from "./gateway"

const compatibleModel = {
  name: "verified.qwn",
  path: "C:/models/verified.qwn",
  type: "qwn",
  compatibility_state: "compatible",
  qwn_validation: { status: "passed" },
  supported_by_qwnrun: true,
  hardware_fit: { status: "fit" },
  recommended: true,
} as const

describe("gateway state", () => {
  it("distinguishes a static server 404 from a stopped gateway", () => {
    expect(classifyGatewayFailure(new Error("HTTP 404: Not Found"))).toBe("wrong-server")
    expect(classifyGatewayFailure(new Error("Failed to fetch"))).toBe("not-running")
  })

  it("accepts the versioned Qwanto health schema", () => {
    expect(gatewayStateFromHealth({ status: "ok", gateway: "qwanto", api_version: "1" })).toBe("connected")
    expect(gatewayStateFromHealth({ status: "ok", gateway: "other", api_version: "1" })).toBe("incompatible-version")
    expect(gatewayStateFromHealth({ status: "ok", gateway: "qwanto", api_version: "2" })).toBe("incompatible-version")
  })
})

describe("model recommendation", () => {
  it("only selects a validated native QWN that fits the host", () => {
    expect(modelIsSelectable(compatibleModel)).toBe(true)
    expect(chooseRecommendedModel([compatibleModel], "")).toMatchObject({ model: compatibleModel })
    expect(chooseRecommendedModel([{ ...compatibleModel, qwn_validation: { status: "failed" } }], "").model).toBeNull()
  })
})
