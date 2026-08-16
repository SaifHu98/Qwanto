import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import { GatewayStatusBanner } from "@/components/GatewayStatusBanner"

describe("gateway status UI", () => {
  it.each([
    ["connected", "Gateway connected"],
    ["wrong-server", "Wrong server selected"],
    ["incompatible-version", "Incompatible gateway version"],
    ["not-running", "Gateway not running"],
  ] as const)("renders the %s state with actionable status", (state, label) => {
    const html = renderToStaticMarkup(<GatewayStatusBanner state={state} message="Test gateway message" onProbe={() => undefined} />)
    expect(html).toContain(label)
    expect(html).toContain("Test gateway message")
    expect(html).toContain(`gateway-${state}`)
  })
})
