import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import { DesktopAgentView } from "@/components/DesktopAgentView"

const props = {
  gateway: { state: "ready", api_url: "http://127.0.0.1:41234", port: 41234, error: null, sidecar_packaged: true },
  gatewayState: "connected" as const,
  gatewayMessage: "Local gateway ready.",
  connected: true,
  onProbe: () => undefined,
  model: "fixture.qwn",
  models: ["fixture.qwn"],
  discoveredModels: [],
  onSelectModel: () => undefined,
  onLoadModel: () => undefined,
  onStopModel: () => undefined,
  loadingModel: false,
  mode: "plan" as const,
  onModeChange: () => undefined,
  messages: [],
  draft: "",
  onDraftChange: () => undefined,
  onSend: () => undefined,
  onStopGeneration: () => undefined,
  onClear: () => undefined,
  loading: false,
  error: "",
}

describe("desktop agent information architecture", () => {
  it("exposes only the five primary agent destinations", () => {
    const html = renderToStaticMarkup(<DesktopAgentView {...props} />)
    const nav = html.match(/<nav[^>]*data-testid="desktop-primary-nav"[\s\S]*?<\/nav>/)?.[0] || ""
    expect(nav).toBeTruthy()
    for (const label of ["Project", "Chats", "Files", "Changes", "Settings"]) expect(nav).toContain(label)
    expect((nav.match(/desktop-nav-item/g) || []).length).toBe(5)
    for (const label of ["Dashboard", "Converter", "Telemetry", "Benchmarks", "Security", "Workbench", "Doctor", "Brain", "Models", "Logs"]) expect(html).not.toContain(`>${label}<`)
    expect(html).toContain("data-testid=\"desktop-agent-shell\"")
  })

  it("starts with a calm no-model state and a collapsible inspector affordance", () => {
    const html = renderToStaticMarkup(<DesktopAgentView {...props} model="" />)
    expect(html).toContain("Choose a validated QWN model")
    expect(html).toContain("Show inspector")
    expect(html).not.toContain("Selected file")
  })
})
