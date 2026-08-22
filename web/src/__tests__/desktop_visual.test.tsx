import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { renderToStaticMarkup } from "react-dom/server"
import { describe, expect, it } from "vitest"
import { DesktopSettingsView } from "@/components/DesktopSettingsView"

const css = readFileSync(resolve(process.cwd(), "src/index.css"), "utf8")

const settingsProps = {
  baseUrl: "http://127.0.0.1:41234/v1",
  apiKey: "",
  gatewayReady: false,
  logs: [],
  model: "",
  models: [],
  onSelectModel: () => undefined,
  onActivateModel: () => undefined,
  loadingModel: false,
  profile: "balanced" as const,
  onProfileChange: () => undefined,
  threadMode: "auto" as const,
  manualThreads: 4,
  onThreadModeChange: () => undefined,
  onManualThreadsChange: () => undefined,
  usage: { promptTokens: null, completionTokens: null, totalTokens: null, elapsedMs: null, ttftMs: null, tokensPerSecond: null, contextUse: null, toolCalls: 0, queueState: "idle" },
}

describe("desktop settings visual contract", () => {
  it.each([
    ["desktop-1280", 1280, 820],
    ["desktop-1440", 1440, 900],
    ["desktop-1920", 1920, 1080],
    ["laptop-short", 1280, 650],
  ])("keeps the %s layout contract at %d x %d", (_name, width, height) => {
    const html = renderToStaticMarkup(<DesktopSettingsView {...settingsProps} />)
    expect(html).toContain('data-testid="desktop-settings-layout"')
    expect(html).toContain('data-settings-tab="models"')
    for (const section of ["models", "runtime", "agent", "memory", "skills", "github", "privacy", "diagnostics", "feedback"]) expect(html).toContain(`data-settings-tab="${section}"`)
    expect(html).toContain('aria-orientation="vertical"')
    expect(css).toContain(".desktop-settings-layout { display: grid; grid-template-columns: 220px")
    expect(css).toContain(".settings-section-nav button.active")
    if (width <= 760) expect(css).toContain(".desktop-settings-layout { grid-template-columns: 1fr;")
    if (height <= 700) expect(css).toContain("@media (max-height: 700px) and (min-width: 761px)")
  })

  it("defines the verification modal status classes", () => {
    expect(css).toContain(".verification-status-bar.passed")
    expect(css).toContain(".verification-status-bar.failed")
    expect(css).toContain(".verification-facts { display: grid")
    expect(css).toContain(".verification-invariants {")
    expect(css).toContain(".verification-section-header")
  })
})
