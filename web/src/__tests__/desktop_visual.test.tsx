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
  usage: { promptTokens: null, completionTokens: null, totalTokens: null, elapsedMs: null, ttftMs: null, tokensPerSecond: null, contextUse: null, toolCalls: 0, queueState: "idle" },
}

describe("desktop settings visual contract", () => {
  it.each([
    ["desktop-1280", 1280, 820],
    ["desktop-1440", 1440, 900],
    ["laptop-short", 1280, 650],
  ])("keeps the %s layout contract at %d x %d", (_name, width, height) => {
    const html = renderToStaticMarkup(<DesktopSettingsView {...settingsProps} />)
    expect(html).toContain('data-testid="desktop-settings-layout"')
    expect(html).toContain('data-settings-tab="models"')
    expect(html).toContain('data-settings-tab="privacy"')
    expect(html).toContain('aria-orientation="vertical"')
    expect(css).toContain(".desktop-settings-layout { display: grid; grid-template-columns: 208px")
    expect(css).toContain(".settings-section-nav button.active")
    if (width <= 760) expect(css).toContain(".desktop-settings-layout { grid-template-columns: 1fr;")
    if (height <= 700) expect(css).toContain("@media (max-height: 700px) and (min-width: 761px)")
  })
})
