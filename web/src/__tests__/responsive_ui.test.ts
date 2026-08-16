import { describe, expect, it } from "vitest"
import { readFileSync } from "node:fs"
import { resolve } from "node:path"

describe("responsive local agent layout", () => {
  it("keeps desktop and laptop breakpoints for both side panels", () => {
    const css = readFileSync(resolve(process.cwd(), "src/index.css"), "utf8")
    expect(css).toContain(".desktop-workspace { display: grid; grid-template-columns: minmax(0, 1fr) 310px")
    expect(css).toContain("@media (max-width: 1050px)")
    expect(css).toContain("@media (max-width: 760px)")
    expect(css).toContain(".desktop-inspector { border-top: 1px solid")
  })
})
