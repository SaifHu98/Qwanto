import { describe, expect, it } from "vitest"
import { isLocalEndpoint } from "../lib/api"

describe("local dashboard boundary", () => {
  it("recognizes the loopback gateway defaults", () => {
    expect(isLocalEndpoint("http://127.0.0.1:8000/v1")).toBe(true)
    expect(isLocalEndpoint("http://localhost:8000/v1")).toBe(true)
    expect(isLocalEndpoint("http://[::1]:8000/v1")).toBe(true)
  })

  it("does not silently classify remote or malformed endpoints as local", () => {
    expect(isLocalEndpoint("https://api.example.test/v1")).toBe(false)
    expect(isLocalEndpoint("not an endpoint")).toBe(false)
  })
})
