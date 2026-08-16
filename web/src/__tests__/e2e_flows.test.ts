import { afterEach, describe, expect, it, vi } from "vitest"
import {
  getBenchmarks,
  getHealth,
  getConversionStatus,
  loadModel,
  startConversion,
  streamChat,
  type ChatMessage,
  type HealthResponse,
} from "../lib/api"

afterEach(() => vi.unstubAllGlobals())

describe("local gateway lifecycle", () => {
  it("probes health, loads a local model, and streams a response", async () => {
    const mockHealth: HealthResponse = {
      status: "ok",
      scheduler: {
        active: 1,
        capacity: 2,
        queued: 0,
        max_queue: 4,
        queue_timeout_seconds: 60,
        admitted: 1,
        completed: 0,
        rejected: 0,
        timed_out: 0,
        cancelled: 0,
      },
      kv_slots: 1,
      hwinfo: {
        cores: 1,
        ram_total_gb: 1,
        ram_avail_gb: 1,
        gpus: 0,
        vram_total_gb: 0,
        cpu: "TEST_FIXTURE CPU",
        gpu: "",
      },
    }

    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.endsWith("/health")) return Promise.resolve(new Response(JSON.stringify(mockHealth)))
      if (url.endsWith("/qwanto/load")) return Promise.resolve(new Response(JSON.stringify({ status: "loaded", model_id: "fixture.qwn", backend: "native" })))
      if (url.endsWith("/chat/completions")) {
        const stream = "data: {\"choices\":[{\"delta\":{\"content\":\"fixture response\"}}]}\n\ndata: [DONE]\n\n"
        return Promise.resolve(new Response(stream, { headers: { "content-type": "text/event-stream" } }))
      }
      return Promise.reject(new Error(`Unknown fixture route: ${url}`))
    })
    vi.stubGlobal("fetch", fetchMock)

    const health = await getHealth("http://localhost:8000/v1", "")
    expect(health.status).toBe("ok")
    expect(health.hwinfo?.gpus).toBe(0)

    const loadRes = await loadModel("http://localhost:8000/v1", "fixture.qwn", "native")
    expect(loadRes.status).toBe("loaded")

    const chunks: string[] = []
    const messages: ChatMessage[] = [{ id: "fixture", role: "user", content: "fixture prompt" }]
    await streamChat({
      baseUrl: "http://localhost:8000/v1",
      apiKey: "",
      model: "fixture.qwn",
      messages,
      temperature: 0,
      maxTokens: 8,
      enableThinking: false,
      signal: new AbortController().signal,
      onDelta: (delta) => chunks.push(delta),
    })
    expect(chunks.join("")).toBe("fixture response")
  })
})

describe("benchmark evidence lifecycle", () => {
  it("renders an unavailable result without inventing metrics", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({
      classification: "UNAVAILABLE",
      evidence: null,
      message: "TEST_FIXTURE: no real qwnrun evidence was supplied",
    }))))

    const report = await getBenchmarks("http://localhost:8000/v1", "")
    expect(report.classification).toBe("UNAVAILABLE")
    expect(report.evidence).toBeNull()
    expect(report.baseline).toBeUndefined()
  })
})

describe("conversion status lifecycle", () => {
  it("preserves unavailable conversion telemetry instead of fabricating a speed", async () => {
    const fetchMock = vi.fn().mockImplementation((url: string, init?: RequestInit) => {
      if (url.endsWith("/qwanto/convert") && init?.method === "POST") {
        return Promise.resolve(new Response(JSON.stringify({ status: "converting", output: "fixture.qwn", message: "TEST_FIXTURE conversion" })))
      }
      if (url.endsWith("/qwanto/convert/status")) {
        return Promise.resolve(new Response(JSON.stringify({
          status: "done",
          progress: 100,
          source: "fixture.safetensors",
          output: "fixture.qwn",
          quant: "q4_0",
          message: "TEST_FIXTURE conversion complete",
          elapsed: null,
          speed_mb_s: null,
        })))
      }
      return Promise.reject(new Error(`Unknown fixture route: ${url}`))
    })
    vi.stubGlobal("fetch", fetchMock)

    const conversion = await startConversion("http://localhost:8000/v1", "", "fixture.safetensors", "fixture.qwn", "q4_0")
    expect(conversion.status).toBe("converting")
    const status = await getConversionStatus("http://localhost:8000/v1", "")
    expect(status.status).toBe("done")
    expect(status.speed_mb_s).toBeNull()
  })
})
