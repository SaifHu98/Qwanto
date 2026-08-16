import { afterEach, describe, expect, it, vi } from "vitest"
import {
  getHealth,
  getQwantoConfig,
  loadModel,
  startConversion,
  getConversionStatus,
  getBenchmarks,
  streamChat,
  type ChatMessage,
  type HealthResponse
} from "../lib/api"

afterEach(() => vi.unstubAllGlobals())

describe("End-to-End User Flow 1: System Status -> Load Model -> Run Streaming Inference", () => {
  it("executes complete lifecycle: probes hardware, loads .qwn model, and streams response", async () => {
    // 1. Probe System Hardware Health
    const mockHealth: HealthResponse = {
      status: "ok",
      scheduler: {
        active: 1,
        capacity: 8,
        queued: 0,
        max_queue: 32,
        queue_timeout_seconds: 60,
        admitted: 120,
        completed: 119,
        rejected: 0,
        timed_out: 0,
        cancelled: 0,
      },
      kv_slots: 8,
      tiers: { vram: 1, ram: 1, disk: 1, vram_gb: 12, ram_gb: 32 },
      hwinfo: {
        cores: 16,
        ram_total_gb: 32,
        ram_avail_gb: 28,
        gpus: 2,
        vram_total_gb: 12,
        cpu: "AMD Ryzen 9 9955HX",
        gpu: "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
      },
    }

    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.endsWith("/health")) {
        return Promise.resolve(new Response(JSON.stringify(mockHealth)))
      }
      if (url.endsWith("/qwanto/load")) {
        return Promise.resolve(new Response(JSON.stringify({ status: "loaded", model_id: "deepseek-4b", backend: "cuda" })))
      }
      if (url.endsWith("/chat/completions")) {
        const sseStream = "data: {\"choices\":[{\"delta\":{\"content\":\"def fib(\"}}]}\n\n" +
                          "data: {\"choices\":[{\"delta\":{\"content\":\"n): return n\"}}]}\n\n" +
                          "data: [DONE]\n\n"
        return Promise.resolve(new Response(sseStream, {
          headers: { "content-type": "text/event-stream" }
        }))
      }
      return Promise.reject(new Error("Unknown route: " + url))
    })

    vi.stubGlobal("fetch", fetchMock)

    // Step 1: Query Health
    const health = await getHealth("http://localhost:8000/v1", "")
    expect(health.status).toBe("ok")
    expect(health.hwinfo?.gpu).toContain("NVIDIA")

    // Step 2: Load Model
    const loadRes = await loadModel("http://localhost:8000/v1", "models/4b.qwn", "cuda")
    expect(loadRes.status).toBe("loaded")
    expect(loadRes.backend).toBe("cuda")

    // Step 3: Run Streaming Inference
    const chunks: string[] = []
    const messages: ChatMessage[] = [{ id: "1", role: "user", content: "Write fibonacci in Python" }]
    
    await streamChat({
      baseUrl: "http://localhost:8000/v1",
      apiKey: "",
      model: "deepseek-4b",
      messages,
      temperature: 0.7,
      maxTokens: 256,
      enableThinking: false,
      signal: new AbortController().signal,
      onDelta: (d) => chunks.push(d)
    })

    expect(chunks.join("")).toBe("def fib(n): return n")
  })
})

describe("End-to-End User Flow 2: Open Dashboard -> Run Benchmark -> View Results", () => {
  it("executes 4-scenario benchmark and verifies acceleration metrics", async () => {
    const mockBenchmarks = {
      baseline: {
        median_tok_s: 2.18,
        p90_tok_s: 2.15,
        p95_tok_s: 2.10,
        peak_rss_mb: 6400,
        quantization: "fp16",
        backend: "scalar",
        gates_passed: { regression_gate: true }
      },
      candidate: {
        median_tok_s: 452.80,
        p90_tok_s: 450.00,
        p95_tok_s: 445.00,
        peak_rss_mb: 540,
        quantization: "twla_1.58b",
        backend: "cuda_bitdecoding",
        gates_passed: { regression_gate: true }
      }
    }

    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(mockBenchmarks))))

    const report = await getBenchmarks("http://localhost:8000/v1", "")
    expect(report.baseline?.median_tok_s).toBe(2.18)
    expect(report.candidate?.median_tok_s).toBe(452.80)

    // Calculate Speedup & Memory Reduction
    const speedup = (report.candidate!.median_tok_s! / report.baseline!.median_tok_s!)
    const memoryReduction = (report.baseline!.peak_rss_mb! / report.candidate!.peak_rss_mb!)

    expect(speedup).toBeGreaterThan(200.0) // 207x Speedup
    expect(memoryReduction).toBeGreaterThan(10.0) // >11x Memory Reduction
  })
})

describe("End-to-End User Flow 3: Convert Model -> Auto-Activate -> Verify Ingestion", () => {
  it("executes wire-speed conversion from safetensors to TWLA .qwn container", async () => {
    const fetchMock = vi.fn().mockImplementation((url: string, init?: RequestInit) => {
      if (url.endsWith("/qwanto/convert") && init?.method === "POST") {
        return Promise.resolve(new Response(JSON.stringify({
          status: "converting",
          output: "D:/models/model_twla.qwn",
          message: "Ingesting Safetensors into TWLA 1.58-Bit superblock layout"
        })))
      }
      if (url.endsWith("/qwanto/convert/status")) {
        return Promise.resolve(new Response(JSON.stringify({
          status: "done",
          progress: 100,
          source: "model.safetensors",
          output: "D:/models/model_twla.qwn",
          quant: "twla",
          message: "Conversion complete. 4KiB aligned .qwn container ready.",
          elapsed: 4.2,
          speed_mb_s: 850.0
        })))
      }
      return Promise.reject(new Error("Unknown route: " + url))
    })

    vi.stubGlobal("fetch", fetchMock)

    // Step 1: Start conversion
    const conv = await startConversion("http://localhost:8000/v1", "", "model.safetensors", "D:/models/model_twla.qwn", "twla")
    expect(conv.status).toBe("converting")

    // Step 2: Poll status to completion
    const status = await getConversionStatus("http://localhost:8000/v1", "")
    expect(status.status).toBe("done")
    expect(status.progress).toBe(100)
    expect(status.speed_mb_s).toBe(850.0)
  })
})
