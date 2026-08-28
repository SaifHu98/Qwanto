import { afterEach, describe, expect, it, vi } from "vitest"
import {
  listModels,
  getHealth,
  getQwantoConfig,
  getAcquisitionProviders,
  listDiscoveredModels,
  getModelPaths,
  addModelPath,
  removeModelPath,
  loadModel,
  downloadModel,
  getDownloadStatus,
  cancelDownloadModel,
  pauseDownloadModel,
  resumeDownloadModel,
  deleteModel,
  configDownload,
  setResourceLimits,
  getResourceLimits,
  getPresets,
  savePreset,
  deletePreset,
  getTelemetry,
  getDoctorReport,
  getBenchmarks,
  getSecurityReport,
  createGitHubIssue,
  startConversion,
  getConversionStatus,
  extractSSE,
  streamChat,
  endpoint,
  serverEndpoint
} from "../lib/api"

afterEach(() => vi.unstubAllGlobals())

describe("API Endpoint Utility Functions", () => {
  it("constructs standard v1 endpoints correctly", () => {
    expect(endpoint("http://127.0.0.1:8000/v1", "models")).toBe("http://127.0.0.1:8000/v1/models")
    expect(endpoint("http://127.0.0.1:8000/v1/", "/chat/completions")).toBe("http://127.0.0.1:8000/v1/chat/completions")
  })

  it("constructs server root endpoints stripping v1 correctly", () => {
    expect(serverEndpoint("http://127.0.0.1:8000/v1", "health")).toBe("http://127.0.0.1:8000/health")
    expect(serverEndpoint("http://127.0.0.1:8000/v1/", "/qwanto/telemetry")).toBe("http://127.0.0.1:8000/qwanto/telemetry")
  })
})

describe("Models and Hardware Discovery API", () => {
  it("listModels fetches and extracts model IDs", async () => {
    const mockResponse = { data: [{ id: "qwanto-4b-twla" }, { id: "qwanto-27b-iq2" }] }
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(mockResponse))))

    const models = await listModels("http://localhost:8000/v1", "test-key")
    expect(models).toEqual(["qwanto-4b-twla", "qwanto-27b-iq2"])
  })

  it("getQwantoConfig retrieves active configuration and hardware capabilities", async () => {
    const mockConfig = {
      model_id: "deepseek-4b",
      model_path: "D:/models/4b.qwn",
      backend: "cuda",
      proxy_url: null,
      kv_slots: 8,
      max_tokens: 4096,
      capabilities: {
        streaming: true,
        tool_calls: true,
        structured_output: true,
        reasoning: true,
        cancellation: true,
        model_discovery: true,
      }
    }
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(mockConfig))))

    const cfg = await getQwantoConfig("http://localhost:8000/v1", "")
    expect(cfg.model_id).toBe("deepseek-4b")
    expect(cfg.backend).toBe("cuda")
    expect(cfg.capabilities.reasoning).toBe(true)
  })

  it("listDiscoveredModels lists available local checkpoints", async () => {
    const mockDiscovery = {
      models: [
        { name: "DeepSeek-V4-Pro-4B.qwn", path: "D:/models/4b.qwn", type: "qwn" },
        { name: "Qwen3.8-27B.gguf", path: "D:/models/27b.gguf", type: "gguf" }
      ],
      search_paths: ["D:/models", "C:/Users/test/models"]
    }
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(mockDiscovery))))

    const res = await listDiscoveredModels("http://localhost:8000/v1", "")
    expect(res.models.length).toBe(2)
    expect(res.models[0].type).toBe("qwn")
  })

  it("reads provider metadata without assuming a remote catalog download", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ providers: [
      { id: "huggingface", name: "Hugging Face public artifacts", network: true, requires_https: true, formats: ["gguf"] },
      { id: "local_file", name: "Local file import", network: false, requires_https: false, formats: ["gguf"] },
    ] }))))
    const providers = await getAcquisitionProviders("http://localhost:8000/v1", "")
    expect(providers.map(provider => provider.id)).toEqual(["huggingface", "local_file"])
  })
})

describe("Model Ingestion & Conversion API", () => {
  it("startConversion initiates wire-speed model compilation", async () => {
    const mockRes = { status: "converting", output: "model.qwn", message: "Conversion started" }
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(mockRes))))

    const res = await startConversion("http://localhost:8000/v1", "", "model.safetensors", "model.qwn", "twla")
    expect(res.status).toBe("converting")
    expect(res.output).toBe("model.qwn")
  })

  it("getConversionStatus reports progress and speed", async () => {
    const mockStatus = {
      status: "converting",
      progress: 65,
      message: "Quantizing tensor blocks into TWLA 1.58-bit superblocks",
      speed_mb_s: 480.5
    }
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(mockStatus))))

    const status = await getConversionStatus("http://localhost:8000/v1", "")
    expect(status.progress).toBe(65)
    expect(status.speed_mb_s).toBe(480.5)
  })
})

describe("GitHub Issue API", () => {
  it("creates an explicitly-consented issue through the local gateway", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({
      status: "created", url: "https://github.com/SaifHu98/Qwanto/issues/42", number: 42
    }))))

    const result = await createGitHubIssue("http://localhost:8000/v1", "local-key", {
      title: "Qwanto Code feedback", body: "A reproducible error.", category: "Bug", consent: true
    })

    expect(result.number).toBe(42)
    expect(fetch).toHaveBeenCalledWith("http://localhost:8000/v1/qwanto/github/issues", expect.objectContaining({
      method: "POST", headers: expect.objectContaining({ Authorization: "Bearer local-key" })
    }))
  })
})

describe("Telemetry, Doctor & Security APIs", () => {
  it("getTelemetry returns full hardware statistics and request counts", async () => {
    const mockTelemetry = {
      request_count: 520,
      total_tokens_generated: 1254000,
      uptime_seconds: 43200,
      uptime_formatted: "12h 00m 00s",
      active_backend: "TEST_FIXTURE backend",
      model_id: "deepseek-4b",
      model_path: "D:/models/4b.qwn",
      hardware: {
        cpu_cores: 16,
        ram_available_gb: 28.5,
        gpus_detected: 1,
        gpu_names: ["TEST_FIXTURE GPU"]
      },
      recent_requests: []
    }
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(mockTelemetry))))

    const data = await getTelemetry("http://localhost:8000/v1", "")
    expect(data.hardware.gpus_detected).toBe(1)
    expect(data.active_backend).toContain("TEST_FIXTURE")
  })

  it("getDoctorReport verifies system integrity", async () => {
    const mockReport = {
      checks: [
        { id: "cpu_avxvnni", status: "pass", summary: "AVX-VNNI acceleration active" },
        { id: "cuda_driver", status: "pass", summary: "NVIDIA Driver 592.02 ready" },
        { id: "nvme_mmap", status: "pass", summary: "NVMe zero-copy mmap enabled" }
      ]
    }
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(mockReport))))

    const doc = await getDoctorReport("http://localhost:8000/v1", "")
    expect(doc.checks.length).toBe(3)
    expect(doc.checks.every(c => c.status === "pass")).toBe(true)
  })

  it("getSecurityReport verifies defense headers and auth", async () => {
    const mockSec = {
      api_key_protected: true,
      constant_time_auth: true,
      cors_wildcard: false,
      cors_allowed_origins: ["http://localhost:5173"],
      security_headers_active: true,
      path_traversal_protection: true,
      max_request_body_bytes: 33554432,
      tls_proxy_supported: true
    }
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(mockSec))))

    const sec = await getSecurityReport("http://localhost:8000/v1", "")
    expect(sec.security_headers_active).toBe(true)
    expect(sec.path_traversal_protection).toBe(true)
  })
})

describe("Error Handling & Resiliency", () => {
  it("throws user-friendly error message on 500 internal server error", async () => {
    const mockErr = { error: { message: "GPU Out Of Memory: required 14.5 GB, available 10.2 GB" } }
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(mockErr), { status: 500, statusText: "Internal Server Error" })))

    await expect(getHealth("http://localhost:8000/v1", "")).rejects.toThrow("GPU Out Of Memory")
  })

  it("handles non-JSON error responses gracefully", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("Gateway Timeout", { status: 504, statusText: "Gateway Timeout" })))

    await expect(getHealth("http://localhost:8000/v1", "")).rejects.toThrow("504 Gateway Timeout")
  })
})
