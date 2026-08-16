export type ChatRole = "system" | "user" | "assistant"

export interface ChatMessage {
  id: string
  role: ChatRole
  content: string
}

interface OpenAIError {
  error?: { message?: string }
}

export interface SchedulerHealth {
  active: boolean | number
  capacity?: number
  queued: number
  max_queue: number
  queue_timeout_seconds: number
  admitted: number
  completed: number
  rejected: number
  timed_out: number
  cancelled: number
}

export interface TiersHealth {
  vram: number
  ram: number
  disk: number
  vram_gb: number
  ram_gb: number
}

export interface HwinfoHealth {
  cores: number
  ram_total_gb: number
  ram_avail_gb: number
  gpus: number
  vram_total_gb: number
  cpu: string
  gpu: string
}

export interface HealthResponse {
  status: string
  gateway?: string
  api_version?: string
  gateway_version?: string
  endpoints?: {
    health: string
    models: string
    config: string
    telemetry: string
  }
  scheduler?: SchedulerHealth
  kv_slots?: number
  tiers?: TiersHealth
  hwinfo?: HwinfoHealth
}

export interface TokenUsage {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
}

export interface StreamChatResult {
  finishReason: string | null
  usage: TokenUsage | null
  requestId: string | null
  queueWaitMs: number | null
}

export function endpoint(baseUrl: string, path: string) {
  return `${baseUrl.replace(/\/+$/, "")}/${path.replace(/^\/+/, "")}`
}

export function serverEndpoint(baseUrl: string, path: string) {
  return endpoint(baseUrl.replace(/\/v1\/?$/, ""), path)
}

export function isLocalEndpoint(baseUrl: string): boolean {
  try {
    const parsed = new URL(baseUrl)
    return parsed.protocol === "http:" && ["127.0.0.1", "localhost", "[::1]", "::1"].includes(parsed.hostname)
  } catch {
    return false
  }
}

function headers(apiKey: string) {
  return {
    "Content-Type": "application/json",
    ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
  }
}

async function responseError(response: Response) {
  const fallback = `${response.status} ${response.statusText}`
  try {
    const body = (await response.json()) as OpenAIError
    return body.error?.message || fallback
  } catch {
    return fallback
  }
}

export async function listModels(baseUrl: string, apiKey: string, signal?: AbortSignal) {
  const response = await fetch(endpoint(baseUrl, "models"), { headers: headers(apiKey), signal })
  if (!response.ok) throw new Error(await responseError(response))
  const body = (await response.json()) as { data?: Array<{ id: string }> }
  return (body.data || []).map((model) => model.id)
}

export async function getHealth(baseUrl: string, apiKey = "", signal?: AbortSignal): Promise<HealthResponse> {
  const response = await fetch(serverEndpoint(baseUrl, "health"), { headers: headers(apiKey), signal })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as HealthResponse
}

export function extractSSE(buffer: string) {
  const frames = buffer.split(/\r?\n\r?\n/)
  const rest = frames.pop() || ""
  const data = frames.flatMap((frame) =>
    frame
      .split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart()),
  )
  return { data, rest }
}

export interface StreamChatOptions {
  baseUrl: string
  apiKey: string
  model: string
  messages: ChatMessage[]
  temperature: number
  maxTokens: number
  enableThinking: boolean
  cacheSlot?: number
  signal: AbortSignal
  onDelta: (text: string) => void
}

export async function streamChat(options: StreamChatOptions): Promise<StreamChatResult> {
  const response = await fetch(endpoint(options.baseUrl, "chat/completions"), {
    method: "POST",
    headers: headers(options.apiKey),
    signal: options.signal,
    body: JSON.stringify({
      model: options.model,
      messages: options.messages.map(({ role, content }) => ({ role, content })),
      temperature: options.temperature,
      max_completion_tokens: options.maxTokens,
      enable_thinking: options.enableThinking,
      ...(options.cacheSlot === undefined ? {} : { cache_slot: options.cacheSlot }),
      stream: true,
      stream_options: { include_usage: true },
    }),
  })
  if (!response.ok) throw new Error(await responseError(response))
  if (!response.body) throw new Error("The server returned an empty stream.")

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  let finishReason: string | null = null
  let usage: TokenUsage | null = null

  const consume = (data: string) => {
    if (data === "[DONE]") return
    const event = JSON.parse(data) as {
      choices?: Array<{ delta?: { content?: string }; finish_reason?: string | null }>
      usage?: TokenUsage | null
    }
    const choice = event.choices?.[0]
    const text = choice?.delta?.content
    if (text) options.onDelta(text)
    if (choice?.finish_reason) finishReason = choice.finish_reason
    if (event.usage) usage = event.usage
  }

  while (true) {
    const { value, done } = await reader.read()
    buffer += decoder.decode(value, { stream: !done })
    const parsed = extractSSE(buffer)
    buffer = parsed.rest
    parsed.data.forEach(consume)
    if (done) break
  }

  const queueWaitHeader = response.headers.get("x-qwanto-queue-wait-ms")
  const parsedQueueWait = queueWaitHeader === null ? null : Number(queueWaitHeader)
  return {
    finishReason,
    usage,
    requestId: response.headers.get("x-request-id"),
    queueWaitMs: parsedQueueWait !== null && Number.isFinite(parsedQueueWait) ? parsedQueueWait : null,
  }
}

export interface QwantoConfig {
  schema_version?: string
  model_id: string | null
  model_path: string
  backend: string
  proxy_url: string | null
  kv_slots: number
  max_tokens: number
  ctx_size?: number
  capabilities: {
    streaming: boolean
    tool_calls: boolean
    structured_output: boolean
    reasoning: boolean
    cancellation: boolean
    model_discovery: boolean
  }
  acquisition?: {
    converter: boolean
    downloader: boolean
    desktop_sidecar: boolean
  }
}

export interface DiscoveredModel {
  name: string
  path: string
  type: "native" | "gguf" | "qwn" | string
  compatibility_state?: string
  qwn_validation?: { status: string; reason?: string }
  supported_by_qwnrun?: boolean
  hardware_fit?: { status: string; reason?: string; available_ram_bytes?: number; available_disk_bytes?: number }
  quantization?: string
  n_tensors?: number | null
  arch_dims?: number[] | null
  recommended?: boolean
  recommendation_reason?: string
}

export interface ModelPathsResponse {
  schema_version?: string
  models: DiscoveredModel[]
  search_paths: string[]
  recommendation?: {
    model: DiscoveredModel | null
    reason: string
    evidence_source?: string | null
    measured_throughput_tok_s?: number | null
    measured_ttft_ms?: number | null
    selection_basis?: string
  }
}

export interface ResourceLimits {
  cpu: number
  ram: number
  vram: number
  disk: number
}

export interface DownloadStatus {
  status: "idle" | "downloading" | "paused" | "completed" | "error"
  filename: string
  dest_path: string
  url: string
  downloaded: number
  total: number
  speed: number
  progress: number
  error: string | null
  connections: number
  speed_limit: number
  chunks_done: number
  chunks_total: number
  provider?: string
  verification?: "verified" | "unverified" | string
  sha256?: string | null
  speed_bytes_per_sec?: number
  eta_seconds?: number | null
  partial_path?: string
  retry_count?: number
}

export interface AcquisitionProvider {
  id: string
  name: string
  network: boolean
  requires_https: boolean
  formats: string[]
  requires_license_confirmation_for_gated?: boolean
}

export async function getQwantoConfig(baseUrl: string, apiKey = ""): Promise<QwantoConfig> {
  const response = await fetch(endpoint(baseUrl, "qwanto/config"), { headers: headers(apiKey) })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as QwantoConfig
}

export async function listDiscoveredModels(baseUrl: string, apiKey = ""): Promise<ModelPathsResponse> {
  const response = await fetch(endpoint(baseUrl, "qwanto/models"), { headers: headers(apiKey) })
  if (!response.ok) throw new Error(await responseError(response))
  const body = (await response.json()) as ModelPathsResponse
  return body || { models: [], search_paths: [] }
}

export async function getModelPaths(baseUrl: string, apiKey = ""): Promise<string[]> {
  const response = await fetch(endpoint(baseUrl, "qwanto/paths"), { headers: headers(apiKey) })
  if (!response.ok) throw new Error(await responseError(response))
  const body = (await response.json()) as { paths: string[] }
  return body.paths || []
}

export async function addModelPath(baseUrl: string, path: string, apiKey = ""): Promise<string[]> {
  const response = await fetch(endpoint(baseUrl, "qwanto/paths"), {
    method: "POST",
    headers: headers(apiKey),
    body: JSON.stringify({ action: "add", path })
  })
  if (!response.ok) throw new Error(await responseError(response))
  const body = (await response.json()) as { paths: string[] }
  return body.paths || []
}

export async function removeModelPath(baseUrl: string, path: string, apiKey = ""): Promise<string[]> {
  const response = await fetch(endpoint(baseUrl, "qwanto/paths"), {
    method: "POST",
    headers: headers(apiKey),
    body: JSON.stringify({ action: "remove", path })
  })
  if (!response.ok) throw new Error(await responseError(response))
  const body = (await response.json()) as { paths: string[] }
  return body.paths || []
}

export interface AccelOptions {
  flashAttention?: boolean
  kvCacheQuant?: string
  speculativeDecoding?: boolean
  draftModelPath?: string
}

export async function loadModel(baseUrl: string, modelPath: string, backend = "auto", backendUrl?: string, apiKey = "", ctxSize?: number, accel?: AccelOptions) {
  const response = await fetch(endpoint(baseUrl, "qwanto/load"), {
    method: "POST",
    headers: headers(apiKey),
    body: JSON.stringify({
      model_path: modelPath, backend, backend_url: backendUrl, ctx_size: ctxSize,
      ...(accel ? {
        flash_attention: accel.flashAttention,
        kv_cache_quant: accel.kvCacheQuant,
        speculative_decoding: accel.speculativeDecoding,
        draft_model_path: accel.draftModelPath || "",
      } : {}),
    })
  })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as { status: string; model_id: string; backend: string }
}

export async function downloadModel(baseUrl: string, url: string, filename?: string, destPath?: string, apiKey = "", options?: { approvedHost?: string; allowLocalhostHttp?: boolean; sha256?: string; expectedSize?: number; overwrite?: boolean }) {
  const response = await fetch(endpoint(baseUrl, "qwanto/download"), {
    method: "POST",
    headers: headers(apiKey),
    body: JSON.stringify({
      url, filename, dest_path: destPath,
      allowed_hosts: options?.approvedHost ? [options.approvedHost] : undefined,
      allow_localhost_http: options?.allowLocalhostHttp || undefined,
      sha256: options?.sha256,
      expected_size: options?.expectedSize,
      overwrite: options?.overwrite || undefined,
    })
  })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as { status: string; message: string }
}

export async function setResourceLimits(baseUrl: string, resources: ResourceLimits, apiKey = "") {
  const response = await fetch(endpoint(baseUrl, "qwanto/resources"), {
    method: "POST",
    headers: headers(apiKey),
    body: JSON.stringify(resources)
  })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as { status: string; resources: ResourceLimits }
}

export async function getResourceLimits(baseUrl: string, apiKey = ""): Promise<ResourceLimits> {
  const response = await fetch(endpoint(baseUrl, "qwanto/resources"), { headers: headers(apiKey) })
  if (!response.ok) throw new Error(await responseError(response))
  const body = (await response.json()) as { resources: ResourceLimits }
  return body.resources || { cpu: 100, ram: 100, vram: 100, disk: 100 }
}

export async function getDownloadStatus(baseUrl: string, apiKey = ""): Promise<DownloadStatus> {
  const response = await fetch(endpoint(baseUrl, "qwanto/download/status"), { headers: headers(apiKey) })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as DownloadStatus
}

export async function cancelDownloadModel(baseUrl: string, apiKey = "") {
  const response = await fetch(endpoint(baseUrl, "qwanto/download/cancel"), {
    method: "POST",
    headers: headers(apiKey)
  })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as { status: string; message: string }
}

export async function pauseDownloadModel(baseUrl: string, apiKey = "") {
  const response = await fetch(endpoint(baseUrl, "qwanto/download/pause"), {
    method: "POST",
    headers: headers(apiKey)
  })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as { status: string; message: string }
}

export async function resumeDownloadModel(baseUrl: string, apiKey = "") {
  const response = await fetch(endpoint(baseUrl, "qwanto/download/resume"), {
    method: "POST",
    headers: headers(apiKey)
  })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as { status: string; message: string }
}

export async function deleteModel(baseUrl: string, modelPath: string, apiKey = "") {
  const response = await fetch(endpoint(baseUrl, "qwanto/delete"), {
    method: "POST",
    headers: headers(apiKey),
    body: JSON.stringify({ path: modelPath })
  })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as { status: string; message: string }
}

export async function configDownload(baseUrl: string, connections?: number, speedLimit?: number, apiKey = "") {
  const body: Record<string, number> = {}
  if (connections !== undefined) body.connections = connections
  if (speedLimit !== undefined) body.speed_limit = speedLimit
  const response = await fetch(endpoint(baseUrl, "qwanto/download/config"), {
    method: "POST",
    headers: headers(apiKey),
    body: JSON.stringify(body)
  })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as { status: string; connections: number; speed_limit: number }
}

export interface SystemPreset {
  id: string
  name: string
  system_prompt: string
  temperature: number
  top_p: number
  description: string
}

export interface TelemetryData {
  request_count: number
  total_tokens_generated: number
  uptime_seconds: number
  uptime_formatted: string
  active_backend: string
  model_id: string
  model_path: string
  hardware: {
    cpu_cores: number
    ram_available_gb: number
    gpus_detected: number
    gpu_names: string[]
    disk_free_bytes?: number
  }
  recent_requests: Array<Record<string, any>>
}

export interface DoctorCheck {
  id: string
  status: "pass" | "fail" | "warn" | "skip"
  summary: string
  details?: Record<string, any>
}

export interface DoctorReport {
  checks: DoctorCheck[]
  plan?: any
}

export async function getPresets(baseUrl: string, apiKey = ""): Promise<SystemPreset[]> {
  const response = await fetch(endpoint(baseUrl, "qwanto/presets"), { headers: headers(apiKey) })
  if (!response.ok) throw new Error(await responseError(response))
  const body = (await response.json()) as { presets: SystemPreset[] }
  return body.presets || []
}

export async function savePreset(baseUrl: string, preset: Partial<SystemPreset>, apiKey = ""): Promise<SystemPreset[]> {
  const response = await fetch(endpoint(baseUrl, "qwanto/presets"), {
    method: "POST",
    headers: headers(apiKey),
    body: JSON.stringify(preset)
  })
  if (!response.ok) throw new Error(await responseError(response))
  const body = (await response.json()) as { presets: SystemPreset[] }
  return body.presets || []
}

export async function deletePreset(baseUrl: string, id: string, apiKey = ""): Promise<SystemPreset[]> {
  const response = await fetch(endpoint(baseUrl, "qwanto/presets/delete"), {
    method: "POST",
    headers: headers(apiKey),
    body: JSON.stringify({ id })
  })
  if (!response.ok) throw new Error(await responseError(response))
  const body = (await response.json()) as { presets: SystemPreset[] }
  return body.presets || []
}

export async function getTelemetry(baseUrl: string, apiKey = ""): Promise<TelemetryData> {
  const response = await fetch(endpoint(baseUrl, "qwanto/telemetry"), { headers: headers(apiKey) })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as TelemetryData
}

export async function getDoctorReport(baseUrl: string, apiKey = ""): Promise<DoctorReport> {
  const response = await fetch(endpoint(baseUrl, "qwanto/doctor"), { headers: headers(apiKey) })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as DoctorReport
}

export interface BenchmarkMetrics {
  median_tok_s?: number
  p90_tok_s?: number
  p95_tok_s?: number
  peak_rss_mb?: number
  quantization?: string
  context_size?: number
  cache_state?: string
  backend?: string
  gates_passed?: Record<string, boolean>
}

export async function getAcquisitionProviders(baseUrl: string, apiKey = ""): Promise<AcquisitionProvider[]> {
  const response = await fetch(endpoint(baseUrl, "qwanto/providers"), { headers: headers(apiKey) })
  if (!response.ok) throw new Error(await responseError(response))
  const body = (await response.json()) as { providers?: AcquisitionProvider[] }
  return body.providers || []
}

export type BenchmarkClassification =
  | "MEASURED"
  | "UNAVAILABLE"
  | "INVALID"
  | "TEST_FIXTURE"
  | "EXPERIMENTAL"
  | "PROJECTED"

export interface BenchmarkEvidence {
  schema_version?: string
  benchmark_id?: string
  timestamp_utc?: string
  evidence_classification: BenchmarkClassification
  error_reason?: string | null
  host_environment?: Record<string, unknown>
  runtime_metadata?: Record<string, unknown>
  model_metadata?: Record<string, unknown>
  benchmark_parameters?: Record<string, unknown>
  execution_evidence?: Record<string, unknown>
  measured_evidence?: {
    generated_tokens?: number
    wall_seconds?: number
    tok_per_sec?: number
    ttft_ms?: number | null
  } | null
  unavailable_metrics?: Record<string, string>
}

export interface BenchmarkReport {
  baseline?: BenchmarkMetrics | null
  candidate?: BenchmarkMetrics | null
  classification?: BenchmarkClassification
  source?: string | null
  evidence?: BenchmarkEvidence | null
  message?: string | null
}

export async function getBenchmarks(baseUrl: string, apiKey = ""): Promise<BenchmarkReport> {
  const response = await fetch(endpoint(baseUrl, "qwanto/benchmarks"), { headers: headers(apiKey) })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as BenchmarkReport
}

export interface SecurityReport {
  api_key_protected: boolean
  constant_time_auth: boolean
  cors_wildcard: boolean
  cors_allowed_origins: string[]
  security_headers_active: boolean
  path_traversal_protection: boolean
  max_request_body_bytes: number
  tls_proxy_supported: boolean
}

export async function getSecurityReport(baseUrl: string, apiKey = ""): Promise<SecurityReport> {
  const response = await fetch(endpoint(baseUrl, "qwanto/security"), { headers: headers(apiKey) })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as SecurityReport
}

export interface ConversionStatus {
  status: "idle" | "converting" | "done" | "error" | "cancelled"
  source?: string
  output?: string
  quant?: string
  progress: number | null
  message: string
  error?: string | null
  elapsed?: number
  speed_mb_s?: number
  stage?: string
  manifest?: Record<string, unknown> | null
}

export async function startConversion(
  baseUrl: string,
  apiKey = "",
  source: string,
  output?: string,
  quant = "q4_0"
): Promise<{ status: string; output: string; message: string }> {
  const response = await fetch(endpoint(baseUrl, "qwanto/convert"), {
    method: "POST",
    headers: headers(apiKey),
    body: JSON.stringify({ source, output, quant })
  })
  if (!response.ok) throw new Error(await responseError(response))
  return await response.json()
}

export async function getConversionStatus(baseUrl: string, apiKey = ""): Promise<ConversionStatus> {
  const response = await fetch(endpoint(baseUrl, "qwanto/convert/status"), { headers: headers(apiKey) })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as ConversionStatus
}

export async function cancelConversion(baseUrl: string, apiKey = "") {
  const response = await fetch(endpoint(baseUrl, "qwanto/convert/cancel"), {
    method: "POST",
    headers: headers(apiKey),
  })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as { status: string; message: string }
}

