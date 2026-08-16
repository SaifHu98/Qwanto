export interface DesktopGatewayStatus {
  state: "starting" | "ready" | "stopped" | "failed" | string
  api_url: string | null
  port: number | null
  error: string | null
  sidecar_packaged: boolean
}

export interface DesktopModelInfo {
  id: string
  name: string
  path: string
  path_alias: string
  size_formatted: string
  size_bytes: number
  format: string
  quantization: string
  compatibility_state: string
  metadata_status: string
  n_tensors?: number | null
  n_layers?: number | null
}

export interface DesktopToolResult {
  success: boolean
  outcome: "executed" | "needs_approval" | "denied" | string
  output: string
  error: string | null
  truncated: boolean
  approval_token: string | null
  action_details?: {
    tool_name: string
    description: string
    target_path?: string | null
    command?: string | null
    diff_preview?: string | null
  } | null
}

export interface ProjectMemory {
  schema_version: number
  workspace_root: string
  enabled: boolean
  summary: string
  architecture_notes: string
  user_conventions: string
  accepted_decisions: string[]
  task_checkpoints: string[]
  updated_at: string
}

export interface AgentStep {
  id: string
  timestamp: string
  step_type: string
  content: string
  tool_name?: string | null
  tool_args?: Record<string, unknown> | null
  tool_result?: unknown
  approval_status?: string | null
}

export interface AgentSession {
  id: string
  title: string
  created_at: string
  updated_at: string
  workspace_root: string
  active_model: string
  mode: string
  steps: AgentStep[]
}

type TauriInternals = {
  invoke: <T>(command: string, args?: Record<string, unknown>) => Promise<T>
}

function internals(): TauriInternals | null {
  if (typeof window === "undefined") return null
  return (window as Window & { __TAURI_INTERNALS__?: TauriInternals }).__TAURI_INTERNALS__ || null
}

export function tauriAvailable(): boolean {
  return Boolean(internals())
}

export async function desktopInvoke<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  const api = internals()
  if (!api) throw new Error("This action is available in Qwanto Desktop only.")
  return api.invoke<T>(command, args)
}
