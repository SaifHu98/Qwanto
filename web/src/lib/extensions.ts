export const PLUGIN_CAPABILITIES = [
  "workspace.read",
  "workspace.write",
  "terminal.execute",
  "git.read",
  "git.write",
  "github.read",
  "github.write",
  "network.search",
  "model.control",
  "diagnostics.read",
  "secrets.access",
] as const

export type PluginCapability = typeof PLUGIN_CAPABILITIES[number]

const DANGEROUS_CAPABILITIES = new Set<PluginCapability>([
  "workspace.write",
  "terminal.execute",
  "git.write",
  "github.write",
  "network.search",
  "model.control",
  "secrets.access",
])

export interface BuiltinSkill {
  id: string
  label: string
  description: string
  capabilities: PluginCapability[]
}

export const BUILTIN_SKILLS: BuiltinSkill[] = [
  { id: "code-review", label: "Code Review", description: "Inspect a change and report actionable correctness, security, and maintainability findings.", capabilities: ["workspace.read", "git.read"] },
  { id: "test-and-fix", label: "Test and Fix", description: "Run the relevant local checks and work through the first real failure with approval for writes.", capabilities: ["workspace.read", "workspace.write", "terminal.execute"] },
  { id: "git-commit-branch-pull-request", label: "Git Commit / Branch / Pull Request", description: "Prepare a reviewable Git change; writes and remote actions remain approval-gated.", capabilities: ["git.read", "git.write", "github.write"] },
  { id: "release-readiness", label: "Release Readiness", description: "Review release gates, artifacts, checksums, documentation, and signing evidence.", capabilities: ["workspace.read", "diagnostics.read", "git.read"] },
  { id: "project-memory", label: "Project Memory", description: "Review and update the selected workspace memory and task checkpoints.", capabilities: ["workspace.read", "workspace.write"] },
  { id: "documentation-writer", label: "Documentation Writer", description: "Draft concise repository documentation while keeping claims tied to local evidence.", capabilities: ["workspace.read", "workspace.write"] },
  { id: "local-benchmark", label: "Local Benchmark", description: "Run reproducible local measurements without uploading code, models, or telemetry.", capabilities: ["diagnostics.read", "terminal.execute"] },
  { id: "optional-web-research", label: "Optional Web Research", description: "Prepare an externally sourced search only after a fresh user approval.", capabilities: ["network.search"] },
  { id: "github-issue-reporter", label: "GitHub Issue Reporter", description: "Prepare a redacted issue draft; issue creation and repository writes require approval.", capabilities: ["github.read", "github.write", "diagnostics.read"] },
]

export interface PluginPublisher {
  name: string
  key_id: string
}

export interface PluginManifest {
  schema_version: 1
  id: string
  name: string
  publisher: PluginPublisher
  version: string
  sha256: string
  requested_capabilities: PluginCapability[]
  entrypoint: string
  signature: string
  source_url?: string
  license?: string
}

export interface InstalledPlugin {
  manifest: PluginManifest
  enabled: boolean
  quarantined: boolean
}

export interface ManifestValidation {
  valid: boolean
  errors: string[]
  dangerousCapabilities: PluginCapability[]
}

const SHA256 = /^[a-f0-9]{64}$/i
const IDENTIFIER = /^[a-z0-9][a-z0-9._-]{1,80}$/

export function capabilityNeedsApproval(capability: PluginCapability | string): boolean {
  return DANGEROUS_CAPABILITIES.has(capability as PluginCapability)
}

export function capabilitiesNeedApproval(capabilities: readonly (PluginCapability | string)[]): boolean {
  return capabilities.some(capabilityNeedsApproval)
}

export function validatePluginManifest(value: unknown, expectedSha256?: string): ManifestValidation {
  const errors: string[] = []
  const manifest = value && typeof value === "object" ? value as Partial<PluginManifest> : null
  if (!manifest) return { valid: false, errors: ["Manifest must be a JSON object."], dangerousCapabilities: [] }
  if (manifest.schema_version !== 1) errors.push("schema_version must be 1.")
  if (typeof manifest.id !== "string" || !IDENTIFIER.test(manifest.id)) errors.push("id must be a short lowercase package identifier.")
  if (typeof manifest.name !== "string" || !manifest.name.trim()) errors.push("name is required.")
  if (!manifest.publisher || typeof manifest.publisher !== "object" || typeof manifest.publisher.name !== "string" || !manifest.publisher.name.trim() || typeof manifest.publisher.key_id !== "string" || !manifest.publisher.key_id.trim()) errors.push("publisher.name and publisher.key_id are required.")
  if (typeof manifest.version !== "string" || !manifest.version.trim()) errors.push("version is required.")
  if (typeof manifest.sha256 !== "string" || !SHA256.test(manifest.sha256)) errors.push("sha256 must be a 64-character hexadecimal digest.")
  if (expectedSha256 && manifest.sha256?.toLowerCase() !== expectedSha256.toLowerCase()) errors.push("sha256 does not match the package bytes.")
  if (!Array.isArray(manifest.requested_capabilities) || manifest.requested_capabilities.length === 0) errors.push("requested_capabilities must contain at least one capability.")
  const capabilities = Array.isArray(manifest.requested_capabilities) ? manifest.requested_capabilities : []
  const unknown = capabilities.filter((capability) => !PLUGIN_CAPABILITIES.includes(capability as PluginCapability))
  if (unknown.length) errors.push(`Unsupported capability: ${unknown.join(", ")}.`)
  if (typeof manifest.entrypoint !== "string" || !manifest.entrypoint.trim() || manifest.entrypoint.startsWith("/") || manifest.entrypoint.includes("..")) errors.push("entrypoint must be a package-relative path.")
  if (typeof manifest.signature !== "string" || !manifest.signature.trim()) errors.push("signature is required; unsigned plugins cannot be enabled.")
  return {
    valid: errors.length === 0,
    errors,
    dangerousCapabilities: capabilities.filter((capability) => capabilityNeedsApproval(capability as PluginCapability)) as PluginCapability[],
  }
}

export interface SkillInvocation {
  skill: BuiltinSkill
  prompt: string
}

export function resolveSkillInvocation(input: string): SkillInvocation | null {
  const match = input.trimStart().match(/^@([a-z0-9][a-z0-9-]*)\b\s*/i)
  if (!match) return null
  const skill = BUILTIN_SKILLS.find((candidate) => candidate.id === match[1].toLowerCase())
  return skill ? { skill, prompt: input.trimStart().slice(match[0].length).trim() } : null
}
