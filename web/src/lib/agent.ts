export type AgentProfile = "fast" | "balanced" | "deep"

export interface AgentProfileConfig {
  id: AgentProfile
  label: string
  description: string
  maxTokens: number
  contextSize: number
  temperature: number
  topP: number
  supportedParameters: string[]
}

export const AGENT_PROFILES: AgentProfileConfig[] = [
  {
    id: "fast",
    label: "Fast",
    description: "Shorter responses with a smaller context window.",
    maxTokens: 256,
    contextSize: 4096,
    temperature: 0.2,
    topP: 0.9,
    supportedParameters: ["max output tokens", "context size", "temperature", "top-p"],
  },
  {
    id: "balanced",
    label: "Balanced",
    description: "The default local coding-agent trade-off.",
    maxTokens: 512,
    contextSize: 8192,
    temperature: 0.7,
    topP: 0.95,
    supportedParameters: ["max output tokens", "context size", "temperature", "top-p"],
  },
  {
    id: "deep",
    label: "Deep",
    description: "More context and output budget for multi-step work.",
    maxTokens: 1024,
    contextSize: 16384,
    temperature: 0.4,
    topP: 0.9,
    supportedParameters: ["max output tokens", "context size", "temperature", "top-p"],
  },
]

export function profileConfig(profile: AgentProfile): AgentProfileConfig {
  return AGENT_PROFILES.find((candidate) => candidate.id === profile) || AGENT_PROFILES[1]
}
