import { describe, expect, it } from "vitest"

import { BUILTIN_SKILLS, capabilityNeedsApproval, resolveSkillInvocation, validatePluginManifest } from "./extensions"

const validManifest = {
  schema_version: 1,
  id: "local-reviewer",
  name: "Local Reviewer",
  publisher: { name: "Qwanto Test Publisher", key_id: "test-key-1" },
  version: "1.0.0",
  sha256: "a".repeat(64),
  requested_capabilities: ["workspace.read", "git.read"],
  entrypoint: "bin/reviewer",
  signature: "signed-manifest-for-tests",
}

describe("skills and plugin manifests", () => {
  it("accepts a signed manifest with an allowlisted capability set", () => {
    expect(validatePluginManifest(validManifest, "a".repeat(64))).toMatchObject({ valid: true, errors: [] })
  })

  it("rejects a package checksum mismatch and dangerous capability remains approval-gated", () => {
    const result = validatePluginManifest(validManifest, "b".repeat(64))
    expect(result.valid).toBe(false)
    expect(result.errors).toContain("sha256 does not match the package bytes.")
    expect(capabilityNeedsApproval("github.write")).toBe(true)
  })

  it("rejects unsigned and unknown-capability plugins", () => {
    const result = validatePluginManifest({ ...validManifest, signature: "", requested_capabilities: ["workspace.read", "shell.root"] })
    expect(result.valid).toBe(false)
    expect(result.errors).toContain("signature is required; unsigned plugins cannot be enabled.")
    expect(result.errors).toContain("Unsupported capability: shell.root.")
  })

  it("resolves a built-in skill invocation without changing the prompt body", () => {
    const invocation = resolveSkillInvocation("  @code-review inspect the current diff")
    expect(invocation?.skill).toEqual(BUILTIN_SKILLS.find((skill) => skill.id === "code-review"))
    expect(invocation?.prompt).toBe("inspect the current diff")
    expect(resolveSkillInvocation("@not-installed do work")).toBeNull()
  })
})
