import { describe, expect, it, beforeEach } from "vitest";
import { PermissionPolicy } from "../../electron/src/lib/permission-policy";

describe("PermissionPolicy Security Engine", () => {
  let policy: PermissionPolicy;

  beforeEach(() => {
    policy = new PermissionPolicy("D:/EcoUni/qwanto/workspace", "agent");
  });

  it("permits safe read operations inside workspace", () => {
    const dec = policy.evaluateFileOperation("read", "D:/EcoUni/qwanto/workspace/src/main.ts");
    expect(dec.allowed).toBe(true);
    expect(dec.riskLevel).toBe("read_only");
    expect(dec.requiresUserApproval).toBe(false);
  });

  it("blocks path traversal outside workspace for writes", () => {
    const dec = policy.evaluateFileOperation("write", "C:/Windows/System32/drivers/etc/hosts");
    expect(dec.allowed).toBe(false);
    expect(dec.riskLevel).toBe("mutation_dangerous");
    expect(dec.requiresUserApproval).toBe(true);
  });

  it("strictly requires plan approval for all mutations in Plan Mode", () => {
    policy.setMode("plan");
    const dec = policy.evaluateFileOperation("write", "D:/EcoUni/qwanto/workspace/src/app.ts");
    expect(dec.allowed).toBe(false);
    expect(dec.requiresUserApproval).toBe(true);
    expect(dec.reason).toContain("Plan Mode active");
  });

  it("flags destructive shell commands as high risk requiring approval", () => {
    const dec = policy.evaluateShellCommand("rm -rf /", "D:/EcoUni/qwanto/workspace");
    expect(dec.allowed).toBe(false);
    expect(dec.riskLevel).toBe("mutation_dangerous");
    expect(dec.requiresUserApproval).toBe(true);
  });

  it("permits read-only inspection git commands without prompts", () => {
    const dec = policy.evaluateShellCommand("git status", "D:/EcoUni/qwanto/workspace");
    expect(dec.allowed).toBe(true);
    expect(dec.riskLevel).toBe("read_only");
    expect(dec.requiresUserApproval).toBe(false);
  });

  it("redacts sensitive tokens and secrets from logged output", () => {
    const raw = "Connecting with Bearer sk-ant-api03-abcdef1234567890abcdef1234567890 and token ghp_123456789012345678901234567890123456";
    const redacted = policy.redactSecrets(raw);
    expect(redacted).not.toContain("sk-ant-api03");
    expect(redacted).not.toContain("ghp_1234567890");
    expect(redacted).toContain("[REDACTED");
  });
});
