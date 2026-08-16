import * as path from "path";
import { log } from "./logger";

export type ExecutionMode = "plan" | "agent";

export type RiskLevel = "read_only" | "mutation_safe" | "mutation_dangerous";

export interface PermissionDecision {
  allowed: boolean;
  requiresUserApproval: boolean;
  reason: string;
  riskLevel: RiskLevel;
}

export interface AuditLogEntry {
  timestamp: string;
  command: string;
  workingDirectory: string;
  approved: boolean;
  riskLevel: RiskLevel;
  exitStatus?: number;
  redactedOutputSnippet?: string;
}

export class PermissionPolicy {
  private workspaceRoot: string = "";
  private mode: ExecutionMode = "agent";
  private auditLog: AuditLogEntry[] = [];

  constructor(initialWorkspaceRoot: string = "", initialMode: ExecutionMode = "agent") {
    this.workspaceRoot = path.resolve(initialWorkspaceRoot || process.cwd());
    this.mode = initialMode;
  }

  public setWorkspaceRoot(newRoot: string): void {
    if (newRoot) {
      this.workspaceRoot = path.resolve(newRoot);
    }
  }

  public getWorkspaceRoot(): string {
    return this.workspaceRoot;
  }

  public setMode(mode: ExecutionMode): void {
    this.mode = mode;
    log("PERMISSION_POLICY", `Execution mode switched to: ${mode}`);
  }

  public getMode(): ExecutionMode {
    return this.mode;
  }

  public isSafePath(targetPath: string): boolean {
    if (!targetPath) return false;
    const resolved = path.resolve(this.workspaceRoot, targetPath);
    // Prevent path traversal outside of workspaceRoot
    return resolved.startsWith(this.workspaceRoot);
  }

  public evaluateFileOperation(
    operation: "read" | "write" | "delete" | "list",
    targetPath: string
  ): PermissionDecision {
    const isInsideWorkspace = this.isSafePath(targetPath);

    if (operation === "read" || operation === "list") {
      return {
        allowed: true,
        requiresUserApproval: false,
        reason: "Read-only file operations are safe",
        riskLevel: "read_only",
      };
    }

    // In Plan Mode, mutations are strictly blocked until user approval
    if (this.mode === "plan") {
      return {
        allowed: false,
        requiresUserApproval: true,
        reason: "Plan Mode active: file mutations require explicit plan approval",
        riskLevel: "mutation_safe",
      };
    }

    if (!isInsideWorkspace) {
      return {
        allowed: false,
        requiresUserApproval: true,
        reason: "Target path is outside the active workspace directory",
        riskLevel: "mutation_dangerous",
      };
    }

    return {
      allowed: true,
      requiresUserApproval: operation === "delete",
      reason: operation === "delete" ? "File deletion requires user confirmation" : "Safe file mutation within workspace",
      riskLevel: operation === "delete" ? "mutation_dangerous" : "mutation_safe",
    };
  }

  public evaluateShellCommand(command: string, cwd: string): PermissionDecision {
    const trimmed = command.trim();

    // Dangerous commands blacklist
    const dangerousPatterns = [
      /\brm\s+(-rf?|-fr)\b/i,
      /\bformat\b/i,
      /\bdiskpart\b/i,
      /\bdd\s+if=/i,
      /\bgit\s+push\s+--force\b/i,
      /\bgit\s+reset\s+--hard\b/i,
      /\bgit\s+clean\s+-fdx\b/i,
      /\bcurl\b/i,
      /\bwget\b/i,
      /\bInvoke-WebRequest\b/i,
      /\bnpm\s+publish\b/i,
    ];

    for (const pattern of dangerousPatterns) {
      if (pattern.test(trimmed)) {
        return {
          allowed: false,
          requiresUserApproval: true,
          reason: `Command contains high-risk/network pattern: ${pattern}`,
          riskLevel: "mutation_dangerous",
        };
      }
    }

    // Read-only shell commands
    const readOnlyPatterns = [
      /^(git\s+(status|diff|log|branch|show)|ls|dir|cat|type|grep|find|pwd)\b/i,
    ];

    for (const pattern of readOnlyPatterns) {
      if (pattern.test(trimmed)) {
        return {
          allowed: true,
          requiresUserApproval: false,
          reason: "Read-only inspection command",
          riskLevel: "read_only",
        };
      }
    }

    if (this.mode === "plan") {
      return {
        allowed: false,
        requiresUserApproval: true,
        reason: "Plan Mode active: shell execution requires approval",
        riskLevel: "mutation_safe",
      };
    }

    return {
      allowed: true,
      requiresUserApproval: false,
      reason: "Safe workspace shell execution",
      riskLevel: "mutation_safe",
    };
  }

  public redactSecrets(content: string): string {
    if (!content) return content;
    let redacted = content;
    // API keys & bearer tokens
    redacted = redacted.replace(/(bearer\s+)[A-Za-z0-9_\-\.]{15,}/gi, "$1[REDACTED_SECRET]");
    redacted = redacted.replace(/(api_key|apikey|secret|password)["']?\s*[:=]\s*["']?[A-Za-z0-9_\-\.]{8,}["']?/gi, "$1=[REDACTED_SECRET]");
    redacted = redacted.replace(/ghp_[A-Za-z0-9]{30,}/g, "ghp_[REDACTED_GH_TOKEN]");
    redacted = redacted.replace(/sk-[A-Za-z0-9]{30,}/g, "sk-[REDACTED_API_KEY]");
    return redacted;
  }

  public recordAudit(entry: Omit<AuditLogEntry, "timestamp">): void {
    const fullEntry: AuditLogEntry = {
      ...entry,
      timestamp: new Date().toISOString(),
      redactedOutputSnippet: entry.redactedOutputSnippet ? this.redactSecrets(entry.redactedOutputSnippet) : undefined,
    };
    this.auditLog.push(fullEntry);
    if (this.auditLog.length > 500) {
      this.auditLog.shift();
    }
  }

  public getAuditLogs(): AuditLogEntry[] {
    return [...this.auditLog];
  }
}
