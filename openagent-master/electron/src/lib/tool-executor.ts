import * as fs from "fs";
import * as path from "path";
import { execFile, exec } from "child_process";
import { PermissionPolicy } from "./permission-policy";
import { log } from "./logger";

export interface ToolResult {
  tool: string;
  success: boolean;
  output: string;
  error?: string;
  durationMs: number;
}

export class ToolExecutor {
  private policy: PermissionPolicy;

  constructor(policy: PermissionPolicy) {
    this.policy = policy;
  }

  public async readFile(filePath: string, startLine?: number, endLine?: number): Promise<ToolResult> {
    const t0 = Date.now();
    const resolved = path.resolve(this.policy.getWorkspaceRoot(), filePath);
    const decision = this.policy.evaluateFileOperation("read", resolved);

    if (!decision.allowed) {
      return { tool: "read_file", success: false, output: "", error: decision.reason, durationMs: Date.now() - t0 };
    }

    try {
      if (!fs.existsSync(resolved)) {
        return { tool: "read_file", success: false, output: "", error: `File not found: ${filePath}`, durationMs: Date.now() - t0 };
      }
      const raw = fs.readFileSync(resolved, "utf8");
      if (startLine !== undefined && endLine !== undefined) {
        const lines = raw.split("\n");
        const sliced = lines.slice(Math.max(0, startLine - 1), endLine).join("\n");
        return { tool: "read_file", success: true, output: sliced, durationMs: Date.now() - t0 };
      }
      return { tool: "read_file", success: true, output: raw, durationMs: Date.now() - t0 };
    } catch (e: any) {
      return { tool: "read_file", success: false, output: "", error: e.message, durationMs: Date.now() - t0 };
    }
  }

  public async writeFile(filePath: string, content: string): Promise<ToolResult> {
    const t0 = Date.now();
    const resolved = path.resolve(this.policy.getWorkspaceRoot(), filePath);
    const decision = this.policy.evaluateFileOperation("write", resolved);

    if (!decision.allowed) {
      return { tool: "write_file", success: false, output: "", error: decision.reason, durationMs: Date.now() - t0 };
    }

    try {
      const parent = path.dirname(resolved);
      if (!fs.existsSync(parent)) {
        fs.mkdirSync(parent, { recursive: true });
      }
      fs.writeFileSync(resolved, content, "utf8");
      this.policy.recordAudit({
        command: `write_file ${filePath}`,
        workingDirectory: this.policy.getWorkspaceRoot(),
        approved: true,
        riskLevel: decision.riskLevel,
        exitStatus: 0,
      });
      return { tool: "write_file", success: true, output: `Successfully wrote ${Buffer.byteLength(content)} bytes to ${filePath}`, durationMs: Date.now() - t0 };
    } catch (e: any) {
      return { tool: "write_file", success: false, output: "", error: e.message, durationMs: Date.now() - t0 };
    }
  }

  public async editFile(filePath: string, target: string, replacement: string): Promise<ToolResult> {
    const t0 = Date.now();
    const resolved = path.resolve(this.policy.getWorkspaceRoot(), filePath);
    const decision = this.policy.evaluateFileOperation("write", resolved);

    if (!decision.allowed) {
      return { tool: "edit_file", success: false, output: "", error: decision.reason, durationMs: Date.now() - t0 };
    }

    try {
      if (!fs.existsSync(resolved)) {
        return { tool: "edit_file", success: false, output: "", error: `File not found: ${filePath}`, durationMs: Date.now() - t0 };
      }
      const existing = fs.readFileSync(resolved, "utf8");
      if (!existing.includes(target)) {
        return { tool: "edit_file", success: false, output: "", error: `Target snippet not found in ${filePath}`, durationMs: Date.now() - t0 };
      }
      const updated = existing.replace(target, replacement);
      fs.writeFileSync(resolved, updated, "utf8");

      this.policy.recordAudit({
        command: `edit_file ${filePath}`,
        workingDirectory: this.policy.getWorkspaceRoot(),
        approved: true,
        riskLevel: decision.riskLevel,
        exitStatus: 0,
      });

      return { tool: "edit_file", success: true, output: `Successfully replaced content in ${filePath}`, durationMs: Date.now() - t0 };
    } catch (e: any) {
      return { tool: "edit_file", success: false, output: "", error: e.message, durationMs: Date.now() - t0 };
    }
  }

  public async listDirectory(dirPath: string = ""): Promise<ToolResult> {
    const t0 = Date.now();
    const resolved = path.resolve(this.policy.getWorkspaceRoot(), dirPath);
    const decision = this.policy.evaluateFileOperation("list", resolved);

    if (!decision.allowed) {
      return { tool: "list_directory", success: false, output: "", error: decision.reason, durationMs: Date.now() - t0 };
    }

    try {
      if (!fs.existsSync(resolved)) {
        return { tool: "list_directory", success: false, output: "", error: `Directory not found: ${dirPath}`, durationMs: Date.now() - t0 };
      }
      const entries = fs.readdirSync(resolved, { withFileTypes: true });
      const formatted = entries.map(e => `${e.isDirectory() ? "[DIR]" : "[FILE]"} ${e.name}`).join("\n");
      return { tool: "list_directory", success: true, output: formatted, durationMs: Date.now() - t0 };
    } catch (e: any) {
      return { tool: "list_directory", success: false, output: "", error: e.message, durationMs: Date.now() - t0 };
    }
  }

  public async executeShell(command: string, customCwd?: string): Promise<ToolResult> {
    const t0 = Date.now();
    const cwd = customCwd ? path.resolve(this.policy.getWorkspaceRoot(), customCwd) : this.policy.getWorkspaceRoot();
    const decision = this.policy.evaluateShellCommand(command, cwd);

    if (!decision.allowed) {
      return { tool: "bash", success: false, output: "", error: decision.reason, durationMs: Date.now() - t0 };
    }

    return new Promise((resolve) => {
      exec(command, { cwd, env: process.env, maxBuffer: 10 * 1024 * 1024 }, (err, stdout, stderr) => {
        const out = stdout || stderr || "";
        const redacted = this.policy.redactSecrets(out);

        this.policy.recordAudit({
          command,
          workingDirectory: cwd,
          approved: true,
          riskLevel: decision.riskLevel,
          exitStatus: err ? (err.code || 1) : 0,
          redactedOutputSnippet: redacted.slice(0, 300),
        });

        if (err) {
          resolve({ tool: "bash", success: false, output: redacted, error: err.message, durationMs: Date.now() - t0 });
        } else {
          resolve({ tool: "bash", success: true, output: redacted, durationMs: Date.now() - t0 });
        }
      });
    });
  }
}
