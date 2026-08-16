import * as fs from "fs";
import * as path from "path";
import { log } from "./logger";

export interface AgentStep {
  id: string;
  timestamp: string;
  type: "plan" | "tool_call" | "tool_result" | "user_message" | "assistant_message" | "error";
  content: string;
  toolName?: string;
  toolArgs?: any;
  toolResult?: any;
  approvalStatus?: "pending" | "approved" | "rejected";
}

export interface QwantoSession {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  workspacePath: string;
  activeModel: string;
  mode: "plan" | "agent";
  steps: AgentStep[];
}

export class SessionStore {
  private storageDir: string = "";

  constructor(customStorageDir?: string) {
    if (customStorageDir) {
      this.storageDir = path.resolve(customStorageDir);
    } else {
      try {
        const electron = require("electron");
        const app = electron?.app;
        const userPath = app?.getPath?.("userData") || process.cwd();
        this.storageDir = path.join(userPath, "qwanto-sessions");
      } catch {
        this.storageDir = path.resolve(process.cwd(), ".qwanto/sessions");
      }
    }

    if (!fs.existsSync(this.storageDir)) {
      try {
        fs.mkdirSync(this.storageDir, { recursive: true });
      } catch (e) {}
    }
  }

  public async listSessions(): Promise<Array<{ id: string; title: string; updatedAt: string; workspacePath: string }>> {
    if (!fs.existsSync(this.storageDir)) return [];
    try {
      const files = fs.readdirSync(this.storageDir);
      const list = [];
      for (const file of files) {
        if (file.endsWith(".json")) {
          const content = fs.readFileSync(path.join(this.storageDir, file), "utf8");
          const session = JSON.parse(content) as QwantoSession;
          list.push({
            id: session.id,
            title: session.title,
            updatedAt: session.updatedAt,
            workspacePath: session.workspacePath,
          });
        }
      }
      return list.sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime());
    } catch (e: any) {
      log("SESSION_STORE", `Error listing sessions: ${e.message}`);
      return [];
    }
  }

  public async getSession(sessionId: string): Promise<QwantoSession | null> {
    const sessionFile = path.join(this.storageDir, `${sessionId}.json`);
    if (!fs.existsSync(sessionFile)) return null;
    try {
      const content = fs.readFileSync(sessionFile, "utf8");
      return JSON.parse(content) as QwantoSession;
    } catch {
      return null;
    }
  }

  public async saveSession(session: QwantoSession): Promise<void> {
    const sessionFile = path.join(this.storageDir, `${session.id}.json`);
    session.updatedAt = new Date().toISOString();
    fs.writeFileSync(sessionFile, JSON.stringify(session, null, 2), "utf8");
  }

  public async deleteSession(sessionId: string): Promise<boolean> {
    const sessionFile = path.join(this.storageDir, `${sessionId}.json`);
    if (fs.existsSync(sessionFile)) {
      fs.unlinkSync(sessionFile);
      return true;
    }
    return false;
  }
}
