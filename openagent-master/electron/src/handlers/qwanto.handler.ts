import { ipcMain, BrowserWindow } from "electron";
import { QwantoRuntimeAdapter } from "../lib/qwanto-runtime";
import { ModelRegistry } from "../lib/model-registry";
import { PermissionPolicy, ExecutionMode } from "../lib/permission-policy";
import { ToolExecutor } from "../lib/tool-executor";
import { HardwareProbe } from "../lib/hardware-probe";
import { SessionStore, QwantoSession } from "../lib/session-store";
import { LocalApiServer } from "../lib/local-api-server";
import { log } from "../lib/logger";

export const runtime = new QwantoRuntimeAdapter();
export const registry = new ModelRegistry();
export const policy = new PermissionPolicy(process.cwd(), "agent");
export const toolExecutor = new ToolExecutor(policy);
export const hardwareProbe = new HardwareProbe();
export const sessionStore = new SessionStore();
export const localApiServer = new LocalApiServer(runtime, registry);

export function register(getMainWindow: () => BrowserWindow | null): void {
  log("QWANTO_HANDLER", "Registering Qwanto Native IPC Handlers...");

  // --- Inference Streaming ---
  ipcMain.handle("qwanto:stream", async (event, options) => {
    const win = getMainWindow();
    const sessionId = options.sessionId || "default";

    return new Promise((resolve, reject) => {
      runtime.streamGenerate(
        {
          modelPath: options.modelPath,
          prompt: options.prompt,
          maxTokens: options.maxTokens,
          mode: options.mode,
          gpuDevice: options.gpuDevice,
          forceCpu: options.forceCpu,
        },
        {
          onToken: (token) => {
            win?.webContents.send(`qwanto:token:${sessionId}`, { token });
          },
          onTelemetry: (tokPerSec, ttftMs, totalTokens) => {
            win?.webContents.send(`qwanto:telemetry:${sessionId}`, { tokPerSec, ttftMs, totalTokens });
          },
          onDone: (totalTokens, fullText) => {
            win?.webContents.send(`qwanto:done:${sessionId}`, { totalTokens, fullText });
            resolve({ success: true, totalTokens, fullText });
          },
          onError: (err) => {
            win?.webContents.send(`qwanto:error:${sessionId}`, { error: err.message });
            reject(err);
          },
        }
      );
    });
  });

  ipcMain.handle("qwanto:cancel", () => {
    runtime.cancel();
    return { success: true };
  });

  // --- Model Registry ---
  ipcMain.handle("qwanto:models:list", async () => {
    return registry.scanModels();
  });

  ipcMain.handle("qwanto:models:inspect", (event, filePath: string) => {
    return registry.inspectModelFile(filePath);
  });

  ipcMain.handle("qwanto:models:addDir", (event, dirPath: string) => {
    registry.addSearchDirectory(dirPath);
    return registry.getSearchDirectories();
  });

  // --- Hardware Telemetry ---
  ipcMain.handle("qwanto:telemetry:get", async () => {
    return hardwareProbe.probeSnapshot();
  });

  // --- Permission Policy & Modes ---
  ipcMain.handle("qwanto:permission:setMode", (event, mode: ExecutionMode) => {
    policy.setMode(mode);
    return { mode: policy.getMode() };
  });

  ipcMain.handle("qwanto:permission:getMode", () => {
    return { mode: policy.getMode() };
  });

  ipcMain.handle("qwanto:permission:setWorkspace", (event, workspaceRoot: string) => {
    policy.setWorkspaceRoot(workspaceRoot);
    return { workspaceRoot: policy.getWorkspaceRoot() };
  });

  ipcMain.handle("qwanto:permission:audit", () => {
    return policy.getAuditLogs();
  });

  // --- Agent Tool Execution ---
  ipcMain.handle("qwanto:tools:readFile", async (event, filePath: string, startLine?: number, endLine?: number) => {
    return toolExecutor.readFile(filePath, startLine, endLine);
  });

  ipcMain.handle("qwanto:tools:writeFile", async (event, filePath: string, content: string) => {
    return toolExecutor.writeFile(filePath, content);
  });

  ipcMain.handle("qwanto:tools:editFile", async (event, filePath: string, target: string, replacement: string) => {
    return toolExecutor.editFile(filePath, target, replacement);
  });

  ipcMain.handle("qwanto:tools:listDirectory", async (event, dirPath?: string) => {
    return toolExecutor.listDirectory(dirPath);
  });

  ipcMain.handle("qwanto:tools:bash", async (event, command: string, customCwd?: string) => {
    return toolExecutor.executeShell(command, customCwd);
  });

  // --- Local Session Store ---
  ipcMain.handle("qwanto:sessions:list", async () => {
    return sessionStore.listSessions();
  });

  ipcMain.handle("qwanto:sessions:get", async (event, sessionId: string) => {
    return sessionStore.getSession(sessionId);
  });

  ipcMain.handle("qwanto:sessions:save", async (event, session: QwantoSession) => {
    await sessionStore.saveSession(session);
    return { success: true };
  });

  ipcMain.handle("qwanto:sessions:delete", async (event, sessionId: string) => {
    return sessionStore.deleteSession(sessionId);
  });

  // --- Localhost API Server ---
  ipcMain.handle("qwanto:apiServer:start", async (event, port?: number) => {
    return localApiServer.start(port);
  });

  ipcMain.handle("qwanto:apiServer:stop", async () => {
    await localApiServer.stop();
    return { success: true };
  });

  ipcMain.handle("qwanto:apiServer:status", () => {
    return localApiServer.getStatus();
  });
}
