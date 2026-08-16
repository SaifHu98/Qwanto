import type { AgentEvent } from "./protocol";
import type {
  LegacySessionInfo, PersistedSession, Project, UIMessage, Space,
  SearchMessageResult, SearchSessionResult,
  GitRepoInfo, GitStatus, GitBranch, GitLogEntry,
  AgentDefinition, McpServerConfig, McpServerStatus,
} from "./ui";
import type { OAPSessionEvent, OAPPermissionEvent, OAPTurnCompleteEvent, OAPConfigOption } from "./oap";

interface SessionListItem {
  id: string;
  projectId: string;
  title: string;
  createdAt: number;
  model?: string;
  totalCost: number;
  engine?: "agent" | "oap";
}

declare global {
  interface Window {
    clientCore: {
      getGlassEnabled: () => Promise<boolean>;
      start: (options?: {
        cwd?: string;
        model?: string;
        permissionMode?: string;
        resume?: string;
        mcpServers?: McpServerConfig[];
        llmProvider?: "openrouter" | "ollama";
        openRouterKey?: string;
        ollamaEndpoint?: string;
      }) => Promise<{ sessionId: string; pid: number }>;
      send: (
        sessionId: string,
        message: { type: string; message: { role: string; content: string | Array<{ type: string; [key: string]: unknown }> } },
      ) => Promise<{ ok?: boolean; error?: string }>;
      stop: (sessionId: string) => Promise<{ ok: boolean }>;
      interrupt: (sessionId: string) => Promise<{ ok?: boolean; error?: string }>;
      mcpStatus: (sessionId: string) => Promise<{ servers: McpServerStatus[]; error?: string }>;
      mcpReconnect: (sessionId: string, serverName: string) => Promise<{ ok?: boolean; error?: string; restarted?: boolean }>;
      restartSession: (sessionId: string, mcpServers?: McpServerConfig[]) => Promise<{ ok?: boolean; error?: string; restarted?: boolean }>;
      readFile: (filePath: string) => Promise<{ content?: string; error?: string }>;
      openInEditor: (filePath: string, line?: number) => Promise<{ ok?: boolean; editor?: string; error?: string }>;
      generateTitle: (
        message: string,
        cwd?: string,
        options?: {
          llmProvider?: "openrouter" | "ollama";
          model?: string;
          openRouterKey?: string;
          ollamaEndpoint?: string;
        },
      ) => Promise<{ title?: string; error?: string }>;
      log: (label: string, data: unknown) => void;
      onEvent: (callback: (event: AgentEvent & { _sessionId: string }) => void) => () => void;
      onStderr: (callback: (data: { data: string; _sessionId: string }) => void) => () => void;
      onExit: (callback: (data: { code: number | null; _sessionId: string }) => void) => () => void;
      onPermissionRequest: (
        callback: (data: {
          _sessionId: string;
          requestId: string;
          toolName: string;
          toolInput: Record<string, unknown>;
          toolUseId: string;
          suggestions?: string[];
          decisionReason?: string;
        }) => void,
      ) => () => void;
      respondPermission: (
        sessionId: string,
        requestId: string,
        behavior: "allow" | "deny",
        toolUseId: string,
        toolInput: Record<string, unknown>,
        newPermissionMode?: string,
      ) => Promise<{ ok?: boolean; error?: string }>;
      setPermissionMode: (
        sessionId: string,
        permissionMode: string,
      ) => Promise<{ ok?: boolean; error?: string }>;
      projects: {
        list: () => Promise<Project[]>;
        create: () => Promise<Project | null>;
        delete: (projectId: string) => Promise<{ ok?: boolean; error?: string }>;
        rename: (projectId: string, name: string) => Promise<{ ok?: boolean; error?: string }>;
        updateSpace: (projectId: string, spaceId: string) => Promise<{ ok?: boolean; error?: string }>;
        reorder: (projectId: string, targetProjectId: string) => Promise<{ ok?: boolean; error?: string }>;
      };
      sessions: {
        save: (data: PersistedSession) => Promise<{ ok?: boolean; error?: string }>;
        load: (projectId: string, sessionId: string) => Promise<PersistedSession | null>;
        list: (projectId: string) => Promise<SessionListItem[]>;
        delete: (projectId: string, sessionId: string) => Promise<{ ok?: boolean; error?: string }>;
        search: (projectIds: string[], query: string) => Promise<{
          messageResults: SearchMessageResult[];
          sessionResults: SearchSessionResult[];
        }>;
      };
      spaces: {
        list: () => Promise<Space[]>;
        save: (spaces: Space[]) => Promise<{ ok?: boolean; error?: string }>;
      };
      legacySessions: {
        list: (projectPath: string) => Promise<LegacySessionInfo[]>;
        import: (projectPath: string, legacySessionId: string) => Promise<{
          messages?: UIMessage[];
          legacySessionId?: string;
          error?: string;
        }>;
      };
      files: {
        list: (cwd: string) => Promise<{ files: string[]; dirs: string[] }>;
        readMultiple: (
          cwd: string,
          paths: string[],
        ) => Promise<
          Array<
            | { path: string; content: string; isDir?: false; error?: undefined }
            | { path: string; isDir: true; tree: string; error?: undefined }
            | { path: string; error: string; content?: undefined; isDir?: undefined }
          >
        >;
      };
      git: {
        discoverRepos: (projectPath: string) => Promise<GitRepoInfo[]>;
        status: (cwd: string) => Promise<GitStatus & { error?: string }>;
        diffStats: (cwd: string) => Promise<{ insertions: number; deletions: number }>;
        stage: (cwd: string, files: string[]) => Promise<{ ok?: boolean; error?: string }>;
        unstage: (cwd: string, files: string[]) => Promise<{ ok?: boolean; error?: string }>;
        stageAll: (cwd: string) => Promise<{ ok?: boolean; error?: string }>;
        unstageAll: (cwd: string) => Promise<{ ok?: boolean; error?: string }>;
        discard: (cwd: string, files: string[]) => Promise<{ ok?: boolean; error?: string }>;
        commit: (cwd: string, message: string) => Promise<{ ok?: boolean; output?: string; error?: string }>;
        branches: (cwd: string) => Promise<GitBranch[] & { error?: string }>;
        checkout: (cwd: string, branch: string) => Promise<{ ok?: boolean; error?: string }>;
        createBranch: (cwd: string, name: string) => Promise<{ ok?: boolean; error?: string }>;
        push: (cwd: string) => Promise<{ ok?: boolean; output?: string; error?: string }>;
        pull: (cwd: string) => Promise<{ ok?: boolean; output?: string; error?: string }>;
        fetch: (cwd: string) => Promise<{ ok?: boolean; output?: string; error?: string }>;
        diffFile: (cwd: string, file: string, staged: boolean) => Promise<{ diff?: string; error?: string }>;
        log: (cwd: string, count?: number) => Promise<GitLogEntry[]>;
        generateCommitMessage: (
          cwd: string,
          options?: {
            llmProvider?: "openrouter" | "ollama";
            model?: string;
            openRouterKey?: string;
            ollamaEndpoint?: string;
          },
        ) => Promise<{ message?: string; error?: string }>;
      };
      terminal: {
        create: (options: { cwd?: string; cols?: number; rows?: number }) => Promise<{ terminalId?: string; error?: string }>;
        write: (terminalId: string, data: string) => Promise<{ ok?: boolean; error?: string }>;
        resize: (terminalId: string, cols: number, rows: number) => Promise<{ ok?: boolean; error?: string }>;
        destroy: (terminalId: string) => Promise<{ ok?: boolean }>;
        onData: (callback: (data: { terminalId: string; data: string }) => void) => () => void;
        onExit: (callback: (data: { terminalId: string; exitCode: number }) => void) => () => void;
      };
      oap: {
        log: (label: string, data: unknown) => void;
        start: (options: { agentId: string; cwd: string; mcpServers?: McpServerConfig[] }) => Promise<{
          sessionId?: string;
          agentSessionId?: string;
          agentName?: string;
          configOptions?: OAPConfigOption[];
          mcpStatuses?: Array<{ name: string; status: string }>;
          error?: string;
        }>;
        prompt: (sessionId: string, text: string, images?: unknown[]) => Promise<{ ok?: boolean; error?: string }>;
        stop: (sessionId: string) => Promise<{ ok?: boolean; error?: string }>;
        reloadSession: (sessionId: string, mcpServers?: McpServerConfig[]) => Promise<{ ok?: boolean; supportsLoad?: boolean; error?: string }>;
        reviveSession: (options: { agentId: string; cwd: string; agentSessionId?: string; mcpServers?: McpServerConfig[] }) => Promise<{ sessionId?: string; agentSessionId?: string; usedLoad?: boolean; configOptions?: OAPConfigOption[]; mcpStatuses?: Array<{ name: string; status: string }>; error?: string }>;
        cancel: (sessionId: string) => Promise<{ ok?: boolean; error?: string }>;
        respondPermission: (sessionId: string, requestId: string, optionId: string) => Promise<{ ok?: boolean; error?: string }>;
        setConfig: (sessionId: string, configId: string, value: string) => Promise<{ configOptions?: OAPConfigOption[]; error?: string }>;
        getConfigOptions: (sessionId: string) => Promise<{ configOptions?: OAPConfigOption[] }>;
        onEvent: (callback: (data: OAPSessionEvent) => void) => () => void;
        onPermissionRequest: (callback: (data: OAPPermissionEvent) => void) => () => void;
        onTurnComplete: (callback: (data: OAPTurnCompleteEvent) => void) => () => void;
        onExit: (callback: (data: { _sessionId: string; code: number | null }) => void) => () => void;
      };
      mcp: {
        list: (projectId: string) => Promise<McpServerConfig[]>;
        add: (projectId: string, server: McpServerConfig) => Promise<{ ok?: boolean; error?: string }>;
        remove: (projectId: string, name: string) => Promise<{ ok?: boolean; error?: string }>;
        authenticate: (serverName: string, serverUrl: string) => Promise<{ ok?: boolean; error?: string }>;
        authStatus: (serverName: string) => Promise<{ hasToken: boolean; expiresAt?: number }>;
        probe: (servers: McpServerConfig[]) => Promise<Array<{ name: string; status: "connected" | "needs-auth" | "failed"; error?: string }>>;
      };
      agents: {
        list: () => Promise<AgentDefinition[]>;
        save: (agent: AgentDefinition) => Promise<{ ok?: boolean; error?: string }>;
        delete: (id: string) => Promise<{ ok?: boolean; error?: string }>;
      };
    };
    qwanto: {
      stream: (options: {
        sessionId: string;
        modelPath: string;
        prompt: string;
        maxTokens?: number;
        mode?: "max-performance" | "balanced" | "max-quality";
        gpuDevice?: number;
        forceCpu?: boolean;
      }) => Promise<{ success: boolean; totalTokens: number; fullText: string }>;
      cancel: () => Promise<{ success: boolean }>;
      onToken: (sessionId: string, callback: (data: { token: string }) => void) => () => void;
      onTelemetry: (sessionId: string, callback: (data: { tokPerSec: number; ttftMs: number; totalTokens: number }) => void) => () => void;
      onDone: (sessionId: string, callback: (data: { totalTokens: number; fullText: string }) => void) => () => void;
      onError: (sessionId: string, callback: (data: { error: string }) => void) => () => void;
      models: {
        list: () => Promise<Array<{
          id: string;
          name: string;
          path: string;
          format: string;
          sizeBytes: number;
          sizeFormatted: string;
          quantization: string;
          paramSizeEstimate?: string;
          role: string;
          isValid: boolean;
        }>>;
        inspect: (filePath: string) => Promise<any>;
        addDir: (dirPath: string) => Promise<string[]>;
      };
      telemetry: {
        get: () => Promise<{
          timestamp: string;
          cpu: { model: string; cores: number; threads: number; loadPct: number };
          memory: { totalGb: number; usedGb: number; freeGb: number; processRssMb: number };
          gpus: Array<{ vendor: string; name: string; vramTotalGb: number; vramUsedGb: number; temperatureC?: number; utilizationPct?: number; isDiscrete: boolean; score: number }>;
          disk: { freeGb: number | null; status: string };
        }>;
      };
      permission: {
        setMode: (mode: "plan" | "agent") => Promise<{ mode: string }>;
        getMode: () => Promise<{ mode: string }>;
        setWorkspace: (root: string) => Promise<{ workspaceRoot: string }>;
        audit: () => Promise<any[]>;
      };
      tools: {
        readFile: (filePath: string, startLine?: number, endLine?: number) => Promise<{ tool: string; success: boolean; output: string; error?: string; durationMs: number }>;
        writeFile: (filePath: string, content: string) => Promise<{ tool: string; success: boolean; output: string; error?: string; durationMs: number }>;
        editFile: (filePath: string, target: string, replacement: string) => Promise<{ tool: string; success: boolean; output: string; error?: string; durationMs: number }>;
        listDirectory: (dirPath?: string) => Promise<{ tool: string; success: boolean; output: string; error?: string; durationMs: number }>;
        bash: (command: string, customCwd?: string) => Promise<{ tool: string; success: boolean; output: string; error?: string; durationMs: number }>;
      };
      sessions: {
        list: () => Promise<Array<{ id: string; title: string; updatedAt: string; workspacePath: string }>>;
        get: (sessionId: string) => Promise<any>;
        save: (session: any) => Promise<{ success: boolean }>;
        delete: (sessionId: string) => Promise<boolean>;
      };
      apiServer: {
        start: (port?: number) => Promise<{ port: number; host: string }>;
        stop: () => Promise<{ success: boolean }>;
        status: () => Promise<{ isRunning: boolean; port: number; host: string }>;
      };
    };
  }
}
