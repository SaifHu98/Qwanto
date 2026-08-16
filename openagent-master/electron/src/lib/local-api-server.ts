import * as http from "http";
import { QwantoRuntimeAdapter } from "./qwanto-runtime";
import { ModelRegistry } from "./model-registry";
import { log } from "./logger";

export class LocalApiServer {
  private server: http.Server | null = null;
  private runtime: QwantoRuntimeAdapter;
  private registry: ModelRegistry;
  private isRunning: boolean = false;
  private port: number = 8000;
  private host: string = "127.0.0.1";

  constructor(runtime: QwantoRuntimeAdapter, registry: ModelRegistry) {
    this.runtime = runtime;
    this.registry = registry;
  }

  public async start(port: number = 8000): Promise<{ port: number; host: string }> {
    if (this.server) {
      return { port: this.port, host: this.host };
    }

    this.port = port;
    return new Promise((resolve, reject) => {
      this.server = http.createServer(async (req, res) => {
        // Strict Localhost & Security Headers
        res.setHeader("Access-Control-Allow-Origin", "http://127.0.0.1");
        res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
        res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.setHeader("X-Content-Type-Options", "nosniff");
        res.setHeader("X-Frame-Options", "DENY");

        if (req.method === "OPTIONS") {
          res.writeHead(204);
          res.end();
          return;
        }

        const url = req.url || "/";

        if (req.method === "GET" && (url === "/health" || url === "/v1/health")) {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ status: "ok", engine: "qwanto-native", local_only: true }));
          return;
        }

        if (req.method === "GET" && (url === "/v1/models" || url === "/models")) {
          const models = await this.registry.scanModels();
          const response = {
            object: "list",
            data: models.map(m => ({
              id: m.name,
              object: "model",
              created: Math.floor(Date.now() / 1000),
              owned_by: "qwanto-local",
              quantization: m.quantization,
            })),
          };
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify(response));
          return;
        }

        if (req.method === "POST" && (url === "/v1/chat/completions" || url === "/chat/completions")) {
          let body = "";
          req.on("data", chunk => { body += chunk; });
          req.on("end", async () => {
            try {
              const parsed = JSON.parse(body);
              const modelName = parsed.model || "default";
              const messages = parsed.messages || [];
              const stream = parsed.stream === true;
              const prompt = messages.map((m: any) => `${m.role}: ${m.content}`).join("\n");

              const models = await this.registry.scanModels();
              const matched = models.find(m => m.name === modelName) || models[0];
              const modelPath = matched ? matched.path : modelName;

              if (stream) {
                res.writeHead(200, {
                  "Content-Type": "text/event-stream",
                  "Cache-Control": "no-cache",
                  "Connection": "keep-alive",
                });

                this.runtime.streamGenerate(
                  { modelPath, prompt, maxTokens: parsed.max_tokens || 512 },
                  {
                    onToken: (tok) => {
                      const payload = {
                        id: `chatcmpl-${Date.now()}`,
                        object: "chat.completion.chunk",
                        created: Math.floor(Date.now() / 1000),
                        model: modelName,
                        choices: [{ delta: { content: tok }, index: 0, finish_reason: null }],
                      };
                      res.write(`data: ${JSON.stringify(payload)}\n\n`);
                    },
                    onDone: () => {
                      res.write("data: [DONE]\n\n");
                      res.end();
                    },
                    onError: (err) => {
                      res.write(`data: {"error": "${err.message}"}\n\n`);
                      res.end();
                    }
                  }
                );
              } else {
                const text = await this.runtime.generate({ modelPath, prompt, maxTokens: parsed.max_tokens || 512 });
                res.writeHead(200, { "Content-Type": "application/json" });
                res.end(JSON.stringify({
                  id: `chatcmpl-${Date.now()}`,
                  object: "chat.completion",
                  created: Math.floor(Date.now() / 1000),
                  model: modelName,
                  choices: [{ message: { role: "assistant", content: text }, finish_reason: "stop" }],
                }));
              }
            } catch (e: any) {
              res.writeHead(400, { "Content-Type": "application/json" });
              res.end(JSON.stringify({ error: { message: e.message } }));
            }
          });
          return;
        }

        res.writeHead(404, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: { message: `Endpoint ${url} not found or unsupported` } }));
      });

      // Bind strictly to 127.0.0.1 (Localhost only)
      this.server.listen(this.port, this.host, () => {
        this.isRunning = true;
        log("LOCAL_API_SERVER", `Server running at http://${this.host}:${this.port}`);
        resolve({ port: this.port, host: this.host });
      });

      this.server.on("error", (err) => {
        this.isRunning = false;
        reject(err);
      });
    });
  }

  public stop(): Promise<void> {
    return new Promise((resolve) => {
      if (this.server) {
        this.server.close(() => {
          this.server = null;
          this.isRunning = false;
          resolve();
        });
      } else {
        resolve();
      }
    });
  }

  public getStatus(): { isRunning: boolean; port: number; host: string } {
    return { isRunning: this.isRunning, port: this.port, host: this.host };
  }
}
