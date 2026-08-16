import { spawn, ChildProcess } from "child_process";
import * as path from "path";
import * as fs from "fs";
import { EventEmitter } from "events";
import { log } from "./logger";

export interface QwantoRunOptions {
  modelPath: string;
  prompt: string;
  maxTokens?: number;
  ctxSize?: number;
  mode?: "max-performance" | "balanced" | "max-quality";
  gpuDevice?: number;
  forceCpu?: boolean;
  autoTune?: boolean;
  speculative?: boolean;
  numThreads?: number;
}

export interface QwantoStreamEvent {
  type: "token" | "telemetry" | "done" | "error";
  token?: string;
  text?: string;
  tokPerSec?: number;
  ttftMs?: number;
  totalTokens?: number;
  model?: string;
  error?: string;
}

export class QwantoRuntimeAdapter extends EventEmitter {
  private activeProcess: ChildProcess | null = null;
  private qwnrunExecutable: string = "";

  constructor(customExecutablePath?: string) {
    super();
    this.qwnrunExecutable = this.resolveQwnrunPath(customExecutablePath);
  }

  public resolveQwnrunPath(customPath?: string): string {
    if (customPath && fs.existsSync(customPath)) {
      return customPath;
    }

    // Standard search locations
    const possiblePaths = [
      path.resolve(__dirname, "../../c/qwnrun.exe"),
      path.resolve(__dirname, "../../../c/qwnrun.exe"),
      path.resolve(process.cwd(), "c/qwnrun.exe"),
      path.resolve(process.cwd(), "../c/qwnrun.exe"),
      path.resolve("D:/EcoUni/qwanto/c/qwnrun_msvc.exe"),
      path.resolve("D:/EcoUni/qwanto/c/qwnrun.exe"),
      "qwnrun.exe",
      "qwnrun"
    ];

    for (const p of possiblePaths) {
      if (fs.existsSync(p)) {
        return p;
      }
    }

    return "qwnrun";
  }

  public getExecutablePath(): string {
    return this.qwnrunExecutable;
  }

  public isAvailable(): boolean {
    return fs.existsSync(this.qwnrunExecutable) || this.qwnrunExecutable === "qwnrun";
  }

  public async generate(options: QwantoRunOptions): Promise<string> {
    return new Promise((resolve, reject) => {
      let fullText = "";
      this.streamGenerate(options, {
        onToken: (token) => {
          fullText += token;
        },
        onDone: () => {
          resolve(fullText);
        },
        onError: (err) => {
          reject(err);
        }
      });
    });
  }

  public streamGenerate(
    options: QwantoRunOptions,
    callbacks: {
      onToken?: (token: string) => void;
      onTelemetry?: (tps: number, ttft: number, tokens: number) => void;
      onDone?: (totalTokens: number, text: string) => void;
      onError?: (err: Error) => void;
    }
  ): { cancel: () => void } {
    if (this.activeProcess) {
      this.cancel();
    }

    const args: string[] = [
      options.modelPath,
      options.prompt,
      String(options.maxTokens || 512),
      String(options.ctxSize || 4096),
    ];

    if (options.mode) {
      args.push("--mode", options.mode);
    }
    if (options.autoTune !== false) {
      args.push("--auto-tune");
    }
    if (options.speculative) {
      args.push("--speculative");
    }
    if (options.numThreads && options.numThreads > 0) {
      args.push("--threads", String(options.numThreads));
    }
    if (options.gpuDevice !== undefined && options.gpuDevice >= 0) {
      args.push("--gpu", "--gpu-device", String(options.gpuDevice));
    }

    const env = { ...process.env };
    if (options.forceCpu) {
      env["QWN_FORCE_CPU"] = "1";
    }
    if (options.gpuDevice !== undefined && options.gpuDevice >= 0) {
      env["QWN_GPU_DEVICE"] = String(options.gpuDevice);
    }

    log("QWANTO_RUNTIME", `Spawning: ${this.qwnrunExecutable} ${args.join(" ")}`);

    let accumulatedText = "";
    let tokenCount = 0;
    let startTime = Date.now();
    let firstTokenTime: number | null = null;

    try {
      const child = spawn(this.qwnrunExecutable, args, {
        env,
        stdio: ["ignore", "pipe", "pipe"],
      });
      this.activeProcess = child;

      child.stdout.on("data", (chunk: Buffer) => {
        const str = chunk.toString("utf8");
        if (!firstTokenTime) {
          firstTokenTime = Date.now();
          const ttft = firstTokenTime - startTime;
          callbacks.onTelemetry?.(0, ttft, 0);
        }

        // Check for telemetry lines from qwnrun
        const lines = str.split("\n");
        for (const line of lines) {
          if (line.includes("Raw Throughput") || line.includes("tok/s")) {
            const match = line.match(/([\d\.]+)\s*tok\/s/);
            if (match) {
              const tps = parseFloat(match[1]);
              const ttft = firstTokenTime ? firstTokenTime - startTime : 0;
              callbacks.onTelemetry?.(tps, ttft, tokenCount);
            }
          } else if (line.startsWith("qwnrun build:") || line.startsWith("Prompt tokens:")) {
            // Internal diagnostic lines
            continue;
          } else {
            // Generated token stream content
            if (line.length > 0) {
              tokenCount++;
              accumulatedText += line + "\n";
              callbacks.onToken?.(line + "\n");
              this.emit("token", line + "\n");
            }
          }
        }
      });

      child.stderr.on("data", (chunk: Buffer) => {
        const errStr = chunk.toString("utf8");
        log("QWANTO_RUNTIME_STDERR", errStr);
      });

      child.on("close", (code) => {
        this.activeProcess = null;
        const totalTimeSec = (Date.now() - startTime) / 1000;
        const finalTps = totalTimeSec > 0 ? tokenCount / totalTimeSec : 0;
        const ttft = firstTokenTime ? firstTokenTime - startTime : 0;

        callbacks.onTelemetry?.(finalTps, ttft, tokenCount);
        callbacks.onDone?.(tokenCount, accumulatedText);
        this.emit("done", { totalTokens: tokenCount, text: accumulatedText });
      });

      child.on("error", (err) => {
        this.activeProcess = null;
        log("QWANTO_RUNTIME_ERROR", err.message);
        callbacks.onError?.(err);
        this.emit("error", err);
      });
    } catch (err: any) {
      this.activeProcess = null;
      callbacks.onError?.(err);
    }

    return {
      cancel: () => this.cancel(),
    };
  }

  public cancel(): void {
    if (this.activeProcess) {
      try {
        log("QWANTO_RUNTIME", "Killing active qwnrun process...");
        this.activeProcess.kill("SIGINT");
      } catch (e) {
        try {
          this.activeProcess.kill("SIGKILL");
        } catch {}
      }
      this.activeProcess = null;
    }
  }
}
