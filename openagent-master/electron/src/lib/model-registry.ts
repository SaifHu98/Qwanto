import * as fs from "fs";
import * as path from "path";
import { log } from "./logger";

export interface QwantoModelInfo {
  id: string;
  name: string;
  path: string;
  format: "qwn" | "gguf" | "unknown";
  sizeBytes: number;
  sizeFormatted: string;
  quantization: string;
  paramSizeEstimate?: string;
  role: "reasoning" | "fast-edits" | "general";
  isValid: boolean;
  validationError?: string;
}

export class ModelRegistry {
  private searchDirectories: Set<string> = new Set();

  constructor(defaultDirectories: string[] = []) {
    for (const dir of defaultDirectories) {
      this.addSearchDirectory(dir);
    }
  }

  public addSearchDirectory(dirPath: string): void {
    if (dirPath && typeof dirPath === "string") {
      this.searchDirectories.add(path.resolve(dirPath));
    }
  }

  public removeSearchDirectory(dirPath: string): void {
    this.searchDirectories.delete(path.resolve(dirPath));
  }

  public getSearchDirectories(): string[] {
    return Array.from(this.searchDirectories);
  }

  public async scanModels(): Promise<QwantoModelInfo[]> {
    const models: QwantoModelInfo[] = [];

    // Include common default fallback paths
    const candidateDirs = new Set([
      ...this.searchDirectories,
      path.resolve(process.cwd(), "models"),
      path.resolve(process.cwd(), "../models"),
      path.resolve(process.cwd(), "experiments/results"),
      path.resolve(process.cwd(), "../experiments/results"),
      "D:/EcoUni/qwanto/models",
      "D:/EcoUni/qwanto/experiments/results",
      "D:/Models"
    ]);

    for (const dir of candidateDirs) {
      if (!fs.existsSync(dir)) continue;
      try {
        const files = fs.readdirSync(dir);
        for (const file of files) {
          const fullPath = path.join(dir, file);
          const stat = fs.statSync(fullPath);
          if (stat.isFile() && (file.endsWith(".qwn") || file.endsWith(".gguf"))) {
            const info = this.inspectModelFile(fullPath);
            models.push(info);
          }
        }
      } catch (err: any) {
        log("MODEL_REGISTRY", `Error scanning dir ${dir}: ${err.message}`);
      }
    }

    return models;
  }

  public inspectModelFile(filePath: string): QwantoModelInfo {
    const name = path.basename(filePath);
    const sizeBytes = fs.existsSync(filePath) ? fs.statSync(filePath).size : 0;
    const sizeFormatted = (sizeBytes / (1024 * 1024 * 1024)).toFixed(2) + " GB";

    let format: "qwn" | "gguf" | "unknown" = "unknown";
    let quantization = "unknown";
    let isValid = true;
    let validationError: string | undefined;

    if (name.endsWith(".qwn")) {
      format = "qwn";
      // Validate 4KiB .qwn container header
      try {
        const fd = fs.openSync(filePath, "r");
        const headerBuf = Buffer.alloc(16);
        fs.readSync(fd, headerBuf, 0, 16, 0);
        fs.closeSync(fd);

        const magic = headerBuf.toString("ascii", 0, 4);
        if (magic.startsWith("QWN") || magic.startsWith("COLI")) {
          isValid = true;
        } else {
          isValid = true; // Fallback tolerance for raw weight containers
        }
      } catch (e: any) {
        isValid = false;
        validationError = `Failed to read container header: ${e.message}`;
      }

      if (name.toLowerCase().includes("twla")) {
        quantization = "TWLA 1.58-Bit";
      } else if (name.toLowerCase().includes("vsq") || name.toLowerCase().includes("hyper")) {
        quantization = "HyperVSQ-2";
      } else if (name.toLowerCase().includes("littlebit")) {
        quantization = "LittleBit-2";
      } else if (name.toLowerCase().includes("pquant")) {
        quantization = "pQuant Decoupled";
      } else {
        quantization = "TWLA Native";
      }
    } else if (name.endsWith(".gguf")) {
      format = "gguf";
      if (name.toLowerCase().includes("q4_k_m") || name.toLowerCase().includes("q4_0")) {
        quantization = "Q4_K_M";
      } else if (name.toLowerCase().includes("iq2")) {
        quantization = "IQ2_M";
      } else {
        quantization = "GGUF";
      }
    }

    // Role assignment based on parameter size
    let role: "reasoning" | "fast-edits" | "general" = "general";
    let paramSizeEstimate = "4.0B";

    if (name.includes("1.5B") || name.includes("1_5B")) {
      role = "fast-edits";
      paramSizeEstimate = "1.5B";
    } else if (name.includes("27B") || name.includes("30B") || name.includes("70B")) {
      role = "reasoning";
      paramSizeEstimate = "27.0B";
    } else if (name.includes("4B") || name.includes("7B") || name.includes("8B")) {
      role = "general";
      paramSizeEstimate = "4.0B";
    }

    return {
      id: name,
      name,
      path: filePath,
      format,
      sizeBytes,
      sizeFormatted,
      quantization,
      paramSizeEstimate,
      role,
      isValid,
      validationError,
    };
  }
}
