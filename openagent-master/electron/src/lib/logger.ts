import path from "path";
import fs from "fs";

let logsDir = path.join(process.cwd(), "logs");
try {
  // Optional electron import with safe fallback for test/node runtime
  const electron = require("electron");
  const app = electron?.app;
  if (app && app.getPath) {
    logsDir = app.isPackaged
      ? path.join(app.getPath("userData"), "logs")
      : path.join(__dirname, "..", "..", "logs");
  }
} catch {
  // Unit test environment
}

try {
  fs.mkdirSync(logsDir, { recursive: true });
} catch {}

const logFile = path.join(logsDir, `main-${Date.now()}.log`);
let logStream: fs.WriteStream | null = null;
try {
  logStream = fs.createWriteStream(logFile, { flags: "a" });
} catch {}

export function log(label: string, data: unknown): void {
  const ts = new Date().toISOString();
  const line = typeof data === "string" ? data : JSON.stringify(data, null, 2);
  const entry = `[${ts}] [${label}] ${line}`;
  if (logStream) {
    logStream.write(`${entry}\n`);
  }
  console.log(entry);
}
