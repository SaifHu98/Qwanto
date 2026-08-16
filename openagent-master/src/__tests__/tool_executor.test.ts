import { describe, expect, it, beforeEach } from "vitest";
import { PermissionPolicy } from "../../electron/src/lib/permission-policy";
import { ToolExecutor } from "../../electron/src/lib/tool-executor";
import * as path from "path";
import * as fs from "fs";

describe("ToolExecutor Safe Operations", () => {
  let policy: PermissionPolicy;
  let executor: ToolExecutor;
  const testDir = path.resolve(__dirname, "../../.tmp_test_ws");

  beforeEach(() => {
    if (!fs.existsSync(testDir)) {
      fs.mkdirSync(testDir, { recursive: true });
    }
    policy = new PermissionPolicy(testDir, "agent");
    executor = new ToolExecutor(policy);
  });

  it("writes and reads files within workspace successfully", async () => {
    const writeRes = await executor.writeFile("sample.txt", "Hello Qwanto Native");
    expect(writeRes.success).toBe(true);

    const readRes = await executor.readFile("sample.txt");
    expect(readRes.success).toBe(true);
    expect(readRes.output).toBe("Hello Qwanto Native");
  });

  it("edits targeted content accurately", async () => {
    await executor.writeFile("code.py", "def old_func():\n    return 42");
    const editRes = await executor.editFile("code.py", "old_func", "new_fast_func");

    expect(editRes.success).toBe(true);
    const readRes = await executor.readFile("code.py");
    expect(readRes.output).toContain("new_fast_func");
  });

  it("blocks file write outside workspace directory", async () => {
    const res = await executor.writeFile("../../secret.txt", "attack");
    expect(res.success).toBe(false);
    expect(res.error).toContain("outside the active workspace");
  });
});
