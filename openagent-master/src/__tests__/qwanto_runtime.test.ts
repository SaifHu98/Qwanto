import { describe, expect, it } from "vitest";
import { QwantoRuntimeAdapter } from "../../electron/src/lib/qwanto-runtime";

describe("QwantoRuntimeAdapter", () => {
  it("initializes and resolves qwnrun executable paths", () => {
    const adapter = new QwantoRuntimeAdapter();
    const exePath = adapter.getExecutablePath();
    expect(exePath).toBeDefined();
    expect(typeof exePath).toBe("string");
  });

  it("handles cancel request gracefully when no active process", () => {
    const adapter = new QwantoRuntimeAdapter();
    expect(() => adapter.cancel()).not.toThrow();
  });
});
