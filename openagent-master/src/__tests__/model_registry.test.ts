import { describe, expect, it } from "vitest";
import { ModelRegistry } from "../../electron/src/lib/model-registry";

describe("ModelRegistry", () => {
  it("identifies model formats and quantizations accurately", () => {
    const registry = new ModelRegistry();
    const info = registry.inspectModelFile("D:/models/DeepSeek-V4-Pro-4B-twla.qwn");

    expect(info.format).toBe("qwn");
    expect(info.quantization).toBe("TWLA 1.58-Bit");
    expect(info.paramSizeEstimate).toBe("4.0B");
  });

  it("classifies roles for fast-edits and reasoning based on model size", () => {
    const registry = new ModelRegistry();
    const fastEdit = registry.inspectModelFile("D:/models/DeepSeek-R1-Distill-1.5B.qwn");
    const reasoning = registry.inspectModelFile("D:/models/Qwen3.8-27B.qwn");

    expect(fastEdit.role).toBe("fast-edits");
    expect(reasoning.role).toBe("reasoning");
  });
});
