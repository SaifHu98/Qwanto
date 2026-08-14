import React, { useState, useEffect, useRef } from "react"
import {
  Zap,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  FileCode,
  HardDrive,
  Clock,
  Sparkles,
  SlidersHorizontal,
  Play,
  ArrowRight,
  TrendingDown,
  Cpu,
  Layers,
  Activity,
  Check
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  startConversion,
  getConversionStatus,
  type ConversionStatus,
  listDiscoveredModels,
  type DiscoveredModel,
  loadModel
} from "@/lib/api"

interface ConverterViewProps {
  baseUrl: string
  apiKey: string
  onModelLoaded?: (modelPath: string) => void
  onNavigateToChat?: () => void
}

export function ConverterView({
  baseUrl,
  apiKey,
  onModelLoaded,
  onNavigateToChat
}: ConverterViewProps) {
  const [sourcePath, setSourcePath] = useState("")
  const [outputPath, setOutputPath] = useState("")
  const [quantMode, setQuantMode] = useState<"vsq" | "q4_0" | "none">("vsq")
  const [autoActivate, setAutoActivate] = useState(true)
  const [status, setStatus] = useState<ConversionStatus>({
    status: "idle",
    progress: 0,
    message: "Ready to convert model"
  })
  const [discoveredModels, setDiscoveredModels] = useState<DiscoveredModel[]>([])
  const [loadingModels, setLoadingModels] = useState(false)
  const [loadingAction, setLoadingAction] = useState(false)
  const [error, setError] = useState("")
  const autoActivatedRef = useRef(false)

  const refreshDiscoveredModels = async () => {
    setLoadingModels(true)
    try {
      const data = await listDiscoveredModels(baseUrl, apiKey)
      setDiscoveredModels(data.models || [])
    } catch {
      // Ignored
    } finally {
      setLoadingModels(false)
    }
  }

  useEffect(() => {
    refreshDiscoveredModels()
  }, [baseUrl, apiKey])

  // Poll conversion status when active and handle auto-activation
  useEffect(() => {
    let timer: any
    const poll = async () => {
      try {
        const data = await getConversionStatus(baseUrl, apiKey)
        setStatus(data)
        if (data.status === "converting") {
          timer = setTimeout(poll, 800)
        } else if (data.status === "done" && autoActivate && !autoActivatedRef.current) {
          autoActivatedRef.current = true
          handleLoadConvertedModel(data.output || outputPath)
        }
      } catch {
        // Ignored
      }
    }
    poll()
    return () => clearTimeout(timer)
  }, [baseUrl, apiKey, status.status, autoActivate, outputPath])

  const handleSelectModel = (model: DiscoveredModel) => {
    setSourcePath(model.path)
    const nameWithoutExt = model.name.replace(/\.[^/.]+$/, "")
    const parentDir = model.path.substring(0, model.path.lastIndexOf(/[/\\]/.exec(model.path)?.[0] || "/"))
    const sep = model.path.includes("\\") ? "\\" : "/"
    setOutputPath(`${parentDir}${sep}${nameWithoutExt}.qwn`)
  }

  const handleStartConversion = async () => {
    if (!sourcePath.trim()) {
      setError("Please select or enter a source model path.")
      return
    }
    setError("")
    setLoadingAction(true)
    autoActivatedRef.current = false
    try {
      await startConversion(baseUrl, apiKey, sourcePath.trim(), outputPath.trim() || undefined, quantMode)
      const data = await getConversionStatus(baseUrl, apiKey)
      setStatus(data)
    } catch (err: any) {
      setError(err?.message || "Failed to start conversion")
    } finally {
      setLoadingAction(false)
    }
  }

  const handleLoadConvertedModel = async (targetPath?: string) => {
    const targetModel = targetPath || status.output || outputPath
    if (!targetModel) return
    setLoadingAction(true)
    try {
      await loadModel(baseUrl, apiKey, targetModel, "qwn")
      onModelLoaded?.(targetModel)
      onNavigateToChat?.()
    } catch (err: any) {
      setError(err?.message || "Failed to load converted model")
    } finally {
      setLoadingAction(false)
    }
  }

  // Model size estimation heuristics
  const modelFormat = sourcePath.toLowerCase().endsWith(".gguf")
    ? "GGUF"
    : sourcePath.toLowerCase().endsWith(".safetensors") || sourcePath.includes("safetensors")
    ? "Safetensors"
    : sourcePath.toLowerCase().endsWith(".pt") || sourcePath.toLowerCase().endsWith(".pth") || sourcePath.toLowerCase().endsWith(".bin")
    ? "PyTorch"
    : sourcePath.toLowerCase().endsWith(".onnx")
    ? "ONNX"
    : sourcePath.toLowerCase().endsWith(".h5") || sourcePath.toLowerCase().endsWith(".keras")
    ? "Keras"
    : "Generic"

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border pb-4">
        <div>
          <h2 className="text-xl font-bold flex items-center gap-2">
            <Zap className="size-5 text-primary" /> Qwanto Native (.qwn) Converter Studio
          </h2>
          <p className="text-xs text-muted-foreground mt-1">
            Transform any checkpoint (.gguf, .safetensors, .pt, .bin, .onnx, .h5) into ultra-fast 4KiB NVMe-aligned .qwn with SIMD quantization.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge className="bg-primary/20 text-primary border-primary/40 text-xs">
            <Sparkles className="size-3 mr-1" /> SIMD Vectorized
          </Badge>
          <Badge className="bg-emerald-950/80 text-emerald-300 border-emerald-800/60 text-xs">
            4KiB NVMe Paged
          </Badge>
          <Badge className="bg-blue-950/80 text-blue-300 border-blue-800/60 text-xs">
            Zero RAM Bloat
          </Badge>
        </div>
      </div>

      {error && (
        <div className="p-3 bg-red-950/50 border border-red-800/60 rounded-lg text-red-300 text-xs flex items-center gap-2">
          <AlertCircle className="size-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Main Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Left Column: Discovered Models */}
        <div className="space-y-4 md:col-span-1">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold flex items-center gap-1.5">
              <HardDrive className="size-4 text-muted-foreground" /> Discovered Models
            </h3>
            <Button
              size="sm"
              variant="ghost"
              className="h-7 px-2 text-xs"
              onClick={refreshDiscoveredModels}
              disabled={loadingModels}
            >
              <RefreshCw className={`size-3 mr-1 ${loadingModels ? "animate-spin" : ""}`} /> Refresh
            </Button>
          </div>

          <div className="border border-border/80 rounded-xl p-2 bg-card/40 max-h-96 overflow-y-auto space-y-1.5">
            {discoveredModels.length === 0 ? (
              <div className="text-center py-8 text-xs text-muted-foreground">
                No raw models found in scanned paths.
              </div>
            ) : (
              discoveredModels.map((m, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSelectModel(m)}
                  className={`w-full text-left p-2.5 rounded-lg text-xs transition border ${
                    sourcePath === m.path
                      ? "bg-primary/15 border-primary/50 text-foreground"
                      : "bg-background/40 hover:bg-muted/50 border-border/50 text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <div className="font-medium truncate">{m.name}</div>
                  <div className="flex items-center gap-1.5 mt-1">
                    <span className="border border-border/60 rounded px-1.5 py-0.5 text-[10px] uppercase font-mono">
                      {m.type}
                    </span>
                    <span className="text-[10px] text-muted-foreground truncate">{m.path}</span>
                  </div>
                </button>
              ))
            )}
          </div>
        </div>

        {/* Right Column: Configuration, Compression Bar & Status */}
        <div className="space-y-5 md:col-span-2">
          <div className="border border-border rounded-xl p-5 bg-card/60 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold flex items-center gap-2">
                <SlidersHorizontal className="size-4 text-primary" /> Model Transformation Settings
              </h3>
              {sourcePath && (
                <Badge className="bg-primary/15 text-primary border-primary/30 text-[10px] font-mono">
                  Source Format: {modelFormat}
                </Badge>
              )}
            </div>

            {/* Source Path Input */}
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground flex items-center gap-1">
                <FileCode className="size-3.5" /> Source Model Path (.gguf, .safetensors, .pt, .bin, .onnx, .h5)
              </label>
              <Input
                placeholder="/path/to/source_model.gguf or safetensors directory"
                value={sourcePath}
                onChange={(e) => {
                  setSourcePath(e.target.value)
                  if (!outputPath && e.target.value) {
                    const name = e.target.value.replace(/\.[^/.]+$/, "")
                    setOutputPath(`${name}.qwn`)
                  }
                }}
                className="font-mono text-xs bg-background/80"
              />
            </div>

            {/* Output Path Input */}
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground flex items-center gap-1">
                <HardDrive className="size-3.5" /> Target Qwanto Native (.qwn) Destination
              </label>
              <Input
                placeholder="/path/to/output_model.qwn"
                value={outputPath}
                onChange={(e) => setOutputPath(e.target.value)}
                className="font-mono text-xs bg-background/80"
              />
            </div>

            {/* Quantization Mode Cards */}
            <div className="space-y-1.5 pt-1">
              <label className="text-xs font-medium text-muted-foreground">Quantization Target</label>
              <div className="grid grid-cols-3 gap-3">
                <button
                  type="button"
                  onClick={() => setQuantMode("vsq")}
                  className={`p-3 rounded-lg border text-left transition ${
                    quantMode === "vsq"
                      ? "bg-primary/15 border-primary text-foreground ring-1 ring-primary/40"
                      : "bg-background/40 border-border text-muted-foreground hover:bg-muted/50"
                  }`}
                >
                  <div className="font-semibold text-xs flex items-center justify-between">
                    <span className="flex items-center gap-1"><Zap className="size-3 text-primary" /> QWN-VSQ</span>
                    <Badge className="text-[9px] bg-primary/30 text-primary border-primary/50">~80% Saved</Badge>
                  </div>
                  <p className="text-[10px] text-muted-foreground mt-1">
                    Proprietary 64-element dual-scale superblock with 10x in-register AVX2 shuffle decoding.
                  </p>
                </button>

                <button
                  type="button"
                  onClick={() => setQuantMode("q4_0")}
                  className={`p-3 rounded-lg border text-left transition ${
                    quantMode === "q4_0"
                      ? "bg-primary/15 border-primary text-foreground"
                      : "bg-background/40 border-border text-muted-foreground hover:bg-muted/50"
                  }`}
                >
                  <div className="font-semibold text-xs flex items-center justify-between">
                    <span>Q4_0 SIMD</span>
                    <Badge className="text-[9px] bg-muted text-muted-foreground">~75% Saved</Badge>
                  </div>
                  <p className="text-[10px] text-muted-foreground mt-1">
                    Standard 32-element 4-bit block representation with FP16 scale.
                  </p>
                </button>

                <button
                  type="button"
                  onClick={() => setQuantMode("none")}
                  className={`p-3 rounded-lg border text-left transition ${
                    quantMode === "none"
                      ? "bg-primary/15 border-primary text-foreground"
                      : "bg-background/40 border-border text-muted-foreground hover:bg-muted/50"
                  }`}
                >
                  <div className="font-semibold text-xs flex items-center justify-between">
                    <span>Full Precision</span>
                    <span className="border border-border/60 rounded px-1.5 py-0.5 text-[9px]">None</span>
                  </div>
                  <p className="text-[10px] text-muted-foreground mt-1">
                    Preserves original F32, F16, and BF16 tensor values without conversion.
                  </p>
                </button>
              </div>
            </div>

            {/* Memory & Size Comparison Card */}
            {(quantMode === "vsq" || quantMode === "q4_0") && (
              <div className="p-3 bg-primary/5 border border-primary/20 rounded-lg space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium text-foreground flex items-center gap-1.5">
                    <TrendingDown className="size-3.5 text-primary" /> Memory & Storage Efficiency
                  </span>
                  <span className="text-emerald-400 font-semibold font-mono text-[11px]">
                    {quantMode === "vsq" ? "~78% - 82% Compression (QWN-VSQ)" : "~71% - 75% Compression (Q4_0)"}
                  </span>
                </div>
                <div className="w-full bg-muted/60 rounded-full h-2 overflow-hidden flex">
                  <div className="bg-primary h-full" style={{ width: quantMode === "vsq" ? "20%" : "27%" }} title="Compressed Size" />
                  <div className="bg-muted-foreground/30 h-full" style={{ width: quantMode === "vsq" ? "80%" : "73%" }} title="Saved Space" />
                </div>
                <div className="flex justify-between text-[10px] text-muted-foreground font-mono">
                  <span>Target: {quantMode === "vsq" ? "QWN-VSQ Container (~20% size)" : "Q4_0 Container (~27% size)"}</span>
                  <span>Uncompressed Baseline (100%)</span>
                </div>
              </div>
            )}

            {/* Auto-Activation Checkbox & Action Button */}
            <div className="pt-2 flex items-center justify-between border-t border-border/60">
              <label className="flex items-center gap-2 text-xs text-muted-foreground cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={autoActivate}
                  onChange={(e) => setAutoActivate(e.target.checked)}
                  className="rounded border-border text-primary focus:ring-0"
                />
                <span>Auto-set as default model & start chat on completion</span>
              </label>

              <Button
                onClick={handleStartConversion}
                disabled={status.status === "converting" || loadingAction || !sourcePath}
                className="gap-2"
              >
                {status.status === "converting" ? (
                  <>
                    <RefreshCw className="size-4 animate-spin" /> Converting...
                  </>
                ) : (
                  <>
                    <Zap className="size-4" /> Convert to .qwn
                  </>
                )}
              </Button>
            </div>
          </div>

          {/* Status & Terminal Panel */}
          <div className="border border-border rounded-xl p-5 bg-card/40 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
                <Clock className="size-3.5" /> Conversion Telemetry
              </h3>
              <Badge
                className={
                  status.status === "done"
                    ? "bg-emerald-950 text-emerald-300 border-emerald-800"
                    : status.status === "converting"
                    ? "bg-blue-950 text-blue-300 border-blue-800 animate-pulse"
                    : status.status === "error"
                    ? "bg-red-950 text-red-300 border-red-800"
                    : "bg-muted text-muted-foreground"
                }
              >
                {status.status.toUpperCase()}
              </Badge>
            </div>

            {/* Progress Bar */}
            {status.status === "converting" && (
              <div className="w-full bg-muted/60 rounded-full h-2 overflow-hidden">
                <div
                  className="bg-primary h-full transition-all duration-300 animate-pulse"
                  style={{ width: `${Math.max(status.progress, 15)}%` }}
                />
              </div>
            )}

            {/* Log & Stats Box */}
            <div className="p-3 bg-black/40 rounded-lg border border-border/40 font-mono text-xs text-muted-foreground space-y-1.5">
              <div className="text-foreground">{status.message}</div>
              {status.elapsed ? (
                <div className="text-[11px] text-muted-foreground/80 flex items-center gap-2">
                  <span>Elapsed Time: <strong className="text-foreground">{status.elapsed}s</strong></span>
                  {status.speed_mb_s ? (
                    <> &bull; <span>Throughput: <strong className="text-emerald-400 font-semibold">{status.speed_mb_s} MB/s</strong></span></>
                  ) : null}
                </div>
              ) : null}
            </div>

            {/* Quick Activation Bar */}
            {status.status === "done" && (
              <div className="p-3 bg-emerald-950/40 border border-emerald-800/60 rounded-lg flex items-center justify-between">
                <div className="text-xs text-emerald-300 flex items-center gap-2">
                  <CheckCircle2 className="size-4 shrink-0 text-emerald-400" />
                  <span>Model ready for ultra-fast native inference!</span>
                </div>
                <Button
                  size="sm"
                  variant="default"
                  onClick={() => handleLoadConvertedModel()}
                  disabled={loadingAction}
                  className="bg-emerald-600 hover:bg-emerald-500 text-white gap-1.5 h-8 text-xs"
                >
                  <Play className="size-3.5 fill-current" /> Load & Start Chat
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
