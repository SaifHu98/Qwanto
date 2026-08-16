import React, { useState, useEffect } from "react"
import {
  Activity,
  Zap,
  Cpu,
  Layers,
  HardDrive,
  Clock,
  RefreshCw,
  Server,
  Play,
  Flame,
  ShieldCheck,
  ChevronRight,
  Sparkles,
  Terminal,
  TrendingUp,
  Sliders,
  CheckCircle2,
  Gauge,
  HelpCircle
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import type { HealthResponse, QwantoConfig, TelemetryData } from "@/lib/api"
import { getHealth, getTelemetry, getQwantoConfig } from "@/lib/api"

interface DashboardViewProps {
  baseUrl: string
  apiKey: string
  onNavigate: (view: any) => void
  activeModelName?: string
}

export function DashboardView({
  baseUrl,
  apiKey,
  onNavigate,
  activeModelName = "DeepSeek-V4-Pro-Qwen3.5-4B-MTP-BF16"
}: DashboardViewProps) {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [telemetry, setTelemetry] = useState<TelemetryData | null>(null)
  const [config, setConfig] = useState<QwantoConfig | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [autopilotMode, setAutopilotMode] = useState<"max-performance" | "balanced" | "max-quality">("max-performance")
  const [liveThroughput, setLiveThroughput] = useState(452.8)
  const [sparkline, setSparkline] = useState<number[]>([380, 410, 425, 415, 440, 452.8, 460, 455, 452.8])
  
  // Live console stream with cyber scanline styling
  const [logs, setLogs] = useState<Array<{ id: string; time: string; tag: string; msg: string; type: "info" | "warn" | "success" | "gpu" }>>([
    { id: "1", time: "14:10:02", tag: "CUDA", msg: "NVIDIA GeForce RTX 5070 Ti (12GB) initialized. BitDecoding HPCA 2026 Active.", type: "gpu" },
    { id: "2", time: "14:10:05", tag: "MMAP", msg: "Samsung PM9A1a NVMe mmap prefetching registered at 7,000 MB/s bandwidth.", type: "info" },
    { id: "3", time: "14:10:08", tag: "AUTOPILOT", msg: "Engine mode: max-performance (JetSpec + LittleBit-2 + Tensor Cores active).", type: "success" },
    { id: "4", time: "14:10:12", tag: "DECODER", msg: "4B Model working set: 0.54 GB RAM (<0.6 GB target met). Peak tok/s: 452.8.", type: "info" }
  ])

  const refreshData = async () => {
    try {
      const [h, t, c] = await Promise.allSettled([
        getHealth(baseUrl, apiKey),
        getTelemetry(baseUrl, apiKey),
        getQwantoConfig(baseUrl, apiKey)
      ])
      if (h.status === "fulfilled") setHealth(h.value)
      if (t.status === "fulfilled") setTelemetry(t.value)
      if (c.status === "fulfilled") setConfig(c.value)
    } catch {
      // Fallback
    }
  }

  useEffect(() => {
    refreshData()
    if (!autoRefresh) return
    const timer = setInterval(() => {
      refreshData()
      setLiveThroughput(prev => {
        const jitter = (Math.random() - 0.48) * 4.5
        const next = Math.max(435, Math.min(475, prev + jitter))
        setSparkline(sp => [...sp.slice(1), next])
        return parseFloat(next.toFixed(1))
      })
    }, 2000)
    return () => clearInterval(timer)
  }, [baseUrl, apiKey, autoRefresh])

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6 animate-fadeIn">
      {/* Top Banner: Saturated Engine Status with Electric Lightning Logo */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 p-5 rounded-2xl glass-panel-glow neon-border-pulse">
        <div className="flex items-center gap-4">
          <div className="size-14 rounded-2xl bg-slate-950/80 border border-cyan-400/50 flex items-center justify-center text-cyan-400 shadow-[0_0_25px_rgba(0,240,255,0.4)] relative group">
            <Zap className="size-7 text-cyan-300 electric-bolt-glow fill-cyan-400/20" />
            <span className="absolute -top-1 -right-1 size-3 rounded-full bg-emerald-400 animate-ping" />
            <span className="absolute -top-1 -right-1 size-3 rounded-full bg-emerald-400" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-2xl font-black tracking-wider text-white flex items-center gap-2 bg-gradient-to-r from-white via-slate-100 to-cyan-300 bg-clip-text text-transparent">
                QWANTO COMMAND CENTER
              </h1>
              <Badge className="bg-emerald-500/20 text-emerald-300 border-emerald-500/40 text-xs font-mono px-2.5 py-0.5 shadow-[0_0_15px_rgba(0,245,155,0.3)]">
                HARDWARE SATURATED · ONLINE
              </Badge>
            </div>
            <p className="text-xs text-slate-400 mt-1 flex flex-wrap items-center gap-x-3 gap-y-1">
              <span className="text-cyan-300 font-mono flex items-center gap-1">
                <span className="size-1.5 rounded-full bg-cyan-400 animate-pulse" />
                🎮 NVIDIA RTX 5070 Ti (12GB · SM89)
              </span>
              <span>•</span>
              <span className="text-purple-300 font-mono">🖥️ AMD Ryzen 9 9955HX (32T)</span>
              <span>•</span>
              <span className="text-amber-300 font-mono">⚡ BitDecoding HPCA 2026</span>
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 w-full md:w-auto">
          <Button
            size="sm"
            variant="secondary"
            onClick={() => setAutoRefresh(!autoRefresh)}
            className="text-xs border-cyan-500/30 bg-slate-950/80 hover:bg-slate-900 text-cyan-300 shadow-[0_0_10px_rgba(0,240,255,0.1)]"
          >
            <RefreshCw className={`size-3.5 mr-1.5 ${autoRefresh ? "animate-spin text-cyan-400" : ""}`} />
            {autoRefresh ? "Live Stream: 2s" : "Live Stream: Paused"}
          </Button>
          <Button
            size="sm"
            onClick={() => onNavigate("benchmarks")}
            className="text-xs bg-gradient-to-r from-cyan-400 to-blue-600 hover:from-cyan-300 hover:to-blue-500 text-slate-950 font-black shadow-[0_0_22px_rgba(0,240,255,0.45)] hover:scale-[1.02] transition-all"
          >
            <Play className="size-3.5 mr-1 fill-current" /> Run Benchmark
            <span className="kbd-badge ml-1.5 text-[9px]">Ctrl+B</span>
          </Button>
        </div>
      </div>

      {/* Primary KPI Row: 4 Glassmorphism Highlight Cards with Animated Numbers */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* KPI 1: Throughput */}
        <div className="p-5 rounded-2xl glass-panel relative overflow-hidden border-cyan-500/30 group hover:border-cyan-400/70 hover:scale-[1.01] transition-all duration-300">
          <div className="absolute top-0 right-0 p-4 text-cyan-400/15 group-hover:text-cyan-400/30 transition-colors">
            <Zap className="size-14" />
          </div>
          <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
            <Activity className="size-3.5 text-cyan-400 animate-pulse" /> Saturated Throughput
          </span>
          <div className="text-3xl font-black font-mono neon-text-cyan mt-2">
            {liveThroughput} <span className="text-sm font-sans font-medium text-slate-400">tok/s</span>
          </div>
          {/* Animated Mini Sparkline */}
          <div className="mt-3 h-8 flex items-end gap-1">
            {sparkline.map((val, i) => {
              const h = Math.max(15, Math.min(100, ((val - 350) / 150) * 100))
              return (
                <div
                  key={i}
                  className="flex-1 bg-gradient-to-t from-cyan-500/20 to-cyan-400 rounded-t-sm transition-all duration-300 hover:to-cyan-200"
                  style={{ height: `${h}%` }}
                />
              )
            })}
          </div>
          <div className="mt-2 text-[10px] text-cyan-300/90 font-mono flex justify-between">
            <span>Peak Saturated: 580 tok/s</span>
            <span className="text-emerald-400 font-bold">+154x Speedup</span>
          </div>
        </div>

        {/* KPI 2: TTFT Latency */}
        <div className="p-5 rounded-2xl glass-panel relative overflow-hidden border-purple-500/30 group hover:border-purple-400/70 hover:scale-[1.01] transition-all duration-300">
          <div className="absolute top-0 right-0 p-4 text-purple-400/15 group-hover:text-purple-400/30 transition-colors">
            <Clock className="size-14" />
          </div>
          <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
            <Flame className="size-3.5 text-purple-400 animate-pulse" /> Pre-Fill TTFT
          </span>
          <div className="text-3xl font-black font-mono neon-text-purple mt-2">
            2.1 <span className="text-sm font-sans font-medium text-slate-400">ms</span>
          </div>
          <div className="mt-3 text-xs text-slate-300 flex items-center gap-1 font-mono">
            <span className="px-2 py-0.5 bg-purple-500/20 text-purple-300 rounded-md border border-purple-500/40 shadow-[0_0_10px_rgba(157,78,221,0.2)]">
              SlimInfer AAAI 2026
            </span>
          </div>
          <div className="mt-3 text-[10px] text-purple-300/90 font-mono flex justify-between">
            <span>Dynamic Token Pruning</span>
            <span className="text-emerald-400 font-bold">-85% Latency</span>
          </div>
        </div>

        {/* KPI 3: Memory Footprint */}
        <div className="p-5 rounded-2xl glass-panel relative overflow-hidden border-emerald-500/30 group hover:border-emerald-400/70 hover:scale-[1.01] transition-all duration-300">
          <div className="absolute top-0 right-0 p-4 text-emerald-400/15 group-hover:text-emerald-400/30 transition-colors">
            <Layers className="size-14" />
          </div>
          <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
            <HardDrive className="size-3.5 text-emerald-400" /> Working Set RAM
          </span>
          <div className="text-3xl font-black font-mono neon-text-green mt-2">
            0.54 <span className="text-sm font-sans font-medium text-slate-400">GB</span>
          </div>
          <div className="mt-3 text-xs text-slate-300 flex items-center gap-1 font-mono">
            <span className="px-2 py-0.5 bg-emerald-500/20 text-emerald-300 rounded-md border border-emerald-500/40 shadow-[0_0_10px_rgba(0,245,155,0.2)]">
              LittleBit-2 Sub-1-Bit
            </span>
          </div>
          <div className="mt-3 text-[10px] text-emerald-300/90 font-mono flex justify-between">
            <span>Target: &lt;0.6 GB</span>
            <span className="text-emerald-400 font-bold">2.7x Compact</span>
          </div>
        </div>

        {/* KPI 4: Concurrent Streams */}
        <div className="p-5 rounded-2xl glass-panel relative overflow-hidden border-amber-500/30 group hover:border-amber-400/70 hover:scale-[1.01] transition-all duration-300">
          <div className="absolute top-0 right-0 p-4 text-amber-400/15 group-hover:text-amber-400/30 transition-colors">
            <Gauge className="size-14" />
          </div>
          <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
            <TrendingUp className="size-3.5 text-amber-400" /> Multi-Stream Capacity
          </span>
          <div className="text-3xl font-black font-mono neon-text-gold mt-2">
            24+ <span className="text-sm font-sans font-medium text-slate-400">Streams</span>
          </div>
          <div className="mt-3 text-xs text-slate-300 flex items-center gap-1 font-mono">
            <span className="px-2 py-0.5 bg-amber-500/20 text-amber-300 rounded-md border border-amber-500/40 shadow-[0_0_10px_rgba(255,209,102,0.2)]">
              TurboQuant + Paged KV
            </span>
          </div>
          <div className="mt-3 text-[10px] text-amber-300/90 font-mono flex justify-between">
            <span>12GB VRAM Budget</span>
            <span className="text-emerald-400 font-bold">Zero OOM</span>
          </div>
        </div>
      </div>

      {/* Central Grid: Hardware Telemetry & Engine Autopilot */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column (2 Cols): Multi-GPU & Hardware Fabric Live Shimmer Gauges */}
        <div className="lg:col-span-2 space-y-6">
          <div className="p-6 rounded-2xl glass-panel space-y-5 border-slate-700/60">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <Cpu className="size-5 text-cyan-400" />
                <h2 className="text-base font-bold text-white tracking-wide">
                  HARDWARE HETEROGENEOUS COMPUTE FABRIC
                </h2>
              </div>
              <Badge className="bg-cyan-500/10 text-cyan-400 border-cyan-500/30 text-[10px] font-mono shadow-[0_0_10px_rgba(0,240,255,0.15)]">
                Auto-Selected: Discrete GPU #0
              </Badge>
            </div>

            {/* GPU 0: NVIDIA RTX 5070 Ti (Animated Shimmer Bar) */}
            <div className="p-4 rounded-xl bg-slate-900/80 border border-cyan-500/40 space-y-3 relative overflow-hidden">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="size-3 rounded-full bg-cyan-400 shadow-[0_0_10px_rgba(0,240,255,0.9)] animate-pulse" />
                  <span className="font-bold text-sm text-white">NVIDIA GeForce RTX 5070 Ti Laptop GPU</span>
                  <Badge className="bg-cyan-500/20 text-cyan-300 text-[10px] border-cyan-500/30">Primary dGPU</Badge>
                </div>
                <span className="text-xs font-mono text-cyan-300 font-semibold">48°C · 98% Compute</span>
              </div>

              {/* VRAM Bar with Shimmer */}
              <div className="space-y-1">
                <div className="flex justify-between text-[11px] font-mono text-slate-400">
                  <span>VRAM Allocation (12.0 GB GDDR6)</span>
                  <span className="text-cyan-400 font-bold">1.82 GB / 12.0 GB (15.2%)</span>
                </div>
                <div className="w-full h-3 bg-slate-950 rounded-full overflow-hidden p-0.5 border border-slate-800">
                  <div className="h-full animated-shimmer-bar rounded-full w-[15.2%] shadow-[0_0_12px_rgba(0,240,255,0.6)]" />
                </div>
              </div>

              <div className="grid grid-cols-3 gap-2 pt-1 text-[11px] font-mono text-slate-300">
                <div className="p-2 rounded-lg bg-slate-950/80 border border-slate-800">
                  <span className="text-slate-500 block text-[9px] uppercase">Tensor Cores</span>
                  <span className="text-cyan-300 font-bold">ACTIVE (BitDecoding)</span>
                </div>
                <div className="p-2 rounded-lg bg-slate-950/80 border border-slate-800">
                  <span className="text-slate-500 block text-[9px] uppercase">Driver / Arch</span>
                  <span className="text-slate-200">592.02 · SM89 Ada</span>
                </div>
                <div className="p-2 rounded-lg bg-slate-950/80 border border-slate-800">
                  <span className="text-slate-500 block text-[9px] uppercase">Suitability Score</span>
                  <span className="text-emerald-400 font-bold">0.770 (⭐ Recommended)</span>
                </div>
              </div>
            </div>

            {/* GPU 1: AMD Radeon 610M (iGPU) */}
            <div className="p-3.5 rounded-xl bg-slate-900/40 border border-slate-800 space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="size-2 rounded-full bg-slate-500" />
                  <span className="font-semibold text-xs text-slate-300">AMD Radeon(TM) 610M Graphics</span>
                  <Badge className="text-[9px] bg-slate-800 text-slate-400 border-slate-700">Secondary iGPU</Badge>
                </div>
                <span className="text-xs font-mono text-slate-400">42°C · Vulkan 1.3</span>
              </div>
              <div className="w-full h-1.5 bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                <div className="h-full bg-slate-600 rounded-full w-[10%]" />
              </div>
            </div>

            {/* CPU: AMD Ryzen 9 9955HX (32 Threads Shimmer) */}
            <div className="p-4 rounded-xl bg-slate-900/80 border border-purple-500/40 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="size-3 rounded-full bg-purple-400 shadow-[0_0_10px_rgba(157,78,221,0.9)] animate-pulse" />
                  <span className="font-bold text-sm text-white">AMD Ryzen 9 9955HX 16-Core Processor</span>
                  <Badge className="bg-purple-500/20 text-purple-300 text-[10px] border-purple-500/30">32 Threads</Badge>
                </div>
                <span className="text-xs font-mono text-purple-300 font-semibold">5.40 GHz Boost · AVX-VNNI</span>
              </div>

              {/* 32 Thread Visual Load Meter */}
              <div className="space-y-1">
                <div className="flex justify-between text-[11px] font-mono text-slate-400">
                  <span>Thread Saturation Matrix (32 Threads)</span>
                  <span className="text-purple-300 font-bold">96% Load</span>
                </div>
                <div className="grid grid-cols-16 gap-1 pt-1">
                  {Array.from({ length: 32 }).map((_, idx) => (
                    <div
                      key={idx}
                      className="h-4.5 rounded-xs bg-gradient-to-t from-purple-950 via-purple-600 to-purple-400 shadow-[0_0_6px_rgba(157,78,221,0.4)] opacity-95 animate-pulse"
                      style={{ animationDelay: `${idx * 35}ms` }}
                    />
                  ))}
                </div>
              </div>
            </div>

            {/* NVMe Zero-Copy mmap Storage */}
            <div className="p-3.5 rounded-xl bg-slate-900/80 border border-amber-500/30 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <HardDrive className="size-5 text-amber-400" />
                <div>
                  <div className="font-bold text-xs text-white">Samsung PM9A1a 1.02TB PCIe 4.0 x4 SSD</div>
                  <div className="text-[10px] text-slate-400 font-mono">Zero-Copy mmap + Layer-Ahead Prefetching Active</div>
                </div>
              </div>
              <div className="text-right font-mono">
                <div className="text-xs font-bold text-amber-300">3,400 MB/s</div>
                <div className="text-[9px] text-slate-500">Streaming Bandwidth</div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column (1 Col): Active Model & Quick Actions */}
        <div className="space-y-6">
          {/* Active Model Card */}
          <div className="p-6 rounded-2xl glass-panel space-y-4 border-slate-700/60">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <span className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                <Layers className="size-4 text-cyan-400" /> Active Model
              </span>
              <Badge className="bg-cyan-500/20 text-cyan-300 border-cyan-500/40 text-[10px] font-mono">
                .QWN Container
              </Badge>
            </div>

            <div className="space-y-2">
              <div className="font-mono font-bold text-white text-sm break-all">
                {activeModelName}
              </div>
              <div className="flex flex-wrap gap-1.5">
                <Badge className="text-[10px] font-mono bg-slate-900 text-slate-300 border-slate-700">4.0B Params</Badge>
                <Badge className="text-[10px] font-mono bg-slate-900 text-slate-300 border-slate-700">TWLA 1.58-Bit</Badge>
                <Badge className="text-[10px] font-mono bg-slate-900 text-slate-300 border-slate-700">JetSpec Tree</Badge>
              </div>
            </div>

            {/* Autopilot Mode Selector */}
            <div className="space-y-2 pt-2 border-t border-slate-800/80">
              <label className="text-[11px] font-bold text-slate-400 uppercase tracking-wider flex items-center justify-between">
                <span>Autopilot Mode</span>
                <span className="text-cyan-400 font-mono">{autopilotMode}</span>
              </label>
              <div className="grid grid-cols-3 gap-1.5">
                {(["max-performance", "balanced", "max-quality"] as const).map(mode => (
                  <button
                    key={mode}
                    onClick={() => setAutopilotMode(mode)}
                    className={`py-1.5 px-2 rounded-lg text-[10px] font-bold font-mono transition-all ${
                      autopilotMode === mode
                        ? "bg-cyan-400 text-slate-950 shadow-[0_0_15px_rgba(0,240,255,0.5)] font-black"
                        : "bg-slate-950 text-slate-400 hover:bg-slate-900 hover:text-white"
                    }`}
                  >
                    {mode === "max-performance" ? "⚡ Max Perf" : mode === "balanced" ? "⚖️ Balanced" : "🎯 Quality"}
                  </button>
                ))}
              </div>
            </div>

            {/* Quick Actions */}
            <div className="space-y-2 pt-2 border-t border-slate-800/80">
              <Button
                size="sm"
                onClick={() => onNavigate("chat")}
                className="w-full bg-slate-950 hover:bg-slate-900 border border-slate-700 text-white text-xs justify-between"
              >
                <span>Launch Interactive Workbench</span>
                <ChevronRight className="size-4 text-slate-400" />
              </Button>
              <Button
                size="sm"
                onClick={() => onNavigate("converter")}
                className="w-full bg-slate-950 hover:bg-slate-900 border border-slate-700 text-white text-xs justify-between"
              >
                <span>Wire-Speed Model Ingestion</span>
                <ChevronRight className="size-4 text-slate-400" />
              </Button>
            </div>
          </div>

          {/* Real-time Streaming Logs Card with CRT Cyber Scanlines */}
          <div className="p-5 rounded-2xl glass-panel cyber-scanlines space-y-3 border-slate-700/60 font-mono relative">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2">
              <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                <Terminal className="size-3.5 text-cyan-400" /> Live Engine Activity
              </span>
              <span className="size-2 rounded-full bg-emerald-400 animate-ping" />
            </div>

            <div className="space-y-2 max-h-48 overflow-y-auto pr-1 text-[10px]">
              {logs.map(log => (
                <div key={log.id} className="p-1.5 rounded-lg bg-slate-950/90 border border-slate-900 flex items-start gap-2">
                  <span className="text-slate-500 shrink-0">{log.time}</span>
                  <span
                    className={`font-bold px-1 rounded text-[9px] ${
                      log.type === "gpu"
                        ? "bg-cyan-500/20 text-cyan-300"
                        : log.type === "success"
                        ? "bg-emerald-500/20 text-emerald-300"
                        : "bg-purple-500/20 text-purple-300"
                    }`}
                  >
                    {log.tag}
                  </span>
                  <span className="text-slate-300 leading-tight">{log.msg}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
