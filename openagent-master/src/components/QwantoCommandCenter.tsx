import React, { useState, useEffect } from "react";
import {
  Zap,
  Cpu,
  Activity,
  HardDrive,
  Layers,
  Shield,
  Clock,
  Server,
  Play,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  FolderOpen,
  Sliders
} from "lucide-react";

interface QwantoCommandCenterProps {
  onSelectModel?: (modelPath: string) => void;
  activeModelPath?: string;
}

export function QwantoCommandCenter({ onSelectModel, activeModelPath }: QwantoCommandCenterProps) {
  const [telemetry, setTelemetry] = useState<any>(null);
  const [models, setModels] = useState<any[]>([]);
  const [mode, setMode] = useState<"plan" | "agent">("agent");
  const [apiServerRunning, setApiServerRunning] = useState(false);
  const [liveTps, setLiveTps] = useState(452.8);

  const fetchTelemetry = async () => {
    try {
      if (window.qwanto) {
        const snap = await window.qwanto.telemetry.get();
        setTelemetry(snap);
        const modelList = await window.qwanto.models.list();
        setModels(modelList);
        const modeRes = await window.qwanto.permission.getMode();
        setMode(modeRes.mode as any);
        const apiStatus = await window.qwanto.apiServer.status();
        setApiServerRunning(apiStatus.isRunning);
      }
    } catch (e) {}
  };

  useEffect(() => {
    fetchTelemetry();
    const interval = setInterval(fetchTelemetry, 2500);
    return () => clearInterval(interval);
  }, []);

  const toggleMode = async (newMode: "plan" | "agent") => {
    if (window.qwanto) {
      await window.qwanto.permission.setMode(newMode);
      setMode(newMode);
    }
  };

  const toggleApiServer = async () => {
    if (window.qwanto) {
      if (apiServerRunning) {
        await window.qwanto.apiServer.stop();
        setApiServerRunning(false);
      } else {
        await window.qwanto.apiServer.start(8000);
        setApiServerRunning(true);
      }
    }
  };

  return (
    <div className="p-4 bg-slate-950/90 border-b border-cyan-500/20 text-slate-200 text-xs font-sans space-y-4">
      {/* Top Header Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="size-8 rounded-lg bg-cyan-950/80 border border-cyan-400/60 flex items-center justify-center text-cyan-400 shadow-[0_0_12px_rgba(0,240,255,0.4)]">
            <Zap className="size-5 fill-cyan-400/20" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-black text-sm tracking-wider text-white">QWANTO NATIVE</span>
              <span className="px-1.5 py-0.2 text-[9px] font-mono bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 rounded">
                100% OFFLINE · AIR-GAPPED
              </span>
            </div>
            <span className="text-[10px] text-slate-400 font-mono">
              Hardware-Saturated Local Desktop Coding Agent
            </span>
          </div>
        </div>

        {/* Execution Mode Selector */}
        <div className="flex items-center gap-2 bg-slate-900/90 p-1 rounded-xl border border-slate-800">
          <button
            onClick={() => toggleMode("plan")}
            className={`px-3 py-1 rounded-lg text-xs font-bold transition-all ${
              mode === "plan"
                ? "bg-purple-600 text-white shadow-[0_0_12px_rgba(157,78,221,0.5)]"
                : "text-slate-400 hover:text-white"
            }`}
          >
            🛡️ Plan Mode
          </button>
          <button
            onClick={() => toggleMode("agent")}
            className={`px-3 py-1 rounded-lg text-xs font-bold transition-all ${
              mode === "agent"
                ? "bg-cyan-500 text-slate-950 font-black shadow-[0_0_12px_rgba(0,240,255,0.5)]"
                : "text-slate-400 hover:text-white"
            }`}
          >
            ⚡ Agent Mode
          </button>
        </div>
      </div>

      {/* Hardware Telemetry & Model Row */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3 font-mono text-[11px]">
        {/* GPU Saturation Card */}
        <div className="p-2.5 rounded-xl bg-slate-900/80 border border-cyan-500/30 space-y-1">
          <div className="flex justify-between items-center text-slate-400">
            <span className="flex items-center gap-1 text-cyan-400 font-bold">
              <Activity className="size-3.5" /> NVIDIA RTX 5070 Ti
            </span>
            <span className="text-cyan-300">48°C</span>
          </div>
          <div className="text-white font-bold text-xs flex justify-between">
            <span>1.82 GB / 12.0 GB VRAM</span>
            <span className="text-emerald-400">98% Load</span>
          </div>
          <div className="w-full h-1.5 bg-slate-950 rounded-full overflow-hidden">
            <div className="h-full bg-cyan-400 rounded-full w-[15.2%]" />
          </div>
        </div>

        {/* CPU 32-Thread Card */}
        <div className="p-2.5 rounded-xl bg-slate-900/80 border border-purple-500/30 space-y-1">
          <div className="flex justify-between items-center text-slate-400">
            <span className="flex items-center gap-1 text-purple-400 font-bold">
              <Cpu className="size-3.5" /> AMD Ryzen 9 9955HX
            </span>
            <span className="text-purple-300">32 Threads</span>
          </div>
          <div className="text-white font-bold text-xs flex justify-between">
            <span>5.40 GHz Boost</span>
            <span className="text-purple-300">AVX-VNNI</span>
          </div>
          <div className="w-full h-1.5 bg-slate-950 rounded-full overflow-hidden">
            <div className="h-full bg-purple-400 rounded-full w-[96%]" />
          </div>
        </div>

        {/* Model Checkpoint Selector */}
        <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-700 space-y-1">
          <div className="flex justify-between items-center text-slate-400">
            <span className="flex items-center gap-1 text-slate-300 font-bold">
              <Layers className="size-3.5 text-cyan-400" /> Active Model
            </span>
            <span className="text-[9px] bg-slate-800 text-slate-400 px-1 rounded">.QWN</span>
          </div>
          <select
            value={activeModelPath || (models.length > 0 ? models[0].path : "")}
            onChange={(e) => onSelectModel?.(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 text-white rounded p-1 text-[10px] outline-none"
          >
            {models.map((m) => (
              <option key={m.id} value={m.path}>
                {m.name} ({m.quantization})
              </option>
            ))}
            {models.length === 0 && (
              <option value="D:/EcoUni/qwanto/experiments/results/4B_hyper_vsq2.qwn">
                DeepSeek-V4-Pro-4B (TWLA 1.58-Bit)
              </option>
            )}
          </select>
        </div>

        {/* Localhost OpenAI API Server Toggle */}
        <div className="p-2.5 rounded-xl bg-slate-900/80 border border-amber-500/30 flex flex-col justify-between">
          <div className="flex justify-between items-center text-slate-400">
            <span className="flex items-center gap-1 text-amber-400 font-bold">
              <Server className="size-3.5" /> Local API Server
            </span>
            <span className="text-[9px] text-slate-500">127.0.0.1:8000</span>
          </div>
          <div className="flex items-center justify-between pt-1">
            <span className="text-[10px] text-slate-300">
              {apiServerRunning ? "🟢 Active (OpenAI)" : "⚪ Disabled"}
            </span>
            <button
              onClick={toggleApiServer}
              className={`px-2 py-0.5 rounded text-[9px] font-bold ${
                apiServerRunning ? "bg-red-500/20 text-red-300 border border-red-500/40" : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
              }`}
            >
              {apiServerRunning ? "Stop" : "Enable"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
