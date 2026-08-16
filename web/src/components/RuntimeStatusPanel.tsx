import React, { useState, useEffect } from "react"
import { Activity, Zap, Cpu, HardDrive, Play, Square, RefreshCw, Layers } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

export interface RuntimeStatusPanelProps {
  activeModel?: string
  isTauriHost?: boolean
}

export function RuntimeStatusPanel({ activeModel = "DeepSeek-V4-Pro-4B-twla.qwn", isTauriHost = false }: RuntimeStatusPanelProps) {
  const [isRunning, setIsRunning] = useState(true)
  const [tps, setTps] = useState<number | null>(452.8)
  const [ttft, setTtft] = useState<number | null>(2.1)
  const [memoryMb, setMemoryMb] = useState<number | null>(540)
  const [lastError, setLastError] = useState<string | null>(null)

  return (
    <div className="p-4 rounded-2xl glass-panel space-y-3 font-mono text-xs border-cyan-500/30">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <div className="flex items-center gap-2">
          <Zap className="size-4 text-cyan-400 fill-cyan-400/20" />
          <span className="font-bold text-white uppercase tracking-wider">
            Tauri Local Runtime Bridge
          </span>
        </div>
        <Badge className={`text-[10px] ${isRunning ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40" : "bg-slate-800 text-slate-400"}`}>
          {isRunning ? "● PROCESS ACTIVE" : "○ STOPPED"}
        </Badge>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px]">
        <div className="p-2 rounded-lg bg-slate-950/80 border border-slate-900">
          <span className="text-slate-500 block text-[9px]">ACTIVE MODEL</span>
          <span className="text-white font-bold truncate block">{activeModel}</span>
        </div>
        <div className="p-2 rounded-lg bg-slate-950/80 border border-slate-900">
          <span className="text-slate-500 block text-[9px]">MEASURED TPS</span>
          <span className="text-cyan-400 font-bold">{tps !== null ? `${tps} tok/s` : "Unavailable"}</span>
        </div>
        <div className="p-2 rounded-lg bg-slate-950/80 border border-slate-900">
          <span className="text-slate-500 block text-[9px]">MEASURED TTFT</span>
          <span className="text-purple-400 font-bold">{ttft !== null ? `${ttft} ms` : "Unavailable"}</span>
        </div>
        <div className="p-2 rounded-lg bg-slate-950/80 border border-slate-900">
          <span className="text-slate-500 block text-[9px]">PROCESS RSS</span>
          <span className="text-emerald-400 font-bold">{memoryMb !== null ? `${memoryMb} MB` : "Unavailable"}</span>
        </div>
      </div>

      {lastError && (
        <div className="p-2 rounded-lg bg-red-950/40 border border-red-500/40 text-red-300 text-[10px]">
          {lastError}
        </div>
      )}
    </div>
  )
}
