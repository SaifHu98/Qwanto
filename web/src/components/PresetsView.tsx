import React, { useState, useEffect } from "react"
import { Sparkles, Plus, Trash2, Check, SlidersHorizontal, MessageSquare, BookOpen, Code2, Zap, Feather } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { getPresets, savePreset, deletePreset, type SystemPreset } from "@/lib/api"

interface PresetsViewProps {
  baseUrl: string
  apiKey: string
  onApplyPreset: (preset: SystemPreset) => void
}

export function PresetsView({ baseUrl, apiKey, onApplyPreset }: PresetsViewProps) {
  const [presets, setPresets] = useState<SystemPreset[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")
  const [appliedId, setAppliedId] = useState<string | null>(null)

  // Form state for creating a new preset
  const [showCreate, setShowCreate] = useState(false)
  const [name, setName] = useState("")
  const [systemPrompt, setSystemPrompt] = useState("")
  const [temperature, setTemperature] = useState(0.7)
  const [topP, setTopP] = useState(0.9)
  const [description, setDescription] = useState("")
  const [saving, setSaving] = useState(false)

  const fetchPresets = async () => {
    setLoading(true)
    setError("")
    try {
      const data = await getPresets(baseUrl, apiKey)
      setPresets(data)
    } catch (err: any) {
      setError(err?.message || "Failed to load presets")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchPresets()
  }, [baseUrl, apiKey])

  const handleApply = (preset: SystemPreset) => {
    onApplyPreset(preset)
    setAppliedId(preset.id)
    setTimeout(() => setAppliedId(null), 2000)
  }

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!name.trim()) return
    setSaving(true)
    try {
      const updated = await savePreset(
        baseUrl,
        {
          name: name.trim(),
          system_prompt: systemPrompt.trim(),
          temperature,
          top_p: topP,
          description: description.trim() || "Custom user preset."
        },
        apiKey
      )
      setPresets(updated)
      setShowCreate(false)
      setName("")
      setSystemPrompt("")
      setDescription("")
    } catch (err: any) {
      setError(err?.message || "Failed to save preset")
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async (id: string) => {
    try {
      const updated = await deletePreset(baseUrl, id, apiKey)
      setPresets(updated)
    } catch (err: any) {
      setError(err?.message || "Failed to delete preset")
    }
  }

  const getPresetIcon = (id: string) => {
    switch (id) {
      case "code_expert": return <Code2 className="size-4 text-emerald-400" />
      case "researcher": return <BookOpen className="size-4 text-blue-400" />
      case "creative": return <Sparkles className="size-4 text-purple-400" />
      case "concise": return <Zap className="size-4 text-amber-400" />
      default: return <Feather className="size-4 text-teal-400" />
    }
  }

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <div className="flex items-center justify-between border-b border-border pb-4">
        <div>
          <h2 className="text-xl font-bold flex items-center gap-2">
            <Sparkles className="size-5 text-primary" /> Prompt Studio & Tuning Presets
          </h2>
          <p className="text-xs text-muted-foreground mt-1">
            Pre-configured generation parameters and specialized system instructions for your models.
          </p>
        </div>
        <Button onClick={() => setShowCreate(!showCreate)} size="sm">
          <Plus className="size-4 mr-1.5" /> New Custom Preset
        </Button>
      </div>

      {error && (
        <div className="p-3 bg-red-950/40 border border-red-800/50 rounded-lg text-xs text-red-300">
          {error}
        </div>
      )}

      {showCreate && (
        <form onSubmit={handleSave} className="p-5 border border-border bg-card rounded-xl space-y-4 shadow-xl">
          <h3 className="text-sm font-semibold flex items-center gap-2">
            <SlidersHorizontal className="size-4 text-primary" /> Create Preset
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-xs text-muted-foreground block mb-1">Preset Name</label>
              <Input
                placeholder="e.g. Technical Reviewer"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
              />
            </div>
            <div>
              <label className="text-xs text-muted-foreground block mb-1">Description</label>
              <Input
                placeholder="Brief summary of preset behavior"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
              />
            </div>
          </div>
          <div>
            <label className="text-xs text-muted-foreground block mb-1">System Prompt</label>
            <Textarea
              placeholder="Instructions provided to the model prior to generation..."
              rows={3}
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
            />
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-xs text-muted-foreground block mb-1">
                Temperature: <code className="text-primary font-mono">{temperature}</code>
              </label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.05"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="w-full accent-primary"
              />
            </div>
            <div>
              <label className="text-xs text-muted-foreground block mb-1">
                Top-P: <code className="text-primary font-mono">{topP}</code>
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={topP}
                onChange={(e) => setTopP(parseFloat(e.target.value))}
                className="w-full accent-primary"
              />
            </div>
          </div>
          <div className="flex justify-end gap-2 pt-2">
            <Button type="button" variant="ghost" size="sm" onClick={() => setShowCreate(false)}>
              Cancel
            </Button>
            <Button type="submit" size="sm" disabled={saving}>
              {saving ? "Saving..." : "Save Preset"}
            </Button>
          </div>
        </form>
      )}

      {loading ? (
        <div className="text-center py-12 text-muted-foreground text-sm">Loading presets...</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {presets.map((preset) => (
            <div
              key={preset.id}
              className="p-5 border border-border bg-card/60 hover:bg-card border-border/80 transition-all rounded-xl flex flex-col justify-between space-y-3"
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2 font-semibold text-sm">
                    {getPresetIcon(preset.id)}
                    <span>{preset.name}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge className="border border-border font-mono text-[10px]">
                      Temp: {preset.temperature}
                    </Badge>
                    <Badge className="border border-border font-mono text-[10px]">
                      Top-P: {preset.top_p}
                    </Badge>
                  </div>
                </div>
                <p className="text-xs text-muted-foreground mb-3">{preset.description}</p>

                {preset.system_prompt && (
                  <div className="p-2.5 bg-background/80 border border-border/50 rounded-lg text-xs font-mono text-muted-foreground/90 overflow-hidden line-clamp-3">
                    {preset.system_prompt}
                  </div>
                )}
              </div>

              <div className="flex items-center justify-between pt-2 border-t border-border/40">
                <Button
                  size="sm"
                  onClick={() => handleApply(preset)}
                  className="w-full text-xs gap-1.5"
                >
                  {appliedId === preset.id ? (
                    <>
                      <Check className="size-3.5 text-emerald-400" /> Applied to Active Session!
                    </>
                  ) : (
                    <>
                      <Sparkles className="size-3.5" /> Apply Preset
                    </>
                  )}
                </Button>

                {!["balanced", "code_expert", "researcher", "creative", "concise"].includes(preset.id) && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleDelete(preset.id)}
                    className="ml-2 text-red-400 hover:text-red-300 hover:bg-red-950/30"
                  >
                    <Trash2 className="size-3.5" />
                  </Button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
