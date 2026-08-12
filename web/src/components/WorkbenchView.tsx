import React, { useState } from "react"
import { Code2, Copy, Check, Terminal, Play, Sparkles } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"

interface WorkbenchViewProps {
  baseUrl: string
  model: string
  apiKey: string
  temperature: number
  maxTokens: number
}

type Lang = "curl" | "python" | "js" | "rust"

export function WorkbenchView({ baseUrl, model, apiKey, temperature, maxTokens }: WorkbenchViewProps) {
  const [lang, setLang] = useState<Lang>("python")
  const [prompt, setPrompt] = useState("Write a Python function to perform binary search on a sorted list.")
  const [systemPrompt, setSystemPrompt] = useState("You are an expert software engineer.")
  const [copied, setCopied] = useState(false)

  const cleanBase = baseUrl.replace(/\/+$/, "")

  const getCodeSnippet = () => {
    switch (lang) {
      case "curl":
        return `curl -X POST "${cleanBase}/chat/completions" \\
  -H "Content-Type: application/json" \\
  ${apiKey ? `-H "Authorization: Bearer ${apiKey}" \\` : ""}
  -d '{
    "model": "${model}",
    "messages": [
      {"role": "system", "content": "${systemPrompt.replace(/"/g, '\\"')}"},
      {"role": "user", "content": "${prompt.replace(/"/g, '\\"')}"}
    ],
    "temperature": ${temperature},
    "max_tokens": ${maxTokens}
  }'`

      case "python":
        return `from openai import OpenAI

client = OpenAI(
    base_url="${cleanBase}",
    api_key="${apiKey || "qwanto-key"}"
)

response = client.chat.completions.create(
    model="${model}",
    messages=[
        {"role": "system", "content": "${systemPrompt.replace(/"/g, '\\"')}"},
        {"role": "user", "content": "${prompt.replace(/"/g, '\\"')}"}
    ],
    temperature=${temperature},
    max_tokens=${maxTokens}
)

print(response.choices[0].message.content)`

      case "js":
        return `const response = await fetch("${cleanBase}/chat/completions", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    ${apiKey ? `"Authorization": "Bearer ${apiKey}"` : ""}
  },
  body: JSON.stringify({
    model: "${model}",
    messages: [
      { role: "system", content: "${systemPrompt.replace(/"/g, '\\"')}" },
      { role: "user", content: "${prompt.replace(/"/g, '\\"')}" }
    ],
    temperature: ${temperature},
    max_tokens: ${maxTokens}
  })
});

const data = await response.json();
console.log(data.choices[0].message.content);`

      case "rust":
        return `use reqwest::Client;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new();
    let res = client.post("${cleanBase}/chat/completions")
        .header("Content-Type", "application/json")
        .json(&json!({
            "model": "${model}",
            "messages": [
                {"role": "system", "content": "${systemPrompt.replace(/"/g, '\\"')}"},
                {"role": "user", "content": "${prompt.replace(/"/g, '\\"')}"}
            ],
            "temperature": ${temperature},
            "max_tokens": ${maxTokens}
        }))
        .send()
        .await?
        .text()
        .await?;

    println!("{}", res);
    Ok(())
}`
    }
  }

  const handleCopy = () => {
    navigator.clipboard.writeText(getCodeSnippet())
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <div className="flex items-center justify-between border-b border-border pb-4">
        <div>
          <h2 className="text-xl font-bold flex items-center gap-2">
            <Code2 className="size-5 text-primary" /> API Workbench & Code Generator
          </h2>
          <p className="text-xs text-muted-foreground mt-1">
            Generate production-ready code snippets in Python, cURL, TypeScript, and Rust for your Qwanto endpoint.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="space-y-4 md:col-span-1 border-r border-border/50 pr-4">
          <div>
            <label className="text-xs text-muted-foreground block mb-1">System Instructions</label>
            <Textarea
              rows={3}
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              className="text-xs font-mono"
            />
          </div>
          <div>
            <label className="text-xs text-muted-foreground block mb-1">User Prompt</label>
            <Textarea
              rows={5}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="text-xs font-mono"
            />
          </div>
        </div>

        <div className="md:col-span-2 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex gap-1.5 p-1 bg-secondary rounded-lg border border-border">
              {(["python", "curl", "js", "rust"] as Lang[]).map((l) => (
                <button
                  key={l}
                  onClick={() => setLang(l)}
                  className={`px-3 py-1 text-xs font-mono rounded-md font-semibold transition-all ${
                    lang === l ? "bg-primary text-primary-foreground shadow" : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {l.toUpperCase()}
                </button>
              ))}
            </div>
            <Button size="sm" variant="secondary" onClick={handleCopy} className="gap-1.5 text-xs">
              {copied ? <Check className="size-3.5 text-emerald-400" /> : <Copy className="size-3.5" />}
              {copied ? "Copied!" : "Copy Code"}
            </Button>
          </div>

          <pre className="p-4 bg-card border border-border rounded-xl text-xs font-mono text-emerald-400 overflow-x-auto min-h-[320px] leading-relaxed shadow-inner">
            {getCodeSnippet()}
          </pre>
        </div>
      </div>
    </div>
  )
}
