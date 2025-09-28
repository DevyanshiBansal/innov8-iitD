"use client"

import { fileSystemManager } from "@/lib/file-system"

interface ToolbarProps {
  language: string
  setLanguage: (lang: string) => void
  theme: string
  setTheme: (theme: string) => void
  problemId: string
  setProblemId: (id: string) => void
  syncBackend: boolean
  setSyncBackend: (sync: boolean) => void
  backendUrl: string
  setBackendUrl: (url: string) => void
  isMaximized: boolean
  setIsMaximized: (max: boolean) => void
  status: string
  onRun: () => void
  onClear: () => void
  onReset: () => void
  onStatusChange: (status: string) => void
}

export function Toolbar({
  language,
  setLanguage,
  theme,
  setTheme,
  problemId,
  setProblemId,
  syncBackend,
  setSyncBackend,
  backendUrl,
  setBackendUrl,
  isMaximized,
  setIsMaximized,
  status,
  onRun,
  onClear,
  onReset,
  onStatusChange,
}: ToolbarProps) {
  const handlePickFolder = async () => {
    const success = await fileSystemManager.chooseDirectory()
    if (success) {
      onStatusChange("Folder selected")
    } else {
      onStatusChange("Folder selection canceled")
    }
  }

  return (
    <div className="mini-ide-toolbar">
      <label>
        <span className="mini-ide-kbd">Lang</span>
        <select value={language} onChange={(e) => setLanguage(e.target.value)}>
          <option value="javascript">JavaScript</option>
          <option value="python">Python</option>
          <option value="java">Java</option>
          <option value="cpp">C++</option>
          <option value="go">Go</option>
        </select>
      </label>

      <label>
        <span className="mini-ide-kbd">Theme</span>
        <select value={theme} onChange={(e) => setTheme(e.target.value)}>
          <option value="vs-dark">VS Dark</option>
          <option value="vs">VS Light</option>
          <option value="hc-black">High Contrast</option>
        </select>
      </label>

      <label>
        <span className="mini-ide-kbd">Problem</span>
        <select value={problemId} onChange={(e) => setProblemId(e.target.value)}>
          <option value="two-sum">Two Sum</option>
          <option value="reverse-string">Reverse String</option>
          <option value="fibonacci">Fibonacci</option>
        </select>
      </label>

      <label className="flex items-center gap-2">
        <span className="mini-ide-kbd">Problem</span>
        <input
          type="text"
          value={problemId}
          onChange={(e) => setProblemId(e.target.value)}
          placeholder="default"
          className="w-36"
        />
      </label>

      <button onClick={onRun} className="primary btn" title="Ctrl+Enter">
        Run
      </button>

      <button onClick={handlePickFolder} className="btn" title="Choose folder to write output files">
        Choose Output Folder
      </button>

      <label className="flex items-center gap-2">
        <input type="checkbox" checked={syncBackend} onChange={(e) => setSyncBackend(e.target.checked)} />
        <span>Backend sync</span>
      </label>

      <input
        type="text"
        value={backendUrl}
        onChange={(e) => setBackendUrl(e.target.value)}
        placeholder="http://localhost:3001"
        className="w-56"
      />

      <button onClick={() => setIsMaximized(!isMaximized)} className="btn" title="Toggle bigger editor">
        {isMaximized ? "Restore Layout" : "Maximize Editor"}
      </button>

      <button onClick={onReset} className="btn" title="Reset to starter template">
        Reset
      </button>

      <button onClick={onClear} className="warn" title="Clear saved data">
        Clear Saved
      </button>

      <span className="mini-ide-status">{status}</span>
    </div>
  )
}
