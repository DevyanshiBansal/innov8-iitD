"use client"

interface ToolbarProps {
  language: string
  setLanguage: (lang: string) => void
  theme: string
  setTheme: (theme: string) => void
  isMaximized: boolean
  setIsMaximized: (max: boolean) => void
  status: string
  onRun: () => void
  onSubmit: () => void
  onClear: () => void
  onReset: () => void
}

export function Toolbar({
  language,
  setLanguage,
  theme,
  setTheme,
  isMaximized,
  setIsMaximized,
  status,
  onRun,
  onSubmit,
  onClear,
  onReset,
}: ToolbarProps) {
  return (
    <div className="mini-ide-toolbar">
      <label>
        <span className="mini-ide-kbd">Lang</span>
        <select value={language} onChange={(e) => setLanguage(e.target.value)}>
          <option value="javascript">JavaScript</option>
          <option value="python">Python</option>
          <option value="java">Java</option>
          <option value="cpp">C++</option>
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

      <button onClick={onRun} className="run" title="Execute code (Ctrl+Enter)">
        Run
      </button>

      <button onClick={onSubmit} className="submit" title="Submit solution for evaluation">
        Submit
      </button>

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
