"use client"

import { useState, useEffect } from "react"
import { MonacoEditor } from "./monaco-editor"
import { AdaptiveProblemPanel } from "./adaptive-problem-panel"
import { ChatPanel } from "./chat-panel"
import { Toolbar } from "./toolbar"
import { ResizableLayout } from "./resizable-layout"

export function MiniIDE() {
  const [language, setLanguage] = useState("python")
  const [theme, setTheme] = useState("vs-dark")
  const [code, setCode] = useState("")
  const [status, setStatus] = useState("Ready")
  const [isMaximized, setIsMaximized] = useState(false)
  const [consoleOutput, setConsoleOutput] = useState("Ready to execute code.")
  const [analysisData, setAnalysisData] = useState<any>(null)

  useEffect(() => {
    // Load saved settings
    const savedTheme = localStorage.getItem("miniIDE:theme") || "vs-dark"
    const savedLanguage = localStorage.getItem("miniIDE:language") || "python"
    
    setTheme(savedTheme)
    setLanguage(savedLanguage)
  }, [])

  useEffect(() => {
    // Save settings
    localStorage.setItem("miniIDE:theme", theme)
    localStorage.setItem("miniIDE:language", language)
  }, [theme, language])

  const handleRun = async () => {
    if (!code.trim()) {
      setStatus("No code to execute")
      return
    }

    setStatus("Executing...")
    setConsoleOutput("Executing code...")

    try {
      const response = await fetch("http://localhost:8000/api/execute", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          code: code,
          language: language,
          stdin: ""
        })
      })

      if (response.ok) {
        const result = await response.json()
        
        if (result.success) {
          setConsoleOutput(result.output || "Execution completed successfully")
          setAnalysisData(result)
          setStatus("Execution completed")
        } else {
          setConsoleOutput(result.output || result.error_analysis || "Execution failed")
          setStatus("Execution failed")
        }
      } else {
        setConsoleOutput(`HTTP Error: ${response.status} ${response.statusText}`)
        setStatus("Connection failed")
      }
    } catch (error) {
      setConsoleOutput(`Connection error: ${error}`)
      setStatus("Backend unavailable")
    }
  }

  const handleSubmit = async () => {
    if (!code.trim()) {
      setStatus("No code to submit")
      return
    }

    setStatus("Submitting solution...")
    
    try {
      const response = await fetch("http://localhost:8000/api/submit", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          code: code,
          language: language,
          stdin: ""
        })
      })

      if (response.ok) {
        const result = await response.json()
        
        setConsoleOutput(`Submission Result:\nScore: ${result.performance_score.toFixed(2)}\nNext Level: ${result.next_difficulty}`)
        setAnalysisData(result.execution_result)
        setStatus("Solution submitted - Loading next question...")
        
        // Trigger automatic next question loading via window global
        if (typeof window !== 'undefined' && (window as any).handleAdaptiveSubmission) {
          (window as any).handleAdaptiveSubmission(result)
        }
        
        // Update status after delay
        setTimeout(() => {
          setStatus("Ready for next challenge")
        }, 3000)
      } else {
        setConsoleOutput(`Submission failed: ${response.status} ${response.statusText}`)
        setStatus("Submission failed")
      }
    } catch (error) {
      setConsoleOutput(`Submission error: ${error}`)
      setStatus("Backend unavailable")
    }
  }

  const handleClear = () => {
    localStorage.clear()
    setStatus("Saved data cleared")
  }

  const handleReset = () => {
    setCode("")
    setConsoleOutput("Ready to execute code.")
    setAnalysisData(null)
    setStatus("Reset complete")
  }

  const clearConsole = () => {
    setConsoleOutput("")
  }

  return (
    <div className={`mini-ide ${isMaximized ? "max-editor" : ""}`}>
      <header className="mini-ide-header">
        <h1>Innov8 Adaptive Coding Interview</h1>
        <Toolbar
          language={language}
          setLanguage={setLanguage}
          theme={theme}
          setTheme={setTheme}
          isMaximized={isMaximized}
          setIsMaximized={setIsMaximized}
          status={status}
          onRun={handleRun}
          onSubmit={handleSubmit}
          onClear={handleClear}
          onReset={handleReset}
        />
      </header>

      <ResizableLayout isMaximized={isMaximized}>
        <AdaptiveProblemPanel />

        <div className="mini-ide-gutter" />

        <section className="flex flex-col min-h-0 h-full editor-wrap">
          <div className="mini-ide-panel flex-1 flex flex-col min-h-0">
            <div className="mini-ide-row justify-between items-baseline mb-1.5">
              <h3>Code Editor</h3>
              <span className="mini-ide-hint">Auto-saves locally every few seconds</span>
            </div>
            <div className="flex-1 min-h-0">
              <MonacoEditor 
                language={language} 
                theme={theme} 
                value={code} 
                onChange={setCode}
                problemId="adaptive"
              />
            </div>
          </div>

          <div className="mini-ide-panel mt-2 flex-shrink-0">
            <div className="mini-ide-row justify-between items-center mb-2">
              <h3>Console Output</h3>
              <div className="flex items-center gap-4">
                <button 
                  onClick={clearConsole}
                  className="text-xs text-muted-foreground hover:text-foreground cursor-pointer"
                  title="Clear console"
                >
                  Clear
                </button>
                <span className="text-xs text-muted-foreground">
                  Ctrl+Enter to Run • Submit for evaluation
                </span>
              </div>
            </div>
            <pre className="mini-ide-console">{consoleOutput}</pre>
          </div>
        </section>

        <div className="mini-ide-gutter" />

        <ChatPanel 
          problemId="adaptive"
          analysisData={analysisData}
          code={code}
          language={language}
        />
      </ResizableLayout>
    </div>
  )
}