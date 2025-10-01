"use client"

import { useState, useEffect, useRef } from "react"
import { MonacoEditor } from "./monaco-editor"
import { AdaptiveProblemPanel } from "./adaptive-problem-panel"
import { AdaptiveProblemPanel } from "./adaptive-problem-panel"
import { ChatPanel } from "./chat-panel"
import { Toolbar } from "./toolbar"
import { ResizableLayout } from "./resizable-layout"
import { getProblemTemplate } from "@/lib/problems"
import { useAutosave } from "@/hooks/use-autosave"
import { fileSystemManager } from "@/lib/file-system"
import { BackendSync } from "@/lib/backend-sync"

export function MiniIDE() {
  const [language, setLanguage] = useState("javascript")
  const [theme, setTheme] = useState("vs-dark")
  const [code, setCode] = useState("")
  const [stdin, setStdin] = useState("")
  const [status, setStatus] = useState("Idle")
  const [problemId, setProblemId] = useState("two-sum")
  const [isMaximized, setIsMaximized] = useState(false)
  const [syncBackend, setSyncBackend] = useState(true) // Enable by default
  const [backendUrl, setBackendUrl] = useState("http://localhost:8000") // Correct FastAPI port
  const [consoleOutput, setConsoleOutput] = useState("Ready.")
  const [analysisData, setAnalysisData] = useState<any>(null) // Store analysis for chat panel

  const sessionId = useRef(crypto.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`)

  const { saveCode } = useAutosave({
    code,
    language,
    problemId,
    sessionId: sessionId.current,
    onStatusChange: setStatus,
  })

  useEffect(() => {
    // Load saved settings
    const savedTheme = localStorage.getItem("miniIDE:theme") || "vs-dark"
    const savedProblemId = localStorage.getItem("miniIDE:problemId") || "two-sum"
    const savedSyncBackend = localStorage.getItem("miniIDE:syncBackend") !== "false" // Default true
    const savedBackendUrl = localStorage.getItem("miniIDE:backendUrl") || "http://localhost:8000"

    setTheme(savedTheme)
    setProblemId(savedProblemId)
    setSyncBackend(savedSyncBackend)
    setBackendUrl(savedBackendUrl)
  }, [])

  useEffect(() => {
    // Save settings
    localStorage.setItem("miniIDE:theme", theme)
    localStorage.setItem("miniIDE:problemId", problemId)
    localStorage.setItem("miniIDE:syncBackend", String(syncBackend))
    localStorage.setItem("miniIDE:backendUrl", backendUrl)
  }, [theme, problemId, syncBackend, backendUrl])

  // Load stdin for current problem
  useEffect(() => {
    const savedStdin = localStorage.getItem(`miniIDE:stdin:${problemId}`) || ""
    setStdin(savedStdin)
  }, [problemId])

  // Save stdin for current problem
  useEffect(() => {
    localStorage.setItem(`miniIDE:stdin:${problemId}`, stdin)
  }, [stdin, problemId])

  const handleRun = async () => {
    setStatus("Running code...")
    setConsoleOutput("Executing code...")
    setAnalysisData(null) // Clear previous analysis

    const runData = {
      type: "run",
      sessionId: sessionId.current,
      lang: language,
      timestamp: new Date().toISOString(),
      problemId,
      stdin: stdin || "",
      code,
    }

    const json = JSON.stringify(runData, null, 2)
    localStorage.setItem("miniIDE:run", json)

    const fileWritten = await fileSystemManager.writeFile("run.json", json + "\n")

    // Map language names to backend format
    const languageMap: Record<string, string> = {
      "javascript": "python", // Fallback to python if JS not supported
      "python": "python",
      "java": "python", // Fallback to python if Java not supported  
      "cpp": "cpp",
      "c": "c",
      "go": "python" // Fallback to python if Go not supported
    }

    const backendLanguage = languageMap[language] || "python"

    // Always try backend execution
    if (backendUrl) {
      try {
        setStatus("Connecting to backend...")
        const backendSync = new BackendSync(backendUrl)
        
        // Check if backend is healthy first
        const isHealthy = await backendSync.healthCheck()
        if (!isHealthy) {
          throw new Error("Backend server is not responding")
        }

        setStatus("Executing on backend...")
        const result = await backendSync.executeCode({
          code: code,
          language: backendLanguage,
          test_sizes: [10, 100, 500, 1000, 10000]
        })

        if (result.success && result.frontend_display) {
          setStatus("Execution completed successfully")
          
          // Display simple console format as requested
          const consoleData = result.frontend_display
          let output = "--- Execution Summary ---\n"
          
          output += `Language: ${consoleData.execution_summary.language}\n`
          output += `Execution Time: ${consoleData.execution_summary.execution_time}s\n`
          output += `Peak Memory: ${consoleData.execution_summary.peak_memory}\n`
          
          // Add the exact output from running the code
          if (consoleData.console_output.trim()) {
            output += `Output: ${consoleData.console_output.trim()}\n`
          } else {
            output += `Output: (no output)\n`
          }
          
          output += `Success: ${consoleData.execution_summary.exit_code === 0 ? '1' : '0'}`

          setConsoleOutput(output)
          
          // Store analysis data for chat panel
          setAnalysisData(result.frontend_display)
          
        } else {
          setStatus("Execution failed")
          let output = "--- Execution Summary ---\n"
          output += `Language: ${backendLanguage}\n`
          output += `Execution Time: 0s\n`
          output += `Peak Memory: 0 B\n`
          output += `Output: ${result.message || "Execution failed"}\n`
          output += `Success: 0`
          setConsoleOutput(output)
          setAnalysisData(null)
        }
      } catch (error) {
        console.warn("Backend run failed", error)
        setStatus("Backend connection failed")
        let output = "--- Execution Summary ---\n"
        output += `Language: ${backendLanguage}\n`
        output += `Execution Time: 0s\n`
        output += `Peak Memory: 0 B\n`
        output += `Output: Backend connection failed - ${(error as Error).message}\nMake sure backend is running: python server.py at http://localhost:8000\n`
        output += `Success: 0`
        setConsoleOutput(output)
        setAnalysisData(null)
      }
    } else {
      setStatus("No backend configured")
      let output = "--- Execution Summary ---\n"
      output += `Language: ${backendLanguage}\n`
      output += `Execution Time: 0s\n`
      output += `Peak Memory: 0 B\n`
      output += `Output: Backend URL not configured\n`
      output += `Success: 0`
      setConsoleOutput(output)
    }
  }

  // Handle AI analysis request from chat panel
  const handleAIAnalysis = async (code: string, language: string, analysisData: any) => {
    try {
      if (!backendUrl) {
        throw new Error("Backend URL not configured")
      }

      console.log("🤖 Requesting AI analysis...", { language, hasCode: !!code, hasAnalysisData: !!analysisData })

      const backendSync = new BackendSync(backendUrl)
      
      const result = await backendSync.requestAIAnalysis({
        code: code,
        language: language,
        execution_result: analysisData?.execution_result || null,
        time_analysis: analysisData?.time_analysis || null,
        space_analysis: analysisData?.space_analysis || null
      })

      console.log("✅ AI analysis completed:", result)

      // The AI analysis result needs to be added to chat messages
      if (result.success && result.data) {
        // Update chat panel with AI analysis result
        const chatKey = `miniIDE:chat:${problemId}`
        const existingMessages = JSON.parse(localStorage.getItem(chatKey) || '[]')
        
        const aiMessage = {
          role: "ai-analysis",
          text: result.data.analysis,
          timestamp: Date.now()
        }

        const updatedMessages = [...existingMessages, aiMessage]
        localStorage.setItem(chatKey, JSON.stringify(updatedMessages))
        
        // Force chat panel to refresh by updating analysisData
        setAnalysisData((prev: any) => ({ ...prev, aiAnalysisResult: result.data }))
      } else {
        throw new Error(result.message || "AI analysis failed")
      }
      
    } catch (error) {
      console.error("❌ AI analysis failed:", error)
      
      // Add error message to chat
      const chatKey = `miniIDE:chat:${problemId}`
      const existingMessages = JSON.parse(localStorage.getItem(chatKey) || '[]')
      
      const errorMessage = {
        role: "assistant",
        text: `I couldn't analyze your code right now. ${error instanceof Error ? error.message : 'Please ensure the RAG system is properly configured with API keys and try again.'}`,
        timestamp: Date.now()
      }

      const updatedMessages = [...existingMessages, errorMessage]
      localStorage.setItem(chatKey, JSON.stringify(updatedMessages))
      
      // Force chat panel to refresh
      setAnalysisData((prev: any) => ({ ...prev, aiAnalysisError: error instanceof Error ? error.message : String(error) }))
      
      throw error
    }
  }

  const handleClear = () => {
    if (!confirm("Clear local saved code and run data?")) return

    // Clear stored data
    const problems = ["two-sum", "reverse-string", "fibonacci"]
    const languages = ["javascript", "python", "java", "cpp", "go"]

    for (const p of problems) {
      for (const l of languages) {
        localStorage.removeItem(`miniIDE:code:${p}:${l}`)
        localStorage.removeItem(`miniIDE:stdin:${p}`)
        localStorage.removeItem(`miniIDE:chat:${p}`) // Also clear chat/analysis data
      }
    }

    localStorage.removeItem("miniIDE:autosave")
    localStorage.removeItem("miniIDE:run")
    setStdin("")
    setStatus("Cleared")
    setConsoleOutput("Ready.")
    setAnalysisData(null) // Clear analysis data
  }

  const clearConsole = () => {
    setConsoleOutput("Ready.")
    setAnalysisData(null)
  }

  const handleReset = () => {
    const template = getProblemTemplate(problemId, language)
    setCode(template)
    setStatus("Reset to template")
  }

  useEffect(() => {
    const handleKeyDown = async (e: KeyboardEvent) => {
      // Ctrl+Enter or Cmd+Enter to run
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault()
        await handleRun()
      }
      // Ctrl+S or Cmd+S to save code
      if ((e.ctrlKey || e.metaKey) && (e.key === "s" || e.key === "S")) {
        e.preventDefault()
        await saveCode()
      }
      // Escape to toggle maximize editor
      if (e.key === "Escape") {
        setIsMaximized((prev) => !prev)
      }
    }

    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [saveCode])

  const handleProblemChange = (newProblemId: string) => {
    setProblemId(newProblemId)
    // Load saved code for new problem or use template
    const saveKey = `miniIDE:code:${newProblemId}:${language}`
    const savedCode = localStorage.getItem(saveKey)
    const codeToLoad = savedCode || getProblemTemplate(newProblemId, language)
    setCode(codeToLoad)
  }

  const handleLanguageChange = (newLanguage: string) => {
    setLanguage(newLanguage)
    // Load saved code for current problem in new language or use template
    const saveKey = `miniIDE:code:${problemId}:${newLanguage}`
    const savedCode = localStorage.getItem(saveKey)
    const codeToLoad = savedCode || getProblemTemplate(problemId, newLanguage)
    setCode(codeToLoad)
  }

  return (
    <div className={`mini-ide ${isMaximized ? "max-editor" : ""}`}>
      <header className="mini-ide-header">
        <h1>Mini IDE</h1>
        <Toolbar
          language={language}
          setLanguage={handleLanguageChange}
          theme={theme}
          setTheme={setTheme}
          problemId={problemId}
          setProblemId={handleProblemChange}
          syncBackend={syncBackend}
          setSyncBackend={setSyncBackend}
          backendUrl={backendUrl}
          setBackendUrl={setBackendUrl}
          isMaximized={isMaximized}
          setIsMaximized={setIsMaximized}
          status={status}
          onRun={handleRun}
          onClear={handleClear}
          onReset={handleReset}
          onStatusChange={setStatus}
        />
      </header>

      <ResizableLayout isMaximized={isMaximized}>
        <ProblemPanel problemId={problemId} />

        <div className="mini-ide-gutter" />

        <section className="flex flex-col min-h-0 h-full editor-wrap">
          <div className="mini-ide-panel flex-1 flex flex-col min-h-0">
            <div className="mini-ide-row justify-between items-baseline mb-1.5">
              <h3>Editor</h3>
              <span className="mini-ide-hint">Autosaves code to localStorage every 3s</span>
            </div>
            <div className="flex-1 min-h-0">
              <MonacoEditor language={language} theme={theme} value={code} onChange={setCode} problemId={problemId} />
            </div>
            <div className="mt-2 flex-shrink-0">
              <div className="mini-ide-row">
                <span className="mini-ide-kbd">stdin</span>
                <span className="mini-ide-hint">Optional input for run request</span>
              </div>
              <div className="mini-ide-row flex-1 mt-1">
                <textarea
                  value={stdin}
                  onChange={(e) => setStdin(e.target.value)}
                  placeholder="Sample input lines..."
                  className="mini-ide-textarea w-full"
                />
              </div>
            </div>
          </div>

          <div className="mini-ide-panel mt-2 flex-shrink-0">
            <div className="mini-ide-row justify-between items-center mb-2">
              <h3>Console</h3>
              <div className="flex items-center gap-4">
                <button 
                  onClick={clearConsole}
                  className="text-xs text-muted-foreground hover:text-foreground cursor-pointer"
                  title="Clear console"
                >
                  Clear
                </button>
                <span className="text-xs text-muted-foreground">
                  Ctrl+Enter to Run • Ctrl+S to Save • Esc to Maximize
                </span>
              </div>
            </div>
            <pre className="mini-ide-console">{consoleOutput}</pre>
          </div>
        </section>

        <div className="mini-ide-gutter" />

        <ChatPanel 
          problemId={problemId} 
          analysisData={analysisData}
          code={code}
          language={language}
          onAIAnalysisRequest={handleAIAnalysis}
        />
      </ResizableLayout>
    </div>
  )
}
