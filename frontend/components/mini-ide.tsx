"use client"

import { useState, useEffect, useRef } from "react"
import { MonacoEditor } from "./monaco-editor"
import { AdaptiveProblemPanel } from "./adaptive-problem-panel"
import { ChatPanel } from "./chat-panel"
import { Toolbar } from "./toolbar"
import { ResizableLayout } from "./resizable-layout"
import { useAutosave } from "@/hooks/use-autosave"
import { fileSystemManager } from "@/lib/file-system"
import { BackendSync } from "@/lib/backend-sync"

// Assuming these utility functions/constants exist elsewhere or are a necessary part of the IDE.
// Since they are not defined, I am commenting out the problematic parts that rely on them
// and defining placeholders for the core functions to make the component runnable.

// Placeholder/Stub for undefined functions/variables found in the fragments
// In a real application, these would be imported or defined.
const getProblemTemplate = (id, lang) => `// Code template for ${id} in ${lang}`
const problemId = "adaptive" // Assuming a fixed problem ID for this 'MiniIDE'

export function MiniIDE() {
  const [language, setLanguage] = useState("javascript")
  const [theme, setTheme] = useState("vs-dark")
  const [code, setCode] = useState("")
  const [status, setStatus] = useState("Idle")
  const [isMaximized, setIsMaximized] = useState(false)
  const [syncBackend, setSyncBackend] = useState(true)
  const [backendUrl, setBackendUrl] = useState("http://localhost:8000")
  const [consoleOutput, setConsoleOutput] = useState("Ready.")
  const [analysisData, setAnalysisData] = useState(null)
  const [stdin, setStdin] = useState("") // Added stdin state based on later fragments
  // Note: problemId and setProblemId are removed as the first fragment didn't use them,
  // but I'll use the 'adaptive' string in places that require it.

  const sessionId = useRef(crypto.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`)

  const { saveCode } = useAutosave({
    code,
    language,
    sessionId: sessionId.current,
    onStatusChange: setStatus,
    // problemId: problemId, // removed due to initial fragment's scope
  })

  // --- Utility Functions ---

  const handleLanguageChange = (newLanguage: string) => {
    setLanguage(newLanguage)
    setStatus(`Language: ${newLanguage}`)
    
    // Logic from the later fragment to load/set code for the new language
    const saveKey = `miniIDE:code:${problemId}:${newLanguage}`
    const savedCode = localStorage.getItem(saveKey)
    const codeToLoad = savedCode || getProblemTemplate(problemId, newLanguage)
    setCode(codeToLoad)
  }

  const handleClear = () => {
    if (!confirm("Clear local saved code and run data?")) return
    
    // Only clear general items from the initial fragment, simplified
    const keysToRemove = [
      "miniIDE:run",
      "miniIDE:autosave", 
      "miniIDE:saved"
      // More specific problem/lang keys from the second fragment are omitted for simplicity
    ]
    
    keysToRemove.forEach(key => localStorage.removeItem(key))
    
    setCode("")
    setConsoleOutput("Ready.")
    setAnalysisData(null)
    setStatus("Cleared")
    setStdin("") // Clear stdin
  }
  
  const handleReset = () => {
    // Use the template reset logic from the later fragment
    const template = getProblemTemplate(problemId, language)
    setCode(template)
    setStatus("Reset to template")
    setConsoleOutput("Ready.")
    setAnalysisData(null)
  }
  
  const clearConsole = () => {
    setConsoleOutput("Ready.")
    setAnalysisData(null)
  }

  // --- Main Execution Logic (Merged from both fragments) ---

  const handleRun = async () => {
    setStatus("Running code...")
    setConsoleOutput("Executing code...")
    setAnalysisData(null) // Clear previous analysis

    const runData = {
      type: "run",
      sessionId: sessionId.current,
      lang: language,
      timestamp: new Date().toISOString(),
      code,
    }

    const json = JSON.stringify(runData, null, 2)
    localStorage.setItem("miniIDE:run", json)

    await fileSystemManager.writeFile("run.json", json + "\n")

    // Map language names to backend format (from second fragment)
    const languageMap: Record<string, string> = {
      "javascript": "python",
      "python": "python",
      "java": "python",
      "cpp": "cpp",
      "c": "c",
      "go": "python"
    }
    const backendLanguage = languageMap[language] || "python"

    // Always try backend execution
    if (backendUrl) {
      try {
        setStatus("Connecting to backend...")
        const backendSync = new BackendSync(backendUrl)
        
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
          
          // Display simple console format as requested (from second fragment)
          const consoleData = result.frontend_display
          let output = "--- Execution Summary ---\n"
          
          output += `Language: ${consoleData.execution_summary.language}\n`
          output += `Execution Time: ${consoleData.execution_summary.execution_time}s\n`
          output += `Peak Memory: ${consoleData.execution_summary.peak_memory}\n`
          
          if (consoleData.console_output.trim()) {
            output += `Output: ${consoleData.console_output.trim()}\n`
          } else {
            output += `Output: (no output)\n`
          }
          
          output += `Success: ${consoleData.execution_summary.exit_code === 0 ? '1' : '0'}`

          setConsoleOutput(output)
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
        output += `Output: Backend connection failed - ${(error as Error).message}\n`
        output += `Success: 0`
        setConsoleOutput(output)
        setAnalysisData(null)
      }
    } else {
      setStatus("No backend configured")
      let output = "--- Execution Summary --