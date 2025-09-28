"use client"

import { useEffect, useRef } from "react"
import { fileSystemManager } from "@/lib/file-system"

interface AutosaveOptions {
  code: string
  language: string
  problemId: string
  sessionId: string
  onStatusChange: (status: string) => void
}

export function useAutosave({
  code,
  language,
  problemId,
  sessionId,
  onStatusChange,
}: AutosaveOptions) {
  const intervalRef = useRef<NodeJS.Timeout>()

  useEffect(() => {
    // Clear existing interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
    }

    // Start autosave loop - just save code to localStorage
    intervalRef.current = setInterval(async () => {
      const autoSaveData = {
        sessionId,
        language,
        timestamp: new Date().toISOString(),
        problemId,
        code,
      }

      const json = JSON.stringify(autoSaveData, null, 2)

      // Save to localStorage for resilience
      localStorage.setItem("miniIDE:autosave", json)

      onStatusChange("Autosaved")
    }, 3000) // Every 3 seconds

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [code, language, problemId, sessionId, onStatusChange])

  // Manual save
  const saveCode = async () => {
    const saveData = {
      sessionId,
      language,
      timestamp: new Date().toISOString(),
      problemId,
      code,
    }

    const json = JSON.stringify(saveData, null, 2)
    localStorage.setItem("miniIDE:saved", json)

    // Try to write to disk if folder chosen
    const fileWritten = await fileSystemManager.writeFile("saved_code.json", json + "\n")

    onStatusChange(fileWritten ? "Code saved to folder" : "Code saved")
  }

  return { saveCode }
}
