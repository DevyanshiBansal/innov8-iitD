"use client"

import { useEffect, useRef } from "react"
import { fileSystemManager } from "@/lib/file-system"
import { BackendSync } from "@/lib/backend-sync"

interface AutosaveOptions {
  code: string
  language: string
  problemId: string
  sessionId: string
  syncBackend: boolean
  backendUrl: string
  onStatusChange: (status: string) => void
}

export function useAutosave({
  code,
  language,
  problemId,
  sessionId,
  syncBackend,
  backendUrl,
  onStatusChange,
}: AutosaveOptions) {
  const intervalRef = useRef<NodeJS.Timeout>()
  const backendSyncRef = useRef<BackendSync>()

  // Update backend sync instance when URL changes
  useEffect(() => {
    if (backendUrl) {
      backendSyncRef.current = new BackendSync(backendUrl)
    }
  }, [backendUrl])

  useEffect(() => {
    // Clear existing interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
    }

    // Start autosave loop
    intervalRef.current = setInterval(async () => {
      const snapshotData = {
        type: "snapshot",
        sessionId,
        lang: language,
        timestamp: new Date().toISOString(),
        problemId,
        code,
      }

      const json = JSON.stringify(snapshotData, null, 2)

      // Save to localStorage for resilience
      localStorage.setItem("miniIDE:snapshot", json)

      // Try to write to disk if folder chosen
      const fileWritten = await fileSystemManager.writeFile("snapshot.json", json + "\n")

      // Optional backend sync
      if (syncBackend && backendSyncRef.current) {
        try {
          await backendSyncRef.current.postSnapshot({
            sessionId,
            problemId,
            language,
            code,
            timestamp: snapshotData.timestamp,
          })
        } catch {
          // Silently fail for autosave
        }
      }

      onStatusChange(fileWritten ? "Autosaved to folder" : "Autosaved")
    }, 3000) // Every 3 seconds

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [code, language, problemId, sessionId, syncBackend, onStatusChange])

  // Manual snapshot save
  const saveSnapshot = async () => {
    const snapshotData = {
      type: "snapshot",
      sessionId,
      lang: language,
      timestamp: new Date().toISOString(),
      problemId,
      code,
    }

    const json = JSON.stringify(snapshotData, null, 2)
    localStorage.setItem("miniIDE:snapshot", json)

    const fileWritten = await fileSystemManager.writeFile("snapshot.json", json + "\n")

    if (syncBackend && backendSyncRef.current) {
      try {
        await backendSyncRef.current.postSnapshot({
          sessionId,
          problemId,
          language,
          code,
          timestamp: snapshotData.timestamp,
        })
      } catch {
        // Ignore errors for manual save
      }
    }

    onStatusChange(fileWritten ? "Snapshot saved to folder" : "Snapshot saved")
  }

  return { saveSnapshot }
}
