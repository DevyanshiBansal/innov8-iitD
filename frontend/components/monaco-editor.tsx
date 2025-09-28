"use client"

import { useEffect, useRef, useState } from "react"
import { getProblemTemplate } from "@/lib/problems"

declare global {
  interface Window {
    monaco: any
    require: any
    MonacoEnvironment: any
  }
}

interface MonacoEditorProps {
  language: string
  theme: string
  value: string
  onChange: (value: string) => void
  problemId: string
}

export function MonacoEditor({ language, theme, value, onChange, problemId }: MonacoEditorProps) {
  const editorRef = useRef<HTMLDivElement>(null)
  const editorInstanceRef = useRef<any>(null)
  const [isLoaded, setIsLoaded] = useState(false)

  // Load Monaco Editor
  useEffect(() => {
    if (typeof window === "undefined") return

    const loadMonaco = async () => {
      // Skip if already loaded
      if (window.monaco) {
        setIsLoaded(true)
        return
      }

      // Load Monaco loader
      const script = document.createElement("script")
      script.src = "https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.44.0/min/vs/loader.min.js"
      script.crossOrigin = "anonymous"
      script.referrerPolicy = "no-referrer"

      script.onload = () => {
        // Configure Monaco paths and workers
        window.require.config({
          paths: { vs: "https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.44.0/min/vs" },
        })

        window.MonacoEnvironment = {
          getWorkerUrl: (moduleId: string, label: string) => {
            const base = "https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.44.0/min/"
            const src = `self.MonacoEnvironment={baseUrl:'${base}'};importScripts('${base}vs/base/worker/workerMain.js');`
            return "data:text/javascript;charset=utf-8," + encodeURIComponent(src)
          },
        }

        window.require(["vs/editor/editor.main"], () => {
          setIsLoaded(true)
        })
      }

      document.head.appendChild(script)
    }

    loadMonaco()
  }, [])

  // Initialize editor
  useEffect(() => {
    if (!isLoaded || !editorRef.current || editorInstanceRef.current) return

    const initialCode = value || getProblemTemplate(problemId, language)

    editorInstanceRef.current = window.monaco.editor.create(editorRef.current, {
      value: initialCode,
      language: langToMonaco(language),
      theme: theme,
      automaticLayout: true,
      fontSize: 16,
      minimap: { enabled: false },
      wordWrap: "on",
      scrollBeyondLastLine: false,
    })

    // Handle content changes
    editorInstanceRef.current.onDidChangeModelContent(() => {
      const currentValue = editorInstanceRef.current.getValue()
      onChange(currentValue)

      // Save to localStorage with debouncing
      const saveKey = `miniIDE:code:${problemId}:${language}`
      clearTimeout((window as any).saveTimeout)
      ;(window as any).saveTimeout = setTimeout(() => {
        localStorage.setItem(saveKey, currentValue)
      }, 250)
    })

    return () => {
      if (editorInstanceRef.current) {
        editorInstanceRef.current.dispose()
        editorInstanceRef.current = null
      }
    }
  }, [isLoaded, language, theme, problemId])

  // Update language
  useEffect(() => {
    if (!editorInstanceRef.current) return

    const model = editorInstanceRef.current.getModel()
    if (model) {
      window.monaco.editor.setModelLanguage(model, langToMonaco(language))
    }
  }, [language])

  // Update theme
  useEffect(() => {
    if (!editorInstanceRef.current) return
    window.monaco.editor.setTheme(theme)
  }, [theme])

  // Load saved code or template when problem/language changes
  useEffect(() => {
    if (!editorInstanceRef.current) return

    const saveKey = `miniIDE:code:${problemId}:${language}`
    const savedCode = localStorage.getItem(saveKey)
    const codeToLoad = savedCode || getProblemTemplate(problemId, language)

    if (editorInstanceRef.current.getValue() !== codeToLoad) {
      editorInstanceRef.current.setValue(codeToLoad)
    }
  }, [problemId, language])

  return (
    <div className="mini-ide-editor flex-1" ref={editorRef}>
      {!isLoaded && (
        <div className="flex items-center justify-center h-full text-muted-foreground">Loading Monaco Editor...</div>
      )}
    </div>
  )
}

function langToMonaco(lang: string): string {
  if (lang === "cpp") return "cpp"
  return lang // javascript, python, java, go are supported
}
