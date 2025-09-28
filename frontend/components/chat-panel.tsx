"use client"

import type React from "react"

import { useState, useEffect, useRef } from "react"
import { BackendSync } from "@/lib/backend-sync"

// Helper function to format bytes
const formatBytes = (bytes: number): string => {
  if (bytes === 0) return "0 B"
  const k = 1024
  const sizes = ["B", "KB", "MB", "GB"]
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i]
}

interface ChatMessage {
  role: "user" | "assistant" | "ai-analysis" | "system"
  text: string
  timestamp?: number
}

interface ChatPanelProps {
  problemId: string
  analysisData?: {
    n_vs_time: Record<string, number>
    n_vs_space: Record<string, number>
    error_analysis: string
    execution_summary: {
      language: string
      file_created: string
      exit_code: number
      execution_time: number
      peak_memory: string
      success: boolean
      test_sizes_used: number[]
    }
  } | null
  code?: string
  language?: string
  onAIAnalysisRequest?: (code: string, language: string, analysisData: any) => void
}

export function ChatPanel({ problemId, analysisData, code, language, onAIAnalysisRequest }: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const [isLoadingAI, setIsLoadingAI] = useState(false)
  const [lastAnalyzedData, setLastAnalyzedData] = useState<string | null>(null)
  const messagesRef = useRef<HTMLDivElement>(null)

  // Load chat messages for current problem
  useEffect(() => {
    const loadMessages = () => {
      const chatKey = `miniIDE:chat:${problemId}`
      try {
        const saved = localStorage.getItem(chatKey)
        if (saved) {
          const loadedMessages = JSON.parse(saved)
          setMessages(loadedMessages)
          console.log(`📬 Loaded ${loadedMessages.length} messages for problem ${problemId}`)
        } else {
          // Add welcome message for new problems
          setMessages([{
            role: "system",
            text: "Welcome to the AI Code Assistant! I can help analyze your code for time complexity, space complexity, and potential errors. Execute your code to get started, or ask me questions about algorithms and data structures.",
            timestamp: Date.now()
          }])
        }
      } catch {
        setMessages([])
      }
    }

    loadMessages()

    // Listen for changes in localStorage to reload messages when AI analysis is added
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === `miniIDE:chat:${problemId}`) {
        loadMessages()
      }
    }

    window.addEventListener('storage', handleStorageChange)
    return () => window.removeEventListener('storage', handleStorageChange)
  }, [problemId])

  // Reload messages when AI analysis results are added
  useEffect(() => {
    if ((analysisData as any)?.aiAnalysisResult || (analysisData as any)?.aiAnalysisError) {
      console.log("🔄 AI analysis result detected, reloading messages...")
      const chatKey = `miniIDE:chat:${problemId}`
      try {
        const saved = localStorage.getItem(chatKey)
        if (saved) {
          const loadedMessages = JSON.parse(saved)
          setMessages(loadedMessages)
          console.log(`✅ Reloaded ${loadedMessages.length} messages after AI analysis`)
        }
      } catch (error) {
        console.error("Failed to reload messages:", error)
      }
    }
  }, [(analysisData as any)?.aiAnalysisResult, (analysisData as any)?.aiAnalysisError, problemId])

  // Clear analysis when switching problems
  useEffect(() => {
    setMessages(prev => prev.filter(msg => msg.role === "system")) // Keep system messages
    setLastAnalyzedData(null) // Reset analysis tracking
  }, [problemId])

  // Show execution summary when code runs (no auto AI analysis)
  useEffect(() => {
    if (analysisData && analysisData.execution_summary) {
      // Create a unique identifier for this execution data
      const currentDataHash = JSON.stringify({
        code: code?.slice(0, 100), // First 100 chars to identify code changes
        exitCode: analysisData.execution_summary.exit_code,
        timestamp: Math.floor(Date.now() / 10000) // Group by 10-second windows
      })

      // Only show message if this is new execution data
      if (currentDataHash !== lastAnalyzedData) {
        setLastAnalyzedData(currentDataHash)
        
        // Show a simple execution summary message
        const executionMessage: ChatMessage = {
          role: "system",
          text: `Ask me questions about your code, algorithms, or request an analysis!`,
          timestamp: Date.now()
        }
        
        setMessages(prev => {
          // Remove old system messages and add new one
          const nonSystemMessages = prev.filter(msg => msg.role !== "system")
          return [...nonSystemMessages, executionMessage]
        })
      }
    }
  }, [analysisData, code, lastAnalyzedData])

  // Save chat messages
  const saveMessages = (newMessages: ChatMessage[]) => {
    const chatKey = `miniIDE:chat:${problemId}`
    localStorage.setItem(chatKey, JSON.stringify(newMessages))
    setMessages(newMessages)
  }

  // Auto-scroll to bottom
  useEffect(() => {
    if (messagesRef.current) {
      messagesRef.current.scrollTop = messagesRef.current.scrollHeight
    }
  }, [messages])

  // Request AI analysis
  const requestAIAnalysis = async () => {
    if (!code || !language || !onAIAnalysisRequest) return

    setIsLoadingAI(true)
    try {
      await onAIAnalysisRequest(code, language, analysisData)
    } catch (error) {
      console.error("AI Analysis request failed:", error)
      const errorMessage = {
        role: "assistant" as const,
        text: "Sorry, I couldn't process your request. Please ensure the RAG system is properly configured with API keys.",
        timestamp: Date.now()
      }
      setMessages(prev => [...prev, errorMessage])
    }
    setIsLoadingAI(false)
  }

  const sendMessage = async () => {
    const text = input.trim()
    if (!text) return

    const userMessage: ChatMessage = {
      role: "user",
      text,
      timestamp: Date.now()
    }

    // Add user message immediately
    const messagesWithUser = [...messages, userMessage]
    setMessages(messagesWithUser)
    setInput("")

    // Send message to chatbot
    try {
      setIsLoadingAI(true)
      const backendSync = new BackendSync("http://localhost:8000") // Replace with your backend URL
      
      const chatResponse = await backendSync.sendChatMessage({
        message: text,
        code: code || undefined,
        language: language || undefined
      })
      
      if (chatResponse.success) {
        const responseMessage: ChatMessage = {
          role: "assistant",
          text: chatResponse.data.response,
          timestamp: Date.now()
        }
        const finalMessages = [...messagesWithUser, responseMessage]
        saveMessages(finalMessages)
      } else {
        const errorMessage: ChatMessage = {
          role: "assistant",
          text: "I'm having trouble responding right now. Please try again later.",
          timestamp: Date.now()
        }
        const finalMessages = [...messagesWithUser, errorMessage]
        saveMessages(finalMessages)
      }
    } catch (error) {
      console.error("Chat error:", error)
      const errorMessage: ChatMessage = {
        role: "assistant",
        text: "Sorry, I encountered an error. Please check your connection and try again.",
        timestamp: Date.now()
      }
      const finalMessages = [...messagesWithUser, errorMessage]
      saveMessages(finalMessages)
    } finally {
      setIsLoadingAI(false)
    }
  }

  const clearAnalysis = () => {
    setMessages([])
    localStorage.removeItem(`miniIDE:chat:${problemId}`)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <aside className="mini-ide-chat-wrap">
      <div className="mini-ide-panel h-full grid grid-rows-[auto_1fr_auto]">
        <div className="mini-ide-row justify-between items-baseline">
          <h3>AI Code Assistant</h3>
              <div className="flex items-center gap-2">
                {code && language && onAIAnalysisRequest && (
                  <button 
                    onClick={requestAIAnalysis}
                    disabled={isLoadingAI}
                    className="text-xs bg-blue-600 text-white px-2 py-1 rounded hover:bg-blue-700 disabled:opacity-50 cursor-pointer"
                    title="Get AI analysis of your code"
                  >
                    {isLoadingAI ? "Analyzing..." : "AI Analyze"}
                  </button>
                )}
                <button 
                  onClick={clearAnalysis}
                  className="text-xs text-muted-foreground hover:text-foreground cursor-pointer"
                  title="Clear chat"
                >
                  Clear
                </button>
                <span className="mini-ide-hint text-xs">AI-Powered Analysis</span>
              </div>
            </div>

        <div className="mini-ide-chat-list" ref={messagesRef}>
              {messages.length === 0 ? (
                <div className="mini-ide-chat-msg assistant">
                  <div className="mini-ide-chat-role">AI Assistant</div>
                  <div className="mini-ide-chat-bubble">
                    👋 Welcome to the AI Code Assistant!
                    <br /><br />
                    I can help you with:
                    <br />• Time & space complexity analysis
                    <br />• Code optimization suggestions
                    <br />• Error identification and fixes
                    <br />• Algorithm explanations
                    <br /><br />
                    Execute your code first, then ask me to analyze it!
                  </div>
                </div>
              ) : (
                messages.map((message, index) => (
                  <div key={index} className={`mini-ide-chat-msg ${message.role}`}>
                    <div className="mini-ide-chat-role">
                      {message.role === "user" ? "You" : 
                       message.role === "ai-analysis" ? "🤖 AI Analysis" :
                       message.role === "system" ? "💡 System" : "🔍 Analysis"}
                    </div>
                    <div 
                      className={`mini-ide-chat-bubble ${
                        message.role === "ai-analysis" ? "bg-blue-50 border-blue-200" :
                        message.role === "system" ? "bg-green-50 border-green-200" : ""
                      }`} 
                      style={{ whiteSpace: 'pre-line' }}
                    >
                      {message.text}
                    </div>
                    {message.timestamp && (
                      <div className="text-xs text-muted-foreground mt-1">
                        {new Date(message.timestamp).toLocaleTimeString()}
                      </div>
                    )}
                  </div>
                ))
              )}
              {isLoadingAI && (
                <div className="mini-ide-chat-msg assistant">
                  <div className="mini-ide-chat-role">🤖 AI Assistant</div>
                  <div className="mini-ide-chat-bubble">
                    <div className="flex items-center gap-2">
                      <div className="animate-spin rounded-full h-4 w-4 border-2 border-blue-500 border-t-transparent"></div>
                      Analyzing your code...
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="mini-ide-chat-input">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask me about algorithms, complexity, or request code analysis..."
                className="min-h-11 max-h-32"
                disabled={isLoadingAI}
              />
              <button 
                onClick={sendMessage} 
                disabled={!input.trim() || isLoadingAI}
                className="primary btn disabled:opacity-50"
              >
                {isLoadingAI ? "..." : "Send"}
              </button>
            </div>
      </div>
    </aside>
  )
}
