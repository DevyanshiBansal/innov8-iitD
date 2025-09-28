// Backend synchronization utilities for FastAPI backend

export interface RunData {
  sessionId: string
  problemId: string
  language: string
  code: string
  timestamp: string
  input: string
}

// Solution execution request
export interface SolutionExecutionRequest {
  code: string
  language: string
  test_input?: string
}

// AI Analysis request (legacy)
export interface AIAnalysisRequest {
  code: string
  language: string
  execution_result?: any
  time_analysis?: any
  space_analysis?: any
}

// Chat request
export interface ChatRequest {
  message: string
  code?: string
  language?: string
}

// Chat response
export interface ChatResponse {
  success: boolean
  message: string
  data: {
    response: string
    message_type: string
    has_code_context: boolean
    language?: string
  }
}

// Updated interface to match FastAPI backend response
export interface BackendResponse {
  success: boolean
  message: string
  data?: {
    execution_output: {
      output: string
      error: string
      exit_code: number
      execution_time: number
      peak_memory: number
      peak_memory_formatted: string
    }
    time_complexity?: {
      time_measurements: Record<string, number>
      memory_measurements: Record<string, number>
    }
    error_analysis: string
  }
  frontend_display?: {
    console_output: string
    console_error: string
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
  }
}

// Solution execution response
export interface SolutionResponse {
  success: boolean
  message: string
  data: {
    console_output: string
    console_error: string
    execution_time: number
    peak_memory: string
    exit_code: number
    solution_found: boolean
  }
}

// Boilerplate response
export interface BoilerplateResponse {
  success: boolean
  language: string
  boilerplate: string
  message: string
}

// AI Analysis response
export interface AIAnalysisResponse {
  success: boolean
  message: string
  data: {
    analysis: string
    retrieved_context: number
    error?: string
    language: string
    has_time_data: boolean
    has_space_data: boolean
    has_execution_error: boolean
  }
}

// New interface for FastAPI execution request
export interface CodeExecutionRequest {
  code: string
  language: string
  test_sizes?: number[]
}

export class BackendSync {
  private baseUrl: string

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl
  }

  private safeJoin(base: string, path: string): string {
    try {
      return new URL(path, base).toString()
    } catch {
      return base.replace(/\/$/, "") + path
    }
  }

  // Health check for FastAPI backend
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(this.safeJoin(this.baseUrl, "/api/health"), {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      })
      return response.ok
    } catch (error) {
      console.warn("Backend health check failed:", error)
      return false
    }
  }

  // Get supported languages from FastAPI backend
  async getSupportedLanguages(): Promise<any> {
    try {
      const response = await fetch(this.safeJoin(this.baseUrl, "/api/supported-languages"), {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      })
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      
      return await response.json()
    } catch (error) {
      console.warn("Backend language fetch failed:", error)
      throw error
    }
  }

  // Execute code using FastAPI backend
  async executeCode(data: CodeExecutionRequest): Promise<BackendResponse> {
    try {
      const response = await fetch(this.safeJoin(this.baseUrl, "/api/execute"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          code: data.code,
          language: data.language,
          test_sizes: data.test_sizes || [10, 100, 500, 1000, 10000]
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }

      return await response.json()
    } catch (error) {
      console.warn("Backend code execution failed:", error)
      throw error
    }
  }

  async postRun(data: RunData): Promise<BackendResponse> {
    // Legacy method - now delegates to executeCode
    return this.executeCode({
      code: data.code,
      language: data.language,
      test_sizes: [10, 100, 500, 1000, 10000]
    })
  }

  // Execute solution code in LeetCode format
  async executeSolution(data: SolutionExecutionRequest): Promise<SolutionResponse> {
    try {
      const response = await fetch(this.safeJoin(this.baseUrl, "/api/execute-solution"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }

      return await response.json()
    } catch (error) {
      console.warn("Backend solution execution failed:", error)
      throw error
    }
  }

  // Get boilerplate code for a language
  async getBoilerplate(language: string): Promise<BoilerplateResponse> {
    try {
      const response = await fetch(this.safeJoin(this.baseUrl, `/api/boilerplate/${language}`), {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }

      return await response.json()
    } catch (error) {
      console.warn("Backend boilerplate fetch failed:", error)
      throw error
    }
  }

  // Request AI analysis of code
  // New chat method
  async sendChatMessage(data: ChatRequest): Promise<ChatResponse> {
    try {
      const response = await fetch(this.safeJoin(this.baseUrl, "/api/chat"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }

      return await response.json()
    } catch (error) {
      console.warn("Backend chat failed:", error)
      throw error
    }
  }

  // Legacy AI analysis method - now converts to chat
  async requestAIAnalysis(data: AIAnalysisRequest): Promise<any> {
    try {
      // Convert AI analysis request to chat message
      let message = "Please analyze this code"
      if (data.code) {
        message += " and explain its time and space complexity."
      }

      const chatRequest: ChatRequest = {
        message: message,
        code: data.code,
        language: data.language
      }

      const response = await this.sendChatMessage(chatRequest)
      
      // Convert chat response to legacy AI analysis format
      return {
        success: response.success,
        message: response.message,
        data: {
          analysis: response.data.response,
          complexity: { time: "See analysis", space: "See analysis" },
          suggestions: [],
          reasoning: response.data.response,
          code_quality: {},
          performance_insights: response.data.response,
          language: data.language,
          has_time_data: false,
          has_space_data: false,
          has_execution_error: false
        }
      }
    } catch (error) {
      console.warn("Backend AI analysis failed:", error)
      throw error
    }
  }

  // Get supported languages for solution format
  async getSolutionLanguages(): Promise<any> {
    try {
      const response = await fetch(this.safeJoin(this.baseUrl, "/api/solution-languages"), {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      })
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      
      return await response.json()
    } catch (error) {
      console.warn("Backend solution languages fetch failed:", error)
      throw error
    }
  }
}
