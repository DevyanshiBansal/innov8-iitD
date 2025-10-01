// Backend synchronization utilities
export interface SnapshotData {
  sessionId: string
  problemId: string
  language: string
  code: string
  timestamp: string
}

export interface RunData extends SnapshotData {
  input: string
}

export interface BackendResponse {
  ok: boolean
  note?: string
  exitCode?: number
  runOut?: string
  runErr?: string
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

  async postSnapshot(data: SnapshotData): Promise<void> {
    try {
      const response = await fetch(this.safeJoin(this.baseUrl, "/api/snapshot"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
    } catch (error) {
      console.warn("Backend snapshot failed:", error)
      throw error
    }
  }

  async postRun(data: RunData): Promise<BackendResponse> {
    try {
      const response = await fetch(this.safeJoin(this.baseUrl, "/api/run"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sessionId: data.sessionId,
          problemId: data.problemId,
          language: data.language,
          code: data.code,
          input: data.input,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }

      return await response.json()
    } catch (error) {
      console.warn("Backend run failed:", error)
      throw error
    }
  }
}
