// File System Access API utilities
export class FileSystemManager {
  private dirHandle: FileSystemDirectoryHandle | null = null

  async chooseDirectory(): Promise<boolean> {
    if (!window.showDirectoryPicker) {
      alert("Your browser does not support File System Access API. Use Chrome/Edge over http(s).")
      return false
    }

    try {
      this.dirHandle = await window.showDirectoryPicker({
        id: "mini-ide-output",
        mode: "readwrite",
        startIn: "documents",
      })
      return true
    } catch (e: any) {
      if (e && e.name !== "AbortError") {
        console.warn("Directory selection failed:", e)
      }
      return false
    }
  }

  async writeFile(name: string, contents: string): Promise<boolean> {
    if (!this.dirHandle) return false

    try {
      const fileHandle = await this.dirHandle.getFileHandle(name, { create: true })
      const writable = await fileHandle.createWritable()
      await writable.write(contents)
      await writable.close()
      return true
    } catch (e) {
      console.warn("File write failed:", e)
      return false
    }
  }

  hasDirectory(): boolean {
    return this.dirHandle !== null
  }
}

// Global instance
export const fileSystemManager = new FileSystemManager()
