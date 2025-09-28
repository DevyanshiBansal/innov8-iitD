// Global type declarations for browser APIs and Monaco Editor
declare global {
  interface Window {
    monaco: any
    require: any
    MonacoEnvironment: any
    showDirectoryPicker?: (options?: {
      id?: string
      mode?: "read" | "readwrite"
      startIn?: "desktop" | "documents" | "downloads" | "music" | "pictures" | "videos"
    }) => Promise<FileSystemDirectoryHandle>
  }

  interface FileSystemDirectoryHandle {
    getFileHandle(name: string, options?: { create?: boolean }): Promise<FileSystemFileHandle>
  }

  interface FileSystemFileHandle {
    createWritable(): Promise<FileSystemWritableFileStream>
  }

  interface FileSystemWritableFileStream {
    write(data: string): Promise<void>
    close(): Promise<void>
  }
}

export {}
