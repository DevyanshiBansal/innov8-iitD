"use client"

import { useRef, useEffect, type ReactNode } from "react"

interface ResizableLayoutProps {
  children: ReactNode
  isMaximized: boolean
}

export function ResizableLayout({ children, isMaximized }: ResizableLayoutProps) {
  const mainRef = useRef<HTMLElement>(null)
  const gutter1Ref = useRef<HTMLDivElement>(null)
  const gutter2Ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!mainRef.current || !gutter1Ref.current || !gutter2Ref.current) return

    const main = mainRef.current
    const gutter1 = gutter1Ref.current
    const gutter2 = gutter2Ref.current

    // Initial sizes
    main.style.gridTemplateColumns = "1fr 6px 1.6fr 6px 1.1fr"
    const MIN_LEFT = 220
    const MIN_EDIT = 420
    const MIN_RIGHT = 240

    const startDrag = (ev: PointerEvent, which: number) => {
      const leftEl = main.querySelector(".mini-ide-problem-wrap") as HTMLElement
      const editEl = main.querySelector(".editor-wrap") as HTMLElement
      const rightEl = main.querySelector(".mini-ide-chat-wrap") as HTMLElement

      if (!leftEl || !editEl || !rightEl) return

      const start = {
        x: ev.clientX,
        left: leftEl.clientWidth || 0,
        edit: editEl.clientWidth || 0,
        right: rightEl.clientWidth || 0,
      }

      const onMove = (e: PointerEvent) => {
        const dx = e.clientX - start.x
        let L = start.left
        let E = start.edit
        let R = start.right

        if (which === 1) {
          // Move between problem and editor
          L = Math.max(MIN_LEFT, start.left + dx)
          E = Math.max(MIN_EDIT, start.edit - (L - start.left))
        } else {
          // Move between editor and chat
          E = Math.max(MIN_EDIT, start.edit + dx)
          R = Math.max(MIN_RIGHT, start.right - (E - start.edit))
        }

        main.style.gridTemplateColumns = `${L}px 6px ${E}px 6px ${R}px`

        if (window.monaco) {
          // Find Monaco editor instance and trigger layout
          const editorElement = main.querySelector(".mini-ide-editor")
          if (editorElement && (editorElement as any)._monacoEditor) {
            ;(editorElement as any)._monacoEditor.layout()
          }
        }
      }

      const onUp = () => {
        window.removeEventListener("pointermove", onMove)
        window.removeEventListener("pointerup", onUp)
        // Reset cursor
        document.body.style.cursor = ""
      }

      // Set cursor for dragging
      document.body.style.cursor = "col-resize"

      window.addEventListener("pointermove", onMove)
      window.addEventListener("pointerup", onUp)
    }

    const handleGutter1 = (ev: PointerEvent) => startDrag(ev, 1)
    const handleGutter2 = (ev: PointerEvent) => startDrag(ev, 2)

    gutter1.addEventListener("pointerdown", handleGutter1)
    gutter2.addEventListener("pointerdown", handleGutter2)

    return () => {
      gutter1.removeEventListener("pointerdown", handleGutter1)
      gutter2.removeEventListener("pointerdown", handleGutter2)
    }
  }, [])

  const childrenWithRefs = (children as any).props?.children || children
  const gutterElements = Array.isArray(childrenWithRefs)
    ? childrenWithRefs.filter((child: any) => child?.props?.className === "mini-ide-gutter")
    : []

  return (
    <main ref={mainRef} className="mini-ide-main">
      {Array.isArray(childrenWithRefs)
        ? childrenWithRefs.map((child: any, index: number) => {
            if (child?.props?.className === "mini-ide-gutter") {
              const ref = index === 1 ? gutter1Ref : gutter2Ref
              return <div key={index} ref={ref} className="mini-ide-gutter" />
            }
            return child
          })
        : children}
    </main>
  )
}
