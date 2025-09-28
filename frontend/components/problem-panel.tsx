"use client"

import { getProblem } from "@/lib/problems"

interface ProblemPanelProps {
  problemId: string
}

export function ProblemPanel({ problemId }: ProblemPanelProps) {
  const problem = getProblem(problemId)

  if (!problem) {
    return (
      <aside className="mini-ide-problem-wrap">
        <div className="mini-ide-panel h-full grid grid-rows-[auto_1fr]">
          <div className="mini-ide-row justify-between items-baseline">
            <h3>Problem</h3>
            <span className="mini-ide-hint text-xs">Like LeetCode</span>
          </div>
          <div className="mini-ide-problem-content">
            <div className="mini-ide-problem-title">Problem Not Found</div>
            <div className="mini-ide-problem-desc">The selected problem could not be found.</div>
          </div>
        </div>
      </aside>
    )
  }

  return (
    <aside className="mini-ide-problem-wrap">
      <div className="mini-ide-panel h-full grid grid-rows-[auto_1fr]">
        <div className="mini-ide-row justify-between items-baseline">
          <h3>Problem</h3>
          <span className="mini-ide-hint text-xs">Like LeetCode</span>
        </div>
        <div className="mini-ide-problem-content">
          <div className="mini-ide-problem-title">{problem.title}</div>
          <div className="mini-ide-problem-desc">
            <div className="mb-2">{problem.description}</div>

            {problem.constraints && (
              <div className="my-2">
                <strong>Constraints:</strong>
                <div className="text-xs mt-1">{problem.constraints}</div>
              </div>
            )}

            {problem.examples && problem.examples.length > 0 && (
              <div className="my-2">
                <strong>Examples:</strong>
                {problem.examples.map((example, index) => (
                  <div key={index} className="my-2">
                    <div>
                      <strong>Input:</strong> <code>{example.input}</code>
                    </div>
                    <div>
                      <strong>Output:</strong> <code>{example.output}</code>
                    </div>
                    {example.explanation && <div className="text-xs mt-1">{example.explanation}</div>}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </aside>
  )
}
