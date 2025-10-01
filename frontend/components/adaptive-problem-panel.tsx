"use client"

import { useState, useEffect } from "react"
import { Timer, Brain, Target, TrendingUp } from "lucide-react"

interface AdaptiveQuestion {
  question_id: string
  title: string
  problem_statement: string
  examples: Array<{
    id: number
    input: string
    output: string
    explanation?: string
  }>
  constraints: string[]
  difficulty: string
  topics: string[]
  time_limit: number
}

interface AdaptiveProblemPanelProps {
  className?: string
}

export function AdaptiveProblemPanel({ className }: AdaptiveProblemPanelProps) {
  const [currentQuestion, setCurrentQuestion] = useState<AdaptiveQuestion | null>(null)
  const [loading, setLoading] = useState(false)
  const [timeRemaining, setTimeRemaining] = useState(0)
  const [sessionStats, setSessionStats] = useState({
    questionsAttempted: 0,
    questionsCompleted: 0,
    currentStreak: 0,
    averageTime: 0,
    currentDifficulty: "Easy"
  })

  // Timer effect
  useEffect(() => {
    if (timeRemaining > 0) {
      const timer = setTimeout(() => setTimeRemaining(timeRemaining - 1), 1000)
      return () => clearTimeout(timer)
    }
  }, [timeRemaining])

  const fetchAdaptiveQuestion = async (nextDifficulty?: string) => {
    setLoading(true)
    try {
      const difficulty = nextDifficulty || sessionStats.currentDifficulty
      const params = new URLSearchParams({ difficulty })
      
      const response = await fetch(`http://localhost:8000/api/adaptive/question?${params}`)
      
      if (response.ok) {
        const question = await response.json()
        setCurrentQuestion(question)
        setTimeRemaining(question.time_limit * 60) // Convert minutes to seconds
        setSessionStats(prev => ({
          ...prev,
          questionsAttempted: prev.questionsAttempted + 1,
          currentDifficulty: question.difficulty
        }))
      } else {
        console.error("Failed to fetch question:", response.statusText)
      }
    } catch (error) {
      console.error("Error fetching adaptive question:", error)
    }
    setLoading(false)
  }

  // Function to handle solution submission and automatically load next question
  const handleSolutionSubmitted = async (submissionResult: any) => {
    if (submissionResult?.next_difficulty) {
      // Update session stats based on submission result
      setSessionStats(prev => ({
        ...prev,
        questionsCompleted: prev.questionsCompleted + (submissionResult.success ? 1 : 0),
        currentStreak: submissionResult.success ? prev.currentStreak + 1 : 0,
        currentDifficulty: submissionResult.next_difficulty
      }))
      
      // Automatically fetch next question with adaptive difficulty
      setTimeout(() => {
        fetchAdaptiveQuestion(submissionResult.next_difficulty)
      }, 2000) // Wait 2 seconds before loading next question
    }
  }

  // Expose the handler for the parent component
  useEffect(() => {
    if (typeof window !== 'undefined') {
      (window as any).handleAdaptiveSubmission = handleSolutionSubmitted
    }
  }, [])

  // Fetch initial question on mount
  useEffect(() => {
    fetchAdaptiveQuestion()
  }, [])

  const formatTime = (seconds: number) => {
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = seconds % 60
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`
  }

  const getTimeColor = () => {
    const totalTime = (currentQuestion?.time_limit || 1) * 60
    const percentRemaining = timeRemaining / totalTime
    if (percentRemaining > 0.5) return "text-green-600"
    if (percentRemaining > 0.25) return "text-yellow-600"
    return "text-red-600"
  }

  const getDifficultyColor = (diff: string) => {
    switch (diff.toLowerCase()) {
      case 'easy': return "text-green-600 bg-green-100"
      case 'medium': return "text-yellow-600 bg-yellow-100"
      case 'hard': return "text-red-600 bg-red-100"
      default: return "text-gray-600 bg-gray-100"
    }
  }

  if (loading && !currentQuestion) {
    return (
      <div className={`mini-ide-problem-wrap ${className}`}>
        <div className="mini-ide-panel h-full flex items-center justify-center">
          <div className="text-center">
            <Brain className="w-8 h-8 animate-spin mx-auto mb-2 text-blue-600" />
            <p>Generating adaptive question...</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className={`mini-ide-problem-wrap ${className}`}>
      <div className="mini-ide-panel h-full grid grid-rows-[auto_1fr]">
        {/* Header */}
        <div className="mini-ide-row justify-between items-baseline">
          <div className="flex items-center gap-2">
            <Brain className="w-4 h-4 text-blue-600" />
            <h3>Adaptive Challenge</h3>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <Timer className={`w-3 h-3 ${getTimeColor()}`} />
            <span className={getTimeColor()}>{formatTime(timeRemaining)}</span>
          </div>
        </div>

        {/* Content */}
        <div className="mini-ide-problem-content">
          {currentQuestion ? (
            <>
              {/* Question Header */}
              <div className="mb-4">
                <div className="flex items-center gap-2 mb-2">
                  <h4 className="mini-ide-problem-title text-sm font-semibold">
                    {currentQuestion.title}
                  </h4>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getDifficultyColor(currentQuestion.difficulty)}`}>
                    {currentQuestion.difficulty}
                  </span>
                </div>
                
                {/* Topics */}
                <div className="flex flex-wrap gap-1 mb-2">
                  {currentQuestion.topics.map((topic, index) => (
                    <span key={index} className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded">
                      {topic}
                    </span>
                  ))}
                </div>
              </div>

              {/* Problem Statement */}
              <div className="mb-4">
                <div className="mini-ide-problem-desc text-sm">
                  {currentQuestion.problem_statement}
                </div>
              </div>

              {/* Examples */}
              {currentQuestion.examples && currentQuestion.examples.length > 0 && (
                <div className="mb-4">
                  <strong className="text-sm">Examples:</strong>
                  {currentQuestion.examples.map((example, index) => (
                    <div key={example.id || index} className="mt-2 p-2 bg-gray-50 rounded text-xs">
                      <div><strong>Input:</strong> {example.input}</div>
                      <div><strong>Output:</strong> {example.output}</div>
                      {example.explanation && (
                        <div className="mt-1 text-gray-600"><strong>Explanation:</strong> {example.explanation}</div>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {/* Constraints */}
              {currentQuestion.constraints && currentQuestion.constraints.length > 0 && (
                <div className="mb-4">
                  <strong className="text-sm">Constraints:</strong>
                  <ul className="text-xs mt-1 space-y-1">
                    {currentQuestion.constraints.map((constraint, index) => (
                      <li key={index} className="text-gray-600">• {constraint}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Auto-Adaptive Info */}
              <div className="mt-4 pt-4 border-t">
                <div className="text-center text-xs text-gray-600">
                  <p>🎯 Questions adapt automatically based on your performance</p>
                  <p className="mt-1">Current Level: <span className={`font-medium ${getDifficultyColor(sessionStats.currentDifficulty)}`}>{sessionStats.currentDifficulty}</span></p>
                </div>
              </div>

              {/* Session Stats */}
              <div className="mt-4 pt-4 border-t">
                <div className="flex items-center gap-2 mb-2">
                  <TrendingUp className="w-3 h-3 text-green-600" />
                  <span className="text-xs font-medium">Session Stats</span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <div className="text-gray-500">Attempted</div>
                    <div className="font-medium">{sessionStats.questionsAttempted}</div>
                  </div>
                  <div>
                    <div className="text-gray-500">Completed</div>
                    <div className="font-medium">{sessionStats.questionsCompleted}</div>
                  </div>
                  <div>
                    <div className="text-gray-500">Streak</div>
                    <div className="font-medium">{sessionStats.currentStreak}</div>
                  </div>
                  <div>
                    <div className="text-gray-500">Avg Time</div>
                    <div className="font-medium">{sessionStats.averageTime}min</div>
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="text-center py-8">
              <Target className="w-8 h-8 mx-auto mb-2 text-gray-400" />
              <p className="text-gray-500 mb-4">No question loaded</p>
              <button
                onClick={() => fetchAdaptiveQuestion()}
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
              >
                Start Challenge
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}