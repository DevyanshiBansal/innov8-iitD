"use client"

import React, { useState, useCallback, useRef, useEffect } from 'react'
import { 
  Mic, 
  MicOff, 
  Play, 
  Square, 
  User, 
  Clock,
  Brain,
  MessageSquare,
  CheckCircle,
  AlertCircle
} from 'lucide-react'

interface InterviewStatus {
  active: boolean
  session_id?: string
  candidate_name?: string
  interview_state?: string
  questions_asked?: number
  start_time?: string
  monitoring_active?: boolean
}

interface InterviewReport {
  // Session metadata
  session_metadata?: {
    session_id: string
    candidate_name: string
    interview_mode: string
    interview_duration: string
    conversation_id?: string
    conducted_by: string
  }
  
  // Legacy fields for compatibility
  session_id?: string
  candidate_name?: string
  interview_duration?: string
  
  // Interview performance
  interview_performance?: {
    questions_attempted: number
    code_submissions: number
    professional_interventions: number
  }
  
  // Legacy fields
  questions_attempted?: number
  code_submissions?: number
  
  // Professional assessment (new structure)
  professional_assessment?: {
    technical_competency: {
      algorithmic_thinking: string
      data_structure_knowledge: string
      coding_implementation: string
      debugging_capability: string
    }
    problem_solving_approach: {
      approach_methodology: string
      edge_case_consideration: string
      optimization_mindset: string
      solution_verification: string
    }
    communication_quality: {
      explanation_clarity: string
      question_asking: string
      professional_demeanor: string
      technical_vocabulary: string
    }
    code_quality_analysis: {
      readability: string
      efficiency: string
      best_practices: string
      maintainability: string
    }
    optimization_awareness: {
      complexity_awareness: string
      performance_considerations: string
      trade_off_recognition: string
      scalability_mindset: string
    }
  }
  
  // Objective scores (Mathematics.py)
  objective_scores?: {
    'Problem-Solving Score': number
    'Coding Proficiency Score': number
    'Resilience Score': number
    'Autonomy Score': number
    'Overall Professional Score': number
    'Professional Assessment': string
    'Detailed Metrics'?: {
      'Base Score': number
      'Difficulty Factor': number
      'Professional Standards Applied': boolean
    }
  }
  
  // Professional recommendation
  professional_recommendation?: string
  
  // Detailed feedback
  detailed_feedback?: {
    strengths_identified: string[]
    improvement_areas: string[]
    development_suggestions: string[]
  }
  
  // Interview conduct notes
  interview_conduct_notes?: Array<{
    timestamp: string
    phase?: string
    type?: string
    content?: string
    level?: number
    context?: string
  }>
  
  // Legacy fields for backward compatibility
  performance_analysis?: {
    correctness: string
    efficiency: string
    code_quality: string
    problem_solving_approach: string
  }
  complexity_mastery?: {
    time_complexity_awareness: string
    space_complexity_awareness: string
    optimization_skills: string
  }
  communication_score?: {
    explanation_clarity: string
    questions_asked: string
    receptive_to_feedback: string
  }
  overall_recommendation?: string
}

interface VoiceInterviewPanelProps {
  className?: string
}

export default function VoiceInterviewPanel({ className }: VoiceInterviewPanelProps) {
  const [candidateName, setCandidateName] = useState('')
  const [interviewMode, setInterviewMode] = useState<'voice' | 'chat'>('voice')
  const [interviewStatus, setInterviewStatus] = useState<InterviewStatus>({ active: false })
  const [interviewReport, setInterviewReport] = useState<InterviewReport | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [voiceEnabled, setVoiceEnabled] = useState(false)
  
  const statusCheckInterval = useRef<NodeJS.Timeout | null>(null)

  // Polling for interview status
  const checkInterviewStatus = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8000/api/interview/status')
      if (response.ok) {
        const status = await response.json()
        setInterviewStatus(status)
      }
    } catch (err) {
      console.error('Failed to check interview status:', err)
    }
  }, [])

  // Start status polling when interview is active
  useEffect(() => {
    if (interviewStatus.active) {
      statusCheckInterval.current = setInterval(checkInterviewStatus, 5000)
    } else {
      if (statusCheckInterval.current) {
        clearInterval(statusCheckInterval.current)
        statusCheckInterval.current = null
      }
    }

    return () => {
      if (statusCheckInterval.current) {
        clearInterval(statusCheckInterval.current)
      }
    }
  }, [interviewStatus.active, checkInterviewStatus])

  const startInterview = async () => {
    if (!candidateName.trim()) {
      setError('Please enter candidate name')
      return
    }

    setIsLoading(true)
    setError(null)
    setInterviewReport(null)

    try {
      const response = await fetch('http://localhost:8000/api/interview/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          candidate_name: candidateName.trim(),
          interview_mode: interviewMode
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to start interview')
      }

      const data = await response.json()
      
      // Update status immediately
      await checkInterviewStatus()
      
      // Enable voice after successful start
      setVoiceEnabled(true)
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start interview')
    } finally {
      setIsLoading(false)
    }
  }

  const endInterview = async () => {
    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch('http://localhost:8000/api/interview/end', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to end interview')
      }

      const data = await response.json()
      setInterviewReport(data.report)
      
      // Update status
      await checkInterviewStatus()
      
      // Disable voice
      setVoiceEnabled(false)
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to end interview')
    } finally {
      setIsLoading(false)
    }
  }

  const toggleVoice = () => {
    setVoiceEnabled(!voiceEnabled)
    // Here you would integrate with the actual ElevenLabs voice interface
    // For now, just visual feedback
  }

  const formatDuration = (startTime: string) => {
    const start = new Date(startTime)
    const now = new Date()
    const diff = Math.floor((now.getTime() - start.getTime()) / 1000)
    const minutes = Math.floor(diff / 60)
    const seconds = diff % 60
    return `${minutes}:${seconds.toString().padStart(2, '0')}`
  }

  const getStateColor = (state: string) => {
    switch (state) {
      case 'starting': return 'bg-yellow-100 text-yellow-800'
      case 'questioning': return 'bg-blue-100 text-blue-800'
      case 'coding': return 'bg-green-100 text-green-800'
      case 'analyzing': return 'bg-purple-100 text-purple-800'
      case 'concluding': return 'bg-gray-100 text-gray-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Interview Control Panel */}
      <div className="border rounded-lg shadow-sm bg-white">
        <div className="border-b p-4">
          <h3 className="flex items-center gap-2 text-lg font-semibold">
            <Brain className="w-5 h-5" />
            AI Voice Interviewer
          </h3>
        </div>
        <div className="p-4 space-y-4">
          {!interviewStatus.active ? (
            // Start Interview Form
            <div className="space-y-3">
              <div className="space-y-2">
                <label htmlFor="candidateName" className="text-sm font-medium">
                  Candidate Name
                </label>
                <input
                  id="candidateName"
                  type="text"
                  className="w-full px-3 py-2 border rounded-md"
                  placeholder="Enter candidate's name"
                  value={candidateName}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setCandidateName(e.target.value)}
                  disabled={isLoading}
                />
              </div>

              {/* Interview Mode Selection */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Interview Mode</label>
                <div className="flex gap-4">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="radio"
                      name="interviewMode"
                      value="voice"
                      checked={interviewMode === 'voice'}
                      onChange={(e) => setInterviewMode('voice')}
                      disabled={isLoading}
                      className="text-blue-500"
                    />
                    <Mic className="w-4 h-4" />
                    <span className="text-sm">Voice Interview (ElevenLabs)</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="radio"
                      name="interviewMode"
                      value="chat"
                      checked={interviewMode === 'chat'}
                      onChange={(e) => setInterviewMode('chat')}
                      disabled={isLoading}
                      className="text-blue-500"
                    />
                    <MessageSquare className="w-4 h-4" />
                    <span className="text-sm">Chat Interview (Gemini)</span>
                  </label>
                </div>
              </div>
              
              <button 
                onClick={startInterview}
                disabled={isLoading || !candidateName.trim()}
                className="w-full bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 text-white px-4 py-2 rounded-md flex items-center justify-center gap-2"
              >
                {isLoading ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
                    Starting Interview...
                  </>
                ) : (
                  <>
                    {interviewMode === 'voice' ? <Mic className="w-4 h-4" /> : <MessageSquare className="w-4 h-4" />}
                    Start {interviewMode === 'voice' ? 'Voice' : 'Chat'} Interview
                  </>
                )}
              </button>
            </div>
          ) : (
            // Active Interview Controls
            <div className="space-y-4">
              {/* Interview Status */}
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" />
                    <span className="font-medium">Interview Active</span>
                  </div>
                  {interviewStatus.start_time && (
                    <span className="bg-gray-100 text-gray-800 px-2 py-1 rounded text-sm flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {formatDuration(interviewStatus.start_time)}
                    </span>
                  )}
                </div>
                
                <div className="text-sm space-y-1">
                  <div className="flex items-center gap-2">
                    <User className="w-4 h-4" />
                    <span>Candidate: {interviewStatus.candidate_name}</span>
                  </div>
                  
                  {interviewStatus.interview_state && (
                    <div className="flex items-center gap-2">
                      <span className={`px-2 py-1 rounded text-xs ${getStateColor(interviewStatus.interview_state)}`}>
                        {interviewStatus.interview_state}
                      </span>
                      <span className="text-gray-600">
                        Questions: {interviewStatus.questions_asked || 0}
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* Voice Controls */}
              <div className="flex gap-2">
                <button
                  onClick={toggleVoice}
                  className={`flex-1 px-4 py-2 rounded-md flex items-center justify-center gap-2 ${
                    voiceEnabled 
                      ? 'bg-blue-500 hover:bg-blue-600 text-white' 
                      : 'bg-gray-100 hover:bg-gray-200 text-gray-700 border'
                  }`}
                >
                  {voiceEnabled ? (
                    <>
                      <Mic className="w-4 h-4" />
                      Voice On
                    </>
                  ) : (
                    <>
                      <MicOff className="w-4 h-4" />
                      Voice Off
                    </>
                  )}
                </button>
                
                <button
                  onClick={endInterview}
                  disabled={isLoading}
                  className="bg-red-500 hover:bg-red-600 disabled:bg-gray-300 text-white px-4 py-2 rounded-md flex items-center justify-center"
                >
                  {isLoading ? (
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
                  ) : (
                    <Square className="w-4 h-4" />
                  )}
                </button>
              </div>
            </div>
          )}

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-3">
              <div className="flex items-center gap-2 text-red-700">
                <AlertCircle className="w-4 h-4" />
                <span className="text-sm">{error}</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Interview Report */}
      {interviewReport && (
        <div className="border rounded-lg shadow-sm bg-white">
          <div className="border-b p-4">
            <h3 className="flex items-center gap-2 text-lg font-semibold">
              <CheckCircle className="w-5 h-5 text-green-500" />
              Interview Report
            </h3>
          </div>
          <div className="p-4 space-y-4">
            {/* Basic Info */}
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-medium">Candidate:</span> {interviewReport?.session_metadata?.candidate_name || interviewReport?.candidate_name || 'N/A'}
              </div>
              <div>
                <span className="font-medium">Duration:</span> {interviewReport?.session_metadata?.interview_duration || interviewReport?.interview_duration || 'N/A'}
              </div>
              <div>
                <span className="font-medium">Questions:</span> {interviewReport?.interview_performance?.questions_attempted || interviewReport?.questions_attempted || 0}
              </div>
              <div>
                <span className="font-medium">Submissions:</span> {interviewReport?.interview_performance?.code_submissions || interviewReport?.code_submissions || 0}
              </div>
              <div>
                <span className="font-medium">Interview Mode:</span> {interviewReport?.session_metadata?.interview_mode || 'Voice'}
              </div>
              <div>
                <span className="font-medium">Session ID:</span> {interviewReport?.session_metadata?.session_id || interviewReport?.session_id || 'N/A'}
              </div>
            </div>

            <hr className="border-gray-200" />

            {/* Professional Assessment - Updated for new structure */}
            <div>
              <h4 className="font-medium mb-2">Professional Assessment</h4>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>Technical Competency: <span className="bg-gray-100 px-2 py-1 rounded text-xs">{interviewReport?.professional_assessment?.technical_competency?.algorithmic_thinking || 'N/A'}</span></div>
                <div>Problem Solving: <span className="bg-gray-100 px-2 py-1 rounded text-xs">{interviewReport?.professional_assessment?.problem_solving_approach?.approach_methodology || 'N/A'}</span></div>
                <div>Code Quality: <span className="bg-gray-100 px-2 py-1 rounded text-xs">{interviewReport?.professional_assessment?.code_quality_analysis?.readability || 'N/A'}</span></div>
                <div>Communication: <span className="bg-gray-100 px-2 py-1 rounded text-xs">{interviewReport?.professional_assessment?.communication_quality?.explanation_clarity || 'N/A'}</span></div>
              </div>
            </div>

            <hr className="border-gray-200" />

            {/* Objective Scores - Mathematics.py */}
            <div>
              <h4 className="font-medium mb-2">Objective Scores</h4>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>Problem-Solving: <span className="bg-green-100 px-2 py-1 rounded text-xs font-medium">{interviewReport?.objective_scores?.['Problem-Solving Score'] || 0}/10</span></div>
                <div>Coding Proficiency: <span className="bg-blue-100 px-2 py-1 rounded text-xs font-medium">{interviewReport?.objective_scores?.['Coding Proficiency Score'] || 0}/10</span></div>
                <div>Resilience: <span className="bg-purple-100 px-2 py-1 rounded text-xs font-medium">{interviewReport?.objective_scores?.['Resilience Score'] || 0}/10</span></div>
                <div>Autonomy: <span className="bg-orange-100 px-2 py-1 rounded text-xs font-medium">{interviewReport?.objective_scores?.['Autonomy Score'] || 0}/10</span></div>
              </div>
              <div className="mt-2 text-center">
                <div className="bg-indigo-100 border border-indigo-200 rounded px-3 py-2">
                  <span className="text-sm font-medium">Overall Score: </span>
                  <span className="text-lg font-bold text-indigo-700">{interviewReport?.objective_scores?.['Overall Professional Score'] || 0}/10</span>
                </div>
              </div>
            </div>

            <hr className="border-gray-200" />

            {/* Professional Assessment Tier */}
            <div>
              <h4 className="font-medium mb-2">Assessment Level</h4>
              <div className="bg-gray-50 border rounded-lg p-3">
                <span className="text-sm font-medium">{interviewReport?.objective_scores?.['Professional Assessment'] || 'Assessment pending'}</span>
              </div>
            </div>

            <hr className="border-gray-200" />

            {/* Detailed Feedback */}
            <div>
              <h4 className="font-medium mb-2">Detailed Feedback</h4>
              <div className="space-y-3">
                <div>
                  <h5 className="text-sm font-medium text-green-700 mb-1">Strengths Identified</h5>
                  <ul className="text-xs text-gray-600 space-y-1">
                    {interviewReport?.detailed_feedback?.strengths_identified?.map((strength: string, index: number) => (
                      <li key={index} className="flex items-start">
                        <span className="text-green-500 mr-1">✓</span>
                        {strength}
                      </li>
                    )) || [<li key="none">No specific strengths recorded</li>]}
                  </ul>
                </div>
                
                <div>
                  <h5 className="text-sm font-medium text-orange-700 mb-1">Areas for Improvement</h5>
                  <ul className="text-xs text-gray-600 space-y-1">
                    {interviewReport?.detailed_feedback?.improvement_areas?.map((area: string, index: number) => (
                      <li key={index} className="flex items-start">
                        <span className="text-orange-500 mr-1">→</span>
                        {area}
                      </li>
                    )) || [<li key="none">No specific improvement areas recorded</li>]}
                  </ul>
                </div>
              </div>
            </div>

            <hr className="border-gray-200" />

            {/* Final Recommendation */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
              <h4 className="font-medium mb-1">Professional Recommendation</h4>
              <p className="text-sm text-gray-700">{interviewReport?.professional_recommendation || interviewReport?.overall_recommendation || 'Recommendation pending'}</p>
            </div>
          </div>
        </div>
      )}

      {/* Voice Interview Instructions */}
      <div className="border rounded-lg shadow-sm bg-white">
        <div className="border-b p-4">
          <h3 className="flex items-center gap-2 text-lg font-semibold">
            <MessageSquare className="w-5 h-5" />
            Interview Instructions
          </h3>
        </div>
        <div className="p-4">
          <div className="text-sm space-y-2 text-gray-600">
            <p>• The AI interviewer will conduct a formal technical interview</p>
            <p>• Code in the editor while speaking with the interviewer</p>
            <p>• The interviewer can see your code and execution results in real-time</p>
            <p>• Ask for hints if you're stuck, but try to solve problems independently</p>
            <p>• Explain your thinking process and approach clearly</p>
            <p>• The interviewer will analyze time/space complexity and provide feedback</p>
          </div>
        </div>
      </div>
    </div>
  )
}