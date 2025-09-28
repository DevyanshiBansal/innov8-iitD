"use client";

import React, { useState, useEffect, useCallback } from 'react';
import { MonacoEditor } from './monaco-editor';
import { 
  Play, 
  Send, 
  Trophy, 
  Target, 
  Brain,
  CheckCircle,
  XCircle,
  Timer,
  BookOpen,
  TrendingUp,
  Star,
  Code,
  Lightbulb,
  Zap
} from 'lucide-react';

interface Problem {
  id: string;
  title: string;
  problem_statement: string;
  examples: Array<{
    input: string;
    output: string;
    explanation?: string;
  }>;
  constraints: string[];
  difficulty: 'Easy' | 'Medium' | 'Hard';
  topics: string[];
  expected_time: number;
}

interface UserStats {
  skill_level: number;
  problems_solved: number;
  strong_topics: string[];
  weak_topics: string[];
  problem_history?: Array<{
    problem_id: string;
    difficulty: string;
    is_solved: boolean;
    time_taken: number;
    attempts: number;
    topics: string[];
  }>;
}

interface ExecutionResult {
  success: boolean;
  data?: {
    execution_output: {
      output: string;
      error: string;
      exit_code: number;
      execution_time: number;
      peak_memory_formatted: string;
    };
  };
  message: string;
}

interface AdaptiveCodingPanelProps {
  onExecutionResult?: (result: ExecutionResult) => void;
  className?: string;
}

const difficultyColors = {
  Easy: "bg-green-100 text-green-800 border-green-200",
  Medium: "bg-yellow-100 text-yellow-800 border-yellow-200",
  Hard: "bg-red-100 text-red-800 border-red-200"
};

export const AdaptiveCodingPanel: React.FC<AdaptiveCodingPanelProps> = ({
  onExecutionResult,
  className = ""
}) => {
  // State management
  const [currentProblem, setCurrentProblem] = useState<Problem | null>(null);
  const [userCode, setUserCode] = useState<string>('');
  const [language, setLanguage] = useState<string>('python');
  const [userStats, setUserStats] = useState<UserStats | null>(null);
  
  // Timing and session tracking
  const [sessionStartTime, setSessionStartTime] = useState<number | null>(null);
  const [currentTime, setCurrentTime] = useState<number>(Date.now());
  const [attempts, setAttempts] = useState<number>(0);
  const [runs, setRuns] = useState<number>(0);
  const [hintsUsed, setHintsUsed] = useState<number>(0);
  
  // Execution states
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [executionResult, setExecutionResult] = useState<ExecutionResult | null>(null);
  const [lastSubmissionResult, setLastSubmissionResult] = useState<any>(null);
  
  // UI states
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [appreciationMessage, setAppreciationMessage] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'problem' | 'stats'>('problem');
  const [showHint, setShowHint] = useState<boolean>(false);
  
  // Real-time timer update
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(Date.now());
    }, 1000);
    
    return () => clearInterval(timer);
  }, []);
  
  // Calculate current session time
  const getSessionTime = useCallback((): number => {
    if (!sessionStartTime) return 0;
    return Math.floor((currentTime - sessionStartTime) / 1000);
  }, [currentTime, sessionStartTime]);
  
  // Format time display
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };
  
  // Load first problem on component mount
  useEffect(() => {
    loadNextProblem();
  }, []);
  
  // Load next adaptive problem
  const loadNextProblem = async (previousProblemId?: string, timeTaken?: number, isSolved?: boolean) => {
    setIsLoading(true);
    try {
      const requestBody = {
        current_problem_id: previousProblemId || null,
        time_taken: timeTaken || null,
        is_solved: isSolved || false,
        attempts: attempts,
        runs: runs,
        user_id: "default_user"
      };
      
      const response = await fetch('http://localhost:8000/api/adaptive/next-question', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.success) {
        setCurrentProblem(data.problem_data);
        setUserStats(data.user_stats);
        setAppreciationMessage(data.appreciation_message);
        
        // Reset session tracking
        setSessionStartTime(Date.now());
        setAttempts(0);
        setRuns(0);
        setHintsUsed(0);
        setUserCode(getDefaultCode(language));
        setExecutionResult(null);
        setLastSubmissionResult(null);
        setShowHint(false);
        
        console.log('✅ Loaded next problem:', data.problem_data.title);
      } else {
        throw new Error(data.message || 'Failed to load problem');
      }
    } catch (error) {
      console.error('Error loading next problem:', error);
      // Show fallback problem if API fails
      setCurrentProblem({
        id: 'fallback_two_sum',
        title: 'Two Sum',
        problem_statement: 'Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.\n\nYou may assume that each input would have exactly one solution, and you may not use the same element twice.\n\nYou can return the answer in any order.',
        examples: [
          {
            input: 'nums = [2,7,11,15], target = 9',
            output: '[0,1]',
            explanation: 'Because nums[0] + nums[1] == 9, we return [0, 1].'
          },
          {
            input: 'nums = [3,2,4], target = 6',
            output: '[1,2]'
          }
        ],
        constraints: [
          '2 <= nums.length <= 10^4',
          '-10^9 <= nums[i] <= 10^9',
          '-10^9 <= target <= 10^9',
          'Only one valid answer exists.'
        ],
        difficulty: 'Easy',
        topics: ['Array', 'Hash Table'],
        expected_time: 900
      });
      setSessionStartTime(Date.now());
      setUserCode(getDefaultCode(language));
    } finally {
      setIsLoading(false);
    }
  };
  
  // Get default code template
  const getDefaultCode = (lang: string): string => {
    const templates = {
      python: `def solution():
    # Write your solution here
    # Example: Two Sum problem
    # nums = [2, 7, 11, 15]
    # target = 9
    # return [0, 1]  # indices where nums[i] + nums[j] = target
    pass

# Test your solution
result = solution()
print(result)`,
      javascript: `function solution() {
    // Write your solution here
    // Example: Two Sum problem
    // const nums = [2, 7, 11, 15];
    // const target = 9;
    // return [0, 1]; // indices where nums[i] + nums[j] = target
    return null;
}

// Test your solution
const result = solution();
console.log(result);`,
      cpp: `#include <iostream>
#include <vector>
using namespace std;

vector<int> solution() {
    // Write your solution here
    // Example: Two Sum problem
    // vector<int> nums = {2, 7, 11, 15};
    // int target = 9;
    // return {0, 1}; // indices where nums[i] + nums[j] = target
    return {};
}

int main() {
    vector<int> result = solution();
    for(int i : result) {
        cout << i << " ";
    }
    return 0;
}`,
      java: `import java.util.*;

public class Solution {
    public static int[] solution() {
        // Write your solution here
        // Example: Two Sum problem
        // int[] nums = {2, 7, 11, 15};
        // int target = 9;
        // return new int[]{0, 1}; // indices where nums[i] + nums[j] = target
        return new int[]{};
    }
    
    public static void main(String[] args) {
        int[] result = solution();
        System.out.println(Arrays.toString(result));
    }
}`
    };
    return templates[lang as keyof typeof templates] || templates.python;
  };
  
  // Handle code run (testing)
  const handleRunCode = async () => {
    if (!currentProblem || isRunning) return;
    
    setIsRunning(true);
    setRuns(prev => prev + 1);
    
    try {
      const response = await fetch('http://localhost:8000/api/adaptive/run-code', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          problem_id: currentProblem.id,
          code: userCode,
          language: language
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.success && data.execution_result) {
        setExecutionResult(data.execution_result);
        onExecutionResult?.(data.execution_result);
      }
      
    } catch (error) {
      console.error('Error running code:', error);
      setExecutionResult({
        success: false,
        message: `Failed to run code: ${error}`
      });
    } finally {
      setIsRunning(false);
    }
  };
  
  // Handle solution submission
  const handleSubmitSolution = async () => {
    if (!currentProblem || isSubmitting) return;
    
    setIsSubmitting(true);
    setAttempts(prev => prev + 1);
    
    try {
      const timeTaken = getSessionTime();
      
      const response = await fetch('http://localhost:8000/api/adaptive/submit-solution', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          problem_id: currentProblem.id,
          code: userCode,
          language: language,
          time_taken: timeTaken,
          attempts: attempts + 1,
          runs: runs
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.success) {
        setLastSubmissionResult(data);
        setUserStats(data.user_stats);
        
        // Load next problem after a brief delay to show results
        setTimeout(() => {
          loadNextProblem(currentProblem.id, timeTaken, data.is_solved);
        }, 3000);
      }
      
    } catch (error) {
      console.error('Error submitting solution:', error);
      setLastSubmissionResult({
        success: false,
        feedback_message: `Failed to submit solution: ${error}`
      });
    } finally {
      setIsSubmitting(false);
    }
  };
  
  // Handle hint request
  const handleGetHint = () => {
    setHintsUsed(prev => prev + 1);
    setShowHint(true);
  };
  
  // Generate hint based on problem
  const getHint = (): string => {
    if (!currentProblem) return "No hint available";
    
    const hints = {
      "Two Sum": [
        "💡 Try using a hash map to store numbers you've seen and their indices",
        "🔍 For each number, check if (target - number) exists in your hash map",
        "⚡ This approach gives you O(n) time complexity instead of O(n²)"
      ],
      "Valid Parentheses": [
        "💡 Use a stack data structure to keep track of opening brackets",
        "🔍 When you see a closing bracket, check if it matches the most recent opening bracket",
        "⚡ The string is valid if the stack is empty at the end"
      ]
    };
    
    const problemHints = hints[currentProblem.title as keyof typeof hints] || [
      "💡 Break the problem down into smaller steps",
      "🔍 Think about what data structures might help",
      "⚡ Consider the time and space complexity of your approach"
    ];
    
    return problemHints[Math.min(hintsUsed - 1, problemHints.length - 1)];
  };
  
  // Render skill level indicator
  const renderSkillLevel = (level: number) => {
    const percentage = Math.min(100, (level / 3.0) * 100);
    const levelName = level <= 1.3 ? 'Beginner' : level <= 2.3 ? 'Intermediate' : 'Advanced';
    
    return (
      <div className="space-y-2">
        <div className="flex justify-between items-center">
          <span className="text-sm font-medium">Skill Level</span>
          <span className="px-2 py-1 bg-gray-100 text-xs rounded border">
            {levelName} ({level.toFixed(1)})
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div 
            className="bg-blue-600 h-2 rounded-full transition-all duration-500"
            style={{ width: `${percentage}%` }}
          ></div>
        </div>
      </div>
    );
  };
  
  // Render problem examples
  const renderExamples = (examples: Problem['examples']) => {
    return (
      <div className="space-y-4">
        {examples.map((example, index) => (
          <div key={index} className="bg-slate-50 border rounded-lg p-4">
            <div className="space-y-2">
              <div>
                <h5 className="font-medium text-sm text-slate-700 mb-1">Input:</h5>
                <code className="text-sm bg-slate-100 px-2 py-1 rounded border block">{example.input}</code>
              </div>
              <div>
                <h5 className="font-medium text-sm text-slate-700 mb-1">Output:</h5>
                <code className="text-sm bg-slate-100 px-2 py-1 rounded border block">{example.output}</code>
              </div>
              {example.explanation && (
                <div>
                  <h5 className="font-medium text-sm text-slate-700 mb-1">Explanation:</h5>
                  <p className="text-sm text-slate-600">{example.explanation}</p>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    );
  };
  
  if (isLoading || !currentProblem) {
    return (
      <div className={`h-full flex items-center justify-center ${className}`}>
        <div className="text-center space-y-4">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
          <p className="text-slate-600">Loading adaptive problem...</p>
        </div>
      </div>
    );
  }
  
  return (
    <div className={`h-full flex flex-col ${className}`}>
      {/* Header with Problem Info and Controls */}
      <div className="border-b bg-slate-50 p-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <h1 className="text-xl font-bold text-slate-900">{currentProblem.title}</h1>
            <span className={`px-2 py-1 text-xs rounded border font-medium ${difficultyColors[currentProblem.difficulty]}`}>
              {currentProblem.difficulty === 'Easy' && <CheckCircle className="w-3 h-3 inline mr-1" />}
              {currentProblem.difficulty === 'Medium' && <Target className="w-3 h-3 inline mr-1" />}
              {currentProblem.difficulty === 'Hard' && <Trophy className="w-3 h-3 inline mr-1" />}
              {currentProblem.difficulty}
            </span>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-sm">
              <Timer className="w-4 h-4" />
              <span className="font-mono">{formatTime(getSessionTime())}</span>
            </div>
            <div className="flex gap-2">
              <button
                onClick={handleGetHint}
                className="px-3 py-2 bg-yellow-50 hover:bg-yellow-100 border border-yellow-200 text-yellow-800 rounded-md text-sm font-medium flex items-center gap-2"
              >
                <Lightbulb className="w-4 h-4" />
                Hint ({hintsUsed})
              </button>
              <button
                onClick={handleRunCode}
                disabled={isRunning || isSubmitting}
                className="px-4 py-2 bg-blue-50 hover:bg-blue-100 border border-blue-200 text-blue-800 rounded-md text-sm font-medium flex items-center gap-2 disabled:opacity-50"
              >
                {isRunning ? (
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                ) : (
                  <Play className="w-4 h-4" />
                )}
                Run ({runs})
              </button>
              <button
                onClick={handleSubmitSolution}
                disabled={isSubmitting || isRunning}
                className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-md text-sm font-medium flex items-center gap-2 disabled:opacity-50"
              >
                {isSubmitting ? (
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                ) : (
                  <Send className="w-4 h-4" />
                )}
                Submit ({attempts})
              </button>
            </div>
          </div>
        </div>
        
        {/* Topic Tags */}
        <div className="flex flex-wrap gap-2 mb-3">
          {currentProblem.topics.map((topic) => (
            <span key={topic} className="px-2 py-1 bg-gray-100 text-gray-800 text-xs rounded border flex items-center gap-1">
              <BookOpen className="w-3 h-3" />
              {topic}
            </span>
          ))}
        </div>
        
        {/* Appreciation Message */}
        {appreciationMessage && (
          <div className="mb-3 p-3 bg-green-50 border border-green-200 rounded-lg flex items-start gap-2">
            <Star className="w-4 h-4 text-green-600 mt-0.5 flex-shrink-0" />
            <p className="text-green-800 text-sm">{appreciationMessage}</p>
          </div>
        )}
        
        {/* Hint Display */}
        {showHint && (
          <div className="mb-3 p-3 bg-yellow-50 border border-yellow-200 rounded-lg flex items-start gap-2">
            <Lightbulb className="w-4 h-4 text-yellow-600 mt-0.5 flex-shrink-0" />
            <p className="text-yellow-800 text-sm">{getHint()}</p>
          </div>
        )}
        
        {/* Last Submission Result */}
        {lastSubmissionResult && (
          <div className={`mb-3 p-3 border rounded-lg flex items-start gap-2 ${
            lastSubmissionResult.is_solved 
              ? 'bg-green-50 border-green-200' 
              : 'bg-red-50 border-red-200'
          }`}>
            {lastSubmissionResult.is_solved ? (
              <CheckCircle className="w-4 h-4 text-green-600 mt-0.5 flex-shrink-0" />
            ) : (
              <XCircle className="w-4 h-4 text-red-600 mt-0.5 flex-shrink-0" />
            )}
            <p className={`text-sm ${lastSubmissionResult.is_solved ? 'text-green-800' : 'text-red-800'}`}>
              {lastSubmissionResult.feedback_message}
            </p>
          </div>
        )}
      </div>
      
      {/* Main Content Area */}
      <div className="flex-1 flex">
        {/* Left Panel - Problem Description */}
        <div className="w-1/2 border-r flex flex-col">
          {/* Tab Navigation */}
          <div className="border-b bg-white">
            <div className="flex">
              <button
                onClick={() => setActiveTab('problem')}
                className={`px-4 py-2 text-sm font-medium border-b-2 ${
                  activeTab === 'problem' 
                    ? 'border-blue-600 text-blue-600 bg-blue-50' 
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                Problem
              </button>
              <button
                onClick={() => setActiveTab('stats')}
                className={`px-4 py-2 text-sm font-medium border-b-2 ${
                  activeTab === 'stats' 
                    ? 'border-blue-600 text-blue-600 bg-blue-50' 
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                Progress
              </button>
            </div>
          </div>
          
          {/* Tab Content */}
          <div className="flex-1 p-4 overflow-auto">
            {activeTab === 'problem' && (
              <div className="space-y-6">
                {/* Problem Statement */}
                <div>
                  <h3 className="font-semibold text-lg mb-3">Problem Statement</h3>
                  <div className="prose prose-sm max-w-none">
                    <p className="text-slate-700 leading-relaxed whitespace-pre-wrap">
                      {currentProblem.problem_statement}
                    </p>
                  </div>
                </div>
                
                <hr className="border-gray-200" />
                
                {/* Examples */}
                {currentProblem.examples.length > 0 && (
                  <div>
                    <h3 className="font-semibold text-lg mb-3">Examples</h3>
                    {renderExamples(currentProblem.examples)}
                  </div>
                )}
                
                {/* Constraints */}
                {currentProblem.constraints.length > 0 && (
                  <>
                    <hr className="border-gray-200" />
                    <div>
                      <h3 className="font-semibold text-lg mb-3">Constraints</h3>
                      <ul className="space-y-1 text-sm text-slate-700">
                        {currentProblem.constraints.map((constraint, index) => (
                          <li key={index} className="flex items-start gap-2">
                            <span className="text-slate-400 mt-1">•</span>
                            <span>{constraint}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </>
                )}
              </div>
            )}
            
            {activeTab === 'stats' && (
              <div className="space-y-6">
                {userStats ? (
                  <>
                    {/* Performance Card */}
                    <div className="border rounded-lg p-4">
                      <h3 className="font-semibold text-lg mb-3 flex items-center gap-2">
                        <Brain className="w-5 h-5" />
                        Performance
                      </h3>
                      {renderSkillLevel(userStats.skill_level)}
                      <div className="mt-4 grid grid-cols-2 gap-4">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-blue-600">{userStats.problems_solved}</div>
                          <div className="text-xs text-slate-500">Problems Solved</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-green-600">
                            {Math.round((userStats.skill_level / 3.0) * 100)}%
                          </div>
                          <div className="text-xs text-slate-500">Skill Progress</div>
                        </div>
                      </div>
                    </div>
                    
                    {/* Current Session */}
                    <div className="border rounded-lg p-4">
                      <h3 className="font-semibold text-lg mb-3 flex items-center gap-2">
                        <Zap className="w-5 h-5 text-orange-600" />
                        Current Session
                      </h3>
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div className="flex justify-between">
                          <span>Time:</span>
                          <span className="font-mono">{formatTime(getSessionTime())}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Attempts:</span>
                          <span className="font-bold text-blue-600">{attempts}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Runs:</span>
                          <span className="font-bold text-green-600">{runs}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Hints:</span>
                          <span className="font-bold text-yellow-600">{hintsUsed}</span>
                        </div>
                      </div>
                    </div>
                    
                    {/* Strong Topics */}
                    {userStats.strong_topics.length > 0 && (
                      <div className="border rounded-lg p-4">
                        <h3 className="font-semibold text-lg mb-3 flex items-center gap-2">
                          <TrendingUp className="w-5 h-5 text-green-600" />
                          Strong Topics
                        </h3>
                        <div className="flex flex-wrap gap-2">
                          {userStats.strong_topics.map((topic) => (
                            <span key={topic} className="px-2 py-1 bg-green-100 text-green-800 text-xs rounded border">
                              {topic}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {/* Recent Activity */}
                    {userStats.problem_history && userStats.problem_history.length > 0 && (
                      <div className="border rounded-lg p-4">
                        <h3 className="font-semibold text-lg mb-3">Recent Activity</h3>
                        <div className="space-y-2">
                          {userStats.problem_history.slice(-5).map((session, index) => (
                            <div key={index} className="flex items-center justify-between p-2 bg-slate-50 rounded border">
                              <div className="flex items-center gap-2">
                                {session.is_solved ? (
                                  <CheckCircle className="w-4 h-4 text-green-600" />
                                ) : (
                                  <XCircle className="w-4 h-4 text-red-600" />
                                )}
                                <span className="text-sm">{session.problem_id}</span>
                                <span className="px-1 py-0.5 bg-gray-100 text-xs rounded border">
                                  {session.difficulty}
                                </span>
                              </div>
                              <div className="text-xs text-slate-500">
                                {formatTime(session.time_taken || 0)}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="text-center text-slate-500 py-8">
                    <Brain className="w-12 h-12 text-slate-300 mx-auto mb-4" />
                    <p>Start solving problems to see your progress!</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
        
        {/* Right Panel - Code Editor & Results */}
        <div className="w-1/2 flex flex-col">
          {/* Code Editor */}
          <div className="flex-1 border-b">
            <div className="h-full">
              <MonacoEditor
                value={userCode}
                onChange={setUserCode}
                language={language}
                className="h-full"
              />
            </div>
          </div>
          
          {/* Execution Results */}
          <div className="h-48 p-4 overflow-hidden bg-slate-50">
            <div className="flex items-center gap-2 mb-3">
              <Code className="w-4 h-4" />
              <h4 className="font-medium">Output</h4>
              {executionResult && (
                <span className={`px-2 py-1 text-xs rounded border ${
                  executionResult.success 
                    ? 'bg-green-100 text-green-800 border-green-200' 
                    : 'bg-red-100 text-red-800 border-red-200'
                }`}>
                  {executionResult.success ? "Success" : "Error"}
                </span>
              )}
            </div>
            
            <div className="h-32 overflow-auto">
              {executionResult ? (
                <div className="space-y-2">
                  {executionResult.data?.execution_output.output && (
                    <div>
                      <h5 className="text-xs font-medium text-slate-700 mb-1">Output:</h5>
                      <pre className="text-xs bg-white p-2 rounded border font-mono overflow-x-auto">
                        {executionResult.data.execution_output.output}
                      </pre>
                    </div>
                  )}
                  
                  {executionResult.data?.execution_output.error && (
                    <div>
                      <h5 className="text-xs font-medium text-red-700 mb-1">Error:</h5>
                      <pre className="text-xs bg-red-50 p-2 rounded border border-red-200 font-mono text-red-800 overflow-x-auto">
                        {executionResult.data.execution_output.error}
                      </pre>
                    </div>
                  )}
                  
                  {executionResult.data?.execution_output && (
                    <div className="flex gap-4 text-xs text-slate-600">
                      <span>Time: {executionResult.data.execution_output.execution_time?.toFixed(3) || 0}s</span>
                      <span>Memory: {executionResult.data.execution_output.peak_memory_formatted || 'N/A'}</span>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center text-slate-500 mt-8">
                  <div className="space-y-2">
                    <p>Ready to code! 🚀</p>
                    <div className="text-xs text-slate-400">
                      <p>• Click "Run" to test your code</p>
                      <p>• Click "Submit" when ready</p>
                      <p>• Use "Hint" if you need help</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};