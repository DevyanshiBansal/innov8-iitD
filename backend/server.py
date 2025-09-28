"""
Simplified FastAPI Backend Server for Adaptive Code Learning

Core Features:
1. Code execution with time/space complexity analysis
2. Adaptive question generation using knowledge base
3. Basic chat functionality
4. Mathematical complexity calculation

Removed features: Voice assistants, professional interview system, dual mode system
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
import tempfile
import uuid
import json
import asyncio
import time
from pathlib import Path
import sys

# Add the testing_repo directory to path to import executor_local
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "testing_repo"))
from executor_local import execute_user_code_local, analyze_time_complexity, format_memory_usage

# Import core modules: solution executor, chatbot, and enhanced adaptive system  
from solution_executor import execute_solution_code, get_language_boilerplate, get_supported_languages
from chatbot import send_chat_message
from enhanced_adaptive_qid import AdaptiveQuestionEngine, UserProfile, ProblemSession

app = FastAPI(
    title="Simplified Code Editor Backend",
    description="Backend for adaptive code learning with complexity analysis",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
adaptive_engine = AdaptiveQuestionEngine()

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class CodeExecutionRequest(BaseModel):
    code: str = Field(..., description="The code to execute")
    language: str = Field(default="python", description="Programming language")
    test_sizes: List[int] = Field(default=[10, 100, 500, 1000], description="Input sizes for complexity analysis")

class CodeExecutionResponse(BaseModel):
    success: bool = Field(description="Whether execution was successful")
    output: str = Field(description="Code execution output")
    error: Optional[str] = Field(description="Error message if execution failed")
    n_vs_time: Dict[str, float] = Field(description="Input size vs execution time mapping")
    n_vs_space: Dict[str, int] = Field(description="Input size vs memory usage mapping") 
    complexity_analysis: Dict[str, Any] = Field(description="Detected time/space complexity")
    frontend_display: Optional[str] = Field(description="Formatted display for frontend")

class ChatRequest(BaseModel):
    message: str = Field(..., description="User's chat message")
    session_id: Optional[str] = Field(description="Chat session identifier")

class ChatResponse(BaseModel):
    response: str = Field(description="AI assistant's response")
    session_id: str = Field(description="Session identifier")

class AdaptiveQuestionRequest(BaseModel):
    user_id: str = Field(default="default_user", description="User identifier")
    current_problem_id: Optional[str] = Field(description="ID of current/completed problem")
    time_taken: Optional[int] = Field(description="Time taken to solve (seconds)")
    is_solved: bool = Field(default=False, description="Whether problem was solved successfully")
    attempts: int = Field(default=0, description="Number of submission attempts")
    runs: int = Field(default=0, description="Number of code runs")

class AdaptiveQuestionResponse(BaseModel):
    success: bool = Field(description="Whether request was successful")
    problem_data: Dict[str, Any] = Field(description="Generated problem data")
    user_stats: Dict[str, Any] = Field(description="Updated user statistics")
    appreciation_message: Optional[str] = Field(description="Motivational message for user")

# =============================================================================
# HEALTH AND BASIC ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Simplified Code Editor Backend API",
        "version": "2.0.0",
        "features": [
            "Code execution with complexity analysis",
            "Adaptive question generation", 
            "Basic chat functionality"
        ],
        "endpoints": {
            "health": "/health",
            "execute": "/api/execute",
            "chat": "/api/chat",
            "adaptive_questions": "/api/adaptive/next-question"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "adaptive_engine": "active"
    }

# =============================================================================
# CODE EXECUTION ENDPOINTS
# =============================================================================

@app.post("/api/execute", response_model=CodeExecutionResponse)
async def execute_code(request: CodeExecutionRequest):
    """
    Execute code and analyze time/space complexity
    
    Features:
    - Executes code with multiple test sizes
    - Analyzes time and space complexity patterns
    - Returns formatted results for frontend display
    """
    try:
        # Execute code with complexity analysis
        result = await execute_user_code_local(
            code=request.code,
            language=request.language,
            test_sizes=request.test_sizes
        )
        
        if not result["success"]:
            return CodeExecutionResponse(
                success=False,
                output="",
                error=result.get("error", "Unknown execution error"),
                n_vs_time={},
                n_vs_space={},
                complexity_analysis={},
                frontend_display=f"❌ Execution failed: {result.get('error', 'Unknown error')}"
            )
        
        # Format the response
        n_vs_time = {str(k): v for k, v in result["n_vs_time"].items()}
        n_vs_space = {str(k): v for k, v in result["n_vs_space"].items()}
        
        # Create frontend display
        frontend_display = f"""✅ Code executed successfully!

📊 Performance Analysis:
{format_memory_usage(result["n_vs_space"])}

⏱️  Time Complexity: {result["complexity_analysis"].get("time_complexity", "Unknown")}
💾 Space Complexity: {result["complexity_analysis"].get("space_complexity", "Unknown")}

📈 Execution Times:
""" + "\n".join([f"n={size}: {time:.4f}s" for size, time in result["n_vs_time"].items()])

        return CodeExecutionResponse(
            success=True,
            output=result["output"],
            error=None,
            n_vs_time=n_vs_time,
            n_vs_space=n_vs_space,
            complexity_analysis=result["complexity_analysis"],
            frontend_display=frontend_display
        )
        
    except Exception as e:
        return CodeExecutionResponse(
            success=False,
            output="",
            error=str(e),
            n_vs_time={},
            n_vs_space={},
            complexity_analysis={},
            frontend_display=f"❌ System error: {str(e)}"
        )

# =============================================================================
# CHAT ENDPOINTS  
# =============================================================================

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_ai(chat_request: ChatRequest):
    """
    Simple chat endpoint for programming assistance
    """
    try:
        response = send_chat_message(chat_request.message)
        session_id = chat_request.session_id or str(uuid.uuid4())
        
        return ChatResponse(
            response=response,
            session_id=session_id
        )
        
    except Exception as e:
        return ChatResponse(
            response=f"Sorry, I encountered an error: {str(e)}",
            session_id=chat_request.session_id or str(uuid.uuid4())
        )

# =============================================================================
# ADAPTIVE QUESTION GENERATION ENDPOINTS
# =============================================================================

@app.post("/api/adaptive/next-question", response_model=AdaptiveQuestionResponse)
async def get_next_adaptive_question(request: AdaptiveQuestionRequest):
    """
    Generate next problem based on user performance and adaptive algorithm
    
    Uses knowledge base (ques_data.json) to select appropriate questions
    based on user's skill level, performance history, and learning patterns
    """
    try:
        # Generate next problem using adaptive engine
        result = adaptive_engine.generate_problem_id(
            user_id=request.user_id,
            current_problem_id=request.current_problem_id,
            time_taken=request.time_taken,
            is_solved=request.is_solved,
            attempts=request.attempts,
            runs=request.runs
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("message", "Failed to generate problem"))
        
        return AdaptiveQuestionResponse(
            success=True,
            problem_data=result["problem_data"],
            user_stats=result["user_stats"],
            appreciation_message=result.get("appreciation_message")
        )
        
    except Exception as e:
        # Return fallback problem in case of error
        fallback_problem = {
            "id": "two_sum",
            "title": "Two Sum",
            "problem_statement": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
            "examples": [
                {
                    "input": "nums = [2,7,11,15], target = 9",
                    "output": "[0,1]",
                    "explanation": "Because nums[0] + nums[1] == 9, we return [0, 1]."
                }
            ],
            "constraints": [
                "2 <= nums.length <= 10^4",
                "-10^9 <= nums[i] <= 10^9",
                "Only one valid answer exists."
            ],
            "difficulty": "Easy",
            "topics": ["Array", "Hash Table"],
            "expected_time": 900
        }
        
        return AdaptiveQuestionResponse(
            success=True,
            problem_data=fallback_problem,
            user_stats={
                "skill_level": 1.0,
                "problems_solved": 0,
                "strong_topics": [],
                "weak_topics": []
            },
            appreciation_message="Let's start with this classic problem!"
        )

@app.get("/api/adaptive/user-stats/{user_id}")
async def get_user_stats(user_id: str):
    """Get user's learning statistics and progress"""
    try:
        stats = adaptive_engine.get_user_profile(user_id)
        return {
            "success": True,
            "user_id": user_id,
            "stats": stats
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "stats": {
                "skill_level": 1.0,
                "problems_solved": 0,
                "strong_topics": [],
                "weak_topics": []
            }
        }

@app.get("/api/adaptive/available-problems")
async def get_available_problems():
    """Get list of all available problems in knowledge base"""
    try:
        problems = adaptive_engine.get_available_problems()
        return {
            "success": True,
            "problems": problems,
            "total_count": len(problems)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "problems": [],
            "total_count": 0
        }

# =============================================================================
# MAIN APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Simplified Code Editor Backend...")
    print("📋 Features: Adaptive Questions + Complexity Analysis")
    print("🔗 Frontend URL: http://localhost:3000")
    print("📖 API Docs: http://localhost:8000/docs")
    
    try:
        uvicorn.run(
            app, 
            host="localhost", 
            port=8000, 
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")