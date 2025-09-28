"""
FastAPI Backend Server for Code Editor with Execution and Analysis

This server provides endpoints for:
1. Code execution with output capture
2. Time complexity analysis (n vs time)
3. Memory usage analysis (n vs peak memory)
4. Error analysis and reporting

Integrated with executor_local.py for local code execution.
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
    title="Code Editor Backend",
    description="FastAPI backend for code execution, analysis, and error handling",
    version="1.0.0"
)

# CORS configuration for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",  # In case frontend runs on different port
        "http://127.0.0.1:3001"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Pydantic models for request/response structure
class CodeExecutionRequest(BaseModel):
    code: str = Field(..., description="The code to execute")
    language: str = Field(default="python", description="Programming language (python, cpp, c)")
    test_sizes: Optional[List[int]] = Field(default=[10, 100, 500, 1000, 10000], description="Input sizes for complexity analysis")

class SolutionExecutionRequest(BaseModel):
    code: str = Field(..., description="The solution code to execute")
    language: str = Field(default="python", description="Programming language")
    test_input: Optional[str] = Field(default="", description="Test input for the solution function")

class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message to the chatbot")
    code: Optional[str] = Field(default=None, description="Optional code context")
    language: Optional[str] = Field(default=None, description="Programming language")

class ExecutionOutput(BaseModel):
    output: str = Field(description="Standard output from code execution")
    error: str = Field(description="Error output if any")
    exit_code: int = Field(description="Exit code of the process")
    execution_time: float = Field(description="Time taken for execution in seconds")
    peak_memory: int = Field(description="Peak memory usage in bytes")
    peak_memory_formatted: str = Field(description="Human-readable memory usage")

class TimeComplexityAnalysis(BaseModel):
    time_measurements: Dict[int, float] = Field(description="Dictionary of input size vs execution time")
    memory_measurements: Dict[int, int] = Field(description="Dictionary of input size vs peak memory")

class AnalysisResult(BaseModel):
    execution_output: ExecutionOutput
    time_complexity: Optional[TimeComplexityAnalysis] = None
    error_analysis: str = Field(description="Error analysis summary")

class CodeExecutionResponse(BaseModel):
    success: bool = Field(description="Whether the operation was successful")
    data: Optional[AnalysisResult] = None
    message: str = Field(description="Success or error message")

# Adaptive Question System Models
class AdaptiveQuestionRequest(BaseModel):
    current_problem_id: Optional[str] = Field(default=None, description="Current problem ID")
    time_taken: Optional[float] = Field(default=None, description="Time taken to solve current problem (seconds)")
    is_solved: bool = Field(default=False, description="Whether current problem was solved")
    attempts: int = Field(default=1, description="Number of attempts made")
    runs: int = Field(default=0, description="Number of code runs before solving")
    user_id: Optional[str] = Field(default="default_user", description="User identifier")

class ProblemSubmissionRequest(BaseModel):
    problem_id: str = Field(..., description="Problem ID being submitted")
    code: str = Field(..., description="User's solution code")
    language: str = Field(default="python", description="Programming language")
    time_taken: float = Field(..., description="Time taken to solve (seconds)")
    attempts: int = Field(default=1, description="Number of attempts")
    runs: int = Field(default=0, description="Number of code runs")

class RunCodeRequest(BaseModel):
    problem_id: str = Field(..., description="Problem ID")
    code: str = Field(..., description="User's code to run/test")
    language: str = Field(default="python", description="Programming language")

class AdaptiveQuestionResponse(BaseModel):
    success: bool = Field(description="Whether the request was successful")
    problem_id: str = Field(description="Generated problem ID")
    problem_data: Dict[str, Any] = Field(description="Complete problem information")
    appreciation_message: Optional[str] = Field(default=None, description="Appreciation for previous performance")
    user_stats: Optional[Dict[str, Any]] = Field(default=None, description="User performance statistics")

# Global storage for temporary files and user sessions
temp_files = {}
user_profiles = {}  # Store user profiles in memory (use database in production)
active_sessions = {}  # Store active problem sessions

# Initialize the Adaptive Question Engine
try:
    adaptive_engine = AdaptiveQuestionEngine()
    print("✅ Adaptive Question Engine initialized successfully")
except Exception as e:
    print(f"⚠️ Failed to initialize Adaptive Question Engine: {e}")
    adaptive_engine = None

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Code Editor Backend API",
        "version": "1.0.0",
        "endpoints": {
            "execute": "/api/execute - Execute code with analysis",
            "health": "/api/health - Health check"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "message": "Code Editor Backend is running",
        "version": "1.0.0",
        "frontend_cors": "http://localhost:3000",
        "endpoints": {
            "execute": "/api/execute",
            "languages": "/api/supported-languages",
            "test_sizes": "/api/default-test-sizes",
            "cleanup": "/api/cleanup"
        }
    }

@app.get("/api/status")
async def get_status():
    """Get detailed server status"""
    return {
        "server": "FastAPI Code Editor Backend",
        "version": "1.0.0",
        "status": "running",
        "features": {
            "code_execution": True,
            "time_complexity_analysis": True,
            "memory_monitoring": True,
            "error_analysis": True,
            "multi_language_support": True
        },
        "supported_languages": ["python", "cpp", "c"],
        "frontend_integration": "http://localhost:3000",
        "cors_enabled": True
    }

def cleanup_temp_file(file_path: str):
    """Clean up temporary file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        # Also clean up any related temp files
        base_name = file_path.replace('.py', '')
        for size in [10, 100, 500, 1000, 10000]:
            temp_variant = f"{base_name}_temp_{size}.py"
            if os.path.exists(temp_variant):
                os.remove(temp_variant)
    except Exception as e:
        print(f"Warning: Could not cleanup temp file {file_path}: {e}")

def create_temporary_file(code: str, language: str) -> str:
    """Create a temporary file with the provided code"""
    # Determine file extension
    ext_map = {
        "python": ".py",
        "cpp": ".cpp",
        "c": ".c"
    }
    
    if language not in ext_map:
        raise ValueError(f"Unsupported language: {language}")
    
    # Create temporary file
    temp_id = str(uuid.uuid4())
    temp_dir = tempfile.gettempdir()
    temp_filename = f"code_editor_{temp_id}{ext_map[language]}"
    temp_path = os.path.join(temp_dir, temp_filename)
    
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        # Store in global dict for cleanup
        temp_files[temp_id] = temp_path
        
        return temp_path
    except Exception as e:
        raise ValueError(f"Could not create temporary file: {str(e)}")

def analyze_errors(error_output: str, exit_code: int) -> str:
    """Analyze errors and provide meaningful feedback"""
    if exit_code == 0 and not error_output.strip():
        return "No Errors"
    
    analysis = []
    
    if exit_code != 0:
        analysis.append(f"Process exited with code {exit_code}")
    
    if error_output:
        error_lines = error_output.strip().split('\n')
        
        # Common error patterns
        error_patterns = {
            'SyntaxError': 'Syntax Error: Check for missing parentheses, colons, or indentation issues',
            'IndentationError': 'Indentation Error: Check your code indentation',
            'NameError': 'Name Error: Variable or function not defined',
            'TypeError': 'Type Error: Check data types and operations',
            'ValueError': 'Value Error: Invalid value for operation',
            'ImportError': 'Import Error: Module not found or import issue',
            'ModuleNotFoundError': 'Module Error: Required module not installed',
            'ZeroDivisionError': 'Division by Zero: Check mathematical operations',
            'IndexError': 'Index Error: List/array index out of range',
            'KeyError': 'Key Error: Dictionary key not found',
            'AttributeError': 'Attribute Error: Object attribute not found',
            'TimeoutExpired': 'Timeout: Code execution took too long (possible infinite loop)',
            'FileNotFoundError': 'File Error: Required file not found',
            'PermissionError': 'Permission Error: Insufficient permissions'
        }
        
        for line in error_lines:
            for error_type, description in error_patterns.items():
                if error_type in line:
                    analysis.append(description)
                    break
        
        # If no specific pattern matched, include raw error
        if not analysis:
            analysis.append(f"Error: {error_output[:200]}...")
    
    return "; ".join(analysis) if analysis else "Unknown error occurred"

def detect_complexity_from_data(n_vs_time: dict) -> str:
    """Simple complexity detection from timing data"""
    if not n_vs_time or len(n_vs_time) < 2:
        return "Unknown"
    
    try:
        # Convert to sorted lists
        n_values = sorted([int(k) for k in n_vs_time.keys()])
        time_values = [n_vs_time[str(n)] for n in n_values]
        
        if len(n_values) < 3:
            return "Insufficient data"
        
        # Simple growth rate analysis
        ratios = []
        for i in range(1, len(n_values)):
            if time_values[i-1] > 0:
                time_ratio = time_values[i] / time_values[i-1]
                n_ratio = n_values[i] / n_values[i-1]
                ratios.append(time_ratio / n_ratio)
        
        if not ratios:
            return "Unknown"
            
        avg_ratio = sum(ratios) / len(ratios)
        
        if avg_ratio < 1.2:
            return "O(1) - Constant"
        elif avg_ratio < 2.0:
            return "O(n) - Linear"
        elif avg_ratio < 5.0:
            return "O(n log n) - Log Linear"
        else:
            return "O(n²) or higher - Quadratic+"
            
    except Exception:
        return "Analysis Error"

@app.post("/api/execute")
async def execute_code(request: CodeExecutionRequest, background_tasks: BackgroundTasks):
    """
    Execute code and return analysis results
    
    This endpoint:
    1. Creates a temporary .py/.cpp/.c file based on language
    2. Executes the code using executor_local.py 
    3. Returns output to console display
    4. Returns analysis for chat panel:
       - n vs time dictionary (sizes: 10,100,500,1000,10000)
       - n vs memory/space dictionary  
       - Error analysis and description
    """
    print(f"🔄 Received code execution request:")
    print(f"   Language: {request.language}")
    print(f"   Code length: {len(request.code)} characters")
    print(f"   Test sizes: {request.test_sizes}")
    
    try:
        # Validate language
        supported_languages = ["python", "cpp", "c"]
        if request.language not in supported_languages:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language: {request.language}. Supported: {supported_languages}"
            )
        
        # Create temporary file
        temp_file_path = create_temporary_file(request.code, request.language)
        print(f"📁 Created temporary {request.language} file: {temp_file_path}")
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, temp_file_path)
        
        # Execute code with analysis
        print(f"⚡ Executing {request.language} code...")
        execution_result = execute_user_code_local(
            temp_file_path, 
            request.language, 
            request.test_sizes
        )
        print(f"✅ Execution completed. Exit code: {execution_result.get('exit_code', 'N/A')}")
        
        # Create execution output
        execution_output = ExecutionOutput(
            output=execution_result.get("output", ""),
            error=execution_result.get("error", ""),
            exit_code=execution_result.get("exit_code", -1),
            execution_time=execution_result.get("execution_time", 0),
            peak_memory=execution_result.get("peak_memory", 0),
            peak_memory_formatted=execution_result.get("peak_memory_formatted", "0 B")
        )
        
        # Extract time complexity analysis
        time_complexity = None
        if execution_result.get("time_complexity") and execution_result["time_complexity"]:
            complexity_data = execution_result["time_complexity"]
            time_complexity = TimeComplexityAnalysis(
                time_measurements=complexity_data.get("time_measurements", {}),
                memory_measurements=complexity_data.get("memory_measurements", {})
            )
        
        # Analyze errors
        error_analysis = analyze_errors(
            execution_result.get("error", ""),
            execution_result.get("exit_code", -1)
        )
        
        print(f"📊 Analysis Results:")
        print(f"   Console Output: {len(execution_result.get('output', ''))} chars")
        print(f"   Time Complexity Data: {bool(time_complexity)}")
        print(f"   Error Analysis: {error_analysis}")
        
        # Create analysis result with all the required information
        analysis_result = AnalysisResult(
            execution_output=execution_output,
            time_complexity=time_complexity,
            error_analysis=error_analysis
        )
        
        response_data = {
            "success": True,
            "data": analysis_result,
            "message": "Code executed successfully",
            # Data specifically formatted for frontend components
            "frontend_display": {
                # For Console/Output Display
                "console_output": execution_result.get("output", ""),
                "console_error": execution_result.get("error", ""),
                
                # For Chat Panel Display - Exactly as requested
                "n_vs_time": time_complexity.time_measurements if time_complexity else {},
                "n_vs_space": time_complexity.memory_measurements if time_complexity else {},  # Changed from n_vs_memory
                "error_analysis": error_analysis,
                
                # Additional execution summary
                "execution_summary": {
                    "language": request.language,
                    "file_created": temp_file_path.split("\\")[-1] if "\\" in temp_file_path else temp_file_path.split("/")[-1],
                    "exit_code": execution_result.get("exit_code", -1),
                    "execution_time": execution_result.get("execution_time", 0),
                    "peak_memory": execution_result.get("peak_memory_formatted", "0 B"),
                    "success": execution_result.get("exit_code", -1) == 0,
                    "test_sizes_used": request.test_sizes
                }
            }
        }
        
        print(f"🎯 Response prepared for frontend:")
        print(f"   Console output ready: {bool(response_data['frontend_display']['console_output'])}")
        print(f"   n_vs_time dict: {len(response_data['frontend_display']['n_vs_time'])} entries")
        print(f"   n_vs_space dict: {len(response_data['frontend_display']['n_vs_space'])} entries")
        
        # Update global execution result for voice interviewer monitoring
        global latest_execution_result
        latest_execution_result.update({
            "timestamp": f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
            "code_executed": True,
            "execution_successful": execution_result.get("exit_code", -1) == 0,
            "n_vs_time": response_data['frontend_display']['n_vs_time'],
            "n_vs_space": response_data['frontend_display']['n_vs_space'],
            "error_analysis": error_analysis,
            "output": execution_result.get("output", ""),
            "complexity_detected": detect_complexity_from_data(response_data['frontend_display']['n_vs_time']),
            "performance_notes": f"Execution time: {execution_result.get('execution_time', 0):.3f}s, Peak memory: {execution_result.get('peak_memory_formatted', '0 B')}"
        })
        
        return response_data
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Backend execution error: {str(e)}")
        return {
            "success": False,
            "data": None,
            "message": f"Execution failed: {str(e)}",
            "frontend_display": {
                "console_output": "",
                "console_error": f"Backend Error: {str(e)}",
                "n_vs_time": {},
                "n_vs_space": {},
                "error_analysis": f"Backend Error: {str(e)}",
                "execution_summary": {
                    "language": request.language if 'request' in locals() else "unknown",
                    "file_created": "error",
                    "exit_code": -1,
                    "execution_time": 0,
                    "peak_memory": "0 B",
                    "success": False,
                    "test_sizes_used": []
                }
            }
        }

@app.get("/api/supported-languages")
async def get_supported_languages():
    """Get list of supported programming languages"""
    return {
        "languages": [
            {
                "id": "python", 
                "name": "Python", 
                "extension": ".py",
                "example": "def main(n):\n    return sum(range(n))\n\nresult = main(10)\nprint(f'Result: {result}')"
            },
            {
                "id": "cpp", 
                "name": "C++", 
                "extension": ".cpp",
                "example": "#include <iostream>\nusing namespace std;\n\nint main() {\n    int n = 10;\n    int sum = 0;\n    for(int i = 0; i < n; i++) {\n        sum += i;\n    }\n    cout << \"Result: \" << sum << endl;\n    return 0;\n}"
            },
            {
                "id": "c", 
                "name": "C", 
                "extension": ".c",
                "example": "#include <stdio.h>\n\nint main() {\n    int n = 10;\n    int sum = 0;\n    for(int i = 0; i < n; i++) {\n        sum += i;\n    }\n    printf(\"Result: %d\\n\", sum);\n    return 0;\n}"
            }
        ]
    }

@app.get("/api/default-test-sizes")
async def get_default_test_sizes():
    """Get default test sizes for complexity analysis"""
    return {
        "default_sizes": [10, 100, 500, 1000, 10000],
        "recommended_sizes": {
            "small": [10, 50, 100],
            "medium": [10, 100, 500, 1000],
            "large": [10, 100, 500, 1000, 5000, 10000],
            "custom": "You can specify any positive integers"
        }
    }

@app.delete("/api/cleanup")
async def cleanup_all_temp_files():
    """Cleanup all temporary files (for testing/maintenance)"""
    cleaned = 0
    errors = []
    
    for temp_id, file_path in list(temp_files.items()):
        try:
            cleanup_temp_file(file_path)
            del temp_files[temp_id]
            cleaned += 1
        except Exception as e:
            errors.append(f"Could not cleanup {file_path}: {str(e)}")
    
    return {
        "cleaned_files": cleaned,
        "errors": errors,
        "message": f"Cleaned up {cleaned} temporary files"
    }

@app.post("/api/execute-solution")
async def execute_solution(request: SolutionExecutionRequest, background_tasks: BackgroundTasks):
    """
    Execute code in LeetCode solution() format
    
    This endpoint enforces that users implement a solution() function
    and provides proper test input handling.
    """
    print(f"🔄 Received solution execution request:")
    print(f"   Language: {request.language}")
    print(f"   Code length: {len(request.code)} characters")
    print(f"   Test input: {request.test_input[:100]}..." if request.test_input and len(request.test_input) > 100 else f"   Test input: {request.test_input}")
    
    try:
        # Execute using solution executor
        result = execute_solution_code(
            code=request.code,
            language=request.language,
            test_input=request.test_input
        )
        
        print(f"✅ Solution execution completed. Exit code: {result.get('exit_code', 'N/A')}")
        
        return {
            "success": result.get("solution_found", False),
            "message": "Solution executed successfully" if result.get("solution_found", False) else "Solution function not found",
            "data": {
                "console_output": result.get("output", ""),
                "console_error": result.get("error", ""),
                "execution_time": result.get("execution_time", 0),
                "peak_memory": result.get("peak_memory_formatted", "0 B"),
                "exit_code": result.get("exit_code", -1),
                "solution_found": result.get("solution_found", False)
            }
        }
        
    except Exception as e:
        print(f"❌ Solution execution error: {str(e)}")
        return {
            "success": False,
            "message": f"Solution execution failed: {str(e)}",
            "data": {
                "console_output": "",
                "console_error": f"Error: {str(e)}",
                "execution_time": 0,
                "peak_memory": "0 B",
                "exit_code": -1,
                "solution_found": False
            }
        }

@app.get("/api/boilerplate/{language}")
async def get_boilerplate(language: str):
    """
    Get boilerplate code for a specific language with solution() function template
    """
    try:
        boilerplate = get_language_boilerplate(language)
        return {
            "success": True,
            "language": language,
            "boilerplate": boilerplate,
            "message": f"Boilerplate code for {language}"
        }
    except Exception as e:
        return {
            "success": False,
            "language": language,
            "boilerplate": f"# Language {language} not supported",
            "message": f"Error: {str(e)}"
        }

@app.get("/api/solution-languages")
async def get_solution_supported_languages():
    """Get list of supported programming languages for solution format"""
    try:
        languages = get_supported_languages()
        return {
            "success": True,
            "languages": languages,
            "message": f"Found {len(languages)} supported languages"
        }
    except Exception as e:
        return {
            "success": False,
            "languages": [],
            "message": f"Error: {str(e)}"
        }

@app.post("/api/chat")
async def chat_with_ai(request: ChatRequest):
    """
    Chat with AI assistant about code and programming concepts
    
    This endpoint provides a simple chatbot interface that can:
    - Answer programming questions
    - Explain code snippets
    - Discuss algorithms and data structures
    - Provide coding assistance
    """
    print(f"💬 Received chat request:")
    print(f"   Message: {request.message[:100]}{'...' if len(request.message) > 100 else ''}")
    print(f"   Has code: {bool(request.code)}")
    print(f"   Language: {request.language}")
    
    try:
        # Send message to chatbot
        chat_result = send_chat_message(
            message=request.message,
            code=request.code,
            language=request.language,
            api_key="AIzaSyDxQZ8K7LjQJ6X9wV5Y2nN8PmB4RqT3uHg"  # Replace with your Gemini API key
        )
        
        print(f"🎯 Chat completed. Success: {chat_result.get('success', False)}")
        
        return {
            "success": chat_result.get("success", False),
            "message": "Chat completed successfully" if chat_result.get("success", False) else "Chat encountered issues",
            "data": {
                "response": chat_result.get("response", "I'm having trouble responding right now."),
                "message_type": chat_result.get("message_type", "assistant"),
                "has_code_context": bool(request.code),
                "language": request.language
            }
        }
        
    except Exception as e:
        print(f"❌ Chat error: {str(e)}")
        return {
            "success": False,
            "message": f"Chat failed: {str(e)}",
            "data": {
                "response": "Sorry, I'm having technical difficulties. Please try again later.",
                "message_type": "error",
                "has_code_context": False,
                "language": request.language
            }
        }

# Keep the old endpoint for backwards compatibility but redirect to chat
@app.post("/api/ai-analyze")
async def ai_analyze_code_legacy(request: ChatRequest):
    """Legacy endpoint - redirects to chat"""
    # Convert analysis request to chat message
    message = "Please analyze this code"
    if request.code:
        message += f" and explain its time and space complexity."
    
    chat_request = ChatRequest(
        message=message,
        code=request.code,
        language=request.language
    )
    
    return await chat_with_ai(chat_request)

# Dual Mode Interview Chat Endpoints

class ChatInterviewMessage(BaseModel):
    candidate_name: str = Field(..., description="Name of the candidate")
    message: str = Field(..., description="Candidate's message")
    include_ide_context: bool = Field(default=True, description="Include current IDE state in analysis")

class ChatInterviewResponse(BaseModel):
    interviewer_response: str = Field(description="Professional interviewer's response")
    guidance_type: Optional[str] = Field(description="Type of guidance provided")
    ide_context_included: bool = Field(description="Whether IDE context was analyzed")

@app.post("/api/interview/chat", response_model=ChatInterviewResponse)
async def process_chat_interview_message(request: ChatInterviewMessage):
    """
    Process candidate's chat message in interview context with IDE awareness
    
    This endpoint:
    - Analyzes candidate's message professionally
    - Includes current IDE context (code, execution results, errors)
    - Provides appropriate interviewer guidance and feedback
    - Maintains professional interview standards
    """
    try:
        dual_mode_system = get_dual_mode_system()
        
        # Get IDE context if requested
        ide_context = None
        if request.include_ide_context:
            ide_context = dual_mode_system.get_ide_context()
        
        # Process the message professionally
        interviewer_response = dual_mode_system.process_chat_message(
            request.message,
            ide_context
        )
        
        return ChatInterviewResponse(
            interviewer_response=interviewer_response,
            guidance_type="contextual",
            ide_context_included=request.include_ide_context
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process chat message: {str(e)}")

class ChatInterviewStart(BaseModel):
    candidate_name: str = Field(..., description="Name of the candidate")
    question_difficulty: str = Field(default="easy", description="Question difficulty: easy, medium, hard")

@app.post("/api/interview/chat/start")
async def start_chat_interview(request: ChatInterviewStart):
    """
    Start a professional chat-based interview session
    
    This endpoint:
    - Initializes chat interview with professional protocols  
    - Presents appropriate coding question based on difficulty
    - Provides structured professional introduction
    """
    try:
        dual_mode_system = get_dual_mode_system()
        
        # Get appropriate question (this could be enhanced to select from database)
        sample_questions = {
            "easy": {
                "title": "Two Sum Problem",
                "description": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
                "expected_complexity": {"time": "O(n)", "space": "O(n)"},
                "professional_intro": "Let's begin with a fundamental algorithmic problem to assess your problem-solving approach."
            },
            "medium": {
                "title": "Longest Substring Without Repeating Characters", 
                "description": "Given a string s, find the length of the longest substring without repeating characters.",
                "expected_complexity": {"time": "O(n)", "space": "O(min(m,n))"},
                "professional_intro": "This problem tests your understanding of sliding window techniques and optimization."
            },
            "hard": {
                "title": "Merge k Sorted Lists",
                "description": "You are given an array of k linked-lists, each sorted in ascending order. Merge all linked-lists into one sorted linked-list.",
                "expected_complexity": {"time": "O(n log k)", "space": "O(1)"},
                "professional_intro": "This advanced problem evaluates your grasp of data structures and algorithmic efficiency."
            }
        }
        
        question = sample_questions.get(request.question_difficulty, sample_questions["easy"])
        
        # Start chat interview
        interview_introduction = dual_mode_system.start_chat_interview(
            request.candidate_name,
            question
        )
        
        return {
            "success": True,
            "interview_introduction": interview_introduction,
            "question": question,
            "candidate_name": request.candidate_name,
            "mode": "chat"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start chat interview: {str(e)}")

@app.post("/api/interview/guidance")
async def provide_interview_guidance():
    """
    Provide professional guidance during interview when candidate needs help
    """
    try:
        dual_mode_system = get_dual_mode_system()
        
        # Get current IDE context to determine appropriate guidance
        ide_context = dual_mode_system.get_ide_context()
        
        guidance_type = "general"
        if ide_context:
            if ide_context.get("errors"):
                guidance_type = "error"
            elif ide_context.get("execution_results", {}).get("code_executed"):
                guidance_type = "optimization"
        
        guidance = dual_mode_system.provide_professional_guidance(guidance_type)
        
        return {
            "success": True,
            "guidance": guidance,
            "guidance_type": guidance_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to provide guidance: {str(e)}")

# =============================================================================
# ADAPTIVE QUESTION GENERATION SYSTEM ENDPOINTS
# =============================================================================

@app.post("/api/adaptive/next-question", response_model=AdaptiveQuestionResponse)
async def get_next_adaptive_question(request: AdaptiveQuestionRequest):
    """
    Generate next problem ID based on current problem performance, time taken,
    problem type, difficulty, and user's adaptive learning progress
    """
    try:
        if not adaptive_engine:
            raise HTTPException(status_code=500, detail="Adaptive Question Engine not available")
        
        # Get or create user profile
        user_id = request.user_id or "default_user"
        if user_id not in user_profiles:
            user_profiles[user_id] = UserProfile(username=user_id)
        
        user_profile = user_profiles[user_id]
        
        # Get session history for this user
        session_history = user_profile.problem_history if user_profile else []
        
        # Generate next problem
        problem_id, problem_data = adaptive_engine.generate_problem_id(
            current_problem_id=request.current_problem_id,
            time_taken=request.time_taken,
            is_solved=request.is_solved,
            attempts=request.attempts,
            user_profile=user_profile,
            session_history=session_history
        )
        
        appreciation_message = None
        if request.current_problem_id and request.time_taken:
            # Generate appreciation for previous performance
            current_problem = adaptive_engine.get_problem_data(request.current_problem_id)
            if current_problem:
                performance_metrics = adaptive_engine._analyze_performance(
                    current_problem, request.time_taken, request.is_solved, 
                    request.attempts, user_profile
                )
                appreciation_message = adaptive_engine.get_appreciation_message(
                    ProblemSession(
                        problem_id=request.current_problem_id,
                        start_time=time.time() - (request.time_taken or 0),
                        end_time=time.time(),
                        attempts=request.attempts,
                        runs=request.runs,
                        is_solved=request.is_solved,
                        time_taken=request.time_taken
                    ),
                    performance_metrics
                )
        
        # Create active session for the new problem
        active_sessions[f"{user_id}_{problem_id}"] = ProblemSession(
            problem_id=problem_id,
            start_time=time.time(),
            difficulty=problem_data.get("difficulty", "Easy"),
            topics=problem_data.get("topics", [])
        )
        
        # User statistics
        user_stats = {
            "skill_level": user_profile.current_skill_level,
            "problems_solved": user_profile.total_problems_solved,
            "strong_topics": user_profile.strong_topics,
            "weak_topics": user_profile.weak_topics
        }
        
        return AdaptiveQuestionResponse(
            success=True,
            problem_id=problem_id,
            problem_data=problem_data,
            appreciation_message=appreciation_message,
            user_stats=user_stats
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate adaptive question: {str(e)}")

@app.post("/api/adaptive/run-code")
async def run_problem_code(request: RunCodeRequest):
    """
    Run/test user's code for a specific problem (not final submission)
    This increments the run counter for adaptive analysis
    """
    try:
        user_id = "default_user"  # In production, get from auth
        session_key = f"{user_id}_{request.problem_id}"
        
        if session_key in active_sessions:
            active_sessions[session_key].runs += 1
        
        # Execute the code
        execution_request = CodeExecutionRequest(
            code=request.code,
            language=request.language
        )
        
        execution_result = await execute_code(execution_request)
        
        return {
            "success": True,
            "execution_result": execution_result,
            "message": "Code executed successfully (test run)",
            "runs": active_sessions.get(session_key, ProblemSession(problem_id="")).runs
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run code: {str(e)}")

@app.post("/api/adaptive/submit-solution")
async def submit_problem_solution(request: ProblemSubmissionRequest):
    """
    Submit final solution for a problem and analyze performance
    This completes the problem session and updates user profile
    """
    try:
        if not adaptive_engine:
            raise HTTPException(status_code=500, detail="Adaptive Question Engine not available")
        
        user_id = "default_user"  # In production, get from auth
        session_key = f"{user_id}_{request.problem_id}"
        
        # Get active session
        if session_key not in active_sessions:
            # Create session if doesn't exist
            problem_data = adaptive_engine.get_problem_data(request.problem_id)
            active_sessions[session_key] = ProblemSession(
                problem_id=request.problem_id,
                start_time=time.time() - request.time_taken,
                difficulty=problem_data.get("difficulty", "Easy") if problem_data else "Easy",
                topics=problem_data.get("topics", []) if problem_data else []
            )
        
        session = active_sessions[session_key]
        session.end_time = time.time()
        session.attempts = request.attempts
        session.runs = request.runs
        session.time_taken = request.time_taken
        
        # Execute code to check correctness
        execution_request = CodeExecutionRequest(
            code=request.code,
            language=request.language
        )
        
        execution_result = await execute_code(execution_request)
        
        # Determine if solution is correct based on execution
        session.is_solved = (execution_result.data and 
                           execution_result.data.execution_output.exit_code == 0 and 
                           not execution_result.data.execution_output.error)
        
        # Get user profile
        if user_id not in user_profiles:
            user_profiles[user_id] = UserProfile(username=user_id)
        
        user_profile = user_profiles[user_id]
        
        # Update user profile with session results
        user_profiles[user_id] = adaptive_engine.update_user_profile(user_profile, session)
        
        # Remove active session
        if session_key in active_sessions:
            del active_sessions[session_key]
        
        # Generate performance feedback
        problem_data = adaptive_engine.get_problem_data(request.problem_id)
        performance_metrics = {}
        feedback_message = "Solution submitted successfully!"
        
        if problem_data:
            performance_metrics = adaptive_engine._analyze_performance(
                problem_data, request.time_taken, session.is_solved, 
                request.attempts, user_profile
            )
            
            feedback_message = adaptive_engine.get_appreciation_message(session, performance_metrics)
        
        return {
            "success": True,
            "execution_result": execution_result,
            "is_solved": session.is_solved,
            "performance_metrics": performance_metrics,
            "feedback_message": feedback_message,
            "user_stats": {
                "skill_level": user_profile.current_skill_level,
                "problems_solved": user_profile.total_problems_solved,
                "strong_topics": user_profile.strong_topics,
                "weak_topics": user_profile.weak_topics
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit solution: {str(e)}")

@app.get("/api/adaptive/problem/{problem_id}")
async def get_problem_details(problem_id: str):
    """Get complete problem details by ID"""
    try:
        if not adaptive_engine:
            raise HTTPException(status_code=500, detail="Adaptive Question Engine not available")
        
        problem_data = adaptive_engine.get_problem_data(problem_id)
        
        if not problem_data:
            raise HTTPException(status_code=404, detail="Problem not found")
        
        return {
            "success": True,
            "problem": problem_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get problem details: {str(e)}")

@app.get("/api/adaptive/user-stats/{user_id}")
async def get_user_statistics(user_id: str):
    """Get user performance statistics and progress"""
    try:
        if user_id not in user_profiles:
            return {
                "success": True,
                "stats": {
                    "skill_level": 1.0,
                    "problems_solved": 0,
                    "strong_topics": [],
                    "weak_topics": [],
                    "problem_history": []
                }
            }
        
        user_profile = user_profiles[user_id]
        
        return {
            "success": True,
            "stats": {
                "skill_level": user_profile.current_skill_level,
                "problems_solved": user_profile.total_problems_solved,
                "strong_topics": user_profile.strong_topics,
                "weak_topics": user_profile.weak_topics,
                "average_time_per_difficulty": user_profile.average_time_per_difficulty,
                "problem_history": [
                    {
                        "problem_id": session.problem_id,
                        "difficulty": session.difficulty,
                        "is_solved": session.is_solved,
                        "time_taken": session.time_taken,
                        "attempts": session.attempts,
                        "topics": session.topics
                    }
                    for session in user_profile.problem_history[-10:]  # Last 10 problems
                ]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user statistics: {str(e)}")

@app.get("/api/adaptive/available-problems")
async def get_available_problems():
    """Get list of all available problems in the knowledge base"""
    try:
        if not adaptive_engine:
            raise HTTPException(status_code=500, detail="Adaptive Question Engine not available")
        
        problems = []
        for problem_id, problem_data in adaptive_engine.problems_data.items():
            problems.append({
                "id": problem_id,
                "title": problem_data.get("title", "Untitled"),
                "difficulty": problem_data.get("difficulty", "Easy"),
                "topics": problem_data.get("topics", []),
                "expected_time": problem_data.get("expected_time", 1800)
            })
        
        return {
            "success": True,
            "problems": problems,
            "total_count": len(problems)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get available problems: {str(e)}")

# =============================================================================
# ENHANCED VOICE ASSISTANT ENDPOINTS
# =============================================================================

@app.post("/api/voice/start-conversation")
async def start_voice_conversation(request: ConversationStartRequest):
    """
    Start a new voice conversation with ElevenLabs AI agent
    Includes optional problem context for coding assistance
    """
    try:
        voice_manager = get_voice_assistant_manager()
        
        # Extract problem context if provided
        context = request.context
        if not context and hasattr(request, 'problem_id'):
            # Try to get problem context from adaptive engine
            if adaptive_engine:
                problem_data = adaptive_engine.get_problem_data(request.problem_id)
                if problem_data:
                    context = f"Problem: {problem_data.get('title', 'Unknown')} - {problem_data.get('problem_statement', '')[:200]}..."
        
        result = await voice_manager.start_conversation(
            agent_id=request.agent_id,
            api_key=request.api_key,
            context=context
        )
        
        if result["success"]:
            return {
                "success": True,
                "conversation_id": result["conversation_id"],
                "message": "Voice conversation started successfully",
                "context_provided": bool(context)
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to start conversation"))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start voice conversation: {str(e)}")

@app.post("/api/voice/send-message")
async def send_voice_message(request: VoiceMessageRequest):
    """
    Send a message to an active voice conversation
    Includes problem context for better AI responses
    """
    try:
        voice_manager = get_voice_assistant_manager()
        
        # Enhance problem context if provided
        problem_context = request.problem_context or {}
        
        # Add current user session info if available
        if request.problem_context and "problem_id" in request.problem_context:
            problem_id = request.problem_context["problem_id"]
            if adaptive_engine:
                problem_data = adaptive_engine.get_problem_data(problem_id)
                if problem_data:
                    problem_context.update({
                        "problem_title": problem_data.get("title"),
                        "difficulty": problem_data.get("difficulty"),
                        "topics": problem_data.get("topics", [])
                    })
        
        result = await voice_manager.send_message(
            conversation_id=request.conversation_id,
            message=request.message,
            problem_context=problem_context
        )
        
        if result["success"]:
            return {
                "success": True,
                "response": result.get("response", ""),
                "message_count": result.get("message_count", 0),
                "message": "Message sent successfully"
            }
        else:
            if "not found" in result.get("error", "").lower():
                raise HTTPException(status_code=404, detail="Conversation not found")
            elif "timeout" in result.get("error", "").lower():
                raise HTTPException(status_code=408, detail="Conversation timed out")
            else:
                raise HTTPException(status_code=500, detail=result.get("error", "Failed to send message"))
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send voice message: {str(e)}")

@app.post("/api/voice/end-conversation/{conversation_id}")
async def end_voice_conversation(conversation_id: str):
    """
    End an active voice conversation
    """
    try:
        voice_manager = get_voice_assistant_manager()
        
        result = await voice_manager.end_conversation(conversation_id)
        
        if result["success"]:
            return {
                "success": True,
                "message": "Voice conversation ended successfully",
                "message_count": result.get("message_count", 0)
            }
        else:
            if "not found" in result.get("error", "").lower():
                raise HTTPException(status_code=404, detail="Conversation not found")
            else:
                raise HTTPException(status_code=500, detail=result.get("error", "Failed to end conversation"))
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to end voice conversation: {str(e)}")

@app.get("/api/voice/active-conversations")
async def get_active_voice_conversations():
    """
    Get list of all active voice conversations
    """
    try:
        voice_manager = get_voice_assistant_manager()
        
        result = await voice_manager.get_active_conversations()
        
        if result["success"]:
            return {
                "success": True,
                "conversations": result.get("active_conversations", []),
                "total_count": result.get("total_count", 0)
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to get conversations"))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get active conversations: {str(e)}")

@app.post("/api/voice/problem-help/{problem_id}")
async def get_voice_problem_help(problem_id: str):
    """
    Start a voice conversation specifically for help with a coding problem
    """
    try:
        voice_manager = get_voice_assistant_manager()
        
        # Get problem details from adaptive engine
        context = None
        if adaptive_engine:
            problem_data = adaptive_engine.get_problem_data(problem_id)
            if problem_data:
                context = f"I need help with this coding problem:\n\n"
                context += f"Title: {problem_data.get('title', 'Unknown')}\n"
                context += f"Difficulty: {problem_data.get('difficulty', 'Unknown')}\n"
                context += f"Topics: {', '.join(problem_data.get('topics', []))}\n\n"
                context += f"Problem Statement:\n{problem_data.get('problem_statement', '')}\n\n"
                
                if problem_data.get('examples'):
                    context += "Examples:\n"
                    for i, example in enumerate(problem_data['examples'][:2]):  # Limit to 2 examples
                        context += f"Example {i+1}:\n"
                        context += f"Input: {example.get('input', '')}\n"
                        context += f"Output: {example.get('output', '')}\n"
                        if example.get('explanation'):
                            context += f"Explanation: {example['explanation']}\n"
                        context += "\n"
                
                context += "Please help me understand this problem and guide me through solving it step by step."
        
        # Start conversation with problem context
        result = await voice_manager.start_conversation(context=context)
        
        if result["success"]:
            return {
                "success": True,
                "conversation_id": result["conversation_id"],
                "problem_id": problem_id,
                "message": "Voice help session started for the problem",
                "context_provided": bool(context)
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to start help session"))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start problem help session: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Code Editor Backend Server...")
    print("📍 Server URL: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔗 Frontend Integration: http://localhost:3000")
    print("=" * 50)
    
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
