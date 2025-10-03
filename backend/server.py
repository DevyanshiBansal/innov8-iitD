"""
FastAPI Backend Server for Adaptive Coding Interview Platform

This server provides:
1. Adaptive question generation from knowledge base
2. Code execution with time/space complexity analysis  
3. Gemini AI professional interviewer integration
4. Interview session management and statistics
5. Code submission and performance analytics

Features:
- Adaptive difficulty adjustment based on user performance
- Professional AI interviewer that guides without revealing answers
- Real-time complexity analysis and performance metrics
- Session tracking and progress monitoring
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import json
import os
import subprocess
import time
import psutil
import random
import tempfile
from datetime import datetime
import google.generativeai as genai
import sys

app = FastAPI(title="Innov8 Adaptive Coding Interview Platform")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini AI
GEMINI_API_KEY = "_"
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
else:
    print("⚠️ Warning: GEMINI_API_KEY not set. AI interviewer will not work.")
    model = None

# Import psutil conditionally for enhanced memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

import threading
import ast
import re
from pathlib import Path

class MemoryMonitor:
    """Monitor peak memory usage during execution"""
    def __init__(self):
        self.peak_memory = 0
        self.monitoring = False
        self.process = None
    
    def start_monitoring(self, process):
        if not PSUTIL_AVAILABLE:
            return
            
        self.process = process
        self.monitoring = True
        self.peak_memory = 0
        self._monitor_thread = threading.Thread(target=self._monitor_memory)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        self.monitoring = False
        if hasattr(self, '_monitor_thread'):
            self._monitor_thread.join(timeout=1.0)
    
    def _monitor_memory(self):
        while self.monitoring:
            try:
                if self.process and self.process.poll() is None:
                    try:
                        psutil_process = psutil.Process(self.process.pid)
                        memory_info = psutil_process.memory_info()
                        current_memory = memory_info.rss  # RSS memory in bytes
                        self.peak_memory = max(self.peak_memory, current_memory)
                        
                        # Also check child processes
                        for child in psutil_process.children(recursive=True):
                            try:
                                child_memory = child.memory_info().rss
                                self.peak_memory = max(self.peak_memory, current_memory + child_memory)
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                time.sleep(0.01)  # Check every 10ms
            except:
                break

def format_memory_usage(bytes_value):
    """Format memory usage in human-readable format"""
    if bytes_value == 0:
        return "0 B"
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} TB"

def extract_test_cases_from_problem(problem_data):
    """Extract test cases from problem examples"""
    test_cases = []
    if problem_data and 'examples' in problem_data:
        for example in problem_data['examples']:
            test_cases.append({
                'input': example.get('input', ''),
                'expected_output': example.get('output', '')
            })
    return test_cases

def run_solution_on_test_cases(solution_code, test_cases):
    """Run the knowledge base solution on test cases to get expected outputs"""
    expected_outputs = []
    for test_case in test_cases:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as temp_file:
                test_code = f"""
{solution_code}

# Extract input and run
{test_case['input']}
result = checkIfPangram(sentence) if 'sentence' in locals() else None
if result is not None:
    print(result)
"""
                temp_file.write(test_code)
                temp_file_path = temp_file.name
            
            result = subprocess.run(
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and result.stdout.strip():
                expected_outputs.append(result.stdout.strip())
            else:
                expected_outputs.append(test_case['expected_output'])
            
            os.unlink(temp_file_path)
            
        except Exception as e:
            expected_outputs.append(test_case['expected_output'])
    
    return expected_outputs

def run_user_code_on_test_cases(user_code, test_cases):
    """Run user code on test cases and return outputs"""
    user_outputs = []
    
    for test_case in test_cases:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as temp_file:
                test_code = f"""
{user_code}

# Extract input and run
{test_case['input']}
if 'main' in globals():
    result = main()
else:
    result = checkIfPangram(sentence) if 'sentence' in locals() else None

if result is not None:
    print(result)
"""
                temp_file.write(test_code)
                temp_file_path = temp_file.name
            
            result = subprocess.run(
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                user_outputs.append(result.stdout.strip())
            else:
                user_outputs.append("ERROR")
            
            os.unlink(temp_file_path)
            
        except Exception as e:
            user_outputs.append("ERROR")
    
    return user_outputs

def evaluate_user_solution(user_code, current_question):
    """Strict evaluation of user solution against knowledge base test cases"""
    try:
        test_cases = extract_test_cases_from_problem(current_question)
        if not test_cases:
            return {"score": 0.0, "passed": 0, "total": 0, "details": "No test cases found"}
        
        expected_outputs = run_solution_on_test_cases(current_question.get('solution', {}).get('code', ''), test_cases)
        user_outputs = run_user_code_on_test_cases(user_code, test_cases)
        
        passed = 0
        total = len(test_cases)
        
        for i in range(total):
            if i < len(user_outputs) and i < len(expected_outputs):
                if str(user_outputs[i]).strip() == str(expected_outputs[i]).strip():
                    passed += 1
        
        score = (passed / total) if total > 0 else 0.0
        
        return {
            "score": round(score, 2),
            "passed": passed,
            "total": total,
            "details": f"Passed {passed}/{total} test cases"
        }
        
    except Exception as e:
        return {"score": 0.0, "passed": 0, "total": 0, "details": f"Evaluation error: {str(e)}"}

def get_next_question_with_gemini(knowledge_graph, current_performance, user_history):
    """Use Gemini to intelligently select next question from knowledge graph"""
    try:
        if not model:
            return select_fallback_question()
        
        nodes_info = []
        for node in knowledge_graph.get('nodes', [])[:50]:
            if 'Problem' in node.get('labels', []):
                nodes_info.append({
                    'id': node['id'],
                    'title': node['properties'].get('title', ''),
                    'difficulty': node['properties'].get('difficulty', ''),
                    'topics': node['properties'].get('topics', [])
                })
        
        prompt = f"""
You are an adaptive coding interview system. Based on the knowledge graph and user performance, select the most appropriate next question.

Knowledge Graph Problems: {json.dumps(nodes_info, indent=2)}

Current Performance: {current_performance}
User History: {user_history}

Rules:
1. If user scored >= 0.8, increase difficulty
2. If user scored < 0.5, maintain or decrease difficulty  
3. Consider topic diversity and learning progression
4. Return ONLY the question ID from the knowledge graph

Select the best next question ID:
"""
        
        response = model.generate_content(prompt)
        selected_id = response.text.strip()
        
        for node in knowledge_graph.get('nodes', []):
            if node['id'] == selected_id and 'Problem' in node.get('labels', []):
                return node['properties']
        
        return select_fallback_question()
        
    except Exception as e:
        print(f"Gemini selection error: {e}")
        return select_fallback_question()

def select_fallback_question():
    """Fallback question selection"""
    try:
        data_json_path = os.path.join(os.path.dirname(__file__), "..", "knowledge_base", "data_json.json")
        with open(data_json_path, "r") as f:
            graph_data = json.load(f)
        
        problem_nodes = [node for node in graph_data.get('nodes', []) if 'Problem' in node.get('labels', [])]
        if problem_nodes:
            return random.choice(problem_nodes)['properties']
    except:
        pass
    
    return {
        "title": "Default Problem",
        "problem_statement": "Solve a basic coding problem",
        "difficulty": "Easy",
        "examples": []
    }
    """Extract arguments passed to main function"""
    try:
        tree = ast.parse(code_content)
        main_calls = []
        
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and 
                isinstance(node.func, ast.Name) and 
                node.func.id == 'main'):
                
                args = []
                for arg in node.args:
                    if isinstance(arg, ast.Constant):
                        args.append(repr(arg.value))
                    elif isinstance(arg, ast.Str):  # For older Python versions
                        args.append(repr(arg.s))
                    elif isinstance(arg, ast.Num):  # For older Python versions
                        args.append(str(arg.n))
                    else:
                        try:
                            args.append(ast.unparse(arg) if hasattr(ast, 'unparse') else '<expression>')
                        except:
                            args.append('<expression>')
                
                main_calls.append(args)
        
        return main_calls
    except:
        # Fallback: use regex to find main() calls
        pattern = r'main\s*\(\s*([^)]*)\s*\)'
        matches = re.findall(pattern, code_content)
        result = []
        for match in matches:
            if match.strip():
                # Split by comma and clean up
                args = [arg.strip().strip('"\'') for arg in match.split(',')]
                result.append(args)
            else:
                result.append([])
        return result

# Global state for interview session
class InterviewSession:
    def __init__(self):
        self.current_question = None
        self.questions_asked = []
        self.user_performance = []
        self.session_start = datetime.now()
        self.is_active = False
        self.difficulty_level = "Easy"
        self.topics_covered = []
        self.hints_given = 0
        self.last_guidance_time = 0
        
interview_session = InterviewSession()

# Load knowledge bases
def load_knowledge_base():
    """Load both knowledge base files"""
    try:
        data_json_path = os.path.join(os.path.dirname(__file__), "..", "knowledge_base", "data_json.json")
        ques_data_path = os.path.join(os.path.dirname(__file__), "..", "knowledge_base", "ques_data.json")
        
        with open(data_json_path, "r") as f:
            graph_data = json.load(f)
        
        with open(ques_data_path, "r") as f:
            content = f.read()
            json_objects = []
            current_obj = ""
            brace_count = 0
            
            for char in content:
                current_obj += char
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and current_obj.strip():
                        try:
                            json_objects.append(json.loads(current_obj.strip()))
                            current_obj = ""
                        except json.JSONDecodeError:
                            pass
            
        return graph_data, json_objects
    except Exception as e:
        print(f"Error loading knowledge base: {e}")
        return None, []

graph_knowledge, flat_questions = load_knowledge_base()

# Pydantic models
class CodeExecutionRequest(BaseModel):
    code: str
    language: str
    stdin: Optional[str] = ""

class CodeExecutionResponse(BaseModel):
    success: bool
    output: str
    execution_time: float
    peak_memory: str
    time_complexity: Dict[str, Any]
    space_complexity: Dict[str, Any]
    error_analysis: str

class AdaptiveQuestionResponse(BaseModel):
    question_id: str
    title: str
    problem_statement: str
    examples: List[Dict]
    constraints: List[str]
    difficulty: str
    topics: List[str]
    time_limit: int

class ChatRequest(BaseModel):
    message: str
    code: Optional[str] = ""
    analysis_data: Optional[Dict] = None

class ChatResponse(BaseModel):
    response: str
    guidance_type: str
    should_continue: bool

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Innov8 Adaptive Coding Interview Platform",
        "timestamp": datetime.now().isoformat(),
        "knowledge_base_loaded": len(flat_questions) > 0 if flat_questions else False
    }

@app.get("/api/adaptive/question")
async def get_adaptive_question(
    difficulty: Optional[str] = None
):
    """Generate adaptive question using Gemini and Knowledge Graph"""
    try:
        if not flat_questions:
            raise HTTPException(status_code=500, detail="Knowledge base not loaded")
        
        # Load knowledge graph for intelligent selection
        graph_data = load_knowledge_base()[1]  # Get graph data
        
        current_performance = {
            "last_scores": [p["score"] for p in interview_session.user_performance[-3:]],
            "difficulty_level": interview_session.difficulty_level,
            "questions_asked": len(interview_session.questions_asked),
            "topics_covered": interview_session.topics_covered
        }
        
        # Use Gemini to select next question intelligently
        if model:
            try:
                available_questions = [q for q in flat_questions if q.get("question", {}).get("title") not in interview_session.questions_asked]
                
                question_summaries = []
                for q in available_questions[:20]:  # Limit for prompt
                    question_summaries.append({
                        "title": q.get("question", {}).get("title", ""),
                        "difficulty": q.get("metadata", {}).get("difficulty", ""),
                        "topics": q.get("metadata", {}).get("topic", [])
                    })
                
                prompt = f"""
You are an adaptive coding interview system. Select the most appropriate next question based on user performance.

Available Questions: {json.dumps(question_summaries, indent=2)}

User Performance: {current_performance}

Rules:
1. If average score >= 85, select Hard difficulty
2. If average score >= 65, select Medium difficulty  
3. If average score < 65, select Easy difficulty
4. Ensure topic diversity
5. Progressive learning path

Return only the exact title of the selected question.
"""
                
                response = model.generate_content(prompt)
                selected_title = response.text.strip().strip('"')
                
                # Find the selected question
                selected = None
                for q in available_questions:
                    if q.get("question", {}).get("title", "").strip() == selected_title.strip():
                        selected = q
                        break
                
                if not selected and available_questions:
                    selected = available_questions[0]  # Fallback
                    
            except Exception as e:
                print(f"Gemini selection failed: {e}")
                selected = random.choice([q for q in flat_questions if q.get("question", {}).get("title") not in interview_session.questions_asked])
        else:
            # Fallback without Gemini
            avg_score = sum([p["score"] for p in interview_session.user_performance[-3:]]) / max(len(interview_session.user_performance[-3:]), 1) if interview_session.user_performance else 50
            
            if avg_score >= 85:
                difficulty = "Hard"
            elif avg_score >= 65:
                difficulty = "Medium"
            else:
                difficulty = "Easy"
                
            filtered_questions = [q for q in flat_questions if q.get("metadata", {}).get("difficulty") == difficulty]
            filtered_questions = [q for q in filtered_questions if q.get("question", {}).get("title") not in interview_session.questions_asked]
            
            if not filtered_questions:
                interview_session.questions_asked = []
                filtered_questions = [q for q in flat_questions if q.get("metadata", {}).get("difficulty") == difficulty]
            
            selected = random.choice(filtered_questions) if filtered_questions else flat_questions[0]
        
        if not selected:
            raise HTTPException(status_code=404, detail="No suitable question found")
        
        question_data = selected["question"]
        metadata = selected["metadata"]
        
        # Update session state
        interview_session.current_question = selected
        interview_session.questions_asked.append(question_data["title"])
        interview_session.is_active = True
        interview_session.difficulty_level = difficulty
        
        # Calculate time limit based on difficulty
        time_limits = {"Easy": 15, "Medium": 25, "Hard": 35}
        time_limit = time_limits.get(difficulty, 20)
        
        # Display question generation in IDE console format
        print(f"\n" + "=" * 60)
        print("ADAPTIVE QUESTION GENERATED")
        print("=" * 60)
        print(f"Title: {question_data['title']}")
        print(f"Difficulty: {metadata['difficulty']}")
        print(f"Topics: {', '.join(metadata['topic'])}")
        print(f"Time Limit: {time_limit} minutes")
        print(f"Total Questions Asked: {len(interview_session.questions_asked)}")
        print(f"Session Active: {interview_session.is_active}")
        print("=" * 60)
        
        return AdaptiveQuestionResponse(
            question_id=str(len(interview_session.questions_asked)),
            title=question_data["title"],
            problem_statement=question_data["problemStatement"],
            examples=question_data["examples"],
            constraints=question_data["constraints"],
            difficulty=metadata["difficulty"],
            topics=metadata["topic"],
            time_limit=time_limit
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating question: {str(e)}")

def execute_code_with_subprocess(code_content, language="python"):
    """Execute code using subprocess and return actual output"""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(code_content)
            temp_file_path = temp_file.name
        
        try:
            process = subprocess.Popen(
                [sys.executable, temp_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=10)
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = "", "Execution timed out"
                exit_code = -1
            
            return {
                "success": exit_code == 0,
                "output": stdout,
                "error": stderr,
                "exit_code": exit_code
            }
            
        finally:
            try:
                os.unlink(temp_file_path)
            except:
                pass
                
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": f"Execution failed: {str(e)}",
            "exit_code": -1
        }

@app.post("/api/execute")
async def execute_code(request: CodeExecutionRequest):
    """Execute code and return actual terminal output"""
    try:
        result = execute_code_with_subprocess(request.code, request.language)
        
        print("OUTPUT:")
        if result['output']:
            print(result['output'])
        else:
            print("(No output)")
            
        if result['error']:
            print(f"ERROR: {result['error']}")
        
        return {
            "success": result["success"],
            "output": result["output"],
            "error": result["error"] if result["error"] else None,
            "execution_time": 0.0,
            "peak_memory": "0 B",
            "execution_output": {
                "success": result["success"],
                "output": result["output"],
                "execution_time": 0.0,
                "peak_memory": "0 B"
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": f"Execution error: {str(e)}",
            "execution_time": 0.0,
            "peak_memory": "0 B",
            "execution_output": {
                "success": False,
                "output": "",
                "execution_time": 0.0,
                "peak_memory": "0 B"
            }
        }

@app.post("/api/chat")
async def chat_with_interviewer(request: ChatRequest):
    """Professional AI interviewer chat with continuous guidance"""
    try:
        print(f"\n" + "=" * 60)
        print("AI INTERVIEWER CHAT")
        print("=" * 60)
        print(f"User Message: {request.message[:100]}{'...' if len(request.message) > 100 else ''}")
        print(f"Has Code Context: {bool(request.code)}")
        
        if not model:
            print("⚠️ AI interviewer unavailable - GEMINI_API_KEY not configured")
            return ChatResponse(
                response="AI interviewer is not available. Please configure GEMINI_API_KEY.",
                guidance_type="warning",
                should_continue=True
            )
        
        # Check if interview should end
        if request.message.upper() == "END":
            interview_session.is_active = False
            
            # Generate final assessment
            assessment = generate_final_assessment()
            
            return ChatResponse(
                response=f"Interview completed! {assessment}",
                guidance_type="encouragement",
                should_continue=False
            )
        
        # Prepare context for the AI interviewer
        context = prepare_interviewer_context(request)
        
        # Generate AI response
        response = model.generate_content(context)
        
        # Determine guidance type
        guidance_type = determine_guidance_type(request.message, request.code)
        
        # Update session state
        update_session_state(request, guidance_type)
        
        # Display AI response in IDE console format
        print(f"\n" + "=" * 60)
        print(f"AI RESPONSE - {guidance_type.upper()}")
        print("=" * 60)
        print(response.text[:300] + ('...' if len(response.text) > 300 else ''))
        print("=" * 60)
        print(f"Guidance Type: {guidance_type}")
        print(f"Interview Active: {interview_session.is_active}")
        print("=" * 60)
        
        return ChatResponse(
            response=response.text,
            guidance_type=guidance_type,
            should_continue=interview_session.is_active
        )
        
    except Exception as e:
        return ChatResponse(
            response="I'm having trouble processing your request. Please try again.",
            guidance_type="warning",
            should_continue=True
        )

def create_python_solution_from_kb(solution_data, question_data):
    """Convert knowledge base solution to executable Python code"""
    solution_code = solution_data.get("code", "")
    
    if not solution_code:
        return None
    
    if solution_data.get("language") == "C++":
        # Convert C++ solution to Python equivalent
        title = question_data.get("title", "")
        
        if "Pangram" in title:
            return """
def checkIfPangram(sentence):
    alphabet = set('abcdefghijklmnopqrstuvwxyz')
    sentence_chars = set(sentence.lower())
    return alphabet.issubset(sentence_chars)
"""
        elif "Two Sum" in title:
            return """
def twoSum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
"""
        elif "Target Array" in title:
            return """
def createTargetArray(nums, index):
    target = []
    for i in range(len(nums)):
        target.insert(index[i], nums[i])
    return target
"""
        else:
            # Generic conversion attempt
            return f"""
def solution():
    # Knowledge base solution (converted from C++)
    # {solution_code}
    return None
"""
    
    return solution_code

def extract_test_inputs_from_examples(examples):
    """Extract actual test inputs from examples"""
    test_inputs = []
    
    for example in examples:
        input_str = example.get("input", "")
        expected_output = example.get("output", "")
        
        # Parse input string to extract variables
        if "sentence =" in input_str:
            sentence_match = input_str.split('sentence = ')[1].strip().strip('"')
            test_inputs.append({
                "input_code": f'sentence = "{sentence_match}"',
                "expected": expected_output.strip().strip('"')
            })
        elif "nums =" in input_str and "target =" in input_str:
            lines = input_str.split(", ")
            nums_part = lines[0].split("nums = ")[1]
            target_part = lines[1].split("target = ")[1]
            test_inputs.append({
                "input_code": f'nums = {nums_part}\ntarget = {target_part}',
                "expected": expected_output.strip()
            })
        elif "nums =" in input_str and "index =" in input_str:
            lines = input_str.split(", ")
            nums_part = lines[0].split("nums = ")[1]
            index_part = lines[1].split("index = ")[1]
            test_inputs.append({
                "input_code": f'nums = {nums_part}\nindex = {index_part}',
                "expected": expected_output.strip()
            })
    
    return test_inputs

def run_code_with_subprocess(code, test_input_code, function_name=None):
    """Run code with subprocess and return output"""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as temp_file:
            full_code = f"""
{code}

# Test input
{test_input_code}

# Execute the function
import sys

try:
    if '{function_name}' in globals() and callable({function_name}):
        if 'sentence' in locals():
            result = {function_name}(sentence)
        elif 'nums' in locals() and 'target' in locals():
            result = {function_name}(nums, target)
        elif 'nums' in locals() and 'index' in locals():
            result = {function_name}(nums, index)
        else:
            result = {function_name}()
        print(result)
    else:
        # Try to find main function or any callable
        for name, obj in globals().items():
            if callable(obj) and not name.startswith('_') and name not in ['print', 'input', 'len', 'range', 'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple']:
                try:
                    if 'sentence' in locals():
                        result = obj(sentence)
                    elif 'nums' in locals() and 'target' in locals():
                        result = obj(nums, target)
                    elif 'nums' in locals() and 'index' in locals():
                        result = obj(nums, index)
                    else:
                        result = obj()
                    print(result)
                    break
                except Exception as e:
                    continue
except Exception as e:
    print("ERROR:", str(e))
"""
            temp_file.write(full_code)
            temp_file_path = temp_file.name
        
        result = subprocess.run(
            [sys.executable, temp_file_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        os.unlink(temp_file_path)
        
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return "ERROR"
            
    except Exception as e:
        return f"ERROR: {str(e)}"

def generate_test_cases_with_gemini(user_code, kb_solution, problem_statement, examples):
    """Generate comprehensive test cases using Gemini by analyzing both codes"""
    try:
        if not model:
            return extract_test_inputs_from_examples(examples)
        
        prompt = f"""
Analyze these two code solutions and generate 8-12 comprehensive test cases:

USER CODE:
{user_code}

KNOWLEDGE BASE SOLUTION:
{kb_solution}

PROBLEM STATEMENT:
{problem_statement}

EXAMPLES:
{examples}

Generate test cases that cover:
1. Basic examples from problem
2. Edge cases (empty inputs, single elements, boundaries)
3. Large inputs
4. Corner cases that might break incorrect solutions

Return test cases in this EXACT format:
INPUT: <variable assignments like: sentence = "hello">
EXPECTED: <expected output like: false>

INPUT: <variable assignments>
EXPECTED: <expected output>

Make sure each INPUT line contains proper Python variable assignments.
"""
        
        response = model.generate_content(prompt)
        test_cases = []
        
        lines = response.text.split('\n')
        current_input = ""
        current_expected = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith('INPUT:'):
                current_input = line.replace('INPUT:', '').strip()
            elif line.startswith('EXPECTED:'):
                current_expected = line.replace('EXPECTED:', '').strip()
                if current_input and current_expected:
                    test_cases.append({
                        'input_code': current_input,
                        'expected': current_expected
                    })
                    current_input = ""
                    current_expected = ""
        
        # If Gemini failed to generate enough test cases, add examples
        if len(test_cases) < 3:
            example_tests = extract_test_inputs_from_examples(examples)
            test_cases.extend(example_tests)
        
        return test_cases[:12]  # Limit to 12 test cases
        
    except Exception as e:
        print(f"Gemini test case generation failed: {e}")
        return extract_test_inputs_from_examples(examples)

def run_code_with_validation(code, test_case, function_name):
    """Run code with subprocess and validate output"""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as temp_file:
            full_code = f"""
import sys
import traceback

try:
    {code}
    
    # Test input
    {test_case['input_code']}
    
    # Find and execute the appropriate function
    result = None
    
    # Try common function names based on problem type
    function_candidates = ['{function_name}', 'solution', 'main']
    
    # Add the detected function name as highest priority
    if '{function_name}' != 'solution':
        function_candidates.insert(0, '{function_name}')
    
    # Add some common alternatives based on variable types
    if 'sentence' in locals():
        function_candidates.extend(['checkIfPangram', 'isPangram'])
    elif 'nums' in locals() and 'target' in locals():
        function_candidates.extend(['twoSum', 'two_sum'])
    elif 'nums' in locals() and 'index' in locals():
        function_candidates.extend(['createTargetArray', 'create_target_array'])
    
    for func_name in function_candidates:
        if func_name in globals() and callable(globals()[func_name]):
            try:
                if 'sentence' in locals():
                    result = globals()[func_name](sentence)
                elif 'nums' in locals() and 'target' in locals():
                    result = globals()[func_name](nums, target)
                elif 'nums' in locals() and 'index' in locals():
                    result = globals()[func_name](nums, index)
                else:
                    result = globals()[func_name]()
                break
            except Exception as e:
                continue
    
    if result is not None:
        print(result)
    else:
        print("NO_RESULT")
        
except Exception as e:
    print(f"ERROR: {{str(e)}}")
    traceback.print_exc()
"""
            temp_file.write(full_code)
            temp_file_path = temp_file.name
        
        result = subprocess.run(
            [sys.executable, temp_file_path],
            capture_output=True,
            text=True,
            timeout=15
        )
        
        os.unlink(temp_file_path)
        
        if result.returncode == 0:
            output = result.stdout.strip()
            return output if output != "NO_RESULT" else "ERROR"
        else:
            return "ERROR"
            
    except Exception as e:
        return f"ERROR: {str(e)}"

@app.post("/api/submit")
async def submit_solution(request: CodeExecutionRequest):
    """Submit solution with Gemini-generated test cases and percentage-based scoring"""
    try:
        print(f"\n🎯 SOLUTION SUBMITTED ({request.language.upper()}):")
        print("=" * 60)
        
        # First execute code to show terminal output
        execution_result = execute_code_with_subprocess(request.code, request.language)
        
        print("OUTPUT:")
        if execution_result['output']:
            print(execution_result['output'])
        else:
            print("(No output)")
            
        if execution_result['error']:
            print(f"ERROR: {execution_result['error']}")
        
        # Check if we have an active question from knowledge base
        if not interview_session.current_question:
            print("Score: 0.00%")
            print("Next Level: Easy")
            return {
                "success": execution_result["success"],
                "output": execution_result["output"],
                "error": execution_result["error"],
                "performance_score": 0.0,
                "next_difficulty": "Easy",
                "execution_result": {
                    "success": execution_result["success"],
                    "output": execution_result["output"],
                    "execution_time": 0.0,
                    "peak_memory": "0 B"
                }
            }
        
        # Get current question and solution from knowledge base
        current_q = interview_session.current_question
        question_data = current_q.get("question", {})
        solution_data = current_q.get("solution", {})
        
        # Create Python solution from knowledge base
        kb_solution = create_python_solution_from_kb(solution_data, question_data)
        if not kb_solution:
            print("Knowledge base solution not available")
            score = 100.0 if execution_result["success"] else 0.0
            next_level = "Medium" if score >= 50 else "Easy"
            
            print(f"Score: {score:.2f}%")
            print(f"Next Level: {next_level}")
            
            return {
                "success": execution_result["success"],
                "output": execution_result["output"],
                "error": execution_result["error"],
                "performance_score": score,
                "next_difficulty": next_level,
                "execution_result": {
                    "success": execution_result["success"],
                    "output": execution_result["output"],
                    "execution_time": 0.0,
                    "peak_memory": "0 B"
                }
            }
        
        # Generate comprehensive test cases using Gemini
        problem_statement = question_data.get("problemStatement", "")
        examples = question_data.get("examples", [])
        
        test_cases = generate_test_cases_with_gemini(
            request.code, 
            kb_solution, 
            problem_statement, 
            examples
        )
        
        if not test_cases:
            print("No test cases generated")
            score = 50.0
            next_level = "Medium"
            
            print(f"Score: {score:.2f}%")
            print(f"Next Level: {next_level}")
            
            return {
                "success": execution_result["success"],
                "output": execution_result["output"],
                "error": execution_result["error"],
                "performance_score": score,
                "next_difficulty": next_level,
                "execution_result": {
                    "success": execution_result["success"],
                    "output": execution_result["output"],
                    "execution_time": 0.0,
                    "peak_memory": "0 B"
                }
            }

        # Use Gemini to intelligently determine function name
        function_name = determine_function_name_with_gemini(
            question_data.get("title", ""),
            question_data.get("problemStatement", ""),
            request.code,
            kb_solution
        )
        
        print(f"\nRunning {len(test_cases)} test cases...")
        print("=" * 40)
        
        # Run both solutions on all test cases
        passed = 0
        total = len(test_cases)
        user_outputs = []
        expected_outputs = []
        
        for i, test_case in enumerate(test_cases):
            print(f"Test {i+1}: ", end="")
            
            # Run user code
            user_output = run_code_with_validation(request.code, test_case, function_name)
            user_outputs.append(user_output)
            
            # Run knowledge base solution
            kb_output = run_code_with_validation(kb_solution, test_case, function_name)
            expected_outputs.append(kb_output)
            
            # Compare outputs (normalize for comparison)
            user_normalized = str(user_output).strip().lower()
            kb_normalized = str(kb_output).strip().lower()
            
            if user_normalized == kb_normalized and user_output != "ERROR":
                passed += 1
                print("PASS")
            else:
                print(f"FAIL (got: {user_output}, expected: {kb_output})")
        
        # Calculate percentage score
        score = round((passed / total) * 100, 2) if total > 0 else 0.0
        
        # Update session with test results
        interview_session.user_performance.append({
            "question": question_data.get("title", "Unknown"),
            "score": score,
            "passed": passed,
            "total": total,
            "timestamp": datetime.now().isoformat()
        })
        
        # Determine next difficulty level based on score percentage
        if score >= 80:
            next_level = "Hard"
        elif score >= 60:
            next_level = "Medium"
        else:
            next_level = "Easy"
        
        print("=" * 40)
        print(f"Score: {score:.2f}% ({passed}/{total} test cases passed)")
        print(f"Next Level: {next_level}")
        print("=" * 40)
        
        return {
            "success": execution_result["success"],
            "output": execution_result["output"],
            "error": execution_result["error"],
            "performance_score": score,
            "next_difficulty": next_level,
            "execution_result": {
                "success": execution_result["success"],
                "output": execution_result["output"],
                "execution_time": 0.0,
                "peak_memory": "0 B"
            },
            "test_results": {
                "passed": passed,
                "total": total,
                "score_percentage": score,
                "user_outputs": user_outputs,
                "expected_outputs": expected_outputs,
                "test_cases": test_cases
            }
        }
        
    except Exception as e:
        print(f"Error in solution submission: {str(e)}")
        return {
            "success": False,
            "output": "",
            "error": f"Submission failed: {str(e)}",
            "performance_score": 0.0,
            "next_difficulty": "Easy",
            "execution_result": {
                "success": False,
                "output": "",
                "execution_time": 0.0,
                "peak_memory": "0 B"
            }
        }
        
def determine_function_name_with_gemini(problem_title, problem_statement, user_code, kb_solution):
    """Use Gemini to intelligently determine the correct function name"""
    try:
        if not model:
            # Fallback to basic parsing
            return extract_function_name_from_code(user_code)
        
        prompt = f"""
Analyze this coding problem and determine the most likely function name that should be called.

PROBLEM TITLE: {problem_title}
PROBLEM STATEMENT: {problem_statement}

USER CODE:
{user_code}

KNOWLEDGE BASE SOLUTION:
{kb_solution}

Look for:
1. Function definitions in the user code (def functionName(...))
2. Common function naming patterns for this type of problem
3. Function names mentioned in the problem statement
4. Standard conventions (camelCase, snake_case)

Return ONLY the function name that should be called, nothing else.
Examples: checkIfPangram, twoSum, createTargetArray, isPalindrome, etc.
"""
        
        response = model.generate_content(prompt)
        function_name = response.text.strip().strip('"\'')
        
        # Validate the function name
        if function_name and function_name.isidentifier():
            return function_name
        else:
            return extract_function_name_from_code(user_code)
            
    except Exception as e:
        print(f"Gemini function name detection failed: {e}")
        return extract_function_name_from_code(user_code)

def extract_function_name_from_code(code):
    """Extract function name from user code using AST parsing"""
    try:
        tree = ast.parse(code)
        
        # Look for function definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip common utility functions
                if node.name not in ['main', 'test', 'print', 'input', '__init__']:
                    return node.name
        
        # If no suitable function found, try common patterns
        if 'def main' in code:
            return 'main'
        elif 'def solution' in code:
            return 'solution'
        else:
            # Return first function found
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return node.name
                    
        return 'solution'  # Default fallback
        
    except Exception as e:
        # Regex fallback
        import re
        pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        matches = re.findall(pattern, code)
        
        if matches:
            for match in matches:
                if match not in ['main', 'test', 'print', 'input']:
                    return match
            return matches[0]
        
        return 'solution'

        # Use Gemini to intelligently determine function name
        function_name = determine_function_name_with_gemini(
            question_data.get("title", ""),
            question_data.get("problemStatement", ""),
            request.code,
            kb_solution
        )
        
        print(f"\nRunning {len(test_cases)} test cases...")
        print("=" * 40)
        
        # Run both solutions on all test cases
        passed = 0
        total = len(test_cases)
        user_outputs = []
        expected_outputs = []
        
        for i, test_case in enumerate(test_cases):
            print(f"Test {i+1}: ", end="")
            
            # Run user code
            user_output = run_code_with_validation(request.code, test_case, function_name)
            user_outputs.append(user_output)
            
            # Run knowledge base solution
            kb_output = run_code_with_validation(kb_solution, test_case, function_name)
            expected_outputs.append(kb_output)
            
            # Compare outputs (normalize for comparison)
            user_normalized = str(user_output).strip().lower()
            kb_normalized = str(kb_output).strip().lower()
            
            if user_normalized == kb_normalized and user_output != "ERROR":
                passed += 1
                print("PASS")
            else:
                print(f"FAIL (got: {user_output}, expected: {kb_output})")
        
        # Calculate percentage score
        score = round((passed / total) * 100, 2) if total > 0 else 0.0
        
        # Update session with test results
        interview_session.user_performance.append({
            "question": question_data.get("title", "Unknown"),
            "score": score,
            "passed": passed,
            "total": total,
            "timestamp": datetime.now().isoformat()
        })
        
        # Determine next difficulty level based on score percentage
        if score >= 80:
            next_level = "Hard"
        elif score >= 60:
            next_level = "Medium"
        else:
            next_level = "Easy"
        
        print("=" * 40)
        print(f"Score: {score:.2f}% ({passed}/{total} test cases passed)")
        print(f"Next Level: {next_level}")
        print("=" * 40)
        
        return {
            "success": execution_result["success"],
            "output": execution_result["output"],
            "error": execution_result["error"],
            "performance_score": score,
            "next_difficulty": next_level,
            "execution_result": {
                "success": execution_result["success"],
                "output": execution_result["output"],
                "execution_time": 0.0,
                "peak_memory": "0 B"
            },
            "test_results": {
                "passed": passed,
                "total": total,
                "score_percentage": score,
                "user_outputs": user_outputs,
                "expected_outputs": expected_outputs,
                "test_cases": test_cases
            }
        }
        
    except Exception as e:
        print(f"Error in solution submission: {str(e)}")
        return {
            "success": False,
            "output": "",
            "error": f"Submission failed: {str(e)}",
            "performance_score": 0.0,
            "next_difficulty": "Easy",
            "execution_result": {
                "success": False,
                "output": "",
                "execution_time": 0.0,
                "peak_memory": "0 B"
            }
        }

# Helper functions
def prepare_interviewer_context(request: ChatRequest) -> str:
    """Prepare context for the AI interviewer"""
    current_time = time.time()
    time_since_start = (current_time - interview_session.last_guidance_time) / 60 if interview_session.last_guidance_time else 0
    
    context = f"""
    You are a professional coding interviewer conducting a live coding interview. 

    STRICT RULES:
    1. NEVER reveal the complete solution or answer directly
    2. Only provide hints, nudges, and guidance
    3. Ask probing questions to assess understanding
    4. Guide the candidate toward the solution through Socratic method
    5. Be encouraging but maintain professional standards
    6. If candidate is stuck, provide increasingly specific hints
    7. Evaluate code for correctness, efficiency, and best practices

    CURRENT SESSION:
    - Time Since Last Guidance: {time_since_start:.1f} minutes
    - Current Question: {interview_session.current_question.get('question', {}).get('title', 'None') if interview_session.current_question else 'None'}
    - Questions Asked: {len(interview_session.questions_asked)}
    - Hints Given: {interview_session.hints_given}
    
    CANDIDATE'S MESSAGE: {request.message}
    
    CANDIDATE'S CURRENT CODE:
    {request.code if request.code else "No code provided yet"}
    
    RESPONSE GUIDELINES:
    - If code has bugs, don't fix them directly - guide them to find it
    - If approach is inefficient, ask about time complexity
    - If solution is correct, ask about edge cases or optimization
    - Maintain interview atmosphere - be professional but supportive
    
    Provide your response as a professional interviewer:
    """
    
    return context

def determine_guidance_type(message: str, code: str = "") -> str:
    """Determine what type of guidance to provide"""
    message_lower = message.lower()
    
    if any(word in message_lower for word in ["stuck", "help", "hint", "confused"]):
        return "hint"
    elif any(word in message_lower for word in ["wrong", "error", "bug", "not working"]):
        return "debug"
    elif code and len(code) > 50:
        return "review"
    elif any(word in message_lower for word in ["done", "finished", "complete"]):
        return "evaluation"
    else:
        return "guidance"

def update_session_state(request: ChatRequest, guidance_type: str):
    """Update interview session state based on interaction"""
    interview_session.last_guidance_time = time.time()
    
    if guidance_type == "hint":
        interview_session.hints_given += 1

def generate_final_assessment() -> str:
    """Generate final interview assessment"""
    total_questions = len(interview_session.questions_asked)
    total_hints = interview_session.hints_given
    session_duration = (datetime.now() - interview_session.session_start).seconds / 60
    
    return f"""
    📊 Interview Summary:
    • Questions Attempted: {total_questions}
    • Hints Provided: {total_hints}
    • Session Duration: {session_duration:.1f} minutes
    • Topics Covered: {', '.join(interview_session.topics_covered) if interview_session.topics_covered else 'Various'}
    
    Thank you for participating in the adaptive coding interview!
    """

def analyze_solution_performance(execution_result: CodeExecutionResponse, code: str) -> float:
    """Analyze solution performance and return score (0-100)"""
    score = 0
    
    # Execution success (40 points)
    if execution_result.success:
        score += 40
    
    # Code quality (30 points)
    if len(code.strip()) > 0:
        score += 20  # Basic implementation
        if "def " in code or "function " in code:
            score += 5  # Function usage
        if any(word in code.lower() for word in ["class", "object", "struct"]):
            score += 5  # OOP usage
    
    # Time complexity (30 points)
    if execution_result.time_complexity.get("analysis"):
        complexity_analysis = execution_result.time_complexity["analysis"].lower()
        if "o(1)" in complexity_analysis or "constant" in complexity_analysis:
            score += 30
        elif "o(log n)" in complexity_analysis or "logarithmic" in complexity_analysis:
            score += 25
        elif "o(n)" in complexity_analysis or "linear" in complexity_analysis:
            score += 20
        elif "o(n log n)" in complexity_analysis:
            score += 15
        elif "o(n^2)" in complexity_analysis or "quadratic" in complexity_analysis:
            score += 10
        else:
            score += 5
    
    return min(score, 100)

def generate_submission_feedback(execution_result: CodeExecutionResponse, score: float) -> str:
    """Generate feedback for submitted solution"""
    if score >= 80:
        return "Excellent solution! Great job on both correctness and efficiency."
    elif score >= 60:
        return "Good solution! Consider optimizing for better time complexity."
    elif score >= 40:
        return "Solution works but needs improvement. Focus on edge cases and efficiency."
    else:
        return "Solution needs significant improvement. Review the problem requirements and test your code."

def determine_next_difficulty(performance_score: float) -> str:
    """Determine next question difficulty based on sequential progression"""
    current_difficulty = interview_session.difficulty_level
    
    # Sequential progression: Easy → Medium → Hard
    if current_difficulty == "Easy":
        return "Medium"
    elif current_difficulty == "Medium":
        return "Hard"
    else:
        return "Hard"  # Stay at Hard level

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Adaptive Coding Interview Platform...")
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
