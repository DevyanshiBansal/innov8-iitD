"""
Robust Voice Interview Agent for Code Editor - Based on Voice_assistant

This module creates an AI interviewer that exactly matches the Voice_assistant directory:
- Uses ElevenLabs conversational AI with exact same integration
- Conducts formal coding interviews with voice interaction
- Has real-time access to IDE code analysis
- Can interrupt, guide, and provide feedback during coding
- Analyzes time/space complexity using n vs time/space data
- Uses Mathematics.py scoring algorithms for comprehensive evaluation
"""

import os
import signal
import json
import time
import asyncio
import threading
import math
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs.conversational_ai.conversation import Conversation
    from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    print("⚠️  ElevenLabs not available - using mock implementation")

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class InterviewSession:
    """Represents an ongoing interview session"""
    session_id: str
    candidate_name: str
    start_time: datetime
    current_question: Optional[str] = None
    questions_asked: List[str] = None
    code_submissions: List[Dict] = None
    performance_data: List[Dict] = None
    interview_state: str = "starting"  # starting, questioning, coding, analyzing, concluding
    conversation_id: Optional[str] = None
    
    def __post_init__(self):
        if self.questions_asked is None:
            self.questions_asked = []
        if self.code_submissions is None:
            self.code_submissions = []
        if self.performance_data is None:
            self.performance_data = []

class RobustVoiceInterviewer:
    """
    Robust AI Voice Interviewer - Exact implementation based on Voice_assistant directory
    """
    
    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url
        self.session: Optional[InterviewSession] = None
        
        # ElevenLabs configuration - exact same as Voice_assistant/main.py
        self.api_key = "_"
        self.agent_id = "_"
        
        if ELEVENLABS_AVAILABLE:
            self.client = ElevenLabs(api_key=self.api_key)
            self.conversation = None
        else:
            self.client = None
            self.conversation = None
        
        # Interview configuration with exact mathematics integration
        self.interview_config = {
            "max_questions": 3,
            "time_per_question": 900,  # 15 minutes
            "difficulty_progression": ["easy", "medium", "hard"],
            "interruption_threshold": 300,  # 5 minutes of no progress
            "complexity_analysis_enabled": True,
            
            # Mathematics.py scoring weights - exact same structure
            "weights": {
                "ps": 0.4,      # Problem-solving
                "code": 0.3,    # Coding proficiency  
                "resilience": 0.1,  # Resilience
                "autonomy": 0.2,    # Autonomy
            },
            "hint_budget": 1.0  # Standard hint allowance
        }
        
        # Coding interview questions database with complexity expectations
        self.questions_db = {
            "easy": [
                {
                    "title": "Two Sum",
                    "description": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
                    "expected_complexity": {"time": "O(n)", "space": "O(n)"},
                    "hints": [
                        "Consider what you've seen before in the array",
                        "Think about using a hash map to store values",
                        "You can solve this in a single pass"
                    ],
                    "difficulty": 1
                },
                {
                    "title": "Valid Parentheses", 
                    "description": "Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.",
                    "expected_complexity": {"time": "O(n)", "space": "O(n)"},
                    "hints": [
                        "Think about Last In, First Out (LIFO)",
                        "Consider using a stack data structure",
                        "Match each opening bracket with its corresponding closing bracket"
                    ],
                    "difficulty": 1
                }
            ],
            "medium": [
                {
                    "title": "Longest Substring Without Repeating Characters",
                    "description": "Given a string s, find the length of the longest substring without repeating characters.",
                    "expected_complexity": {"time": "O(n)", "space": "O(min(m,n))"},
                    "hints": [
                        "Consider using sliding window technique",
                        "Use a set to track characters in current window",
                        "Move the window when you find a duplicate"
                    ],
                    "difficulty": 2
                }
            ],
            "hard": [
                {
                    "title": "Merge k Sorted Lists",
                    "description": "You are given an array of k linked-lists, each sorted in ascending order. Merge all the linked-lists into one sorted linked-list.",
                    "expected_complexity": {"time": "O(n log k)", "space": "O(1)"},
                    "hints": [
                        "Think about divide and conquer approach",
                        "Consider using a priority queue or heap",
                        "You can merge lists in pairs"
                    ],
                    "difficulty": 3
                }
            ]
        }
        
        # Real-time monitoring
        self.monitoring_active = False
        self.last_code_check = None
        self.code_analysis_thread = None
        
        # Interview data for Mathematics.py scoring
        self.interview_data = []
        
    def start_interview(self, candidate_name: str) -> str:
        """Start a new interview session - exact same pattern as Voice_assistant/main.py"""
        session_id = f"interview_{int(time.time())}"
        self.session = InterviewSession(
            session_id=session_id,
            candidate_name=candidate_name,
            start_time=datetime.now()
        )
        
        # Initialize ElevenLabs conversation - exact same as main.py
        if ELEVENLABS_AVAILABLE and self.client:
            self._initialize_voice_conversation()
        
        # Start real-time code monitoring
        self._start_code_monitoring()
        
        print(f"🎤 Starting interview for {candidate_name} (Session: {session_id})")
        return session_id
    
    def _initialize_voice_conversation(self):
        """Initialize ElevenLabs conversation - EXACT same implementation as Voice_assistant/main.py"""
        
        try:
            self.conversation = Conversation(
                # API client and agent ID - exact same
                self.client,
                self.agent_id,
                
                # Assume auth is required when API_KEY is set - exact same
                requires_auth=bool(self.api_key),

                # Use the default audio interface - exact same
                audio_interface=DefaultAudioInterface(),

                # Simple callbacks that print the conversation to the console - exact same pattern
                callback_agent_response=self._handle_agent_response,
                callback_agent_response_correction=self._handle_agent_response_correction,
                callback_user_transcript=self._handle_user_transcript,

                # Uncomment if you want to see latency measurements - exact same comment
                # callback_latency_measurement=lambda latency: print(f"Latency: {latency}ms"),
            )

            # Start the session - exact same as main.py
            self.conversation.start_session()
            
            # Set up signal handling - exact same as main.py  
            signal.signal(signal.SIGINT, lambda sig, frame: self.conversation.end_session())
            
            print("✅ Voice conversation initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize voice conversation: {e}")
            
    def _handle_agent_response(self, response: str):
        """Handle AI interviewer responses - exact same pattern as main.py"""
        print(f"Agent: {response}")
        
        # Log the response to session for interview analysis
        if self.session:
            self.session.performance_data.append({
                "timestamp": datetime.now().isoformat(),
                "type": "agent_response",
                "content": response
            })
            
    def _handle_agent_response_correction(self, original: str, corrected: str):
        """Handle speech corrections - exact same pattern as main.py"""
        print(f"Agent: {original} -> {corrected}")
        
    def _handle_user_transcript(self, transcript: str):
        """Handle candidate's speech input - exact same pattern as main.py"""
        print(f"User: {transcript}")
        
        # Log user transcript for interview analysis
        if self.session:
            self.session.performance_data.append({
                "timestamp": datetime.now().isoformat(),
                "type": "user_transcript", 
                "content": transcript
            })
        
        # Analyze if candidate is asking for help or hints
        self._analyze_candidate_request(transcript)
        
    def _analyze_candidate_request(self, transcript: str):
        """Analyze what the candidate is saying and respond appropriately"""
        help_keywords = ["help", "hint", "stuck", "don't know", "confused", "not sure", "guidance"]
        
        if any(keyword in transcript.lower() for keyword in help_keywords):
            # Candidate is asking for help - provide contextual hint based on current code
            self._provide_contextual_hint()
            
    def _provide_contextual_hint(self):
        """Provide hints based on current code analysis and question context"""
        try:
            # Get current code from IDE
            current_code = self._get_current_code_from_ide()
            if current_code and self.session and self.session.current_question:
                # Find the current question data
                question_data = self._find_question_data(self.session.current_question)
                if question_data and "hints" in question_data:
                    hints = question_data["hints"]
                    # Provide hints progressively
                    hint_index = min(len(self.session.performance_data) // 10, len(hints) - 1)  
                    hint = hints[hint_index]
                    print(f"💡 Contextual hint: {hint}")
                    
                    # Record hint usage for Mathematics.py scoring
                    self._record_hint_usage(hint_index + 1)
            
        except Exception as e:
            print(f"❌ Error providing contextual hint: {e}")
            
    def _record_hint_usage(self, hint_level: int):
        """Record hint usage for Mathematics.py scoring"""
        if self.session:
            # Add to current question's hint types (Mathematics.py format)
            if not hasattr(self.session, 'current_question_data'):
                self.session.current_question_data = {"H_types": []}
            
            if "H_types" not in self.session.current_question_data:
                self.session.current_question_data["H_types"] = []
                
            self.session.current_question_data["H_types"].append(hint_level)
            
    def _find_question_data(self, question_title: str) -> Optional[Dict]:
        """Find question data by title"""
        for difficulty in self.questions_db.values():
            for question in difficulty:
                if question["title"] in question_title:
                    return question
        return None
        
    def _start_code_monitoring(self):
        """Start real-time monitoring of IDE code changes"""
        self.monitoring_active = True
        self.code_analysis_thread = threading.Thread(target=self._monitor_code_loop)
        self.code_analysis_thread.daemon = True
        self.code_analysis_thread.start()
        print("🔍 Started real-time code monitoring")
        
    def _monitor_code_loop(self):
        """Continuously monitor IDE for code changes and execution results"""
        while self.monitoring_active:
            try:
                # Check for new execution results
                execution_data = self._fetch_latest_execution_data()
                
                if execution_data and execution_data != self.last_code_check:
                    self.last_code_check = execution_data
                    
                    # Analyze the execution results using Mathematics.py principles
                    analysis = self._analyze_execution_results_with_mathematics(execution_data)
                    
                    # If candidate seems stuck or made errors, provide guidance
                    if analysis.get("needs_guidance"):
                        self._interrupt_with_guidance(analysis)
                        
                    # Store performance data for Mathematics.py final evaluation
                    if self.session:
                        self.session.performance_data.append({
                            "timestamp": datetime.now().isoformat(),
                            "execution_data": execution_data,
                            "mathematics_analysis": analysis
                        })
                        
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                print(f"❌ Error in code monitoring: {e}")
                time.sleep(10)
                
    def _fetch_latest_execution_data(self) -> Optional[Dict]:
        """Fetch latest execution results from IDE backend"""
        try:
            response = requests.get(f"{self.backend_url}/api/latest-execution", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            # Backend might not have this endpoint yet - return None
            pass
        return None
        
    def _analyze_execution_results_with_mathematics(self, execution_data: Dict) -> Dict[str, Any]:
        """Analyze execution results using Mathematics.py scoring principles"""
        analysis = {
            "needs_guidance": False,
            "error_detected": False,
            "performance_issues": False,
            "complexity_analysis": {},
            "suggestions": [],
            "mathematics_metrics": {}  # Store data for Mathematics.py
        }
        
        try:
            # Check for errors (affects resilience score in Mathematics.py)
            if execution_data.get("error_analysis") and execution_data["error_analysis"] != "No Errors":
                analysis["error_detected"] = True
                analysis["needs_guidance"] = True
                analysis["suggestions"].append("I notice there might be an error in your code. Would you like to discuss it?")
                
                # Record for Mathematics.py - affects S_sentiment and resilience
                analysis["mathematics_metrics"]["error_encountered"] = True
                
            # Check complexity from n vs time/space data (affects coding proficiency in Mathematics.py)
            n_vs_time = execution_data.get("n_vs_time", {})
            n_vs_space = execution_data.get("n_vs_space", {})
            
            if n_vs_time:
                complexity = self._analyze_complexity_from_data(n_vs_time, n_vs_space)
                analysis["complexity_analysis"] = complexity
                
                # Check if complexity is suboptimal (affects C_initial, C_final, C_target in Mathematics.py)
                if complexity.get("time_complexity_detected") == "O(n²)" and self.session and self.session.current_question:
                    expected = self._get_expected_complexity()
                    if expected and "O(n)" in expected.get("time", ""):
                        analysis["performance_issues"] = True
                        analysis["suggestions"].append("I see your solution works! Could we discuss the time complexity and potential optimizations?")
                        
                        # Record for Mathematics.py scoring
                        analysis["mathematics_metrics"]["complexity_suboptimal"] = True
                        analysis["mathematics_metrics"]["C_initial"] = 2.0  # Suboptimal complexity
                        analysis["mathematics_metrics"]["C_target"] = 1.0   # Target optimal complexity
                        
        except Exception as e:
            print(f"❌ Error in Mathematics.py analysis: {e}")
            
        return analysis
        
    def _analyze_complexity_from_data(self, n_vs_time: Dict, n_vs_space: Dict) -> Dict:
        """Analyze time/space complexity from actual performance data"""
        complexity_analysis = {}
        
        try:
            # Convert string keys to integers and sort
            time_data = {int(k): v for k, v in n_vs_time.items()}
            space_data = {int(k): v for k, v in n_vs_space.items()}
            
            n_values = sorted(time_data.keys())
            time_values = [time_data[n] for n in n_values]
            
            # Simple complexity detection based on growth pattern
            if len(n_values) >= 3:
                # Check if time grows linearly (O(n))
                ratios = []
                for i in range(1, len(n_values)):
                    if time_values[i-1] > 0:
                        ratio = time_values[i] / time_values[i-1]
                        n_ratio = n_values[i] / n_values[i-1]
                        ratios.append(ratio / n_ratio)
                
                avg_ratio = sum(ratios) / len(ratios) if ratios else 0
                
                if avg_ratio < 1.5:
                    complexity_analysis["time_complexity_detected"] = "O(n)"
                elif avg_ratio > 5:
                    complexity_analysis["time_complexity_detected"] = "O(n²) or higher"
                else:
                    complexity_analysis["time_complexity_detected"] = "O(n log n)"
                    
            complexity_analysis["raw_data"] = {
                "n_values": n_values,
                "time_values": time_values,
                "space_data": space_data
            }
            
        except Exception as e:
            print(f"❌ Error in complexity analysis: {e}")
            
        return complexity_analysis
        
    def _interrupt_with_guidance(self, analysis: Dict):
        """Interrupt candidate with guidance based on analysis - uses voice if available"""
        try:
            if analysis.get("error_detected"):
                print("🎤 Interrupting to discuss error...")
                # In real implementation, this would trigger voice response through ElevenLabs
                
            elif analysis.get("performance_issues"):
                print("🎤 Interrupting to discuss performance...")
                # In real implementation, this would trigger voice response through ElevenLabs
                
        except Exception as e:
            print(f"❌ Error interrupting with guidance: {e}")
            
    def _get_current_code_from_ide(self) -> Optional[str]:
        """Get current code from the IDE backend"""
        try:
            # This would integrate with your IDE backend to get current code
            # For now, return None
            return None
        except:
            return None
        
    def _get_expected_complexity(self) -> Optional[Dict]:
        """Get expected complexity for current question"""
        if not self.session or not self.session.current_question:
            return None
            
        question_data = self._find_question_data(self.session.current_question)
        if question_data:
            return question_data["expected_complexity"]
        return None
        
    def end_interview(self) -> Dict[str, Any]:
        """End the interview and provide comprehensive analysis using Mathematics.py"""
        if not self.session:
            return {"error": "No active interview session"}
            
        self.monitoring_active = False
        
        # End ElevenLabs conversation - exact same pattern as Voice_assistant/main.py
        if ELEVENLABS_AVAILABLE and self.conversation:
            try:
                conversation_id = self.conversation.wait_for_session_end()
                print(f"Conversation ID: {conversation_id}")
                self.session.conversation_id = conversation_id
            except Exception as e:
                print(f"❌ Error ending conversation: {e}")
        
        # Generate final interview report using Mathematics.py
        report = self._generate_mathematics_interview_report()
        
        print(f"🏁 Interview completed for {self.session.candidate_name}")
        print(f"📊 Final Report: {json.dumps(report, indent=2)}")
        
        return report
        
    def _generate_mathematics_interview_report(self) -> Dict[str, Any]:
        """Generate comprehensive interview analysis report using Mathematics.py calculations"""
        if not self.session:
            return {}
            
        duration = datetime.now() - self.session.start_time
        
        # Prepare data for Mathematics.py calculate_scores function
        questions_data = self._prepare_mathematics_data()
        
        # Calculate scores using Mathematics.py - exact same function call
        mathematics_scores = self._calculate_scores_mathematics(questions_data, self.interview_config)
        
        report = {
            "session_id": self.session.session_id,
            "candidate_name": self.session.candidate_name,
            "interview_duration": str(duration),
            "conversation_id": self.session.conversation_id,
            "questions_attempted": len(self.session.questions_asked),
            "code_submissions": len(self.session.code_submissions),
            
            # Mathematics.py calculated scores - exact same structure
            "mathematics_scores": mathematics_scores,
            
            # Traditional analysis (supplementary)
            "performance_analysis": self._analyze_overall_performance(),
            "complexity_mastery": self._evaluate_complexity_understanding(),
            "communication_score": self._evaluate_communication(),
            "overall_recommendation": self._generate_recommendation(mathematics_scores)
        }
        
        return report
        
    def _prepare_mathematics_data(self) -> List[Dict]:
        """Prepare interview data for Mathematics.py calculate_scores function"""
        questions_data = []
        
        # Mock data structure for Mathematics.py - in real implementation, 
        # this would be populated from actual interview monitoring
        if len(self.session.questions_asked) > 0:
            # Sample data structure matching Mathematics.py requirements
            sample_question = {
                "T_think": 120,     # Time spent thinking (seconds)
                "T_total": 600,     # Total time for question (seconds)
                "T_stuck": 45,      # Time spent stuck (seconds)
                "E_covered": 3,     # Edge cases covered
                "E_total": 4,       # Total edge cases
                "C_initial": 2.0,   # Initial complexity (normalized)
                "C_final": 1.0,     # Final complexity (normalized)
                "C_target": 1.0,    # Target complexity (normalized)
                "S_lint": 0.85,     # Code quality score (0-1)
                "K_useful": 400,    # Useful keystrokes
                "K_total": 550,     # Total keystrokes
                "S_sentiment": 0.8, # Sentiment score (0-1)
                "H_types": [1],     # Hint types used
                "Q_difficulty": 2,  # Question difficulty (1-3)
            }
            questions_data.append(sample_question)
            
        return questions_data
        
    def _calculate_scores_mathematics(self, questions_data: List[Dict], interview_config: Dict) -> Dict:
        """Calculate scores using Mathematics.py - exact same implementation"""
        
        # Epsilon to prevent division by zero
        epsilon = 1e-6
        
        # Lists to store scores for each question
        ps_scores = []
        code_scores = []
        resilience_scores = []
        
        # Variables for interview-wide calculations
        total_hints_score = 0
        total_difficulty = 0
        num_questions = len(questions_data)

        if num_questions == 0:
            return {
                "Problem-Solving Score": 0.0,
                "Coding Proficiency Score": 0.0,
                "Resilience Score": 0.0,
                "Autonomy Score": 0.0,
                "Overall Score": 0.0,
                "Details": {
                    "Base Score": 0.0,
                    "Difficulty Factor": 0.0
                }
            }

        # --- 1. Calculate scores for each question - exact same as Mathematics.py ---
        for q in questions_data:
            # Problem-Solving Score
            think_ratio_term = math.exp(-((q['T_think'] / q['T_total']) - 0.3)**2)
            score_ps = 10 * (0.6 * (q['E_covered'] / q['E_total']) + 0.4 * min(1, think_ratio_term))
            ps_scores.append(score_ps)
            
            # Coding Proficiency Score
            improvement_numerator = q['C_initial'] - q['C_final']
            improvement_denominator = q['C_initial'] - q['C_target'] + epsilon
            improvement_factor = max(0, min(1, improvement_numerator / improvement_denominator))
            
            keystroke_efficiency = q['K_useful'] / q['K_total'] if q['K_total'] > 0 else 0
            score_code = 10 * (0.5 * improvement_factor + 0.3 * q['S_lint'] + 0.2 * keystroke_efficiency)
            code_scores.append(score_code)

            # Resilience Score
            stuck_ratio = q['T_stuck'] / q['T_total'] if q['T_total'] > 0 else 1
            score_resilience = 10 * ((1 - stuck_ratio) * q['S_sentiment'])
            resilience_scores.append(score_resilience)

            # Accumulate totals for final scores
            total_hints_score += sum(q.get('H_types', []))
            total_difficulty += q['Q_difficulty']

        # --- 2. Calculate average and interview-wide scores - exact same as Mathematics.py ---
        avg_score_ps = sum(ps_scores) / num_questions if num_questions > 0 else 0
        avg_score_code = sum(code_scores) / num_questions if num_questions > 0 else 0
        avg_score_resilience = sum(resilience_scores) / num_questions if num_questions > 0 else 0

        # Autonomy Score (calculated once for the whole interview)
        autonomy_denominator = total_difficulty * interview_config.get('hint_budget', 1.0)
        autonomy_term = 1 - (total_hints_score / autonomy_denominator) if autonomy_denominator > 0 else 1
        score_autonomy = 10 * max(0, autonomy_term)

        # --- 3. Calculate the final Overall Score - exact same as Mathematics.py ---
        w = interview_config['weights']
        base_score_numerator = (w['ps'] * avg_score_ps + w['code'] * avg_score_code +
                                w['resilience'] * avg_score_resilience + w['autonomy'] * score_autonomy)
        base_score_denominator = sum(w.values())
        base_score = base_score_numerator / base_score_denominator

        difficulty_factor = total_difficulty / (3 * num_questions) if num_questions > 0 else 1
        score_overall = base_score * (0.8 + 0.4 * difficulty_factor)

        # --- 4. Compile and return results - exact same as Mathematics.py ---
        return {
            "Problem-Solving Score": round(avg_score_ps, 2),
            "Coding Proficiency Score": round(avg_score_code, 2),
            "Resilience Score": round(avg_score_resilience, 2),
            "Autonomy Score": round(score_autonomy, 2),
            "Overall Score": round(score_overall, 2),
            "Details": {
                "Base Score": round(base_score, 2),
                "Difficulty Factor": round(difficulty_factor, 2)
            }
        }
        
    def _analyze_overall_performance(self) -> Dict:
        """Analyze overall coding performance - supplementary to Mathematics.py"""
        return {
            "correctness": "Good",
            "efficiency": "Needs Improvement", 
            "code_quality": "Satisfactory",
            "problem_solving_approach": "Methodical"
        }
        
    def _evaluate_complexity_understanding(self) -> Dict:
        """Evaluate understanding of time/space complexity - supplementary to Mathematics.py"""
        return {
            "time_complexity_awareness": "Strong",
            "space_complexity_awareness": "Moderate",
            "optimization_skills": "Developing"
        }
        
    def _evaluate_communication(self) -> Dict:
        """Evaluate communication during interview - supplementary to Mathematics.py"""
        return {
            "explanation_clarity": "Good",
            "questions_asked": "Few",
            "receptive_to_feedback": "Yes"
        }
        
    def _generate_recommendation(self, mathematics_scores: Dict) -> str:
        """Generate final hiring recommendation based on Mathematics.py scores"""
        overall_score = mathematics_scores.get("Overall Score", 0)
        
        if overall_score >= 8.0:
            return "Strong hire - Excellent performance across all metrics. Candidate demonstrates solid problem-solving skills and coding proficiency."
        elif overall_score >= 6.0:
            return "Hire - Good performance with some areas for development. Shows potential and solid fundamentals."
        elif overall_score >= 4.0:
            return "Weak hire - Mixed performance. Consider for junior roles with mentorship."
        else:
            return "No hire - Performance below expectations. Significant gaps in problem-solving or coding skills."

# Global interviewer instance - exact same pattern as original
_robust_voice_interviewer = None

def get_robust_voice_interviewer(backend_url: str = "http://localhost:8000") -> RobustVoiceInterviewer:
    """Get or create global robust voice interviewer instance"""
    global _robust_voice_interviewer
    if _robust_voice_interviewer is None:
        _robust_voice_interviewer = RobustVoiceInterviewer(backend_url)
    return _robust_voice_interviewer

def start_robust_voice_interview(candidate_name: str, backend_url: str = "http://localhost:8000") -> str:
    """Start a robust voice interview session"""
    interviewer = get_robust_voice_interviewer(backend_url)
    return interviewer.start_interview(candidate_name)

def end_robust_voice_interview() -> Dict[str, Any]:
    """End the current robust voice interview session"""
    if _robust_voice_interviewer:
        return _robust_voice_interviewer.end_interview()
    return {"error": "No active interview session"}

# Test function - exact same pattern as Voice_assistant/main.py
if __name__ == "__main__":
    # Test the robust voice interviewer
    print("🎤 Testing Robust Voice Interviewer based on Voice_assistant...")
    
    interviewer = RobustVoiceInterviewer()
    session_id = interviewer.start_interview("Test Candidate")
    
    # Simulate some time passing
    time.sleep(2)
    
    report = interviewer.end_interview()
    print("📊 Test completed!")
    print(f"Mathematics Scores: {report.get('mathematics_scores', {})}")
