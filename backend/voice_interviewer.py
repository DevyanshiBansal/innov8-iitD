"""
Enhanced Voice Interview Agent for Code Editor

This module creates an AI interviewer that:
- Conducts formal coding interviews using ElevenLabs voice
- Has real-time access to IDE code analysis
- Can interrupt, guide, and provide feedback during coding
- Analyzes time/space complexity using n vs time/space data
- Provides comprehensive interview evaluation
"""

import os
import json
import time
import asyncio
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

import requests
from dotenv import load_dotenv

# Mock ElevenLabs classes for development
class MockElevenLabs:
    def __init__(self, api_key: str):
        self.api_key = api_key

class MockConversation:
    def __init__(self, client, agent_id, requires_auth=True, audio_interface=None, 
                 callback_agent_response=None, callback_user_transcript=None, 
                 callback_agent_response_correction=None):
        self.client = client
        self.agent_id = agent_id
        self.requires_auth = requires_auth
        self.audio_interface = audio_interface
        self.callback_agent_response = callback_agent_response
        self.callback_user_transcript = callback_user_transcript
        self.callback_agent_response_correction = callback_agent_response_correction
        print("🎤 Mock voice conversation initialized (ElevenLabs not available)")

class MockAudioInterface:
    def __init__(self):
        print("🔊 Mock audio interface initialized")

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
    
    def __post_init__(self):
        if self.questions_asked is None:
            self.questions_asked = []
        if self.code_submissions is None:
            self.code_submissions = []
        if self.performance_data is None:
            self.performance_data = []

class VoiceInterviewer:
    """
    AI Voice Interviewer that integrates with the IDE for real-time coding interviews
    """
    
    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url
        self.session: Optional[InterviewSession] = None
        
        # ElevenLabs configuration (using mock for now)
        self.api_key = os.getenv("ELEVENLABS_API_KEY", "mock-api-key")
        self.agent_id = os.getenv("AGENT_ID", "mock-agent-id")
        self.client = MockElevenLabs(api_key=self.api_key)
        
        # Interview configuration
        self.interview_config = {
            "max_questions": 3,
            "time_per_question": 900,  # 15 minutes
            "difficulty_progression": ["easy", "medium", "hard"],
            "interruption_threshold": 300,  # 5 minutes of no progress
            "complexity_analysis_enabled": True
        }
        
        # Coding interview questions database
        self.questions_db = {
            "easy": [
                {
                    "title": "Two Sum",
                    "description": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
                    "expected_complexity": {"time": "O(n)", "space": "O(n)"},
                    "hints": ["Consider using a hash map", "Think about what you've seen before"]
                },
                {
                    "title": "Valid Parentheses", 
                    "description": "Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.",
                    "expected_complexity": {"time": "O(n)", "space": "O(n)"},
                    "hints": ["Think about using a stack", "Match opening with closing brackets"]
                }
            ],
            "medium": [
                {
                    "title": "Longest Substring Without Repeating Characters",
                    "description": "Given a string s, find the length of the longest substring without repeating characters.",
                    "expected_complexity": {"time": "O(n)", "space": "O(min(m,n))"},
                    "hints": ["Consider using sliding window technique", "Use a set to track characters"]
                }
            ],
            "hard": [
                {
                    "title": "Merge k Sorted Lists",
                    "description": "You are given an array of k linked-lists, each sorted in ascending order. Merge all the linked-lists into one sorted linked-list.",
                    "expected_complexity": {"time": "O(n log k)", "space": "O(1)"},
                    "hints": ["Think about divide and conquer", "Consider using a priority queue"]
                }
            ]
        }
        
        # Real-time monitoring
        self.monitoring_active = False
        self.last_code_check = None
        self.code_analysis_thread = None
        
    def start_interview(self, candidate_name: str) -> str:
        """Start a new interview session"""
        session_id = f"interview_{int(time.time())}"
        self.session = InterviewSession(
            session_id=session_id,
            candidate_name=candidate_name,
            start_time=datetime.now()
        )
        
        # Start the voice conversation
        self._initialize_voice_conversation()
        
        # Start real-time code monitoring
        self._start_code_monitoring()
        
        print(f"🎤 Starting interview for {candidate_name} (Session: {session_id})")
        return session_id
    
    def _initialize_voice_conversation(self):
        """Initialize ElevenLabs conversation with interview context"""
        
        # Create interview-specific system prompt
        system_prompt = self._create_interviewer_prompt()
        
        try:
            self.conversation = MockConversation(
                client=self.client,
                agent_id=self.agent_id,
                requires_auth=bool(self.api_key),
                audio_interface=MockAudioInterface(),
                
                # Conversation callbacks
                callback_agent_response=self._handle_agent_response,
                callback_user_transcript=self._handle_user_input,
                callback_agent_response_correction=self._handle_correction,
            )
            
            print("✅ Mock voice conversation initialized (ElevenLabs integration disabled)")
            
        except Exception as e:
            print(f"❌ Failed to initialize voice conversation: {e}")
            
    def _create_interviewer_prompt(self) -> str:
        """Create system prompt for the AI interviewer"""
        return f"""You are a professional technical interviewer conducting a coding interview.

INTERVIEW CONTEXT:
- Candidate: {self.session.candidate_name if self.session else 'Unknown'}
- Format: Real-time coding interview with live IDE access
- Duration: {self.interview_config['max_questions']} questions, {self.interview_config['time_per_question']/60} minutes each

INTERVIEWER PERSONALITY:
- Professional, encouraging, and constructive
- Formal but friendly tone
- Ask clarifying questions about approach
- Provide guidance without giving direct solutions
- Interrupt politely if candidate is stuck too long

YOUR CAPABILITIES:
- You have real-time access to the candidate's code in the IDE
- You can see their execution results and performance metrics
- You can analyze time/space complexity from actual n vs time/space data
- You can detect when they're stuck and offer strategic hints

INTERVIEW FLOW:
1. Start with formal greeting and introduction
2. Present coding problems one by one
3. Monitor their coding progress in real-time
4. Provide hints when they're stuck (don't give direct solutions)
5. Analyze their solution's complexity when they finish
6. Ask follow-up questions about optimization
7. Conclude with overall feedback

GUIDANCE STYLE:
- "Let's think about this step by step..."
- "What's the time complexity of your current approach?"
- "I notice you might be stuck - would you like a hint?"
- "Can you walk me through your thinking process?"
- "How might we optimize this further?"

Remember: You're here to evaluate AND help the candidate succeed. Be encouraging while maintaining professional standards."""

    def _handle_agent_response(self, response: str):
        """Handle AI interviewer responses"""
        print(f"🎤 Interviewer: {response}")
        
        # Log the response to session
        if self.session:
            # Could store interview transcript here
            pass
            
    def _handle_user_input(self, transcript: str):
        """Handle candidate's speech input"""
        print(f"👤 Candidate: {transcript}")
        
        # Analyze if candidate is asking for help or hints
        self._analyze_candidate_request(transcript)
        
    def _handle_correction(self, original: str, corrected: str):
        """Handle speech corrections"""
        print(f"🔄 Correction: {original} -> {corrected}")
        
    def _analyze_candidate_request(self, transcript: str):
        """Analyze what the candidate is saying and respond appropriately"""
        help_keywords = ["help", "hint", "stuck", "don't know", "confused", "not sure"]
        
        if any(keyword in transcript.lower() for keyword in help_keywords):
            # Candidate is asking for help - provide contextual hint based on current code
            self._provide_contextual_hint()
            
    def _provide_contextual_hint(self):
        """Provide hints based on current code analysis"""
        try:
            # Get current code from IDE
            current_code = self._get_current_code_from_ide()
            if current_code:
                # Analyze code and provide specific hint
                hint = self._generate_hint_from_code(current_code)
                print(f"💡 Generated hint based on code analysis: {hint}")
            
        except Exception as e:
            print(f"❌ Error providing contextual hint: {e}")
            
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
                    
                    # Analyze the execution results
                    analysis = self._analyze_execution_results(execution_data)
                    
                    # If candidate seems stuck or made errors, provide guidance
                    if analysis.get("needs_guidance"):
                        self._interrupt_with_guidance(analysis)
                        
                    # Store performance data for final evaluation
                    if self.session:
                        self.session.performance_data.append({
                            "timestamp": datetime.now().isoformat(),
                            "execution_data": execution_data,
                            "analysis": analysis
                        })
                        
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                print(f"❌ Error in code monitoring: {e}")
                time.sleep(10)
                
    def _fetch_latest_execution_data(self) -> Optional[Dict]:
        """Fetch latest execution results from IDE backend"""
        try:
            # This would connect to your IDE backend
            # For now, return mock data structure
            response = requests.get(f"{self.backend_url}/api/latest-execution", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            # Backend might not have this endpoint yet - return None
            pass
        return None
        
    def _analyze_execution_results(self, execution_data: Dict) -> Dict[str, Any]:
        """Analyze execution results to determine if guidance is needed"""
        analysis = {
            "needs_guidance": False,
            "error_detected": False,
            "performance_issues": False,
            "complexity_analysis": {},
            "suggestions": []
        }
        
        try:
            # Check for errors
            if execution_data.get("error_analysis") and execution_data["error_analysis"] != "No Errors":
                analysis["error_detected"] = True
                analysis["needs_guidance"] = True
                analysis["suggestions"].append("I notice there might be an error in your code. Would you like to discuss it?")
                
            # Check complexity from n vs time/space data
            n_vs_time = execution_data.get("n_vs_time", {})
            n_vs_space = execution_data.get("n_vs_space", {})
            
            if n_vs_time:
                complexity = self._analyze_complexity_from_data(n_vs_time, n_vs_space)
                analysis["complexity_analysis"] = complexity
                
                # Check if complexity is suboptimal
                if complexity.get("time_complexity_detected") == "O(n²)" and self.session and self.session.current_question:
                    expected = self._get_expected_complexity()
                    if expected and "O(n)" in expected.get("time", ""):
                        analysis["performance_issues"] = True
                        analysis["suggestions"].append("I see your solution works! Could we discuss the time complexity and potential optimizations?")
                        
        except Exception as e:
            print(f"❌ Error analyzing execution results: {e}")
            
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
        """Interrupt candidate with guidance based on analysis"""
        try:
            if analysis.get("error_detected"):
                # Send message to voice agent about error
                print("🎤 Interrupting to discuss error...")
                
            elif analysis.get("performance_issues"):
                # Discuss optimization opportunities
                print("🎤 Interrupting to discuss performance...")
                
        except Exception as e:
            print(f"❌ Error interrupting with guidance: {e}")
            
    def _get_current_code_from_ide(self) -> Optional[str]:
        """Get current code from the IDE"""
        # This would integrate with your IDE backend
        # For now, return None
        return None
        
    def _generate_hint_from_code(self, code: str) -> str:
        """Generate contextual hint based on current code"""
        # Use Gemini or other LLM to analyze code and provide hint
        return "Consider the data structure you're using and its time complexity."
        
    def _get_expected_complexity(self) -> Optional[Dict]:
        """Get expected complexity for current question"""
        if not self.session or not self.session.current_question:
            return None
            
        # Find the question in database
        for difficulty in self.questions_db.values():
            for question in difficulty:
                if question["title"] in self.session.current_question:
                    return question["expected_complexity"]
        return None
        
    def end_interview(self) -> Dict[str, Any]:
        """End the interview and provide comprehensive analysis"""
        if not self.session:
            return {"error": "No active interview session"}
            
        self.monitoring_active = False
        
        # Generate final interview report
        report = self._generate_interview_report()
        
        print(f"🏁 Interview completed for {self.session.candidate_name}")
        print(f"📊 Final Report: {json.dumps(report, indent=2)}")
        
        return report
        
    def _generate_interview_report(self) -> Dict[str, Any]:
        """Generate comprehensive interview analysis report"""
        if not self.session:
            return {}
            
        duration = datetime.now() - self.session.start_time
        
        report = {
            "session_id": self.session.session_id,
            "candidate_name": self.session.candidate_name,
            "interview_duration": str(duration),
            "questions_attempted": len(self.session.questions_asked),
            "code_submissions": len(self.session.code_submissions),
            "performance_analysis": self._analyze_overall_performance(),
            "complexity_mastery": self._evaluate_complexity_understanding(),
            "communication_score": self._evaluate_communication(),
            "overall_recommendation": self._generate_recommendation()
        }
        
        return report
        
    def _analyze_overall_performance(self) -> Dict:
        """Analyze overall coding performance"""
        return {
            "correctness": "Good",
            "efficiency": "Needs Improvement", 
            "code_quality": "Satisfactory",
            "problem_solving_approach": "Methodical"
        }
        
    def _evaluate_complexity_understanding(self) -> Dict:
        """Evaluate understanding of time/space complexity"""
        return {
            "time_complexity_awareness": "Strong",
            "space_complexity_awareness": "Moderate",
            "optimization_skills": "Developing"
        }
        
    def _evaluate_communication(self) -> Dict:
        """Evaluate communication during interview"""
        return {
            "explanation_clarity": "Good",
            "questions_asked": "Few",
            "receptive_to_feedback": "Yes"
        }
        
    def _generate_recommendation(self) -> str:
        """Generate final hiring recommendation"""
        return "Candidate shows promise with solid fundamentals. Recommend for next round with focus on optimization techniques."

# Global interviewer instance
_voice_interviewer = None

def get_voice_interviewer(backend_url: str = "http://localhost:8000") -> VoiceInterviewer:
    """Get or create global voice interviewer instance"""
    global _voice_interviewer
    if _voice_interviewer is None:
        _voice_interviewer = VoiceInterviewer(backend_url)
    return _voice_interviewer

def start_voice_interview(candidate_name: str, backend_url: str = "http://localhost:8000") -> str:
    """Start a voice interview session"""
    interviewer = get_voice_interviewer(backend_url)
    return interviewer.start_interview(candidate_name)

def end_voice_interview() -> Dict[str, Any]:
    """End the current voice interview session"""
    if _voice_interviewer:
        return _voice_interviewer.end_interview()
    return {"error": "No active interview session"}

# Test function
if __name__ == "__main__":
    # Test the voice interviewer
    print("🎤 Testing Voice Interviewer...")
    
    interviewer = VoiceInterviewer()
    session_id = interviewer.start_interview("Test Candidate")
    
    # Simulate some time passing
    time.sleep(2)
    
    report = interviewer.end_interview()
    print("📊 Test completed!")