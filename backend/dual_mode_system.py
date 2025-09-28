"""
Dual Mode Interview System - Chat and Voice Integration

This module implements the dual interview system:
- Chat Mode: Uses Gemini for IDE-aware text-based interviews
- Voice Mode: Uses ElevenLabs for voice-based interviews
- Both modes provide real-time IDE monitoring and guidance
"""

import os
import json
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

class DualModeInterviewSystem:
    """
    Dual Mode Interview System supporting both Chat (Gemini) and Voice (ElevenLabs) modes
    """
    
    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url
        
        # Initialize Gemini for chat mode
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.chat_model = genai.GenerativeModel('gemini-pro')
            print("✅ Gemini Chat Model initialized successfully")
        else:
            print("⚠️  Gemini API key not found - Chat mode disabled")
            self.chat_model = None
        
        # Professional interview prompts
        self.chat_interview_system_prompt = """
        You are a professional AI technical interviewer conducting a formal coding interview.
        
        Your role:
        - Conduct professional, structured technical interviews
        - Ask clear, well-defined coding problems
        - Provide constructive guidance and hints when appropriate
        - Analyze code quality, efficiency, and problem-solving approach
        - Maintain formal but supportive tone throughout
        - Focus on both correctness and optimization
        
        Interview structure:
        1. Present coding problems with clear requirements
        2. Monitor candidate's approach and progress
        3. Provide hints if candidate is stuck (limit: 2-3 hints per problem)
        4. Discuss time/space complexity and optimizations
        5. Evaluate communication and technical explanation skills
        
        Current interview context: {context}
        """
        
        self.current_chat_context = []
        
    def start_chat_interview(self, candidate_name: str, question: Dict) -> str:
        """Start a chat-based interview session"""
        
        context = f"""
        Candidate: {candidate_name}
        Current Question: {question['title']}
        Description: {question['description']}
        Expected Complexity: {question['expected_complexity']}
        Professional Introduction: {question.get('professional_intro', '')}
        """
        
        # Initialize chat with professional introduction
        intro_prompt = f"""
        {self.chat_interview_system_prompt.format(context=context)}
        
        Please provide a professional introduction for candidate {candidate_name} and present the first coding problem:
        
        Problem: {question['title']}
        {question['description']}
        
        Expected Time/Space Complexity: {question['expected_complexity']}
        
        Begin the interview with a warm but professional greeting, explain the structure, and present the problem clearly.
        """
        
        try:
            if self.chat_model:
                response = self.chat_model.generate_content(intro_prompt)
                interview_intro = response.text
                
                # Store context
                self.current_chat_context = [
                    {"role": "system", "content": intro_prompt},
                    {"role": "assistant", "content": interview_intro}
                ]
                
                return interview_intro
            else:
                return "Chat mode not available - Gemini API not configured"
                
        except Exception as e:
            print(f"❌ Error starting chat interview: {e}")
            return f"Error starting chat interview: {str(e)}"
    
    def process_chat_message(self, candidate_message: str, ide_context: Optional[Dict] = None) -> str:
        """Process candidate's chat message with IDE awareness"""
        
        # Add IDE context if available
        ide_info = ""
        if ide_context:
            current_code = ide_context.get("current_code", "")
            execution_results = ide_context.get("execution_results", {})
            errors = ide_context.get("errors", "")
            
            ide_info = f"""
            
            CURRENT IDE STATE:
            Code being worked on:
            ```
            {current_code}
            ```
            
            Execution Results: {execution_results}
            Errors: {errors if errors else "None"}
            """
        
        # Construct professional response prompt
        response_prompt = f"""
        Continue the professional technical interview. 
        
        Candidate's latest message: "{candidate_message}"
        {ide_info}
        
        As a professional interviewer:
        1. Respond to the candidate's question or comment professionally
        2. If they're asking for help, provide appropriate guidance (limited hints)
        3. If they've submitted code, analyze it for correctness and efficiency
        4. If there are errors in their IDE, help them understand and fix them
        5. If they seem stuck, provide encouragement and a helpful hint
        6. If they've solved it, discuss complexity and potential optimizations
        7. Maintain professional, supportive tone throughout
        
        Respond as the interviewer would in a real professional interview.
        """
        
        try:
            if self.chat_model:
                response = self.chat_model.generate_content(response_prompt)
                interviewer_response = response.text
                
                # Update context
                self.current_chat_context.extend([
                    {"role": "user", "content": candidate_message + ide_info},
                    {"role": "assistant", "content": interviewer_response}
                ])
                
                return interviewer_response
            else:
                return "Chat mode not available - Gemini API not configured"
                
        except Exception as e:
            print(f"❌ Error processing chat message: {e}")
            return f"I apologize, but I'm experiencing a technical issue. Could you please repeat your message?"
    
    def get_ide_context(self) -> Optional[Dict]:
        """Get current IDE context for interview awareness"""
        try:
            # Get latest execution results
            execution_response = requests.get(f"{self.backend_url}/api/latest-execution", timeout=5)
            execution_data = execution_response.json() if execution_response.status_code == 200 else {}
            
            # Get current code (if endpoint exists)
            try:
                code_response = requests.get(f"{self.backend_url}/api/current-code", timeout=5)
                current_code = code_response.json().get("code", "") if code_response.status_code == 200 else ""
            except:
                current_code = ""
            
            return {
                "current_code": current_code,
                "execution_results": execution_data,
                "errors": execution_data.get("error_analysis", ""),
                "complexity_detected": execution_data.get("complexity_detected", "Unknown"),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error getting IDE context: {e}")
            return None
    
    def provide_professional_guidance(self, guidance_type: str = "general") -> str:
        """Provide professional guidance based on type"""
        
        guidance_prompts = {
            "stuck": """
            The candidate seems to be stuck. As a professional interviewer, provide encouraging guidance:
            - Acknowledge their effort
            - Ask about their current approach
            - Provide a subtle hint to help them progress
            - Maintain supportive professional tone
            """,
            "error": """
            The candidate has encountered an error. As a professional interviewer:
            - Help them understand what the error means
            - Guide them toward the solution without giving it away
            - Encourage systematic debugging approach
            - Remain patient and supportive
            """,
            "optimization": """
            The candidate has a working solution. As a professional interviewer:
            - Congratulate them on getting it working
            - Discuss the current time/space complexity
            - Ask if they see opportunities for optimization
            - Guide the discussion toward better solutions if needed
            """,
            "general": """
            Provide general professional encouragement and guidance:
            - Check on their progress
            - Ask if they have any questions
            - Offer to clarify problem requirements if needed
            - Maintain professional supportive presence
            """
        }
        
        prompt = guidance_prompts.get(guidance_type, guidance_prompts["general"])
        
        try:
            if self.chat_model:
                response = self.chat_model.generate_content(prompt)
                return response.text
            else:
                return "I'm here to help - what specific question do you have about the problem?"
                
        except Exception as e:
            print(f"❌ Error providing guidance: {e}")
            return "I'm here to support you. What would you like to discuss about the current problem?"
    
    def analyze_candidate_performance(self, session_data: Dict) -> Dict:
        """Analyze candidate performance for both chat and voice modes"""
        
        analysis_prompt = f"""
        As a professional technical interviewer, analyze this candidate's performance:
        
        Session Data: {json.dumps(session_data, indent=2)}
        
        Provide professional analysis covering:
        1. Problem-solving approach and methodology
        2. Code quality and technical implementation
        3. Communication effectiveness and clarity
        4. Response to guidance and adaptability
        5. Time management and efficiency
        6. Technical depth and understanding
        
        Format as a structured professional assessment.
        """
        
        try:
            if self.chat_model:
                response = self.chat_model.generate_content(analysis_prompt)
                return {
                    "ai_analysis": response.text,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "analysis_type": "professional_interview_assessment"
                }
            else:
                return {
                    "ai_analysis": "Performance analysis not available - AI model not configured",
                    "analysis_timestamp": datetime.now().isoformat(),
                    "analysis_type": "manual_assessment_required"
                }
                
        except Exception as e:
            print(f"❌ Error analyzing performance: {e}")
            return {
                "ai_analysis": f"Analysis error: {str(e)}",
                "analysis_timestamp": datetime.now().isoformat(),
                "analysis_type": "error"
            }
    
    def generate_next_question(self, difficulty_level: str, previous_performance: Dict) -> Dict:
        """Generate next appropriate question based on performance"""
        
        question_prompt = f"""
        As a professional technical interviewer, suggest the next coding problem:
        
        Target Difficulty: {difficulty_level}
        Previous Performance: {json.dumps(previous_performance, indent=2)}
        
        Provide a coding problem with:
        1. Clear title and description
        2. Appropriate difficulty for the candidate's demonstrated level
        3. Expected time/space complexity
        4. Professional introduction/context
        5. 2-3 progressive hints for guidance
        
        Format as a structured problem definition.
        """
        
        try:
            if self.chat_model:
                response = self.chat_model.generate_content(question_prompt)
                
                # Parse AI response into structured format
                return {
                    "title": f"AI-Generated {difficulty_level.title()} Problem",
                    "description": response.text,
                    "expected_complexity": {"time": "TBD", "space": "TBD"},
                    "difficulty": 1 if difficulty_level == "easy" else 2 if difficulty_level == "medium" else 3,
                    "professional_intro": "This problem is tailored to your demonstrated skill level.",
                    "hints": ["Consider your approach step by step", "Think about edge cases", "Optimize for the expected complexity"],
                    "ai_generated": True,
                    "generation_timestamp": datetime.now().isoformat()
                }
            else:
                # Fallback to predefined questions
                return {
                    "title": "Standard Array Problem",
                    "description": "Work with arrays and basic algorithms.",
                    "expected_complexity": {"time": "O(n)", "space": "O(1)"},
                    "difficulty": 2,
                    "professional_intro": "Let's work on a fundamental array problem.",
                    "hints": ["Consider iteration", "Think about efficiency", "Handle edge cases"]
                }
                
        except Exception as e:
            print(f"❌ Error generating question: {e}")
            return {
                "title": "Error - Fallback Problem",
                "description": "Please solve a basic coding problem of your choice.",
                "expected_complexity": {"time": "O(n)", "space": "O(1)"},
                "difficulty": 1,
                "professional_intro": "Let's continue with a coding exercise.",
                "hints": ["Take your time", "Think step by step", "Ask if you need help"]
            }

# Global dual mode system instance
_dual_mode_system = None

def get_dual_mode_system(backend_url: str = "http://localhost:8000") -> DualModeInterviewSystem:
    """Get or create global dual mode interview system"""
    global _dual_mode_system
    if _dual_mode_system is None:
        _dual_mode_system = DualModeInterviewSystem(backend_url)
    return _dual_mode_system

if __name__ == "__main__":
    print("🎯 Testing Dual Mode Interview System...")
    
    # Test the system
    system = DualModeInterviewSystem()
    
    # Test question
    test_question = {
        "title": "Two Sum Problem",
        "description": "Given an array of integers and a target, return indices of two numbers that add up to target.",
        "expected_complexity": {"time": "O(n)", "space": "O(n)"},
        "professional_intro": "Let's start with a fundamental algorithmic problem."
    }
    
    # Test chat interview
    intro = system.start_chat_interview("Test Candidate", test_question)
    print("📋 Chat Interview Introduction:")
    print(intro)
    
    # Test message processing
    response = system.process_chat_message("I'm not sure how to approach this problem.", None)
    print("💬 Chat Response:")
    print(response)