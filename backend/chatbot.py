"""
Simple Gemini Chatbot for Code Editor

This module provides a simple chatbot interface using Google's Gemini API
to answer questions about code, algorithms, and programming concepts.
"""

import os
import json
from typing import Dict, Any, Optional
import google.generativeai as genai

class SimpleChatbot:
    """
    Simple Gemini-based chatbot that can answer questions about code, algorithms,
    and programming concepts.
    """
    
    def __init__(self, api_key: str = None):
        # Use provided Gemini API key - you'll need to provide your key here
        self.gemini_api_key = "AIzaSyA-epFo2vqg96Hc7RV88mJB7BOqqhEI-aw"
        
        # Initialize Gemini client
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
            print("✅ Gemini chatbot initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Gemini chatbot: {e}")
            self.gemini_model = None
    
    def chat(self, message: str, code: str = None, language: str = None) -> Dict[str, Any]:
        """
        Send a message to the chatbot and get a response
        
        Args:
            message: User's message/question
            code: Optional code snippet for context
            language: Optional programming language
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            if not self.gemini_model:
                return {
                    "success": False,
                    "response": "Chatbot not initialized. Please check the API key.",
                    "error": "Model not available"
                }
            
            # Create context-aware prompt
            prompt = self._create_chat_prompt(message, code, language)
            
            # Make API call
            response = self._call_gemini_api(prompt)
            
            if response and response.get("success"):
                return {
                    "success": True,
                    "response": response["content"],
                    "message_type": "assistant"
                }
            else:
                return {
                    "success": False,
                    "response": "I'm having trouble responding right now. Please try again.",
                    "error": "API call failed"
                }
                
        except Exception as e:
            print(f"❌ Chat error: {e}")
            return {
                "success": False,
                "response": f"Sorry, I encountered an error: {str(e)}",
                "error": str(e)
            }
    
    def _create_chat_prompt(self, message: str, code: str = None, language: str = None) -> str:
        """Create a context-aware prompt for the chatbot"""
        
        system_prompt = """You are a helpful AI assistant specializing in programming and computer science. 
        You can help with:
        - Code explanation and debugging
        - Algorithm and data structure questions
        - Time and space complexity analysis
        - Programming best practices
        - Code optimization suggestions
        
        Provide clear, concise, and helpful responses. If you see code, explain it clearly.
        If asked about complexity, give specific Big O notation with explanations.
        
        Just give 1 single line response within 10-20words
        """
        
        user_prompt = f"User question: {message}"
        
        if code and language:
            user_prompt += f"\n\nCode context ({language}):\n```{language}\n{code}\n```"
        elif code:
            user_prompt += f"\n\nCode context:\n```\n{code}\n```"
        
        return f"{system_prompt}\n\n{user_prompt}"
    
    def _call_gemini_api(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Make API call to Gemini"""
        try:
            print(f"🤖 Sending message to Gemini...")
            
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,  # More creative for conversations
                    max_output_tokens=2000,
                    top_p=0.9
                )
            )
            
            print(f"✅ Gemini response received")
            return {
                "success": True,
                "content": response.text
            }
            
        except Exception as e:
            print(f"❌ Gemini API error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Global chatbot instance
_chatbot = None

def get_chatbot(api_key: str = None) -> SimpleChatbot:
    """Get or create the global chatbot instance"""
    global _chatbot
    if _chatbot is None:
        _chatbot = SimpleChatbot(api_key)
    return _chatbot

def send_chat_message(message: str, code: str = None, language: str = None, api_key: str = None) -> Dict[str, Any]:
    """
    Send a message to the chatbot
    
    Args:
        message: User's message
        code: Optional code context
        language: Optional programming language
        api_key: Optional API key for initialization
        
    Returns:
        Chatbot response dictionary
    """
    try:
        chatbot = get_chatbot(api_key)
        return chatbot.chat(message, code, language)
    except Exception as e:
        return {
            "success": False,
            "response": f"Chatbot error: {str(e)}",
            "error": str(e)
        }

# Test function
if __name__ == "__main__":
    # Test the chatbot
    test_message = "What is the time complexity of bubble sort?"
    test_code = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
    """
    
    result = send_chat_message(
        message=test_message,
        code=test_code,
        language="python"
    )
    
    print("Chatbot Test Result:")
    print(json.dumps(result, indent=2))