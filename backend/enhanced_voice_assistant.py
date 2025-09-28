"""
Enhanced Voice Assistant Integration for Adaptive Coding System

This module integrates the ElevenLabs voice assistant with the adaptive coding platform,
providing voice-based problem-solving guidance and real-time conversation support.
"""

from typing import Dict, Optional, Any
from pydantic import BaseModel
from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface
import os
from dotenv import load_dotenv
import asyncio
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceAssistantConfig:
    """Configuration for the voice assistant"""
    AGENT_ID: str = os.getenv("AGENT_ID", "agent_7201k669wge0e0faqczt5h5qqcja")
    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "sk_bf68cd37c9636d8d41a5b7b9b6351965db0ca54993a57f14")
    TIMEOUT_SECONDS: int = 300  # 5 minutes timeout

class ConversationStartRequest(BaseModel):
    """Request model for starting a conversation"""
    agent_id: Optional[str] = None
    api_key: Optional[str] = None
    context: Optional[str] = None  # Problem context for the conversation

class VoiceMessageRequest(BaseModel):
    """Request model for sending voice messages"""
    conversation_id: str
    message: str
    problem_context: Optional[Dict[str, Any]] = None

class VoiceAssistantManager:
    """
    Enhanced Voice Assistant Manager for Adaptive Coding System
    
    Manages ElevenLabs conversational AI integration with coding problem guidance
    """
    
    def __init__(self):
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        self.config = VoiceAssistantConfig()
        self.client = None
        
        try:
            self.client = ElevenLabs(api_key=self.config.ELEVENLABS_API_KEY)
            logger.info("✅ Voice Assistant initialized successfully")
        except Exception as e:
            logger.error(f"⚠️ Failed to initialize Voice Assistant: {e}")
    
    async def start_conversation(
        self, 
        agent_id: Optional[str] = None, 
        api_key: Optional[str] = None,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Start a new voice conversation with optional problem context
        """
        try:
            if not self.client:
                self.client = ElevenLabs(api_key=api_key or self.config.ELEVENLABS_API_KEY)
            
            # Create conversation with problem context
            conversation = Conversation(
                client=self.client,
                agent_id=agent_id or self.config.AGENT_ID,
                requires_auth=bool(api_key or self.config.ELEVENLABS_API_KEY),
                audio_interface=DefaultAudioInterface(),
            )
            
            # Start the session
            conversation.start_session()
            conversation_id = str(conversation.session_id)
            
            # Store conversation with context
            self.active_conversations[conversation_id] = {
                "conversation": conversation,
                "context": context,
                "start_time": asyncio.get_event_loop().time(),
                "message_count": 0
            }
            
            # Send initial context if provided
            if context:
                initial_message = f"I'm working on a coding problem. Here's the context: {context}. Please help me understand and solve this problem step by step."
                try:
                    conversation.send_message(initial_message)
                except Exception as e:
                    logger.warning(f"Failed to send initial context: {e}")
            
            logger.info(f"✅ Started voice conversation: {conversation_id}")
            return {
                "success": True,
                "conversation_id": conversation_id,
                "message": "Voice conversation started successfully"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to start conversation: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to start voice conversation"
            }
    
    async def send_message(
        self, 
        conversation_id: str, 
        message: str, 
        problem_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send a message to an active conversation with optional problem context
        """
        try:
            if conversation_id not in self.active_conversations:
                return {
                    "success": False,
                    "error": "Conversation not found",
                    "message": "Please start a new conversation first"
                }
            
            conversation_data = self.active_conversations[conversation_id]
            conversation = conversation_data["conversation"]
            
            # Check timeout
            current_time = asyncio.get_event_loop().time()
            if current_time - conversation_data["start_time"] > self.config.TIMEOUT_SECONDS:
                await self.end_conversation(conversation_id)
                return {
                    "success": False,
                    "error": "Conversation timeout",
                    "message": "Conversation has timed out. Please start a new one."
                }
            
            # Enhance message with problem context
            enhanced_message = message
            if problem_context:
                context_info = []
                if problem_context.get("problem_title"):
                    context_info.append(f"Problem: {problem_context['problem_title']}")
                if problem_context.get("difficulty"):
                    context_info.append(f"Difficulty: {problem_context['difficulty']}")
                if problem_context.get("time_spent"):
                    context_info.append(f"Time spent: {problem_context['time_spent']} minutes")
                if problem_context.get("attempts"):
                    context_info.append(f"Attempts: {problem_context['attempts']}")
                
                if context_info:
                    enhanced_message = f"[Context: {', '.join(context_info)}] {message}"
            
            # Send message and get response
            response = conversation.send_message(enhanced_message)
            conversation_data["message_count"] += 1
            
            logger.info(f"✅ Message sent to conversation: {conversation_id}")
            return {
                "success": True,
                "response": response,
                "message_count": conversation_data["message_count"]
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to send message: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to send message"
            }
    
    async def end_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        End an active voice conversation
        """
        try:
            if conversation_id not in self.active_conversations:
                return {
                    "success": False,
                    "error": "Conversation not found",
                    "message": "Conversation does not exist"
                }
            
            conversation_data = self.active_conversations[conversation_id]
            conversation = conversation_data["conversation"]
            
            # End the conversation
            conversation.end_session()
            
            # Remove from active conversations
            del self.active_conversations[conversation_id]
            
            logger.info(f"✅ Ended voice conversation: {conversation_id}")
            return {
                "success": True,
                "message": "Voice conversation ended successfully",
                "message_count": conversation_data["message_count"]
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to end conversation: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to end conversation"
            }
    
    async def get_active_conversations(self) -> Dict[str, Any]:
        """
        Get list of all active conversations
        """
        try:
            conversations_info = []
            current_time = asyncio.get_event_loop().time()
            
            for conv_id, conv_data in self.active_conversations.items():
                conversations_info.append({
                    "conversation_id": conv_id,
                    "duration": int(current_time - conv_data["start_time"]),
                    "message_count": conv_data["message_count"],
                    "context": conv_data.get("context", "No context")
                })
            
            return {
                "success": True,
                "active_conversations": conversations_info,
                "total_count": len(conversations_info)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get active conversations: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to retrieve active conversations"
            }
    
    async def cleanup_expired_conversations(self):
        """
        Clean up expired conversations (background task)
        """
        try:
            current_time = asyncio.get_event_loop().time()
            expired_conversations = []
            
            for conv_id, conv_data in self.active_conversations.items():
                if current_time - conv_data["start_time"] > self.config.TIMEOUT_SECONDS:
                    expired_conversations.append(conv_id)
            
            for conv_id in expired_conversations:
                await self.end_conversation(conv_id)
                logger.info(f"🧹 Cleaned up expired conversation: {conv_id}")
            
            return len(expired_conversations)
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup conversations: {e}")
            return 0

# Global instance
voice_assistant_manager = None

def get_voice_assistant_manager() -> VoiceAssistantManager:
    """Get or create the voice assistant manager instance"""
    global voice_assistant_manager
    if voice_assistant_manager is None:
        voice_assistant_manager = VoiceAssistantManager()
    return voice_assistant_manager

# Test the voice assistant
async def test_voice_assistant():
    """Test function for the voice assistant"""
    manager = get_voice_assistant_manager()
    
    print("🚀 Testing Voice Assistant Integration")
    print("=" * 50)
    
    # Test starting conversation
    result = await manager.start_conversation(
        context="I'm working on a Two Sum problem. Given an array of integers and a target, I need to find two numbers that add up to the target."
    )
    
    if result["success"]:
        conv_id = result["conversation_id"]
        print(f"✅ Started conversation: {conv_id}")
        
        # Test sending message
        message_result = await manager.send_message(
            conv_id, 
            "Can you help me understand the optimal approach for this problem?",
            {
                "problem_title": "Two Sum",
                "difficulty": "Easy",
                "time_spent": 5,
                "attempts": 1
            }
        )
        
        if message_result["success"]:
            print(f"✅ Message sent successfully")
            print(f"Response: {message_result.get('response', 'No response')}")
        
        # Test ending conversation
        end_result = await manager.end_conversation(conv_id)
        if end_result["success"]:
            print(f"✅ Conversation ended successfully")
    
    print("🎯 Voice Assistant test completed!")

if __name__ == "__main__":
    # Run test
    asyncio.run(test_voice_assistant())