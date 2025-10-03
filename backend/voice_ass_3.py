from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import Dict, Optional
from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Settings configuration
class Settings(BaseSettings):
    AGENT_ID: str = "_"
    ELEVENLABS_API_KEY: str = "_"
    
    class Config:
        env_file = ".env"

settings = Settings()

# FastAPI app initialization
app = FastAPI(title="Voice Assistant API")

# Store active conversations
active_conversations: Dict[str, Conversation] = {}

# Request Models
class ConversationConfig(BaseModel):
    agent_id: str
    api_key: str

class MessageRequest(BaseModel):
    conversation_id: str
    message: str

# API Routes
@app.get("/")
async def root():
    return {"message": "Welcome to Voice Assistant API. Visit /docs for API documentation."}

@app.post("/api/start")
async def start_conversation(config: ConversationConfig):
    client = ElevenLabs(api_key=config.api_key)
    
    conversation = Conversation(
        client=client,
        agent_id=config.agent_id,
        requires_auth=bool(config.api_key),
        audio_interface=DefaultAudioInterface(),
    )
    
    conversation.start_session()
    conversation_id = str(conversation.session_id)
    active_conversations[conversation_id] = conversation
    
    return {"conversation_id": conversation_id}

@app.post("/api/message")
async def send_message(request: MessageRequest):
    if request.conversation_id not in active_conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
        
    conversation = active_conversations[request.conversation_id]
    response = conversation.send_message(request.message)
    
    return {"response": response}

@app.post("/api/end/{conversation_id}")
async def end_conversation(conversation_id: str):
    if conversation_id not in active_conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
        
    conversation = active_conversations[conversation_id]
    conversation.end_session()
    del active_conversations[conversation_id]
    
    return {"status": "conversation ended"}

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
