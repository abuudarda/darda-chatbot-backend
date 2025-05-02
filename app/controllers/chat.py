from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from app.models.chat_session import ChatSession
from app.models.vector_db import VectorDB
from app.services.gemini import GeminiService

# Create router
router = APIRouter()

# Initialize services
vector_db = VectorDB()
gemini_service = GeminiService()

# Pydantic models for request/response validation
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str

@router.post("/chat/", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Process a chat message and return a response
    
    Args:
        request (ChatRequest): Chat request with session_id and message
        
    Returns:
        ChatResponse: Chat response with session_id and response text
    """
    # Create or retrieve session
    session_id = request.session_id or ChatSession.create_session()
    
    # Get existing chat history or initialize new
    chat_history = ChatSession.get_session(session_id) or []
    
    # Store the user message
    ChatSession.add_message(session_id, "user", request.message)
    
    # Retrieve relevant context from vector DB
    context = vector_db.similarity_search(request.message, k=3)
    
    try:
        # Get response from Gemini
        response = gemini_service.query(
            prompt=request.message,
            context=context,
            history=chat_history[:-1]  # Exclude the latest message which we just added
        )
        
        # Store the assistant's response
        ChatSession.add_message(session_id, "assistant", response)
        
        return ChatResponse(session_id=session_id, response=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error from LLM: {str(e)}")

@router.get("/history/{session_id}", tags=["Chat"])
async def get_history(session_id: str):
    """
    Get chat history for a session
    
    Args:
        session_id (str): Session ID
        
    Returns:
        dict: Chat history
    """
    history = ChatSession.get_session(session_id)
    if not history:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"session_id": session_id, "history": history}