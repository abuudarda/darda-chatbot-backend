from typing import List, Dict
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import os
import tempfile

from app.models.chat_session import ChatSession
from app.models.vector_db import VectorDB
from app.utils.pdf_utils import extract_text_from_pdf, chunk_text

# Create router for admin-only endpoints
router = APIRouter()

# Get a shared instance of VectorDB
vector_db = VectorDB()

@router.post("/admin/upload-pdf/", tags=["Admin"])
async def admin_upload_pdf(files: List[UploadFile] = File(...)):
    """
    Admin-only endpoint to upload and process PDF files
    
    Args:
        files (List[UploadFile]): List of PDF files to process
        
    Returns:
        JSONResponse: Upload status
    """
    if len(files) > 5:  # Allowing more files for admin
        raise HTTPException(status_code=400, detail="Maximum 5 PDF files allowed")
    
    try:
        all_chunks = []
        processed_files = []
        
        for pdf_file in files:
            # Check if file is PDF
            if not pdf_file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File {pdf_file.filename} is not a PDF")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(await pdf_file.read())
                temp_path = temp_file.name
            
            # Process the PDF
            text = extract_text_from_pdf(temp_path)
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
            processed_files.append(pdf_file.filename)
            
            # Delete the temporary file
            os.unlink(temp_path)
        
        # Add to vector database with persistence
        vector_db.add_texts(all_chunks)
        
        return JSONResponse(
            content={
                "message": f"Successfully processed {len(files)} PDF files",
                "files": processed_files,
                "chunks": len(all_chunks)
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDFs: {str(e)}")

@router.get("/admin/sessions/", tags=["Admin"])
async def get_all_chat_sessions():
    """
    Admin-only endpoint to get all chat sessions
    
    Returns:
        Dict: All chat sessions with metadata
    """
    sessions = ChatSession.get_all_sessions()
    
    # Add summary metadata for each session
    session_summaries = {}
    for session_id, messages in sessions.items():
        # Count messages
        user_msgs = sum(1 for msg in messages if msg["role"] == "user")
        assistant_msgs = sum(1 for msg in messages if msg["role"] == "assistant")
        
        # Get first and last message timestamps if available
        created_at = messages[0].get("timestamp", "unknown") if messages else "N/A"
        last_activity = messages[-1].get("timestamp", "unknown") if messages else "N/A"
        
        # Get first user message for context
        first_user_msg = next((msg["content"] for msg in messages if msg["role"] == "user"), "N/A")
        if len(first_user_msg) > 50:
            first_user_msg = first_user_msg[:50] + "..."
        
        session_summaries[session_id] = {
            "message_count": len(messages),
            "user_messages": user_msgs,
            "assistant_messages": assistant_msgs,
            "created_at": created_at,
            "last_activity": last_activity,
            "first_user_message": first_user_msg,
            "history": messages
        }
    
    return {"sessions": session_summaries}

@router.get("/admin/vector-db/stats", tags=["Admin"])
async def get_vector_db_stats():
    """
    Admin-only endpoint to get vector database statistics
    
    Returns:
        Dict: Stats about the vector database
    """
    return {
        "document_count": len(vector_db.documents),
        "index_size": vector_db.index.ntotal if hasattr(vector_db.index, "ntotal") else "Unknown"
    }