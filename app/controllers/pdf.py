import os
import tempfile
from typing import List
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse

from app.utils.pdf_utils import extract_text_from_pdf, chunk_text
from app.models.vector_db import VectorDB

# Create router
router = APIRouter()

# Get a shared instance of VectorDB
vector_db = VectorDB()

@router.post("/upload-pdf/", tags=["PDF Management"])
async def upload_pdf(files: List[UploadFile] = File(...)):
    """
    Upload and process PDF files, adding their content to the vector database
    
    Args:
        files (List[UploadFile]): List of PDF files to process
        
    Returns:
        JSONResponse: Upload status
    """
    if len(files) > 2:
        raise HTTPException(status_code=400, detail="Maximum 2 PDF files allowed")
    
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
        
        # Add to vector database
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