# darda-chatbot-backend
# Gemini PDF Chatbot - MVC Architecture

Below are the individual files for the PDF chatbot backend with MVC architecture.

## app/main.py
```python
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.controllers.chat import router as chat_router
from app.controllers.pdf import router as pdf_router

# Initialize FastAPI app
app = FastAPI(title="Gemini PDF Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(pdf_router, tags=["PDF Management"])
app.include_router(chat_router, tags=["Chat"])

@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "Gemini PDF Chatbot API is running"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
```

## app/models/\_\_init\_\_.py
```python
# Initialize package
```

## app/models/vector_db.py
```python
from typing import List
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class VectorDB:
    def __init__(self):
        # Initialize the sentence transformer model for embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # Dimension of the embeddings from the model
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        
    def add_texts(self, texts: List[str]) -> None:
        """
        Add text chunks to the vector database
        
        Args:
            texts (List[str]): List of text chunks to add
        """
        if not texts:
            return
        
        # Generate embeddings
        embeddings = self.model.encode(texts)
        
        # Add to FAISS index
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Store documents
        self.documents.extend(texts)
    
    def similarity_search(self, query: str, k: int = 3) -> List[str]:
        """
        Search for the most similar documents to the query
        
        Args:
            query (str): The search query
            k (int): Number of results to return
            
        Returns:
            List[str]: List of the most similar documents
        """
        if len(self.documents) == 0:
            return []
            
        # Generate query embedding
        query_embedding = self.model.encode([query])
        
        # Search in the index
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), 
            k=min(k, len(self.documents))
        )
        
        # Return the most similar documents
        results = [self.documents[idx] for idx in indices[0]]
        return results
```

## app/models/chat_session.py
```python
from typing import Dict, List, Optional
import uuid

class ChatSession:
    # Using a class attribute as a simple in-memory database
    # In a production app, use a real database
    _sessions: Dict[str, List[Dict[str, str]]] = {}
    
    @classmethod
    def create_session(cls) -> str:
        """
        Create a new chat session
        
        Returns:
            str: Session ID
        """
        session_id = str(uuid.uuid4())
        cls._sessions[session_id] = []
        return session_id
    
    @classmethod
    def get_session(cls, session_id: str) -> Optional[List[Dict[str, str]]]:
        """
        Get a chat session by ID
        
        Args:
            session_id (str): Session ID
            
        Returns:
            Optional[List[Dict[str, str]]]: Chat history or None if not found
        """
        return cls._sessions.get(session_id)
    
    @classmethod
    def add_message(cls, session_id: str, role: str, content: str) -> None:
        """
        Add a message to a chat session
        
        Args:
            session_id (str): Session ID
            role (str): Message role ('user' or 'assistant')
            content (str): Message content
        """
        if session_id not in cls._sessions:
            cls._sessions[session_id] = []
            
        cls._sessions[session_id].append({
            "role": role,
            "content": content
        })
    
    @classmethod
    def get_all_sessions(cls) -> Dict[str, List[Dict[str, str]]]:
        """
        Get all chat sessions
        
        Returns:
            Dict[str, List[Dict[str, str]]]: All sessions
        """
        return cls._sessions
```

## app/controllers/\_\_init\_\_.py
```python
# Initialize package
```

## app/controllers/chat.py
```python
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
```

## app/controllers/pdf.py
```python
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
```

## app/services/\_\_init\_\_.py
```python
# Initialize package
```

## app/services/gemini.py
```python
import os
from typing import List, Dict
import google.generativeai as genai

# Initialize the Gemini API with API key
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "your_api_key_here")
genai.configure(api_key=GOOGLE_API_KEY)

class GeminiService:
    def __init__(self):
        # Configure the model
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
        
        # Initialize the model
        self.model = genai.GenerativeModel(
            model_name="gemini-pro",
            generation_config=self.generation_config
        )
    
    def query(self, prompt: str, context: List[str], history: List[Dict]) -> str:
        """
        Query the Gemini model with prompt, context, and history
        
        Args:
            prompt (str): User's question
            context (List[str]): Retrieved context from vector DB
            history (List[Dict]): Chat history
            
        Returns:
            str: Generated response
        """
        # Format the system prompt with context
        system_prompt = f"""
        You are a helpful assistant that answers questions based on provided knowledge.
        Use the following context to answer the user's question:
        
        {' '.join(context)}
        
        If you don't know the answer based on the provided context, say so.
        Keep your answers concise and to the point.
        """
        
        # Format chat history for Gemini
        chat_history = []
        for msg in history:
            if msg["role"] == "user":
                chat_history.append({"role": "user", "parts": [msg["content"]]})
            else:
                chat_history.append({"role": "model", "parts": [msg["content"]]})
        
        # Create a new chat session
        chat = self.model.start_chat(history=chat_history)
        
        # Add the system prompt as a "hidden" first message
        # For Gemini, we'll use a user message with the system prompt followed by the actual query
        response = chat.send_message([
            f"{system_prompt}\n\nUser question: {prompt}"
        ])
        
        return response.text
```

## app/utils/\_\_init\_\_.py
```python
# Initialize package
```

## app/utils/pdf_utils.py
```python
import re
from typing import List
import PyPDF2

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text
    """
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + " "
    
    return text

def chunk_text(text: str) -> List[str]:
    """
    Split text into meaningful chunks
    
    Args:
        text (str): Text to chunk
        
    Returns:
        List[str]: List of text chunks
    """
    # Clean the text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Simple chunking by paragraphs
    # In a production app, consider more sophisticated chunking strategies
    chunks = []
    paragraphs = re.split(r'\n\n+', text)
    
    for para in paragraphs:
        if len(para.strip()) > 50:  # Only keep substantive paragraphs
            chunks.append(para.strip())
    
    return chunks
```

## requirements.txt
```
fastapi==0.95.0
uvicorn==0.21.1
PyPDF2==3.0.1
faiss-cpu==1.7.4
sentence-transformers==2.2.2
google-generativeai==0.3.1
python-multipart==0.0.6
```

## README.md
```markdown
# Gemini PDF Chatbot API

A backend-only API for a chatbot that uses Google's Gemini LLM, with PDF knowledge base and vector search.

## Features

- **PDF Knowledge Base**: Upload and process PDF documents to use as knowledge sources
- **Vector Search**: Find the most relevant context for user queries using FAISS
- **Chat Sessions**: Maintain conversation history for contextual responses
- **Gemini Integration**: Generate responses using Google's Gemini Pro model

## Project Structure

The project follows an MVC (Model-View-Controller) architecture:

```
app/
├── main.py              # Entry point
├── models/              # Data models
│   ├── chat_session.py  # Chat session model
│   └── vector_db.py     # Vector database model
├── controllers/         # API routes and logic
│   ├── chat.py          # Chat endpoints
│   └── pdf.py           # PDF upload endpoints
├── services/            # External services
│   └── gemini.py        # Gemini LLM service
└── utils/               # Utility functions
    └── pdf_utils.py     # PDF processing utilities
```

## Setup

1. Create the project structure:
   ```
   mkdir -p app/models app/controllers app/services app/utils
   touch app/__init__.py app/models/__init__.py app/controllers/__init__.py app/services/__init__.py app/utils/__init__.py
   ```

2. Copy each file from this document into the appropriate location in the project structure.

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set your Google API key:
   ```
   export GOOGLE_API_KEY="your_api_key_here"
   ```

5. Run the application:
   ```
   python -m app.main
   ```

## API Endpoints

### PDF Management

- `POST /upload-pdf/`: Upload up to 2 PDF files to use as knowledge base

### Chat

- `POST /chat/`: Send a message and get a response
- `GET /history/{session_id}`: Get chat history for a specific session

## Example Usage

1. Upload PDF files:
   ```
   curl -X POST -F "files=@document1.pdf" -F "files=@document2.pdf" http://localhost:8000/upload-pdf/
   ```

2. Send a chat message:
   ```
   curl -X POST -H "Content-Type: application/json" -d '{"message": "What information is in the documents?"}' http://localhost:8000/chat/
   ```

3. Continue the conversation:
   ```
   curl -X POST -H "Content-Type: application/json" -d '{"session_id": "previously_returned_session_id", "message": "Tell me more about topic X"}' http://localhost:8000/chat/
   ```

4. Get chat history:
   ```
   curl http://localhost:8000/history/your_session_id
   ```
```

## setup.py
```python
import os
import shutil

def create_project_structure():
    """
    Create the project directory structure and files
    """
    # Create directories
    directories = [
        "app",
        "app/models",
        "app/controllers",
        "app/services",
        "app/utils"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Create empty __init__.py files
    init_files = [
        "app/__init__.py",
        "app/models/__init__.py",
        "app/controllers/__init__.py",
        "app/services/__init__.py",
        "app/utils/__init__.py"
    ]
    
    for init_file in init_files:
        with open(init_file, "w") as f:
            f.write("# Initialize package\n")
    
    print("Project structure created successfully!")

if __name__ == "__main__":
    create_project_structure()
```