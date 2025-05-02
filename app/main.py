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