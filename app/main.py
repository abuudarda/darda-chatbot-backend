import uvicorn
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from functools import wraps

from app.controllers.chat import router as chat_router
from app.controllers.admin import router as admin_router

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

# Include public router
app.include_router(chat_router, tags=["Chat"])

# Include admin router - not exposed to public API docs
app.include_router(admin_router, include_in_schema=True)

@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "Gemini PDF Chatbot API is running"}

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 4000))
#     uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)