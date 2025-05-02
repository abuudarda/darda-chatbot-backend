import os
from typing import List, Dict
import google.generativeai as genai

# Initialize the Gemini API with API key
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyA1PsSqIjtjpferh3wxxw5AN8DRHbgdr34")
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
            model_name="gemini-2.0-flash",
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