from typing import Dict, List, Optional
import uuid
import os
import json
from datetime import datetime

class ChatSession:
    # File path for storing sessions
    SESSIONS_FILE = "data/chat_sessions.json"
    
    # Using a class attribute as a simple in-memory database
    # Will be synchronized with disk
    _sessions: Dict[str, List[Dict[str, str]]] = {}
    
    @classmethod
    def _initialize(cls) -> None:
        """Initialize the sessions store from disk if it exists"""
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(cls.SESSIONS_FILE), exist_ok=True)
        
        # Load existing sessions if available
        if os.path.exists(cls.SESSIONS_FILE):
            try:
                with open(cls.SESSIONS_FILE, 'r') as f:
                    cls._sessions = json.load(f)
            except Exception as e:
                print(f"Error loading sessions: {str(e)}")
                cls._sessions = {}
    
    @classmethod
    def create_session(cls) -> str:
        """
        Create a new chat session
        
        Returns:
            str: Session ID
        """
        # Initialize from disk if not already loaded
        if not cls._sessions:
            cls._initialize()
            
        session_id = str(uuid.uuid4())
        cls._sessions[session_id] = []
        cls._save_to_disk()
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
        # Initialize from disk if not already loaded
        if not cls._sessions:
            cls._initialize()
            
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
        # Initialize from disk if not already loaded
        if not cls._sessions:
            cls._initialize()
            
        if session_id not in cls._sessions:
            cls._sessions[session_id] = []
        
        # Add timestamp to messages
        timestamp = datetime.now().isoformat()
            
        cls._sessions[session_id].append({
            "role": role,
            "content": content,
            "timestamp": timestamp
        })
        
        # Save changes to disk
        cls._save_to_disk()
    
    @classmethod
    def get_all_sessions(cls) -> Dict[str, List[Dict[str, str]]]:
        """
        Get all chat sessions
        
        Returns:
            Dict[str, List[Dict[str, str]]]: All sessions
        """
        # Initialize from disk if not already loaded
        if not cls._sessions:
            cls._initialize()
            
        return cls._sessions
    
    @classmethod
    def _save_to_disk(cls) -> None:
        """Save all sessions to disk"""
        try:
            with open(cls.SESSIONS_FILE, 'w') as f:
                json.dump(cls._sessions, f, indent=2)
        except Exception as e:
            print(f"Error saving sessions: {str(e)}")