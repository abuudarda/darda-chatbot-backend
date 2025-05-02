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