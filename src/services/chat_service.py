import logging
from typing import List, Dict, Optional
from datetime import datetime

from ..models.conversation import ConversationSession, ChatMessage, ChatResponse
from .rag_service import RAGService

logger = logging.getLogger(__name__)


class ChatService:
    """Service for managing chat conversations and session state."""
    
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self.sessions: Dict[str, ConversationSession] = {}
        self.current_session_id: Optional[str] = None
    
    def create_session(self) -> ConversationSession:
        """Create a new conversation session."""
        session = ConversationSession()
        self.sessions[session.id] = session
        self.current_session_id = session.id
        
        logger.info(f"Created new chat session: {session.id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get a conversation session by ID."""
        return self.sessions.get(session_id)
    
    def get_current_session(self) -> ConversationSession:
        """Get the current active session, creating one if needed."""
        if self.current_session_id and self.current_session_id in self.sessions:
            return self.sessions[self.current_session_id]
        else:
            return self.create_session()
    
    def send_message(self, message: str, session_id: Optional[str] = None) -> ChatResponse:
        """
        Send a message and get a response using the RAG pipeline.
        
        Args:
            message: User's message
            session_id: Optional session ID, uses current session if not provided
        
        Returns:
            ChatResponse with AI response and sources
        """
        # Get or create session
        if session_id:
            session = self.get_session(session_id)
            if not session:
                session = self.create_session()
        else:
            session = self.get_current_session()
        
        try:
            # Add user message to session
            user_message = session.add_message(message, "user")
            
            # Prepare conversation history for context
            recent_messages = session.get_recent_messages(10)
            conversation_history = []
            
            for msg in recent_messages[:-1]:  # Exclude the current message
                conversation_history.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # Generate response using RAG
            chat_response = self.rag_service.generate_response(
                query=message,
                conversation_history=conversation_history
            )
            
            # Add assistant response to session
            session.add_message(chat_response.content, "assistant")
            
            logger.info(f"Processed message in session {session.id}")
            return chat_response
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            
            # Add error response to session
            error_response = f"I apologize, but I encountered an error: {str(e)}"
            session.add_message(error_response, "assistant")
            
            return ChatResponse(
                content=error_response,
                sources=[],
                response_time=0.0,
                token_usage=None
            )
    
    def clear_session(self, session_id: Optional[str] = None) -> bool:
        """Clear a conversation session."""
        if session_id:
            if session_id in self.sessions:
                del self.sessions[session_id]
                if self.current_session_id == session_id:
                    self.current_session_id = None
                logger.info(f"Cleared session {session_id}")
                return True
            return False
        else:
            # Clear current session
            if self.current_session_id:
                return self.clear_session(self.current_session_id)
            return False
    
    def clear_all_sessions(self) -> None:
        """Clear all conversation sessions."""
        self.sessions.clear()
        self.current_session_id = None
        logger.info("Cleared all chat sessions")
    
    def get_session_history(self, session_id: Optional[str] = None) -> List[ChatMessage]:
        """Get message history for a session."""
        session = self.get_session(session_id) if session_id else self.get_current_session()
        return session.messages if session else []
    
    def get_chat_statistics(self) -> Dict:
        """Get statistics about chat usage."""
        total_sessions = len(self.sessions)
        total_messages = sum(len(session.messages) for session in self.sessions.values())
        
        # Count messages by role
        user_messages = 0
        assistant_messages = 0
        
        for session in self.sessions.values():
            for message in session.messages:
                if message.role == "user":
                    user_messages += 1
                elif message.role == "assistant":
                    assistant_messages += 1
        
        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "active_session": self.current_session_id
        }