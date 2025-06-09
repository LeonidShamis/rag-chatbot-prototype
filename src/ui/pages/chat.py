import streamlit as st
import logging

from ...services.chat_service import ChatService
from ..components.chat_interface import render_chat_interface, render_conversation_history

logger = logging.getLogger(__name__)


def render_chat_page(chat_service: ChatService) -> None:
    """Render the main chat page."""
    
    # Check if chat service is available
    if chat_service is None or chat_service.rag_service is None:
        st.error("❌ Chat service not available: OpenAI API key not configured")
        st.info("Please set your OpenAI API key in the .env file and restart the app")
        return
    
    # Check if there are any documents in the knowledge base
    if hasattr(st.session_state, 'vector_service'):
        db_stats = st.session_state.vector_service.get_database_stats()
        logger.info(f"Chat page - Vector DB stats: {db_stats}")
        
        if db_stats["total_vectors"] == 0:
            st.warning("⚠️ No documents in knowledge base. Please upload and process documents first.")
            
            # Show debug info
            with st.expander("Debug Info"):
                st.json(db_stats)
                doc_stats = st.session_state.document_service.get_knowledge_base_stats()
                st.json(doc_stats)
            
            if st.button("Go to Documents Page"):
                st.session_state.page = "Documents"
                st.rerun()
            
            return
    else:
        st.error("Vector service not initialized")
        return
    
    # Create tabs for different chat views
    tab1, tab2 = st.tabs(["💬 Chat", "📜 History"])
    
    with tab1:
        render_chat_interface(chat_service)
    
    with tab2:
        render_conversation_history(chat_service)