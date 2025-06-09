import streamlit as st
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
from config.logging import setup_logging
from config.settings import get_settings

# Import services
from src.services.document_service import DocumentService
from src.services.embedding_service import EmbeddingService
from src.services.vector_service import VectorService
from src.services.rag_service import RAGService
from src.services.chat_service import ChatService

# Import UI components
from src.ui.components.sidebar import render_sidebar, render_footer
from src.ui.pages.chat import render_chat_page
from src.ui.pages.documents import render_documents_page
from src.ui.pages.settings import render_settings_page


def initialize_services():
    """Initialize all services and store in session state."""
    
    settings = get_settings()
    
    # Initialize services
    if 'document_service' not in st.session_state:
        st.session_state.document_service = DocumentService()
    
    if 'vector_service' not in st.session_state:
        st.session_state.vector_service = VectorService()
    
    if 'embedding_service' not in st.session_state:
        if not settings.openai_api_key:
            st.error("❌ OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
            st.info("The app will run in limited mode. You can browse documents but cannot process new ones or chat.")
            # Create a dummy service that will fail gracefully
            st.session_state.embedding_service = None
        else:
            st.session_state.embedding_service = EmbeddingService(settings.openai_api_key)
    
    if 'rag_service' not in st.session_state:
        if settings.openai_api_key:
            st.session_state.rag_service = RAGService(
                api_key=settings.openai_api_key,
                vector_service=st.session_state.vector_service
            )
        else:
            st.session_state.rag_service = None
    
    if 'chat_service' not in st.session_state:
        if st.session_state.rag_service:
            st.session_state.chat_service = ChatService(st.session_state.rag_service)
        else:
            st.session_state.chat_service = None


def main():
    """Main application entry point."""
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Configure Streamlit page
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stAlert > div {
        padding: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f3e5f5;
    }
    </style>
    """, unsafe_allow_html=True)
    
    try:
        # Initialize services
        initialize_services()
        
        # Render sidebar and get selected page
        page = render_sidebar(
            st.session_state.document_service,
            st.session_state.vector_service,
            st.session_state.embedding_service
        )
        
        # Render selected page
        if page == "Chat":
            render_chat_page(st.session_state.chat_service)
        elif page == "Documents":
            render_documents_page(
                st.session_state.document_service,
                st.session_state.embedding_service,
                st.session_state.vector_service
            )
        elif page == "Settings":
            render_settings_page()
        
        # Render footer in sidebar
        with st.sidebar:
            render_footer()
    
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"An error occurred: {str(e)}")
        
        # Show error details in expander
        with st.expander("Error Details"):
            st.exception(e)


if __name__ == "__main__":
    main()