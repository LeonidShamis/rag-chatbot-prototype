import streamlit as st
from typing import Optional

from ...services.document_service import DocumentService
from ...services.vector_service import VectorService
from ...services.embedding_service import EmbeddingService
from config.settings import get_settings


def render_sidebar(
    document_service: DocumentService,
    vector_service: VectorService,
    embedding_service: EmbeddingService
) -> str:
    """
    Render the sidebar with navigation and system information.
    
    Returns:
        Selected page name
    """
    
    with st.sidebar:
        st.title("🤖 RAG Chatbot")
        st.caption("Retrieval-Augmented Generation")
        
        st.divider()
        
        # Navigation
        st.subheader("📋 Navigation")
        page = st.radio(
            "Select Page",
            ["Chat", "Documents", "Settings"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # System status
        render_system_status(document_service, vector_service, embedding_service)
        
        st.divider()
        
        # Quick actions
        render_quick_actions(document_service, vector_service)
        
        return page


def render_system_status(
    document_service: DocumentService,
    vector_service: VectorService,
    embedding_service: EmbeddingService
) -> None:
    """Render system status information."""
    
    st.subheader("📊 System Status")
    
    # Knowledge base stats
    kb_stats = document_service.get_knowledge_base_stats()
    db_stats = vector_service.get_database_stats()
    emb_stats = embedding_service.get_embedding_stats()
    
    # Documents
    st.metric(
        "📄 Documents",
        kb_stats["total_documents"],
        help="Number of processed documents"
    )
    
    # Chunks/Vectors
    st.metric(
        "🔢 Text Chunks",
        kb_stats["total_chunks"],
        help="Number of text chunks in knowledge base"
    )
    
    # Vector database
    if db_stats["total_vectors"] > 0:
        st.metric(
            "🔍 Vectors",
            db_stats["total_vectors"],
            help="Number of vectors in database"
        )
        
        # Index size
        if db_stats["index_size"] > 0:
            size_mb = db_stats["index_size"] / 1024 / 1024
            st.metric(
                "💾 Index Size",
                f"{size_mb:.1f} MB",
                help="Vector database file size"
            )
    
    # Embedding stats
    if emb_stats["total_embeddings"] > 0:
        st.metric(
            "🧠 Embeddings",
            emb_stats["total_embeddings"],
            help="Number of stored embeddings"
        )
        
        st.metric(
            "📏 Embedding Dim",
            emb_stats["embedding_dimension"],
            help="Embedding vector dimension"
        )


def render_quick_actions(
    document_service: DocumentService,
    vector_service: VectorService
) -> None:
    """Render quick action buttons."""
    
    st.subheader("⚡ Quick Actions")
    
    # Save database
    if st.button("💾 Save Database", use_container_width=True):
        if vector_service.save_index():
            st.success("Database saved!")
        else:
            st.error("Failed to save database")
    
    # Refresh stats
    if st.button("🔄 Refresh Stats", use_container_width=True):
        st.rerun()
    
    # API key status
    settings = get_settings()
    if hasattr(settings, 'openai_api_key') and settings.openai_api_key:
        if settings.openai_api_key.startswith('sk-'):
            st.success("✅ OpenAI API Key Set")
        else:
            st.warning("⚠️ Check API Key Format")
    else:
        st.error("❌ OpenAI API Key Missing")


def render_settings_info() -> None:
    """Render settings information in sidebar."""
    
    st.subheader("⚙️ Current Settings")
    
    settings = get_settings()
    
    # Model settings
    with st.expander("🤖 Model Settings"):
        st.text(f"Embedding Model: {settings.embedding_model}")
        st.text(f"Chat Model: {settings.chat_model}")
        st.text(f"Temperature: {settings.temperature}")
        st.text(f"Max Tokens: {settings.max_tokens}")
    
    # Processing settings
    with st.expander("📝 Processing Settings"):
        st.text(f"Chunk Size: {settings.chunk_size}")
        st.text(f"Chunk Overlap: {settings.chunk_overlap}")
        st.text(f"Retrieval Count: {settings.retrieval_count}")
        st.text(f"Max File Size: {settings.max_file_size / 1024 / 1024:.1f} MB")


def render_help_info() -> None:
    """Render help information."""
    
    st.subheader("❓ Help & Tips")
    
    with st.expander("📖 How to Use"):
        st.markdown("""
        **Getting Started:**
        1. Upload PDF documents using the Documents page
        2. Wait for processing and embedding generation
        3. Start chatting with your documents on the Chat page
        
        **Tips:**
        - Upload multiple related documents for better context
        - Ask specific questions for more focused answers
        - Check the sources to verify information
        - Clear chat history to start fresh conversations
        """)
    
    with st.expander("🔧 Troubleshooting"):
        st.markdown("""
        **Common Issues:**
        - **No API Key**: Set OPENAI_API_KEY in .env file
        - **Upload Fails**: Check file size (max 50MB) and format (PDF only)
        - **Slow Responses**: Large documents may take time to process
        - **No Results**: Ensure documents are processed and indexed
        
        **Performance Tips:**
        - Process documents one at a time for large files
        - Clear old documents to reduce memory usage
        - Save database regularly to persist changes
        """)


def render_footer() -> None:
    """Render sidebar footer."""
    
    st.divider()
    
    st.caption("🤖 RAG Chatbot Prototype")
    st.caption("Built with Streamlit & LangChain")
    
    # Version info
    st.caption("v1.0.0")