import streamlit as st
import os
import logging

from config.settings import get_settings

logger = logging.getLogger(__name__)


def render_settings_page() -> None:
    """Render the settings and configuration page."""
    
    st.header("⚙️ Settings & Configuration")
    
    settings = get_settings()
    
    # API Configuration
    st.subheader("🔑 API Configuration")
    
    with st.container():
        # OpenAI API Key status
        if hasattr(settings, 'openai_api_key') and settings.openai_api_key:
            if settings.openai_api_key.startswith('sk-'):
                st.success("✅ OpenAI API Key is configured")
                st.code(f"API Key: {settings.openai_api_key[:12]}...{settings.openai_api_key[-4:]}")
            else:
                st.error("❌ Invalid API Key format")
        else:
            st.error("❌ OpenAI API Key not found")
            st.info("Please set OPENAI_API_KEY in your .env file")
        
        # Environment file info
        env_path = ".env"
        if os.path.exists(env_path):
            st.info(f"📄 Environment file found: {env_path}")
        else:
            st.warning(f"⚠️ Environment file not found. Create {env_path} from .env.example")
    
    st.divider()
    
    # Model Configuration
    st.subheader("🤖 Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Embedding Model", settings.embedding_model)
        st.metric("Chat Model", settings.chat_model)
        st.metric("Temperature", settings.temperature)
    
    with col2:
        st.metric("Max Tokens", settings.max_tokens)
        st.metric("Retrieval Count", settings.retrieval_count)
    
    st.divider()
    
    # Processing Configuration
    st.subheader("📝 Processing Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Chunk Size", f"{settings.chunk_size} chars")
        st.metric("Chunk Overlap", f"{settings.chunk_overlap} chars")
    
    with col2:
        st.metric("Max File Size", f"{settings.max_file_size / 1024 / 1024:.1f} MB")
    
    st.divider()
    
    # Storage Paths
    st.subheader("💾 Storage Paths")
    
    paths_info = {
        "Vector Database": settings.vector_db_path,
        "File Uploads": settings.uploads_path,
        "Logs": settings.logs_path
    }
    
    for name, path in paths_info.items():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.text(name)
        with col2:
            if os.path.exists(path):
                st.success(f"✅ {path}")
            else:
                st.warning(f"⚠️ {path} (will be created)")
    
    st.divider()
    
    # Environment Variables
    st.subheader("🔧 Environment Variables")
    
    with st.expander("Show Environment Configuration"):
        st.code(f"""
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (with defaults)
CHUNK_SIZE={settings.chunk_size}
CHUNK_OVERLAP={settings.chunk_overlap}
RETRIEVAL_COUNT={settings.retrieval_count}
TEMPERATURE={settings.temperature}
MAX_TOKENS={settings.max_tokens}
        """, language="bash")
    
    # Configuration Tips
    st.subheader("💡 Configuration Tips")
    
    with st.expander("Model Selection"):
        st.markdown("""
        **Embedding Models:**
        - `text-embedding-ada-002`: Best balance of performance and cost
        - `text-embedding-3-small`: Newer, more efficient model
        - `text-embedding-3-large`: Highest performance, higher cost
        
        **Chat Models:**
        - `gpt-3.5-turbo`: Fast and cost-effective
        - `gpt-4`: Higher quality responses, slower and more expensive
        - `gpt-4-turbo`: Latest GPT-4 with better performance
        """)
    
    with st.expander("Processing Parameters"):
        st.markdown("""
        **Chunk Size:**
        - Smaller chunks (500-800): Better for specific information
        - Larger chunks (1000-1500): Better for context and relationships
        
        **Chunk Overlap:**
        - 10-20% of chunk size is usually optimal
        - More overlap = better context preservation
        - Less overlap = more unique content
        
        **Retrieval Count:**
        - 3-5 results: Focused, specific answers
        - 5-10 results: Comprehensive, detailed answers
        """)
    
    with st.expander("Performance Tuning"):
        st.markdown("""
        **For Better Performance:**
        - Use smaller chunk sizes for faster processing
        - Reduce retrieval count for faster responses
        - Lower temperature for more consistent responses
        
        **For Better Quality:**
        - Use larger chunk sizes for more context
        - Increase retrieval count for comprehensive answers
        - Higher temperature for more creative responses
        """)
    
    # System Information
    st.divider()
    st.subheader("ℹ️ System Information")
    
    import sys
    import streamlit as st
    
    system_info = {
        "Python Version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "Streamlit Version": st.__version__,
        "Working Directory": os.getcwd(),
    }
    
    for key, value in system_info.items():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.text(key)
        with col2:
            st.code(value)