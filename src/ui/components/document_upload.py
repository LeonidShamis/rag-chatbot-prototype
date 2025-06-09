import streamlit as st
import logging
from typing import List, Optional

from ...services.document_service import DocumentService
from ...services.embedding_service import EmbeddingService
from ...services.vector_service import VectorService

logger = logging.getLogger(__name__)


def render_document_upload(
    document_service: DocumentService,
    embedding_service: EmbeddingService,
    vector_service: VectorService
) -> None:
    """Render the document upload interface."""
    
    st.header("📄 Document Upload")
    
    # Check if services are available
    if embedding_service is None:
        st.error("❌ Cannot upload documents: OpenAI API key not configured")
        st.info("Please set your OpenAI API key in the .env file and restart the app")
        return
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to build your knowledge base"
    )
    
    if uploaded_files:
        # Process uploaded files
        process_button = st.button(
            f"Process {len(uploaded_files)} file(s)",
            type="primary"
        )
        
        if process_button:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    # Read file data
                    file_data = uploaded_file.read()
                    
                    # Process document
                    result = document_service.process_uploaded_file(
                        file_data=file_data,
                        filename=uploaded_file.name
                    )
                    
                    if result.success:
                        # Generate embeddings
                        status_text.text(f"Generating embeddings for {uploaded_file.name}...")
                        
                        chunk_embeddings = embedding_service.generate_chunk_embeddings(
                            result.chunks
                        )
                        
                        # Add to vector database
                        status_text.text(f"Adding to vector database...")
                        
                        success = vector_service.add_embeddings(
                            chunks=result.chunks,
                            embeddings=chunk_embeddings,
                            document_name=uploaded_file.name
                        )
                        
                        if not success:
                            st.error(f"Failed to add {uploaded_file.name} to vector database")
                            continue
                        
                        # Save vector database
                        vector_service.save_index()
                        
                        st.success(f"✅ Successfully processed {uploaded_file.name} ({result.document.total_chunks} chunks)")
                        
                        # Debug: Log vector database stats
                        db_stats = vector_service.get_database_stats()
                        logger.info(f"Vector DB stats after processing: {db_stats}")
                    
                    else:
                        st.error(f"❌ Failed to process {uploaded_file.name}: {result.error_message}")
                
                except Exception as e:
                    logger.error(f"Error processing {uploaded_file.name}: {e}")
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("Processing complete!")
            
            # Force update of session state to refresh sidebar
            if 'last_update' not in st.session_state:
                st.session_state.last_update = 0
            st.session_state.last_update += 1
            
            st.rerun()


def render_document_management(document_service: DocumentService) -> None:
    """Render the document management interface."""
    
    st.header("📚 Knowledge Base")
    
    # Get all documents
    documents = document_service.get_all_documents()
    
    if not documents:
        st.info("No documents uploaded yet. Use the upload section above to add documents to your knowledge base.")
        return
    
    # Knowledge base statistics
    stats = document_service.get_knowledge_base_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", stats["total_documents"])
    with col2:
        st.metric("Total Chunks", stats["total_chunks"])
    with col3:
        st.metric("Total Size", f"{stats['total_size'] / 1024 / 1024:.1f} MB")
    
    st.divider()
    
    # Document list
    st.subheader("Uploaded Documents")
    
    for doc in sorted(documents, key=lambda x: x.upload_timestamp, reverse=True):
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.text(doc.filename)
                st.caption(f"Uploaded: {doc.upload_timestamp.strftime('%Y-%m-%d %H:%M')}")
            
            with col2:
                # Status badge
                if doc.processing_status == "completed":
                    st.success("✅ Ready")
                elif doc.processing_status == "processing":
                    st.info("⏳ Processing")
                elif doc.processing_status == "failed":
                    st.error("❌ Failed")
                else:
                    st.warning("⏸️ Pending")
            
            with col3:
                st.text(f"{doc.total_chunks} chunks")
                st.caption(f"{doc.file_size / 1024:.1f} KB")
            
            with col4:
                if st.button("🗑️", key=f"delete_{doc.id}", help="Remove document"):
                    document_service.remove_document(doc.id)
                    st.rerun()
        
        # Show error message if processing failed
        if doc.processing_status == "failed" and doc.error_message:
            st.error(f"Error: {doc.error_message}")
        
        st.divider()
    
    # Clear all button
    if st.button("🗑️ Clear All Documents", type="secondary"):
        if st.session_state.get("confirm_clear_all", False):
            document_service.clear_all_documents()
            st.session_state.confirm_clear_all = False
            st.success("All documents cleared!")
            st.rerun()
        else:
            st.session_state.confirm_clear_all = True
            st.warning("Click again to confirm clearing all documents")


def render_vector_database_info(vector_service: VectorService) -> None:
    """Render vector database information."""
    
    st.header("🔍 Vector Database")
    
    # Get database stats
    stats = vector_service.get_database_stats()
    
    if stats["total_vectors"] == 0:
        st.info("Vector database is empty. Upload and process documents to populate it.")
        return
    
    # Database statistics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Vectors", stats["total_vectors"])
        st.metric("Index Size", f"{stats['index_size'] / 1024 / 1024:.1f} MB")
    
    with col2:
        st.metric("Unique Documents", stats["unique_documents"])
        st.metric("Index Type", stats["index_type"])
    
    # Database management
    st.subheader("Database Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save Index", help="Save current vector database to disk"):
            if vector_service.save_index():
                st.success("Vector database saved successfully!")
            else:
                st.error("Failed to save vector database")
    
    with col2:
        if st.button("🗑️ Clear Database", type="secondary", help="Clear all vectors from database"):
            if st.session_state.get("confirm_clear_db", False):
                vector_service.clear_index()
                st.session_state.confirm_clear_db = False
                st.success("Vector database cleared!")
                st.rerun()
            else:
                st.session_state.confirm_clear_db = True
                st.warning("Click again to confirm clearing the vector database")