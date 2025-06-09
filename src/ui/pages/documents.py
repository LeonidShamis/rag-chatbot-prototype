import streamlit as st
import logging

from ...services.document_service import DocumentService
from ...services.embedding_service import EmbeddingService
from ...services.vector_service import VectorService
from ..components.document_upload import (
    render_document_upload,
    render_document_management,
    render_vector_database_info
)

logger = logging.getLogger(__name__)


def render_documents_page(
    document_service: DocumentService,
    embedding_service: EmbeddingService,
    vector_service: VectorService
) -> None:
    """Render the documents management page."""
    
    # Create tabs for different document views
    tab1, tab2, tab3 = st.tabs(["📤 Upload", "📚 Manage", "🔍 Database"])
    
    with tab1:
        render_document_upload(document_service, embedding_service, vector_service)
    
    with tab2:
        render_document_management(document_service)
    
    with tab3:
        render_vector_database_info(vector_service)