import logging
import io
import os
from typing import List, Optional
from datetime import datetime

from ..models.document import Document, DocumentChunk, ProcessingResult
from ..utils.pdf_utils import extract_text_from_pdf, validate_pdf_file
from ..utils.text_utils import create_text_chunks
from config.settings import get_settings

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for handling document upload, processing, and management."""
    
    def __init__(self):
        self.settings = get_settings()
        self.processed_documents: dict[str, Document] = {}
        self.document_chunks: dict[str, List[DocumentChunk]] = {}
    
    def process_uploaded_file(
        self,
        file_data: bytes,
        filename: str
    ) -> ProcessingResult:
        """
        Process an uploaded PDF file and extract text chunks.
        
        Args:
            file_data: Raw file data
            filename: Original filename
        
        Returns:
            ProcessingResult with document and chunks
        """
        logger.info(f"Processing uploaded file: {filename}")
        
        # Create document record
        document = Document(
            filename=filename,
            file_size=len(file_data),
            processing_status="processing"
        )
        
        try:
            # Validate file
            validation_error = validate_pdf_file(
                file_data, filename, self.settings.max_file_size
            )
            if validation_error:
                document.processing_status = "failed"
                document.error_message = validation_error
                return ProcessingResult(
                    document=document,
                    chunks=[],
                    success=False,
                    error_message=validation_error
                )
            
            # Extract text from PDF
            pdf_file = io.BytesIO(file_data)
            extracted_text, extraction_error = extract_text_from_pdf(pdf_file, filename)
            
            if extraction_error:
                document.processing_status = "failed"
                document.error_message = extraction_error
                return ProcessingResult(
                    document=document,
                    chunks=[],
                    success=False,
                    error_message=extraction_error
                )
            
            # Create text chunks
            chunks = create_text_chunks(
                text=extracted_text,
                document_id=document.id,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap
            )
            
            # Update document status
            document.processing_status = "completed"
            document.total_chunks = len(chunks)
            
            # Store in memory
            self.processed_documents[document.id] = document
            self.document_chunks[document.id] = chunks
            
            logger.info(f"Successfully processed {filename}: {len(chunks)} chunks created")
            
            return ProcessingResult(
                document=document,
                chunks=chunks,
                success=True
            )
        
        except Exception as e:
            error_msg = f"Unexpected error processing {filename}: {str(e)}"
            logger.error(error_msg)
            
            document.processing_status = "failed"
            document.error_message = error_msg
            
            return ProcessingResult(
                document=document,
                chunks=[],
                success=False,
                error_message=error_msg
            )
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self.processed_documents.get(document_id)
    
    def get_document_chunks(self, document_id: str) -> List[DocumentChunk]:
        """Get all chunks for a document."""
        return self.document_chunks.get(document_id, [])
    
    def get_all_documents(self) -> List[Document]:
        """Get all processed documents."""
        return list(self.processed_documents.values())
    
    def get_all_chunks(self) -> List[DocumentChunk]:
        """Get all chunks from all documents."""
        all_chunks = []
        for chunks in self.document_chunks.values():
            all_chunks.extend(chunks)
        return all_chunks
    
    def remove_document(self, document_id: str) -> bool:
        """Remove a document and its chunks."""
        if document_id in self.processed_documents:
            del self.processed_documents[document_id]
            if document_id in self.document_chunks:
                del self.document_chunks[document_id]
            logger.info(f"Removed document {document_id}")
            return True
        return False
    
    def clear_all_documents(self) -> None:
        """Clear all documents and chunks."""
        self.processed_documents.clear()
        self.document_chunks.clear()
        logger.info("Cleared all documents")
    
    def get_knowledge_base_stats(self) -> dict:
        """Get statistics about the current knowledge base."""
        total_documents = len(self.processed_documents)
        total_chunks = sum(len(chunks) for chunks in self.document_chunks.values())
        total_size = sum(doc.file_size for doc in self.processed_documents.values())
        
        status_counts = {}
        for doc in self.processed_documents.values():
            status_counts[doc.processing_status] = status_counts.get(doc.processing_status, 0) + 1
        
        return {
            "total_documents": total_documents,
            "total_chunks": total_chunks,
            "total_size": total_size,
            "status_counts": status_counts,
            "last_updated": datetime.now().isoformat() if total_documents > 0 else None
        }