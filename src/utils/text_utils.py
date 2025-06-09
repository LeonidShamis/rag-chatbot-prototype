import re
import logging
from typing import List
from ..models.document import DocumentChunk

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page headers/footers patterns (common ones)
    text = re.sub(r'--- Page \d+ ---\n?', '', text)
    
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def create_text_chunks(
    text: str,
    document_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[DocumentChunk]:
    """
    Split text into overlapping chunks for better semantic retrieval.
    
    Args:
        text: Text content to chunk
        document_id: ID of the parent document
        chunk_size: Maximum characters per chunk
        chunk_overlap: Number of characters to overlap between chunks
    
    Returns:
        List of DocumentChunk objects
    """
    chunks = []
    
    if not text or not text.strip():
        logger.warning(f"Empty text provided for document {document_id}")
        return chunks
    
    # Clean the text first
    clean_content = clean_text(text)
    
    # Split into chunks with overlap
    start = 0
    chunk_index = 0
    
    while start < len(clean_content):
        # Calculate end position
        end = min(start + chunk_size, len(clean_content))
        
        # Try to break at sentence boundary if we're not at the end
        if end < len(clean_content):
            # Look for sentence endings within the last 100 characters
            sentence_break = max(
                clean_content.rfind('.', start, end),
                clean_content.rfind('!', start, end),
                clean_content.rfind('?', start, end)
            )
            
            # If we found a sentence break, use it
            if sentence_break > start + chunk_size // 2:
                end = sentence_break + 1
        
        # Extract chunk content
        chunk_content = clean_content[start:end].strip()
        
        if chunk_content:
            chunk = DocumentChunk(
                document_id=document_id,
                chunk_index=chunk_index,
                content=chunk_content,
                start_char=start,
                end_char=end,
                metadata={
                    "chunk_size": len(chunk_content),
                    "original_length": len(clean_content)
                }
            )
            chunks.append(chunk)
            chunk_index += 1
        
        # Move start position for next chunk
        if end >= len(clean_content):
            break
        
        # Calculate next start with overlap
        start = max(start + chunk_size - chunk_overlap, end - chunk_overlap)
        
        # Ensure we make progress
        if start <= chunks[-1].start_char if chunks else False:
            start = end
    
    logger.info(f"Created {len(chunks)} chunks for document {document_id}")
    return chunks


def extract_page_number(text: str) -> int:
    """Extract page number from text if available."""
    page_match = re.search(r'--- Page (\d+) ---', text)
    if page_match:
        return int(page_match.group(1))
    return None