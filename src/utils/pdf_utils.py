import logging
import io
from typing import List, Optional
import PyPDF2
from ..models.document import Document, DocumentChunk

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_file: io.BytesIO, filename: str) -> tuple[str, Optional[str]]:
    """
    Extract text content from a PDF file.
    
    Returns:
        tuple: (extracted_text, error_message)
    """
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text_content = ""
        
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text.strip():
                    text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                else:
                    logger.warning(f"No text found on page {page_num + 1} of {filename}")
            except Exception as e:
                logger.error(f"Error extracting text from page {page_num + 1} of {filename}: {e}")
                continue
        
        if not text_content.strip():
            return "", "No text content could be extracted from the PDF"
        
        return text_content.strip(), None
        
    except Exception as e:
        error_msg = f"Failed to process PDF {filename}: {str(e)}"
        logger.error(error_msg)
        return "", error_msg


def validate_pdf_file(file_data: bytes, filename: str, max_size: int) -> Optional[str]:
    """
    Validate PDF file before processing.
    
    Returns:
        Optional[str]: Error message if validation fails, None if valid
    """
    # Check file size
    if len(file_data) > max_size:
        return f"File size ({len(file_data)} bytes) exceeds maximum allowed size ({max_size} bytes)"
    
    # Check file extension
    if not filename.lower().endswith('.pdf'):
        return "File must be a PDF (.pdf extension)"
    
    # Try to read PDF structure
    try:
        pdf_file = io.BytesIO(file_data)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Check if PDF has pages
        if len(pdf_reader.pages) == 0:
            return "PDF file appears to be empty (no pages found)"
            
    except Exception as e:
        return f"Invalid or corrupted PDF file: {str(e)}"
    
    return None