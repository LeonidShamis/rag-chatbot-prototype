from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
import uuid


class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    file_size: int
    upload_timestamp: datetime = Field(default_factory=datetime.now)
    processing_status: str = "pending"  # pending, processing, completed, failed
    total_chunks: int = 0
    error_message: Optional[str] = None
    metadata: dict = {}

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DocumentChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    chunk_index: int
    content: str
    start_char: int
    end_char: int
    page_number: Optional[int] = None
    metadata: dict = {}


class ProcessingResult(BaseModel):
    document: Document
    chunks: List[DocumentChunk]
    success: bool
    error_message: Optional[str] = None