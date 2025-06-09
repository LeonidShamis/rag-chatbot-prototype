from typing import List, Optional
from pydantic import BaseModel
from .document import DocumentChunk


class SearchResult(BaseModel):
    chunk: DocumentChunk
    relevance_score: float
    document_name: str
    

class SearchQuery(BaseModel):
    query: str
    top_k: int = 5
    relevance_threshold: float = 0.0
    

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    search_time: float