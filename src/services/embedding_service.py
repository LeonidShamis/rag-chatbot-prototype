import logging
import time
from typing import List, Dict, Optional
import numpy as np

from ..models.document import DocumentChunk
from ..utils.openai_utils import OpenAIClient
from config.settings import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating and managing vector embeddings using OpenAI."""
    
    def __init__(self, api_key: str):
        self.settings = get_settings()
        self.openai_client = OpenAIClient(api_key)
        self.chunk_embeddings: Dict[str, List[float]] = {}
    
    def generate_chunk_embeddings(self, chunks: List[DocumentChunk]) -> Dict[str, List[float]]:
        """
        Generate embeddings for a list of document chunks.
        
        Args:
            chunks: List of DocumentChunk objects to embed
        
        Returns:
            Dictionary mapping chunk IDs to embedding vectors
        """
        if not chunks:
            return {}
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        start_time = time.time()
        
        try:
            # Extract text content from chunks
            texts = [chunk.content for chunk in chunks]
            chunk_ids = [chunk.id for chunk in chunks]
            
            # Generate embeddings in batch
            embeddings = self.openai_client.get_embeddings_batch(
                texts=texts,
                model=self.settings.embedding_model
            )
            
            # Map chunk IDs to embeddings
            chunk_embeddings = {}
            for chunk_id, embedding in zip(chunk_ids, embeddings):
                chunk_embeddings[chunk_id] = embedding
                self.chunk_embeddings[chunk_id] = embedding
            
            elapsed_time = time.time() - start_time
            logger.info(f"Generated {len(embeddings)} embeddings in {elapsed_time:.2f} seconds")
            
            return chunk_embeddings
        
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        
        Args:
            query: Search query text
        
        Returns:
            Embedding vector as list of floats
        """
        try:
            embedding = self.openai_client.get_embedding(
                text=query,
                model=self.settings.embedding_model
            )
            return embedding
        
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise
    
    def get_chunk_embedding(self, chunk_id: str) -> Optional[List[float]]:
        """Get stored embedding for a specific chunk."""
        return self.chunk_embeddings.get(chunk_id)
    
    def get_all_embeddings(self) -> Dict[str, List[float]]:
        """Get all stored chunk embeddings."""
        return self.chunk_embeddings.copy()
    
    def remove_chunk_embedding(self, chunk_id: str) -> bool:
        """Remove embedding for a specific chunk."""
        if chunk_id in self.chunk_embeddings:
            del self.chunk_embeddings[chunk_id]
            return True
        return False
    
    def clear_all_embeddings(self) -> None:
        """Clear all stored embeddings."""
        self.chunk_embeddings.clear()
        logger.info("Cleared all embeddings")
    
    def get_embedding_stats(self) -> Dict:
        """Get statistics about stored embeddings."""
        if not self.chunk_embeddings:
            return {
                "total_embeddings": 0,
                "embedding_dimension": 0,
                "memory_usage_mb": 0
            }
        
        # Get embedding dimension from first embedding
        first_embedding = next(iter(self.chunk_embeddings.values()))
        embedding_dim = len(first_embedding)
        
        # Estimate memory usage (rough calculation)
        total_floats = len(self.chunk_embeddings) * embedding_dim
        memory_mb = (total_floats * 4) / (1024 * 1024)  # 4 bytes per float
        
        return {
            "total_embeddings": len(self.chunk_embeddings),
            "embedding_dimension": embedding_dim,
            "memory_usage_mb": round(memory_mb, 2)
        }