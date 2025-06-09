import logging
import os
import pickle
import time
from typing import List, Dict, Tuple, Optional
import numpy as np
import faiss

from ..models.document import DocumentChunk
from ..models.search import SearchResult, SearchQuery, SearchResponse
from config.settings import get_settings

logger = logging.getLogger(__name__)


class VectorService:
    """Service for managing FAISS vector database operations."""
    
    def __init__(self):
        self.settings = get_settings()
        self.index: Optional[faiss.Index] = None
        self.chunk_metadata: Dict[int, DocumentChunk] = {}
        self.document_names: Dict[str, str] = {}  # document_id -> filename
        self.index_path = os.path.join(self.settings.vector_db_path, "faiss_index.bin")
        self.metadata_path = os.path.join(self.settings.vector_db_path, "metadata.pkl")
        
        # Ensure vector db directory exists
        os.makedirs(self.settings.vector_db_path, exist_ok=True)
        
        # Try to load existing index
        self.load_index()
    
    def create_index(self, embedding_dimension: int) -> None:
        """Create a new FAISS index with the specified dimension."""
        logger.info(f"Creating new FAISS index with dimension {embedding_dimension}")
        
        # Use IndexFlatIP for cosine similarity (inner product on normalized vectors)
        self.index = faiss.IndexFlatIP(embedding_dimension)
        self.chunk_metadata.clear()
        self.document_names.clear()
    
    def add_embeddings(
        self,
        chunks: List[DocumentChunk],
        embeddings: Dict[str, List[float]],
        document_name: str
    ) -> bool:
        """
        Add embeddings and chunks to the vector database.
        
        Args:
            chunks: List of DocumentChunk objects
            embeddings: Dictionary mapping chunk IDs to embedding vectors
            document_name: Name of the source document
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not chunks or not embeddings:
                logger.warning("No chunks or embeddings provided")
                return False
            
            # Prepare embeddings matrix
            embedding_vectors = []
            valid_chunks = []
            
            for chunk in chunks:
                if chunk.id in embeddings:
                    embedding = embeddings[chunk.id]
                    # Normalize for cosine similarity
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        normalized_embedding = np.array(embedding) / norm
                        embedding_vectors.append(normalized_embedding)
                        valid_chunks.append(chunk)
                    else:
                        logger.warning(f"Zero-norm embedding for chunk {chunk.id}")
                else:
                    logger.warning(f"No embedding found for chunk {chunk.id}")
            
            if not embedding_vectors:
                logger.error("No valid embeddings to add")
                return False
            
            # Convert to numpy array
            embeddings_matrix = np.array(embedding_vectors, dtype=np.float32)
            
            # Create index if it doesn't exist
            if self.index is None:
                self.create_index(embeddings_matrix.shape[1])
            
            # Add to FAISS index
            start_idx = self.index.ntotal
            self.index.add(embeddings_matrix)
            
            # Store metadata
            for i, chunk in enumerate(valid_chunks):
                faiss_idx = start_idx + i
                self.chunk_metadata[faiss_idx] = chunk
                self.document_names[chunk.document_id] = document_name
            
            logger.info(f"Added {len(valid_chunks)} embeddings to vector database")
            logger.info(f"Index now has {self.index.ntotal} total vectors")
            return True
        
        except Exception as e:
            logger.error(f"Error adding embeddings to vector database: {e}")
            return False
    
    def search(self, query_embedding: List[float], query: SearchQuery) -> SearchResponse:
        """
        Perform semantic search against the vector database.
        
        Args:
            query_embedding: Query embedding vector
            query: Search query parameters
        
        Returns:
            SearchResponse with results
        """
        start_time = time.time()
        
        try:
            if self.index is None or self.index.ntotal == 0:
                logger.warning("No vectors in database for search")
                return SearchResponse(
                    query=query.query,
                    results=[],
                    total_results=0,
                    search_time=0.0
                )
            
            # Normalize query embedding for cosine similarity
            query_vector = np.array(query_embedding, dtype=np.float32)
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm
            
            # Perform search
            scores, indices = self.index.search(
                query_vector.reshape(1, -1),
                min(query.top_k, self.index.ntotal)
            )
            
            # Build results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                
                if score >= query.relevance_threshold:
                    chunk = self.chunk_metadata.get(idx)
                    if chunk:
                        document_name = self.document_names.get(chunk.document_id, "Unknown")
                        
                        result = SearchResult(
                            chunk=chunk,
                            relevance_score=float(score),
                            document_name=document_name
                        )
                        results.append(result)
            
            search_time = time.time() - start_time
            
            logger.info(f"Search completed: {len(results)} results in {search_time:.3f}s")
            
            return SearchResponse(
                query=query.query,
                results=results,
                total_results=len(results),
                search_time=search_time
            )
        
        except Exception as e:
            logger.error(f"Error performing vector search: {e}")
            return SearchResponse(
                query=query.query,
                results=[],
                total_results=0,
                search_time=time.time() - start_time
            )
    
    def save_index(self) -> bool:
        """Save the FAISS index and metadata to disk."""
        try:
            if self.index is None:
                logger.warning("No index to save")
                return False
            
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)
            
            # Save metadata
            metadata = {
                "chunk_metadata": self.chunk_metadata,
                "document_names": self.document_names
            }
            
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Saved vector database to {self.index_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving vector database: {e}")
            return False
    
    def load_index(self) -> bool:
        """Load the FAISS index and metadata from disk."""
        try:
            if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
                logger.info("No existing vector database found")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(self.index_path)
            
            # Load metadata
            with open(self.metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.chunk_metadata = metadata.get("chunk_metadata", {})
                self.document_names = metadata.get("document_names", {})
            
            logger.info(f"Loaded vector database with {self.index.ntotal} vectors")
            return True
        
        except Exception as e:
            logger.error(f"Error loading vector database: {e}")
            return False
    
    def clear_index(self) -> None:
        """Clear the vector database."""
        self.index = None
        self.chunk_metadata.clear()
        self.document_names.clear()
        
        # Remove files
        for path in [self.index_path, self.metadata_path]:
            if os.path.exists(path):
                os.remove(path)
        
        logger.info("Cleared vector database")
    
    def get_database_stats(self) -> Dict:
        """Get statistics about the vector database."""
        if self.index is None:
            return {
                "total_vectors": 0,
                "index_size": 0,
                "unique_documents": 0,
                "index_type": None
            }
        
        index_size = 0
        if os.path.exists(self.index_path):
            index_size = os.path.getsize(self.index_path)
        
        return {
            "total_vectors": self.index.ntotal,
            "index_size": index_size,
            "unique_documents": len(set(self.document_names.values())),
            "index_type": type(self.index).__name__
        }