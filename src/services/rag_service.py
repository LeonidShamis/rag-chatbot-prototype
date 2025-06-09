import logging
import time
from typing import List, Dict, Any, Optional

from ..models.conversation import ChatResponse, SourceCitation
from ..models.search import SearchQuery
from ..utils.openai_utils import OpenAIClient
from .vector_service import VectorService
from config.settings import get_settings

logger = logging.getLogger(__name__)


class RAGService:
    """Service implementing Retrieval-Augmented Generation pipeline."""
    
    def __init__(self, api_key: str, vector_service: VectorService):
        self.settings = get_settings()
        self.openai_client = OpenAIClient(api_key)
        self.vector_service = vector_service
        
        # System prompt for RAG responses
        self.system_prompt = """You are a helpful AI assistant that answers questions based on provided document context. 

Guidelines:
- Use only the information provided in the context to answer questions
- If the context doesn't contain enough information to answer a question, say so clearly
- Be concise but comprehensive in your responses
- When referencing information, mention which document it came from when possible
- If multiple documents provide relevant information, synthesize the information appropriately
- Maintain a professional and helpful tone"""
    
    def generate_response(
        self,
        query: str,
        conversation_history: List[Dict[str, str]] = None
    ) -> ChatResponse:
        """
        Generate a response using retrieval-augmented generation.
        
        Args:
            query: User's question
            conversation_history: Recent conversation messages for context
        
        Returns:
            ChatResponse with answer and source citations
        """
        start_time = time.time()
        
        try:
            # Step 1: Generate query embedding
            query_embedding = self.openai_client.get_embedding(
                text=query,
                model=self.settings.embedding_model
            )
            
            # Step 2: Perform semantic search
            search_query = SearchQuery(
                query=query,
                top_k=self.settings.retrieval_count,
                relevance_threshold=0.1  # Low threshold to get diverse results
            )
            
            search_response = self.vector_service.search(query_embedding, search_query)
            
            # Step 3: Prepare context from retrieved chunks
            context_chunks = []
            source_citations = []
            
            for result in search_response.results:
                chunk = result.chunk
                context_chunks.append({
                    "content": chunk.content,
                    "document": result.document_name,
                    "score": result.relevance_score
                })
                
                # Create source citation
                citation = SourceCitation(
                    document_id=chunk.document_id,
                    document_name=result.document_name,
                    chunk_id=chunk.id,
                    content=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                    relevance_score=result.relevance_score,
                    page_number=chunk.page_number
                )
                source_citations.append(citation)
            
            # Step 4: Build prompt with context
            context_text = self._build_context_text(context_chunks)
            
            # Step 5: Prepare messages for LLM
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            # Add conversation history if provided
            if conversation_history:
                messages.extend(conversation_history[-6:])  # Last 6 messages for context
            
            # Add current query with context
            user_message = f"""Context from documents:
{context_text}

Question: {query}

Please answer the question based on the provided context. If the context doesn't contain sufficient information to answer the question, please state that clearly."""
            
            messages.append({"role": "user", "content": user_message})
            
            # Step 6: Generate response
            response_text, token_usage = self.openai_client.chat_completion(
                messages=messages,
                model=self.settings.chat_model,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens
            )
            
            response_time = time.time() - start_time
            
            logger.info(f"Generated RAG response in {response_time:.2f}s with {len(source_citations)} sources")
            
            return ChatResponse(
                content=response_text,
                sources=source_citations,
                response_time=response_time,
                token_usage=token_usage
            )
        
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            response_time = time.time() - start_time
            
            return ChatResponse(
                content=f"I apologize, but I encountered an error while processing your question: {str(e)}",
                sources=[],
                response_time=response_time,
                token_usage=None
            )
    
    def _build_context_text(self, context_chunks: List[Dict]) -> str:
        """Build formatted context text from retrieved chunks."""
        if not context_chunks:
            return "No relevant context found in the uploaded documents."
        
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(
                f"[Document: {chunk['document']}, Relevance: {chunk['score']:.3f}]\n"
                f"{chunk['content']}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def get_simple_answer(self, query: str) -> str:
        """
        Get a simple text answer without full ChatResponse structure.
        Useful for testing or simple integrations.
        """
        try:
            response = self.generate_response(query)
            return response.content
        except Exception as e:
            logger.error(f"Error getting simple answer: {e}")
            return f"Error: {str(e)}"