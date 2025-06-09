import time
import logging
from typing import List, Dict, Any
import openai
from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Wrapper for OpenAI API with retry logic and error handling."""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.max_retries = 3
        self.base_delay = 1.0
    
    def get_embedding(self, text: str, model: str = "text-embedding-ada-002") -> List[float]:
        """
        Get embedding for a single text with retry logic.
        
        Args:
            text: Text to embed
            model: OpenAI embedding model to use
        
        Returns:
            List of floats representing the embedding vector
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model=model
                )
                return response.data[0].embedding
            
            except openai.RateLimitError as e:
                wait_time = self.base_delay * (2 ** attempt)
                logger.warning(f"Rate limit hit, waiting {wait_time}s (attempt {attempt + 1})")
                time.sleep(wait_time)
                if attempt == self.max_retries - 1:
                    raise
            
            except openai.APIError as e:
                logger.error(f"OpenAI API error: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.base_delay)
            
            except Exception as e:
                logger.error(f"Unexpected error getting embedding: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.base_delay)
    
    def get_embeddings_batch(self, texts: List[str], model: str = "text-embedding-ada-002") -> List[List[float]]:
        """
        Get embeddings for multiple texts in batch with retry logic.
        
        Args:
            texts: List of texts to embed
            model: OpenAI embedding model to use
        
        Returns:
            List of embedding vectors
        """
        # OpenAI API has limits on batch size, so we process in chunks
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            for attempt in range(self.max_retries):
                try:
                    response = self.client.embeddings.create(
                        input=batch,
                        model=model
                    )
                    
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    break
                
                except openai.RateLimitError as e:
                    wait_time = self.base_delay * (2 ** attempt)
                    logger.warning(f"Rate limit hit, waiting {wait_time}s (attempt {attempt + 1})")
                    time.sleep(wait_time)
                    if attempt == self.max_retries - 1:
                        raise
                
                except openai.APIError as e:
                    logger.error(f"OpenAI API error: {e}")
                    if attempt == self.max_retries - 1:
                        raise
                    time.sleep(self.base_delay)
                
                except Exception as e:
                    logger.error(f"Unexpected error getting embeddings: {e}")
                    if attempt == self.max_retries - 1:
                        raise
                    time.sleep(self.base_delay)
        
        return all_embeddings
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> tuple[str, Dict[str, Any]]:
        """
        Get chat completion with retry logic.
        
        Args:
            messages: List of message dictionaries
            model: OpenAI chat model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        
        Returns:
            Tuple of (response_text, usage_info)
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                content = response.choices[0].message.content
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
                
                return content, usage
            
            except openai.RateLimitError as e:
                wait_time = self.base_delay * (2 ** attempt)
                logger.warning(f"Rate limit hit, waiting {wait_time}s (attempt {attempt + 1})")
                time.sleep(wait_time)
                if attempt == self.max_retries - 1:
                    raise
            
            except openai.APIError as e:
                logger.error(f"OpenAI API error: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.base_delay)
            
            except Exception as e:
                logger.error(f"Unexpected error in chat completion: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.base_delay)