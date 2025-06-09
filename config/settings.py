import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: Optional[str] = None
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_count: int = 5
    temperature: float = 0.7
    max_tokens: int = 1000
    embedding_model: str = "text-embedding-ada-002"
    chat_model: str = "gpt-3.5-turbo"
    max_file_size: int = 52428800  # 50MB in bytes
    vector_db_path: str = "data/vector_db"
    uploads_path: str = "data/uploads"
    logs_path: str = "data/logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


def get_settings() -> Settings:
    return Settings()