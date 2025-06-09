from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
import uuid


class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    role: str  # user, assistant, system
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict = {}

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SourceCitation(BaseModel):
    document_id: str
    document_name: str
    chunk_id: str
    content: str
    relevance_score: float
    page_number: Optional[int] = None


class ChatResponse(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    sources: List[SourceCitation] = []
    response_time: float
    timestamp: datetime = Field(default_factory=datetime.now)
    token_usage: Optional[dict] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConversationSession(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[ChatMessage] = []
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: dict = {}

    def add_message(self, content: str, role: str) -> ChatMessage:
        message = ChatMessage(content=content, role=role)
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message

    def get_recent_messages(self, count: int = 10) -> List[ChatMessage]:
        return self.messages[-count:] if self.messages else []