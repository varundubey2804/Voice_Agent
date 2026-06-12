from pydantic import BaseModel
from typing import Optional, List

class ChatRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    user_text: str
    agent_text: str
    audio_url: Optional[str] = None

class IngestionResponse(BaseModel):
    status: str
    message: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

class SearchResponse(BaseModel):
    results: List[str]
