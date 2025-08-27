from pydantic import BaseModel, Field, EmailStr
from enum import Enum
from typing import List, Optional

class ChunkStrategy(str, Enum):
    sliding_window = "sliding_window"
    semantic_split = "semantic_split"

class VectorBackend(str, Enum):
    qdrant = "qdrant"
    pinecone = "pinecone"
    weaviate = "weaviate"
    milvus = "milvus"

class DBBackend(str, Enum):
    postgres = "postgres"
    mongodb = "mongodb"

class IngestResponse(BaseModel):
    status: str = "success"
    chunks: int
    vector_backend: VectorBackend

class ChatQuery(BaseModel):
    session_id: str = Field(..., description="Chat session key")
    query: str
    top_k: int = 4

class ChatResponse(BaseModel):
    response: str
    context: List[str]

class BookingDetails(BaseModel):
    name: str
    email: EmailStr
    datetime_iso: str
    notes: Optional[str] = None

class BookingResponse(BaseModel):
    status: str
