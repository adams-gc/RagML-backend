
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache

class Settings(BaseSettings):
    # API
    APP_NAME: str = "rag-backend"
    APP_ENV: str = "dev"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    # Embeddings
    EMBEDDING_PROVIDER: str = Field("sentence_transformers", description="openai | sentence_transformers")
    ST_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    OPENAI_API_KEY: str | None = None

    # Vector backends
    VECTOR_BACKEND: str = Field("qdrant", description="qdrant | pinecone | weaviate | milvus")

    # Qdrant
    QDRANT_URL: str = "http://qdrant:6333"
    QDRANT_COLLECTION: str = "documents"

    # Pinecone
    PINECONE_API_KEY: str | None = None
    PINECONE_INDEX: str = "documents"

    # Weaviate
    WEAVIATE_URL: str = "http://weaviate:8080"
    WEAVIATE_API_KEY: str | None = None
    WEAVIATE_COLLECTION: str = "Document"

    # Milvus
    MILVUS_URI: str = "http://milvus-standalone:19530"
    MILVUS_TOKEN: str = "root:Milvus"
    MILVUS_COLLECTION: str = "documents"

    # Databases
    DB_BACKEND: str = Field("postgres", description="postgres | mongodb")
    POSTGRES_DSN: str = "postgresql+psycopg2://postgres:postgres@postgres:5432/postgres"
    # POSTGRES_DSN:str  = "postgresql+psycopg2://user:pass@localhost:5432/dbname", example .
    MONGODB_URI: str = "mongodb://mongodb:27017"
    MONGODB_DB: str = "rag"

    # Redis (chat memory)
    REDIS_URL: str = "redis://redis:6379/0"

    # Email
    EMAIL_SENDER: str = "no-reply@example.com"
    SMTP_HOST: str | None = None
    SMTP_PORT: int | None = 587
    SMTP_USERNAME: str | None = None
    SMTP_PASSWORD: str | None = None
    SENDGRID_API_KEY: str | None = None

    # Misc
    MAX_CHUNK_TOKENS: int = 400
    SLIDING_OVERLAP: int = 60

    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache
def get_settings() -> Settings:
    return Settings()
