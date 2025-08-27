from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from ..core.config import get_settings

settings = get_settings()
_engine = create_engine(settings.POSTGRES_DSN, future=True, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)

def save_metadata(filename: str, strategy: str, backend: str) -> None:
    with _engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ingestions (
                id SERIAL PRIMARY KEY,
                filename TEXT,
                strategy TEXT,
                vector_backend TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """))
        conn.execute(text("INSERT INTO ingestions(filename, strategy, vector_backend) VALUES (:f,:s,:b)"),
                     {"f": filename, "s": strategy, "b": backend})
