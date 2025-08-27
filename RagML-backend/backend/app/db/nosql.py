from ..core.config import get_settings
from motor.motor_asyncio import AsyncIOMotorClient

settings = get_settings()
_client: AsyncIOMotorClient | None = None

def _mongo() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(settings.MONGODB_URI)
    return _client

async def save_metadata(filename: str, strategy: str, backend: str) -> None:
    db = _mongo()[settings.MONGODB_DB]
    await db.ingestions.insert_one({"filename": filename, "strategy": strategy, "vector_backend": backend})
