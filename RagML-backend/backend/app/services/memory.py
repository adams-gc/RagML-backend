import json
from typing import List, Tuple
import redis
from ..core.config import get_settings

settings = get_settings()

_r: redis.Redis | None = None

def _redis() -> redis.Redis:
    global _r
    if _r is None:
        _r = redis.from_url(settings.REDIS_URL)
    return _r

def append_message(session_id: str, role: str, content: str) -> None:
    key = f"chat:{session_id}"
    _redis().rpush(key, json.dumps({"role": role, "content": content}))

def get_history(session_id: str, limit: int = 10) -> List[Tuple[str, str]]:
    key = f"chat:{session_id}"
    vals = _redis().lrange(key, -limit, -1)
    out: List[Tuple[str, str]] = []
    for v in vals:
        d = json.loads(v)
        out.append((d["role"], d["content"]))
    return out
