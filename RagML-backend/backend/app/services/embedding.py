from __future__ import annotations
from typing import List
import numpy as np
from ..core.config import get_settings

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from sentence_transformers import SentenceTransformer

settings = get_settings()

_st_model: SentenceTransformer | None = None

def _st() -> SentenceTransformer:
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer(settings.ST_MODEL_NAME)
    return _st_model

def encode_texts(texts: List[str]) -> np.ndarray:
    if settings.EMBEDDING_PROVIDER == "openai":
        if OpenAI is None or not settings.OPENAI_API_KEY:
            raise RuntimeError("OpenAI embeddings requested but OPENAI_API_KEY not set")
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
        return np.array([d.embedding for d in resp.data], dtype=np.float32)
    vecs = _st().encode(texts, normalize_embeddings=True)
    return np.array(vecs, dtype=np.float32)
