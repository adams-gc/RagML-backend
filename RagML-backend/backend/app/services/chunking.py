from __future__ import annotations
from typing import List
from pydantic import BaseModel
from ..core.config import get_settings

settings = get_settings()

class Chunk(BaseModel):
    id: str
    text: str

import re
def sliding_window(text: str, max_tokens: int, overlap: int) -> List[Chunk]:
    words = text.split()
    chunks: List[Chunk] = []
    i = 0
    idx = 0
    step = max(max_tokens - overlap, 1)
    while i < len(words):
        window = words[i:i+max_tokens]
        chunk_text = " ".join(window)
        chunks.append(Chunk(id=f"c{idx}", text=chunk_text))
        idx += 1
        i += step
    return chunks

from sentence_transformers import SentenceTransformer, util
_model: SentenceTransformer | None = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.ST_MODEL_NAME)
    return _model

def semantic_split(text: str, target_tokens: int) -> List[Chunk]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []
    model = _get_model()
    embeds = model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
    groups: List[List[str]] = []
    current = [sentences[0]]
    for i in range(1, len(sentences)):
        sim = util.cos_sim(embeds[i-1], embeds[i]).item()
        if sim >= 0.55 and sum(len(w.split()) for w in current) < target_tokens:
            current.append(sentences[i])
        else:
            groups.append(current)
            current = [sentences[i]]
    groups.append(current)
    chunks = [Chunk(id=f"s{i}", text=" ".join(g)) for i, g in enumerate(groups)]
    return chunks

def chunk_text(text: str, strategy: str) -> List[Chunk]:
    if strategy == "semantic_split":
        return semantic_split(text, settings.MAX_CHUNK_TOKENS)
    return sliding_window(text, settings.MAX_CHUNK_TOKENS, settings.SLIDING_OVERLAP)
