from __future__ import annotations
from typing import List, Dict, Tuple
import uuid
import numpy as np
from ..core.config import get_settings
from pydantic import BaseModel

settings = get_settings()

class VectorItem(BaseModel):
    id: str
    vector: List[float]
    text: str
    metadata: Dict[str, str] | None = None

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

_qdrant: QdrantClient | None = None

def _qdrant_client() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=settings.QDRANT_URL)
        try:
            _qdrant.get_collection(settings.QDRANT_COLLECTION)
        except Exception:
            _qdrant.recreate_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
    return _qdrant

try:
    import pinecone
except Exception:
    pinecone = None

try:
    import weaviate
except Exception:
    weaviate = None

try:
    from pymilvus import MilvusClient
except Exception:
    MilvusClient = None

def upsert_vectors(items: List[VectorItem], backend: str = "qdrant") -> None:
    if backend == "qdrant":
        cli = _qdrant_client()
        points = [
            PointStruct(id=item.id, vector=item.vector, payload={"text": item.text, **(item.metadata or {})})
            for item in items
        ]
        cli.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)
        return
    if backend == "pinecone":
        if pinecone is None or not settings.PINECONE_API_KEY:
            raise RuntimeError("Pinecone not configured")
        pinecone.init(api_key=settings.PINECONE_API_KEY)
        index = pinecone.Index(settings.PINECONE_INDEX)
        index.upsert(vectors=[(it.id, it.vector, {"text": it.text, **(it.metadata or {})}) for it in items])
        return
    if backend == "weaviate":
        if weaviate is None:
            raise RuntimeError("Weaviate client missing")
        client = weaviate.connect_to_custom(url=settings.WEAVIATE_URL, auth_client_secret=weaviate.auth.AuthApiKey(settings.WEAVIATE_API_KEY) if settings.WEAVIATE_API_KEY else None)
        try:
            client.collections.get(settings.WEAVIATE_COLLECTION)
        except Exception:
            client.collections.create(name=settings.WEAVIATE_COLLECTION)
        coll = client.collections.get(settings.WEAVIATE_COLLECTION)
        with coll.batch.dynamic() as batch:
            for it in items:
                batch.add_object(properties={"text": it.text, **(it.metadata or {})}, vector=it.vector)
        client.close()
        return
    if backend == "milvus":
        if MilvusClient is None:
            raise RuntimeError("Milvus client missing")
        client = MilvusClient(uri=settings.MILVUS_URI, token=settings.MILVUS_TOKEN)
        try:
            client.get_collection(settings.MILVUS_COLLECTION)
        except Exception:
            client.create_collection(collection_name=settings.MILVUS_COLLECTION, dimension=len(items[0].vector))
        client.insert(collection_name=settings.MILVUS_COLLECTION, data={
            "id": [it.id for it in items],
            "vector": [it.vector for it in items],
            "text": [it.text for it in items],
        })
        return
    raise ValueError(f"Unsupported vector backend: {backend}")

def search_vectors(query_vec: List[float], top_k: int = 4, backend: str = "qdrant") -> List[Tuple[str, float, Dict]]:
    if backend == "qdrant":
        cli = _qdrant_client()
        res = cli.search(collection_name=settings.QDRANT_COLLECTION, query_vector=query_vec, limit=top_k, with_payload=True)
        return [(str(r.id), float(r.score), r.payload) for r in res]
    if backend == "pinecone":
        index = pinecone.Index(settings.PINECONE_INDEX)
        res = index.query(vector=query_vec, top_k=top_k, include_metadata=True)
        return [(m["id"], float(m["score"]), m["metadata"]) for m in res["matches"]]
    if backend == "weaviate":
        client = weaviate.connect_to_custom(url=settings.WEAVIATE_URL, auth_client_secret=weaviate.auth.AuthApiKey(settings.WEAVIATE_API_KEY) if settings.WEAVIATE_API_KEY else None)
        coll = client.collections.get(settings.WEAVIATE_COLLECTION)
        res = coll.query.near_vector(query_vec, limit=top_k, return_metadata=["distance"]).objects
        out = []
        for o in res:
            out.append((o.uuid, 1.0 - float(o.metadata.distance), {"text": o.properties.get("text", "")}))
        client.close()
        return out
    if backend == "milvus":
        client = MilvusClient(uri=settings.MILVUS_URI, token=settings.MILVUS_TOKEN)
        res = client.search(collection_name=settings.MILVUS_COLLECTION, data=[query_vec], limit=top_k, output_fields=["text"])
        out = []
        for hit in res[0]:
            out.append((str(hit["id"]), float(hit["distance"]), {"text": hit["entity"]["text"]}))
        return out
    raise ValueError(f"Unsupported vector backend: {backend}")
