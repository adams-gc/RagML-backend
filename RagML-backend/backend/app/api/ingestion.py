from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from ..models.schemas import ChunkStrategy, VectorBackend, DBBackend, IngestResponse
from ..services.chunking import chunk_text
from ..services.embedding import encode_texts
from ..services.vector_store import VectorItem, upsert_vectors
from ..core.config import get_settings
from ..db import sql as sql_db
from ..db import nosql as nosql_db
import uuid

from pypdf import PdfReader

router = APIRouter(prefix="/api", tags=["ingestion"])
settings = get_settings()

def extract_text(file: UploadFile) -> str:
    if file.filename.lower().endswith(".txt"):
        return file.file.read().decode("utf-8", errors="ignore")
    if file.filename.lower().endswith(".pdf"):
        reader = PdfReader(file.file)
        text = []
        for p in reader.pages:
            t = p.extract_text() or ""
            text.append(t)
        return "\n".join(text)
    raise HTTPException(status_code=400, detail="Only .pdf and .txt supported")

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    strategy: ChunkStrategy = Form(...),
    vector_backend: VectorBackend = Form(...),
    db_backend: DBBackend = Form(...),
):
    raw = extract_text(file)
    chunks = chunk_text(raw, strategy.value)
    vecs = encode_texts([c.text for c in chunks])
    items = [VectorItem(id=str(uuid.uuid4()), vector=v.tolist(), text=c.text, metadata={"filename": file.filename}) for c, v in zip(chunks, vecs)]
    upsert_vectors(items, backend=vector_backend.value)
    if db_backend.value == "postgres":
        sql_db.save_metadata(file.filename, strategy.value, vector_backend.value)
    else:
        await nosql_db.save_metadata(file.filename, strategy.value, vector_backend.value)
    return IngestResponse(chunks=len(chunks), vector_backend=vector_backend)
