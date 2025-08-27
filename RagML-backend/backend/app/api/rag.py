from fastapi import APIRouter
from ..models.schemas import ChatQuery, ChatResponse, BookingDetails, BookingResponse
from ..services.memory import get_history, append_message
from ..services.embedding import encode_texts
from ..services.vector_store import search_vectors
from ..services.booking import save_booking, send_confirmation
from ..core.config import get_settings

settings = get_settings()
router = APIRouter(prefix="/api", tags=["rag"])

def synthesize_answer(query: str, context_texts: list[str], history: list[tuple[str, str]]) -> str:
    prelude = "\n".join([f"{r}: {c}" for r, c in history[-6:]])
    ctx = "\n---\n".join(context_texts)
    return f"Based on the docs, here is a concise answer to: '{query}'.\n\nContext used:\n{ctx[:1200]}\n\n(History considered: {len(history)} turns)"

@router.post("/chat", response_model=ChatResponse)
async def chat(query: ChatQuery) -> ChatResponse:
    history = get_history(query.session_id, limit=12)
    qvec = encode_texts([query.query])[0].tolist()
    results = search_vectors(qvec, top_k=query.top_k, backend=settings.VECTOR_BACKEND)
    ctx_texts = [r[2].get("text", "") for r in results]
    answer = synthesize_answer(query.query, ctx_texts, history)
    append_message(query.session_id, "user", query.query)
    append_message(query.session_id, "assistant", answer)
    return ChatResponse(response=answer, context=ctx_texts)

@router.post("/book", response_model=BookingResponse)
async def book(details: BookingDetails) -> BookingResponse:
    save_booking(details)
    send_confirmation(details)
    return BookingResponse(status="confirmed")
