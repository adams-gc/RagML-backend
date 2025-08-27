from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import ingestion, rag
from .core.config import get_settings

settings = get_settings()

app = FastAPI(title=settings.APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingestion.router)
app.include_router(rag.router)

@app.get("/")  # health check endpoint
def health():
    return {"status": "ok"} # this is a simple health check endpoint that should return a 200 OK response for the API to be consideread the healthy and operational 
def read_root():
    return {"message": "Welcome to the RAG ML API. Use the /docs endpoint to explore the API."}
  # fast aip automatically gives your swagger docs at /docs
@app.get("/docs", include_in_schema=False)
def get_docs():
    return {"message": "API documentation is available at /docs. Use this endpoint to explore the API."}
