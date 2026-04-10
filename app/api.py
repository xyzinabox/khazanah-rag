"""
Section 4 — FastAPI Backend
Endpoints: /health, /query, /extract
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import time

app = FastAPI(
    title="Khazanah Annual Review RAG API",
    description="Query and extract insights from Khazanah's Annual Review using AI.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, examples=["What was Khazanah's TWRR?"])

class SourceInfo(BaseModel):
    page: int
    type: str
    excerpt: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    confidence: float
    latency_seconds: float

class HealthResponse(BaseModel):
    status: str
    vector_store_ready: bool
    chunks_count: int


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check if the system is ready."""
    try:
        import chromadb
        from app.config import CHROMA_DIR
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_collection("annual_review")
        count = collection.count()
        return HealthResponse(status="ok", vector_store_ready=True, chunks_count=count)
    except Exception:
        return HealthResponse(status="ok", vector_store_ready=False, chunks_count=0)


@app.post("/query", response_model=QueryResponse)
def query_annual_review(req: QueryRequest):
    """Ask a natural-language question about the Annual Review."""
    from app.rag import query_rag
    start = time.time()
    try:
        result = query_rag(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    latency = round(time.time() - start, 2)
    return QueryResponse(
        answer=result["answer"],
        sources=[SourceInfo(**s) for s in result["sources"]],
        confidence=result["confidence"],
        latency_seconds=latency,
    )


@app.get("/extract")
def extract_structured():
    """Return structured data extracted from the Annual Review."""
    from app.extraction import extract_structured_data
    try:
        data = extract_structured_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return data
