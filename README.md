# Khazanah Annual Review — RAG Query Tool

## What is this?

A tool that lets you ask questions about Khazanah's Annual Review in plain English and get answers with source citations. No need to dig through the PDF manually.

You can ask stuff like:
- "What was Khazanah's TWRR?"
- "Which sectors did Khazanah increase exposure to?"
- "Summarise the sustainability initiatives"

There's also a structured data tab that extracts portfolio companies, financial metrics, and sector allocations into clean tables.

Every answer tells you which page and which document it came from, so you can verify it yourself.

## How it works

```
  PDFs (3 files)
       │
       ▼
  Ingestion Pipeline ──► ChromaDB (vector store)
  (parse, chunk, embed)         │
                                ▼
  Streamlit UI ◄──► FastAPI ◄──► RAG Engine ◄──► Groq LLM
```

Basically:

1. The PDFs get parsed (text via PyMuPDF, tables via pdfplumber), split into chunks, turned into embeddings, and stored in ChromaDB.
2. When you ask a question, the system finds the most relevant chunks, feeds them to the LLM with your question, and returns an answer with citations.
3. There's a separate extraction flow that pulls structured data (companies, financials) into JSON.

## Setup

### What you need
- Python 3.11 (I used Miniconda)
- A free [Groq API key](https://console.groq.com)

### Steps

```bash
# clone it
git clone https://github.com/YOUR_USERNAME/khazanah-rag.git
cd khazanah-rag

# create conda env
conda create -n khazanah-rag python=3.11 -y
conda activate khazanah-rag

# install deps
pip install -r requirements.txt

# set up env vars
cp .env.example .env
# edit .env — add your GROQ_API_KEY and set LLM_PROVIDER=groq

# download the Annual Review PDFs from:
# https://www.khazanah.com.my/media-downloads/khazanah-annual-review/
# put them in data/

# run ingestion
python -m app.ingestion

# start API (terminal 1)
uvicorn app.api:app --host 0.0.0.0 --port 8000

# start UI (terminal 2)
conda activate khazanah-rag
streamlit run app/frontend.py
```

### Docker (alternative)

```bash
# make sure .env and data/*.pdf are in place
docker compose up --build
```

Then open:
- UI → http://localhost:8501
- API docs → http://localhost:8000/docs

### API Endpoints

| Endpoint | Method | What it does |
|----------|--------|-------------|
| `/health` | GET | Check if vector store is ready |
| `/query` | POST | Ask a question (`{"question": "..."}`) |
| `/extract` | GET | Get structured data as JSON |
| `/docs` | GET | Swagger docs |

## Design Decisions

**Why two PDF parsers?** PyMuPDF is fast for flowing text but bad at tables. pdfplumber is great at tables but slower. Using both gives the best coverage for a report like this that mixes paragraphs, charts, and financial tables.

**Why hybrid chunking?** I first try to split on section headings (so you don't cut a topic in half), then fall back to fixed-size chunks with overlap for long sections. Tables are kept whole because splitting a table across chunks makes it useless.

**Why all-MiniLM-L6-v2?** It runs locally, it's free, it's fast. Not the best for financial text specifically, but good enough for an MVP. With more time I'd look into a finance-specific embedding model.

**Why ChromaDB?** Zero config, runs locally, persists to disk. For a POC this is perfect. For production I'd swap to something like Qdrant or Pinecone.

**Why Groq?** Fast inference, generous free tier, and the Llama 3.1 model is solid. The LLM abstraction in `llm.py` supports both Groq and Gemini, so switching is just an env var change.

**Confidence scoring** — the system checks cosine similarity of retrieved chunks. If the best match is too weak, it refuses to answer instead of hallucinating. It's not perfect (it's just a retrieval proxy, not true confidence), but it helps.

### What I'd do with more time

- Use Unstructured.io or LlamaParse for better PDF parsing (especially charts/images)
- Add a re-ranker (cross-encoder) to improve retrieval quality
- Set up RAGAS evaluation to actually measure answer quality
- Add caching for repeated queries
- Multi-year comparison view

## Known Limitations

- Charts and images aren't processed — text and tables only. If something is only in a graph, the tool won't catch it.
- Structured extraction depends on the LLM output — sometimes it gets things wrong with complex tables. Always double-check against the PDF.
- Confidence scores are approximate. High confidence doesn't guarantee correctness.
- Groq free tier has rate limits. If you're hammering it, you might get throttled.
- Structured extraction can hit Groq's free tier payload and rate limits on large documents. 
  With more time I'd chunk the extraction into smaller calls or switch to a model with a 
  larger context window (e.g., Gemini Flash with 1M tokens). The extraction logic works — 
  it's a provider constraint, not a code issue.

## Design Questions

### 1. What would you change for production at Khazanah?

Swap ChromaDB for a managed vector DB (Qdrant Cloud or Pinecone). Add auth — probably SSO/LDAP since Khazanah would have internal identity management. Put ingestion behind a task queue (Celery + Redis) so it doesn't block the API. Add proper logging and monitoring. Use a self-hosted LLM instead of external APIs since you don't want to send internal data outside the network. Support multiple years of Annual Reviews with versioned indexes.

### 2. A user shared a wrong answer in a presentation. How do you prevent this?

This is a layered problem. First, improve retrieval — add a re-ranker so the LLM gets better context. Second, make the confidence threshold stricter and show it prominently in the UI so users think twice. Third, every answer already cites sources — make it a hard rule that answers without strong source matches get flagged as uncertain. Fourth, add a "faithfulness check" where a second LLM call verifies the answer is actually supported by the cited sources. Finally, on the process side — add a disclaimer like "AI-generated, verify before sharing externally" and log all queries for audit.

### 3. How do you roll out a pipeline update safely with 20 active analysts?

Run the new pipeline alongside the old one (blue-green). Version the vector index so both coexist. Route maybe 10-20% of traffic to the new version first, monitor for issues, then gradually increase. Keep the old version ready for instant rollback — just a config flag switch, no rebuild needed. Let analysts know ahead of time, give them a changelog, and set up a quick feedback channel so they can flag issues during the rollout.

## Tech Stack

| Component | Tool |
|-----------|------|
| PDF Parsing | PyMuPDF + pdfplumber |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB |
| LLM | Groq (Llama 3.1 8B) |
| Backend | FastAPI |
| Frontend | Streamlit |