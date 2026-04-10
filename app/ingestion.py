"""
Section 1 — Data Ingestion & Pipeline
Parses the Annual Review PDF, chunks it semantically, embeds, and stores in ChromaDB.
"""
import os, re, json
import pdfplumber
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
from app.config import (
    PDF_DIR, CHROMA_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
)
import glob

# ── PDF Parsing ──────────────────────────────────────────────────────────────

def extract_text_pymupdf(pdf_path: str) -> list[dict]:
    """Extract text per page using PyMuPDF (good for flowing text)."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({"page": i + 1, "text": text.strip()})
    doc.close()
    return pages


def extract_tables_pdfplumber(pdf_path: str) -> list[dict]:
    """Extract tables per page using pdfplumber (better for tabular data)."""
    tables_out = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for t_idx, table in enumerate(tables):
                # Convert table rows to readable text
                rows = []
                for row in table:
                    cells = [str(c).strip() if c else "" for c in row]
                    rows.append(" | ".join(cells))
                table_text = "\n".join(rows)
                tables_out.append({
                    "page": i + 1,
                    "table_index": t_idx,
                    "text": table_text,
                })
    return tables_out


def parse_pdf(pdf_path: str = "") -> list[dict]:
    """
    Combined parsing: page text + tables.
    Returns a list of {'page': int, 'type': str, 'text': str}.
    """
    print(f"📄 Parsing PDF: {pdf_path}")
    pages = extract_text_pymupdf(pdf_path)
    tables = extract_tables_pdfplumber(pdf_path)

    documents = []
    for p in pages:
        if p["text"]:
            documents.append({"page": p["page"], "type": "text", "text": p["text"]})
    for t in tables:
        if t["text"].strip():
            documents.append({
                "page": t["page"],
                "type": "table",
                "text": f"[TABLE on page {t['page']}]\n{t['text']}",
            })
    print(f"   → Extracted {len(pages)} pages, {len(tables)} tables")
    return documents


# ── Chunking ─────────────────────────────────────────────────────────────────

def _heading_pattern():
    """Regex for common heading-like lines (ALL CAPS or short bold-like lines)."""
    return re.compile(r"^[A-Z][A-Z &/,\-]{4,}$", re.MULTILINE)


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Hybrid chunking strategy:
    1. Try to split on section headings (semantic boundaries).
    2. Fall back to overlapping fixed-size chunks for long sections.
    Tables are kept as single chunks (they lose meaning when split).
    """
    chunks = []
    heading_re = _heading_pattern()

    for doc in documents:
        page = doc["page"]
        doc_type = doc["type"]
        text = doc["text"]

        # Tables → keep whole
        if doc_type == "table":
            chunks.append({"page": page, "type": "table", "text": text})
            continue

        # Split on headings first
        sections = heading_re.split(text)
        headings = heading_re.findall(text)

        parts = []
        for idx, section in enumerate(sections):
            heading = headings[idx - 1] if idx > 0 and idx - 1 < len(headings) else ""
            section_text = (heading + "\n" + section).strip() if heading else section.strip()
            if section_text:
                parts.append(section_text)

        # If no headings found, treat whole page text as one part
        if len(parts) == 0:
            parts = [text]

        # Sub-chunk long parts with overlap
        for part in parts:
            approx_tokens = len(part) // 4
            if approx_tokens <= CHUNK_SIZE:
                chunks.append({"page": page, "type": doc_type, "text": part})
            else:
                words = part.split()
                step = CHUNK_SIZE - CHUNK_OVERLAP
                for start in range(0, len(words), step):
                    chunk_words = words[start : start + CHUNK_SIZE]
                    chunks.append({
                        "page": page,
                        "type": doc_type,
                        "text": " ".join(chunk_words),
                    })

    print(f"✂️  Created {len(chunks)} chunks")
    return chunks


# ── Embedding & Storage ──────────────────────────────────────────────────────

def build_vector_store(chunks: list[dict], persist_dir: str = CHROMA_DIR):
    """Embed chunks and store in ChromaDB."""
    print(f"🧠 Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["text"] for c in chunks]
    print(f"🔢 Generating embeddings for {len(texts)} chunks …")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32).tolist()

    # Persist ChromaDB
    client = chromadb.PersistentClient(path=persist_dir)
    # Delete existing collection if re-ingesting
    try:
        client.delete_collection("annual_review")
    except Exception:
        pass
    collection = client.create_collection(
        name="annual_review",
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"page": c["page"], "type": c["type"]} for c in chunks]

    # ChromaDB has a batch limit; add in batches of 500
    batch = 500
    for start in range(0, len(ids), batch):
        end = start + batch
        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

    print(f"✅ Vector store built at {persist_dir} ({collection.count()} vectors)")
    return collection


# ── Main ─────────────────────────────────────────────────────────────────────

def run_ingestion(pdf_dir="data/"):
    all_chunks = []
    for pdf_path in glob.glob(os.path.join(pdf_dir, "*.pdf")):
        documents = parse_pdf(pdf_path)
        # Tag each chunk with its source file
        for doc in documents:
            doc["source_file"] = os.path.basename(pdf_path)
        chunks = chunk_documents(documents)
        all_chunks.extend(chunks)
    build_vector_store(all_chunks)


if __name__ == "__main__":
    run_ingestion()
