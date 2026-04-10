"""
Section 2 — RAG-Powered Query Engine
Retrieves relevant chunks and generates answers with source citations.
"""
import chromadb
from sentence_transformers import SentenceTransformer
from app.config import CHROMA_DIR, EMBEDDING_MODEL, TOP_K, SIMILARITY_THRESHOLD
from app.llm import call_llm

# Cache model & collection at module level
_model = None
_collection = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection("annual_review")
    return _collection


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Return the top-k most relevant chunks with distances."""
    model = _get_model()
    embedding = model.encode([query]).tolist()
    collection = _get_collection()

    results = collection.query(
        query_embeddings=embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "page": results["metadatas"][0][i]["page"],
            "type": results["metadatas"][0][i]["type"],
            "distance": results["distances"][0][i],
        })
    return chunks


def query_rag(question: str) -> dict:
    """
    Full RAG pipeline: retrieve → build prompt → generate answer.
    Returns {answer, sources, confidence}.
    """
    chunks = retrieve(question)

    # Confidence check: if best chunk is too far, refuse gracefully
    best_distance = chunks[0]["distance"] if chunks else 1.0
    # cosine distance: 0 = identical, 2 = opposite.  Lower is better.
    if best_distance > (1 - SIMILARITY_THRESHOLD):
        return {
            "answer": (
                "I could not find enough relevant information in the Annual Review "
                "to confidently answer this question. Please try rephrasing or ask "
                "about a topic covered in the report."
            ),
            "sources": [],
            "confidence": round(1 - best_distance, 3),
        }

    # Build context
    context_parts = []
    for i, c in enumerate(chunks):
        label = f"[Source {i+1} — Page {c['page']}, {c['type']}]"
        context_parts.append(f"{label}\n{c['text']}")
    context = "\n\n---\n\n".join(context_parts)

    system = (
        "You are an expert analyst assistant for Khazanah Nasional Berhad's Annual Review. "
        "Answer the user's question based ONLY on the provided context. "
        "If the context does not contain enough information, say so clearly. "
        "Always cite your sources using the [Source N] labels provided."
    )

    prompt = f"""Context from the Annual Review:
{context}

Question: {question}

Instructions:
- Answer concisely and accurately using ONLY the context above.
- Reference sources as [Source 1], [Source 2], etc.
- If the information is not in the context, say "This information is not available in the provided sections of the Annual Review."
"""

    answer = call_llm(prompt, system=system)

    sources = [
        {"page": c["page"], "type": c["type"], "excerpt": c["text"][:200] + "…"}
        for c in chunks
    ]
    confidence = round(1 - best_distance, 3)

    return {"answer": answer, "sources": sources, "confidence": confidence}
