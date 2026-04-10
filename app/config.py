"""Centralised configuration."""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
PDF_DIR = DATA_DIR  # all PDFs in data/
EXTRACTED_JSON = os.path.join(DATA_DIR, "structured_data.json")

# --- LLM ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# --- Embedding ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- RAG ---
CHUNK_SIZE = 500        # tokens (approx chars / 4)
CHUNK_OVERLAP = 100
TOP_K = 5
SIMILARITY_THRESHOLD = 0.25   # below this → "not enough info"