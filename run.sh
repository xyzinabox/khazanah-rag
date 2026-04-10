#!/usr/bin/env bash
set -e

echo "============================================"
echo "  Khazanah Annual Review RAG — Setup"
echo "============================================"

# 1. Check .env
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Copying from .env.example …"
    cp .env.example .env
    echo "👉 Please edit .env and add your GEMINI_API_KEY, then re-run this script."
    exit 1
fi

# 2. Install dependencies
echo "📦 Installing Python dependencies …"
pip install -r requirements.txt -q
# Note: If using conda, activate your env first:
#   conda create -n khazanah-rag python=3.11 -y
#   conda activate khazanah-rag
#   pip install -r requirements.txt

# 3. Check PDF
if [ ! -f data/annual_review.pdf ]; then
    echo ""
    echo "📥 Please download the latest Khazanah Annual Review PDF from:"
    echo "   https://www.khazanah.com.my/media-downloads/khazanah-annual-review/"
    echo "   and save it as:  data/annual_review.pdf"
    echo ""
    exit 1
fi

# 4. Run ingestion
echo "🔄 Running ingestion pipeline …"
python -m app.ingestion

# 5. Start backend + frontend
echo ""
echo "🚀 Starting API server (port 8000) and Streamlit UI (port 8501) …"
echo "   API docs: http://localhost:8000/docs"
echo "   Frontend:  http://localhost:8501"
echo ""

uvicorn app.api:app --host 0.0.0.0 --port 8000 &
API_PID=$!
streamlit run app/frontend.py --server.port 8501 --server.headless true &
UI_PID=$!

trap "kill $API_PID $UI_PID 2>/dev/null" EXIT
wait
