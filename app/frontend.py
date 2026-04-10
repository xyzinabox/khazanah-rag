"""
Streamlit frontend for the Khazanah Annual Review RAG tool.
"""
import streamlit as st
import requests
import json

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Khazanah Annual Review AI", page_icon="🏦", layout="wide")
st.title("🏦 Khazanah Annual Review — AI Insights")
st.caption("Ask questions about Khazanah Nasional Berhad's Annual Review or explore extracted data.")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    with st.expander("System Status"):
        try:
            health = requests.get(f"{API_URL}/health", timeout=5).json()
            if health["vector_store_ready"]:
                st.success(f"✅ Ready — {health['chunks_count']} chunks indexed")
            else:
                st.warning("⚠️ Vector store not built yet.")
        except Exception:
            st.error("❌ API not reachable.")

    with st.expander("Sample Questions"):
        st.markdown("- What was Khazanah's TWRR?")
        st.markdown("- Which sectors did Khazanah increase exposure to?")
        st.markdown("- Summarise sustainability initiatives.")
        st.markdown("- What are the key financial highlights?")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_query, tab_data = st.tabs(["💬 Ask a Question", "📊 Structured Data"])

# -- Query Tab --
with tab_query:
    question = st.text_input("Your question:", placeholder="e.g. What was Khazanah's TWRR?")

    if st.button("Ask", type="primary") and question:
        with st.spinner("Searching the Annual Review…"):
            try:
                resp = requests.post(
                    f"{API_URL}/query",
                    json={"question": question},
                    timeout=120,
                ).json()

                # Answer
                st.markdown("### Answer")
                st.markdown(resp["answer"])

                # Confidence
                conf = resp.get("confidence", 0)
                color = "green" if conf > 0.6 else "orange" if conf > 0.4 else "red"
                st.markdown(f"**Confidence:** :{color}[{conf:.1%}]  |  **Latency:** {resp.get('latency_seconds', '?')}s")

                # Sources
                if resp.get("sources"):
                    st.markdown("### 📄 Sources")
                    for i, src in enumerate(resp["sources"], 1):
                        with st.expander(f"Source {i} — Page {src['page']} ({src['type']})"):
                            st.text(src["excerpt"])
            except Exception as e:
                st.error(f"Error: {e}")

# -- Structured Data Tab --
with tab_data:
    st.markdown("### Extracted Structured Data")
    st.caption("Financial metrics, portfolio companies, and more — extracted from the Annual Review using AI.")

    if st.button("Load / Refresh Data"):
        with st.spinner("Extracting structured data (this may take a minute on first run)…"):
            try:
                data = requests.get(f"{API_URL}/extract", timeout=180).json()

                if "error" in data:
                    st.error(data["error"])
                    st.code(data.get("raw_response", ""))
                else:
                    # Financial Metrics
                    if "financial_metrics" in data:
                        st.markdown("#### 💰 Key Financial Metrics")
                        fm = data["financial_metrics"]
                        cols = st.columns(3)
                        cols[0].metric("TWRR", fm.get("twrr", "N/A"))
                        cols[1].metric("Total Assets", fm.get("total_assets", "N/A"))
                        cols[2].metric("Realisable Asset Value", fm.get("realisable_asset_value", "N/A"))

                    # Portfolio Companies
                    if "portfolio_companies" in data and data["portfolio_companies"]:
                        st.markdown("#### 🏢 Portfolio Companies")
                        st.dataframe(data["portfolio_companies"], use_container_width=True)

                    # Sector Allocation
                    if "sector_allocation" in data and data["sector_allocation"]:
                        st.markdown("#### 📊 Sector Allocation")
                        st.dataframe(data["sector_allocation"], use_container_width=True)

                    # Investment Highlights
                    if "investment_highlights" in data and data["investment_highlights"]:
                        st.markdown("#### 🌟 Investment Highlights")
                        for h in data["investment_highlights"]:
                            st.markdown(f"- {h}")

                    # Sustainability
                    if "sustainability_initiatives" in data and data["sustainability_initiatives"]:
                        st.markdown("#### 🌱 Sustainability Initiatives")
                        for s in data["sustainability_initiatives"]:
                            st.markdown(f"- {s}")

                    # Raw JSON
                    with st.expander("View raw JSON"):
                        st.json(data)
            except Exception as e:
                st.error(f"Error: {e}")
