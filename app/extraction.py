"""
Section 3 — Structured Data Extraction
Uses the LLM to extract portfolio companies, financial metrics, etc. into clean JSON.
"""
import json, os, glob
from app.config import EXTRACTED_JSON, PDF_DIR, DATA_DIR
from app.llm import call_llm
from app.ingestion import parse_pdf


def extract_structured_data(force: bool = False) -> dict:
    """Extract structured data from the Annual Review using LLM."""
    # Return cached if available
    if not force and os.path.exists(EXTRACTED_JSON):
        with open(EXTRACTED_JSON) as f:
            return json.load(f)

    # Parse all PDFs to get raw text
    documents = []
    for pdf_path in glob.glob(os.path.join(PDF_DIR, "*.pdf")):
        documents.extend(parse_pdf(pdf_path))
    # Combine all text (truncate if needed to fit context window)
    full_text = "\n\n".join(d["text"] for d in documents)
    full_text = full_text[:15000]

    system = (
        "You are a data extraction specialist. Extract structured data from "
        "Khazanah Nasional Berhad's Annual Review. Return ONLY valid JSON, "
        "no markdown backticks, no explanation."
    )

    prompt = f"""From the following Annual Review text, extract ALL of the following into a JSON object:

{{
  "portfolio_companies": [
    {{
      "name": "Company Name",
      "sector": "Sector",
      "ownership_stake_percent": null or number,
      "notes": "any relevant detail"
    }}
  ],
  "financial_metrics": {{
    "twrr": "value as string with period",
    "total_assets": "value",
    "realisable_asset_value": "value",
    "net_worth_adjusted": "value if available",
    "other_metrics": {{}}
  }},
  "investment_highlights": [
    "key highlight 1",
    "key highlight 2"
  ],
  "sustainability_initiatives": [
    "initiative 1",
    "initiative 2"
  ],
  "sector_allocation": [
    {{"sector": "name", "percentage": number_or_null}}
  ]
}}

Rules:
- Use null for unavailable values; do NOT invent data.
- For ownership stakes, only include if explicitly stated.
- Include as many portfolio companies as you can find.
- All monetary values should include their unit (e.g., "RM 100 billion").

Annual Review Text:
{full_text}
"""

    raw = call_llm(prompt, system=system)

    # Clean and parse
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # If JSON parse fails, return raw text for debugging
        data = {"raw_response": raw, "error": "Failed to parse LLM output as JSON"}

    # Cache
    os.makedirs(os.path.dirname(EXTRACTED_JSON), exist_ok=True)
    with open(EXTRACTED_JSON, "w") as f:
        json.dump(data, f, indent=2)

    return data


if __name__ == "__main__":
    result = extract_structured_data(force=True)
    print(json.dumps(result, indent=2))