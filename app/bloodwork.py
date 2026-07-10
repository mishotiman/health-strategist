"""Bloodwork PDF ingestion — pull lab values out of a report with Claude (Haiku)
and feed them through the same normalized ingestion layer as WHOOP.

Lab PDFs vary wildly in layout, so we extract the text (PyMuPDF) and let a cheap
model map it into a fixed JSON schema. Structured outputs guarantee the model
returns exactly our shape — it can't hand back malformed data. The metric_type
enum means only tests in our canonical vocabulary come through; everything else
is dropped automatically.
"""

from __future__ import annotations

import json

import anthropic
import fitz  # PyMuPDF

from app.ingestion import upsert_metrics

_claude = anthropic.Anthropic()
EXTRACT_MODEL = "claude-haiku-4-5"  # cheap; extraction is a simple task

# Canonical bloodwork metric_types the model is allowed to emit.
BLOODWORK_TYPES = [
    "vitamin_d", "ferritin", "glucose", "hdl", "ldl", "triglycerides",
    "total_cholesterol", "hba1c", "tsh", "testosterone", "crp",
]

_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["report_date", "metrics"],
    "properties": {
        "report_date": {"type": "string",
                        "description": "Sample collection date as YYYY-MM-DD"},
        "metrics": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["metric_type", "value", "unit"],
                "properties": {
                    "metric_type": {"type": "string", "enum": BLOODWORK_TYPES},
                    "value": {"type": "number"},
                    "unit": {"type": "string",
                             "description": "unit exactly as printed, e.g. ng/mL"},
                },
            },
        },
    },
}


def extract_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def extract_metrics(text: str) -> dict:
    prompt = (
        "Extract laboratory test results from this bloodwork report.\n"
        f"Only include tests whose metric_type is one of: {BLOODWORK_TYPES}.\n"
        "Ignore every other test. Use the numeric value and the unit exactly as "
        "printed. Also extract the sample collection date as YYYY-MM-DD.\n\n"
        f"REPORT TEXT:\n{text}"
    )
    msg = _claude.messages.create(
        model=EXTRACT_MODEL,
        max_tokens=1024,
        output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(b.text for b in msg.content if b.type == "text")
    return json.loads(raw)


def ingest_pdf(user_id: int, pdf_bytes: bytes) -> dict:
    extracted = extract_metrics(extract_text(pdf_bytes))
    date = extracted["report_date"]
    records = [
        {"date": date, "metric_type": m["metric_type"],
         "value": m["value"], "unit": m["unit"]}
        for m in extracted["metrics"]
    ]
    written = upsert_metrics(user_id, "bloodwork", records)
    return {
        "source": "bloodwork",
        "report_date": date,
        "metrics_written": written,
        "extracted": extracted["metrics"],
    }
