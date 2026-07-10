"""Bloodwork PDF ingestion — pull lab values out of a report with Claude and
feed them through the same normalized ingestion layer as WHOOP.

Lab PDFs vary wildly in layout AND in units (one lab reports testosterone in
ng/dL, another in nmol/L; vitamin D in ng/mL vs nmol/L). We extract the text
(PyMuPDF) and let the model both read the values and CONVERT each into our
canonical unit, so everything downstream is directly comparable. Structured
outputs guarantee the model returns exactly our shape; the metric_type enum
means only canonical tests come through.
"""

from __future__ import annotations

import json

import anthropic
import fitz  # PyMuPDF

from app.ingestion import CANONICAL_UNITS, upsert_metrics

_claude = anthropic.Anthropic()
EXTRACT_MODEL = "claude-sonnet-5"  # stronger reader for messy layouts + conversions

# Canonical bloodwork metric_types + their target units (subset of CANONICAL_UNITS).
BLOODWORK_TYPES = [
    "vitamin_d", "ferritin", "glucose", "hdl", "ldl", "triglycerides",
    "total_cholesterol", "hba1c", "tsh", "testosterone", "crp",
]
TARGET_UNITS = {t: CANONICAL_UNITS[t] for t in BLOODWORK_TYPES}

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
                "required": ["metric_type", "value", "unit", "reported"],
                "properties": {
                    "metric_type": {"type": "string", "enum": BLOODWORK_TYPES},
                    "value": {"type": "number",
                              "description": "value CONVERTED into the canonical unit"},
                    "unit": {"type": "string",
                             "description": "the canonical unit (must match target)"},
                    "reported": {"type": "string",
                                 "description": "value and unit exactly as printed, e.g. '47.65 nmol/L'"},
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
    unit_lines = "\n".join(f"  {t}: {u}" for t, u in TARGET_UNITS.items())
    prompt = (
        "Extract laboratory test results from this bloodwork report.\n"
        "Only include tests matching one of these metric_types, each with its "
        "CANONICAL unit:\n"
        f"{unit_lines}\n\n"
        "Rules:\n"
        "- Match each value to its correct test name carefully (layouts can be "
        "misaligned; do not grab a neighbouring test's number).\n"
        "- Convert the value into the canonical unit using standard clinical "
        "conversion factors (e.g. testosterone nmol/L x 28.84 = ng/dL; "
        "vitamin D nmol/L x 0.4006 = ng/mL). Put the converted number in 'value' "
        "and the canonical unit in 'unit'.\n"
        "- Put the exact printed value and unit in 'reported'.\n"
        "- If you cannot confidently identify a test or its value, omit it.\n"
        "- Extract the sample collection date as YYYY-MM-DD.\n\n"
        f"REPORT TEXT:\n{text}"
    )
    msg = _claude.messages.create(
        model=EXTRACT_MODEL,
        max_tokens=2048,
        output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(b.text for b in msg.content if b.type == "text")
    return json.loads(raw)


def ingest_pdf(user_id: int, pdf_bytes: bytes) -> dict:
    extracted = extract_metrics(extract_text(pdf_bytes))
    date = extracted["report_date"]
    records = [
        # store the canonical unit we defined, not the model's echo, for consistency
        {"date": date, "metric_type": m["metric_type"], "value": m["value"],
         "unit": TARGET_UNITS.get(m["metric_type"])}
        for m in extracted["metrics"]
    ]
    written = upsert_metrics(user_id, "bloodwork", records)
    return {
        "source": "bloodwork",
        "report_date": date,
        "metrics_written": written,
        "extracted": extracted["metrics"],  # includes 'reported' for audit
    }
