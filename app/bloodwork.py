"""Bloodwork PDF ingestion — pull lab values out of a report with Claude and
feed them through the same normalized ingestion layer as WHOOP.

Lab PDFs vary wildly in layout AND in units (one lab reports testosterone in
ng/dL, another in nmol/L; vitamin D in ng/mL vs nmol/L). We split the job by
reliability:

  * the MODEL reads the messy layout and returns each value with the unit
    exactly as printed (no arithmetic);
  * PYTHON converts that value into our canonical unit with a fixed factor
    table, and range-checks the result.

Asking an LLM to multiply by clinical conversion factors is fragile — it's
arithmetic, on health data, with no audit trail. Deterministic conversion in
code is exact, reviewable, and unit-testable. Anything with an unrecognized
unit or an out-of-range value is rejected (with a reason) rather than stored.
"""

from __future__ import annotations

import json

import anthropic
import fitz  # PyMuPDF

from app.ingestion import CANONICAL_UNITS, upsert_metrics

_claude = anthropic.Anthropic()
EXTRACT_MODEL = "claude-sonnet-5"  # stronger reader for messy layouts

# Canonical bloodwork metric_types + their target units (subset of CANONICAL_UNITS).
BLOODWORK_TYPES = [
    "vitamin_d", "ferritin", "glucose", "hdl", "ldl", "triglycerides",
    "total_cholesterol", "hba1c", "tsh", "testosterone", "crp",
]
TARGET_UNITS = {t: CANONICAL_UNITS[t] for t in BLOODWORK_TYPES}

# --------------------------------------------------------------------------- #
# Deterministic unit conversion
# --------------------------------------------------------------------------- #
# For each metric_type: normalized source unit -> multiplicative factor into the
# canonical unit. Identity (factor 1.0) covers the canonical unit and any unit
# equivalent to it (e.g. ferritin ug/L == ng/mL). hba1c is affine, handled below.
_FACTORS: dict[str, dict[str, float]] = {
    "vitamin_d":         {"ng/ml": 1.0, "nmol/l": 0.4006},
    "ferritin":          {"ng/ml": 1.0, "ug/l": 1.0, "mcg/l": 1.0},
    "glucose":           {"mg/dl": 1.0, "mmol/l": 18.0156},
    "hdl":               {"mg/dl": 1.0, "mmol/l": 38.67},
    "ldl":               {"mg/dl": 1.0, "mmol/l": 38.67},
    "triglycerides":     {"mg/dl": 1.0, "mmol/l": 88.57},
    "total_cholesterol": {"mg/dl": 1.0, "mmol/l": 38.67},
    "tsh":               {"miu/l": 1.0, "uiu/ml": 1.0, "miu/ml": 1.0, "mu/l": 1.0},
    "testosterone":      {"ng/dl": 1.0, "nmol/l": 28.84, "ng/ml": 100.0},
    "crp":               {"mg/l": 1.0, "mg/dl": 10.0},
}

# Physiologically plausible bounds in the CANONICAL unit. A value outside these
# almost always means a misread or a bad conversion, so we drop it.
_RANGES: dict[str, tuple[float, float]] = {
    "vitamin_d":         (0, 150),     # ng/mL
    "ferritin":          (0, 3000),    # ng/mL
    "glucose":           (20, 800),    # mg/dL
    "hdl":               (5, 150),     # mg/dL
    "ldl":               (10, 500),    # mg/dL
    "triglycerides":     (5, 3000),    # mg/dL
    "total_cholesterol": (40, 600),    # mg/dL
    "hba1c":             (3, 20),      # %
    "tsh":               (0, 100),     # mIU/L
    "testosterone":      (0, 2000),    # ng/dL
    "crp":               (0, 500),     # mg/L
}


def _norm_unit(unit: str) -> str:
    """Normalize a printed unit for lookup: lowercase, drop spaces, fold the
    micro sign (µ/μ) to 'u' so 'µIU/mL' and 'uIU/mL' compare equal."""
    return (unit or "").strip().lower().replace("µ", "u").replace("μ", "u").replace(" ", "")


def convert_to_canonical(metric_type: str, value: float, unit: str) -> float:
    """Convert a printed value into this metric's canonical unit.

    Raises ValueError if the source unit isn't recognized for the metric — we
    never guess a conversion.
    """
    u = _norm_unit(unit)
    if metric_type == "hba1c":
        if u in ("%", "percent"):
            return float(value)
        if u == "mmol/mol":  # IFCC -> NGSP %
            return float(value) * 0.0915 + 2.15
        raise ValueError(f"unrecognized hba1c unit {unit!r}")

    factors = _FACTORS.get(metric_type)
    if factors is None:
        raise ValueError(f"no conversion table for metric {metric_type!r}")
    if u not in factors:
        raise ValueError(f"unrecognized unit {unit!r} for {metric_type}")
    return float(value) * factors[u]


def in_range(metric_type: str, canonical_value: float) -> bool:
    lo, hi = _RANGES[metric_type]
    return lo <= canonical_value <= hi


def normalize_metrics(extracted_metrics: list[dict]) -> tuple[list[dict], list[dict]]:
    """Convert + validate model-extracted metrics.

    Input items look like {metric_type, value, unit} with the unit AS PRINTED.
    Returns (accepted, rejected). Accepted items carry the canonical value/unit
    plus a 'reported' string for audit; rejected items carry a 'reason'.
    """
    accepted: list[dict] = []
    rejected: list[dict] = []
    for m in extracted_metrics:
        mt = m.get("metric_type")
        reported = f"{m.get('value')} {m.get('unit')}".strip()
        if mt not in TARGET_UNITS:
            rejected.append({**m, "reason": f"unknown metric_type {mt!r}"})
            continue
        try:
            value = convert_to_canonical(mt, float(m["value"]), m["unit"])
        except (ValueError, TypeError, KeyError) as e:
            rejected.append({**m, "reason": str(e)})
            continue
        if not in_range(mt, value):
            rejected.append({**m, "reason": f"{round(value, 2)} out of plausible range"})
            continue
        accepted.append({
            "metric_type": mt,
            "value": round(value, 3),
            "unit": TARGET_UNITS[mt],
            "reported": reported,  # exactly what the report printed
        })
    return accepted, rejected


# --------------------------------------------------------------------------- #
# Model extraction (reading only — no arithmetic)
# --------------------------------------------------------------------------- #
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
                    "value": {"type": "number",
                              "description": "the numeric value EXACTLY as printed (do not convert)"},
                    "unit": {"type": "string",
                             "description": "the unit EXACTLY as printed, e.g. 'nmol/L', 'ng/mL'"},
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
    metric_list = ", ".join(BLOODWORK_TYPES)
    prompt = (
        "Extract laboratory test results from this bloodwork report.\n"
        f"Only include tests matching one of these metric_types: {metric_list}.\n\n"
        "Rules:\n"
        "- Match each value to its correct test name carefully (layouts can be "
        "misaligned; do not grab a neighbouring test's number).\n"
        "- Report the numeric value and its unit EXACTLY as printed. DO NOT "
        "convert units or do any arithmetic — that happens downstream.\n"
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
    accepted, rejected = normalize_metrics(extracted["metrics"])
    records = [
        {"date": date, "metric_type": m["metric_type"], "value": m["value"],
         "unit": m["unit"]}
        for m in accepted
    ]
    written = upsert_metrics(user_id, "bloodwork", records)
    return {
        "source": "bloodwork",
        "report_date": date,
        "metrics_written": written,
        "extracted": accepted,   # canonical values + 'reported' for audit
        "rejected": rejected,    # anything dropped, with a reason
    }
