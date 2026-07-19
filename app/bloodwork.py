"""Bloodwork PDF ingestion — pull lab values out of a report with Claude and
feed them through the same normalized ingestion layer as WHOOP.

Lab PDFs vary wildly in layout AND in units (one lab reports testosterone in
ng/dL, another in nmol/L; vitamin D in ng/mL vs nmol/L). We split the job by
reliability:

  * the MODEL reads the messy layout and returns each value with the unit
    exactly as printed (no arithmetic), plus a classification of the document;
  * PYTHON converts each value into our canonical unit with a fixed factor
    table, and range-checks the result.

Asking an LLM to multiply by clinical conversion factors is fragile — it's
arithmetic, on health data, with no audit trail. Deterministic conversion in
code is exact, reviewable, and unit-testable. Anything with an unrecognized
unit or an out-of-range value is rejected (with a reason) rather than stored.

Numeric markers live in MARKERS (canonical unit + source-unit factors + a
plausibility range). Qualitative markers (microbiology: negative/positive) can't
be numbers, so they're stored via a text_value instead.
"""

from __future__ import annotations

import datetime as dt
import json

import anthropic
import fitz  # PyMuPDF

from app.ingestion import CANONICAL_UNITS, upsert_metrics

_claude = anthropic.Anthropic()
EXTRACT_MODEL = "claude-sonnet-5"  # stronger reader for messy layouts

# --------------------------------------------------------------------------- #
# Marker catalogue — the single source of truth for numeric bloodwork markers.
#   unit    : canonical unit (what we store)
#   factors : normalized-source-unit -> multiplicative factor into the canonical
#             unit. Identity (1.0) covers the canonical unit and any equivalent
#             unit. Units not listed are rejected (we never guess a conversion).
#   range   : physiologically plausible (lo, hi) in the CANONICAL unit; a value
#             outside almost always means a misread or bad conversion, so drop it.
# hba1c is affine (IFCC mmol/mol -> NGSP %), handled specially below.
# --------------------------------------------------------------------------- #
MARKERS: dict[str, dict] = {
    # ---- lipids / metabolic ----
    "glucose":           {"unit": "mg/dL", "factors": {"mg/dl": 1.0, "mmol/l": 18.0156}, "range": (20, 800)},
    "hdl":               {"unit": "mg/dL", "factors": {"mg/dl": 1.0, "mmol/l": 38.67}, "range": (5, 150)},
    "ldl":               {"unit": "mg/dL", "factors": {"mg/dl": 1.0, "mmol/l": 38.67}, "range": (10, 500)},
    "triglycerides":     {"unit": "mg/dL", "factors": {"mg/dl": 1.0, "mmol/l": 88.57}, "range": (5, 3000)},
    "total_cholesterol": {"unit": "mg/dL", "factors": {"mg/dl": 1.0, "mmol/l": 38.67}, "range": (40, 600)},
    # ---- endocrine / stores ----
    "tsh":               {"unit": "mIU/L", "factors": {"miu/l": 1.0, "uiu/ml": 1.0, "miu/ml": 1.0, "mu/l": 1.0}, "range": (0, 100)},
    "testosterone":      {"unit": "ng/dL", "factors": {"ng/dl": 1.0, "nmol/l": 28.84, "ng/ml": 100.0}, "range": (0, 2000)},
    "ferritin":          {"unit": "ng/mL", "factors": {"ng/ml": 1.0, "ug/l": 1.0, "mcg/l": 1.0}, "range": (0, 3000)},
    # ---- inflammation / immunology ----
    "crp":               {"unit": "mg/L", "factors": {"mg/l": 1.0, "mg/dl": 10.0}, "range": (0, 500)},
    "hs_crp":            {"unit": "mg/L", "factors": {"mg/l": 1.0, "mg/dl": 10.0}, "range": (0, 500)},
    "calprotectin":      {"unit": "µg/g", "factors": {"ug/g": 1.0, "mcg/g": 1.0, "mg/kg": 1.0}, "range": (0, 6000)},
    # ---- CBC / hematology ----
    "esr":               {"unit": "mm/h", "factors": {"mm/h": 1.0, "mm/hr": 1.0}, "range": (0, 200)},
    "wbc":               {"unit": "10^9/L", "factors": {"10^9/l": 1.0, "g/l": 1.0, "10^3/ul": 1.0, "k/ul": 1.0}, "range": (0, 200)},
    "neutrophils_pct":   {"unit": "%", "factors": {"%": 1.0}, "range": (0, 100)},
    "neutrophils_abs":   {"unit": "10^9/L", "factors": {"10^9/l": 1.0, "g/l": 1.0, "10^3/ul": 1.0}, "range": (0, 100)},
    "eosinophils_pct":   {"unit": "%", "factors": {"%": 1.0}, "range": (0, 100)},
    "eosinophils_abs":   {"unit": "10^9/L", "factors": {"10^9/l": 1.0, "g/l": 1.0, "10^3/ul": 1.0}, "range": (0, 50)},
    "lymphocytes_pct":   {"unit": "%", "factors": {"%": 1.0}, "range": (0, 100)},
    "lymphocytes_abs":   {"unit": "10^9/L", "factors": {"10^9/l": 1.0, "g/l": 1.0, "10^3/ul": 1.0}, "range": (0, 100)},
    "monocytes_pct":     {"unit": "%", "factors": {"%": 1.0}, "range": (0, 100)},
    "monocytes_abs":     {"unit": "10^9/L", "factors": {"10^9/l": 1.0, "g/l": 1.0, "10^3/ul": 1.0}, "range": (0, 50)},
    "basophils_pct":     {"unit": "%", "factors": {"%": 1.0}, "range": (0, 100)},
    "basophils_abs":     {"unit": "10^9/L", "factors": {"10^9/l": 1.0, "g/l": 1.0, "10^3/ul": 1.0}, "range": (0, 20)},
    "rbc":               {"unit": "10^12/L", "factors": {"10^12/l": 1.0, "t/l": 1.0, "10^6/ul": 1.0, "m/ul": 1.0}, "range": (0, 15)},
    "hemoglobin":        {"unit": "g/L", "factors": {"g/l": 1.0, "g/dl": 10.0}, "range": (0, 300)},
    "hematocrit":        {"unit": "L/L", "factors": {"l/l": 1.0, "%": 0.01}, "range": (0, 1)},
    "mcv":               {"unit": "fL", "factors": {"fl": 1.0}, "range": (30, 180)},
    "mch":               {"unit": "pg", "factors": {"pg": 1.0}, "range": (5, 60)},
    "mchc":              {"unit": "g/L", "factors": {"g/l": 1.0, "g/dl": 10.0}, "range": (150, 450)},
    "rdw":               {"unit": "%", "factors": {"%": 1.0}, "range": (5, 40)},
    "platelets":         {"unit": "10^9/L", "factors": {"10^9/l": 1.0, "g/l": 1.0, "10^3/ul": 1.0, "k/ul": 1.0}, "range": (0, 2000)},
    "mpv":               {"unit": "fL", "factors": {"fl": 1.0}, "range": (3, 25)},
    "pdw":               {"unit": "fL", "factors": {"fl": 1.0}, "range": (3, 40)},
    # ---- biochemistry (kidney / liver / electrolytes / iron) ----
    "creatinine":        {"unit": "µmol/L", "factors": {"umol/l": 1.0, "mg/dl": 88.42}, "range": (10, 2000)},
    "total_protein":     {"unit": "g/L", "factors": {"g/l": 1.0, "g/dl": 10.0}, "range": (20, 150)},
    "albumin":           {"unit": "g/L", "factors": {"g/l": 1.0, "g/dl": 10.0}, "range": (5, 80)},
    "urea":              {"unit": "mmol/L", "factors": {"mmol/l": 1.0}, "range": (0, 60)},
    "ast":               {"unit": "U/L", "factors": {"u/l": 1.0}, "range": (0, 5000)},
    "alt":               {"unit": "U/L", "factors": {"u/l": 1.0}, "range": (0, 5000)},
    "ggt":               {"unit": "U/L", "factors": {"u/l": 1.0}, "range": (0, 5000)},
    "alp":               {"unit": "U/L", "factors": {"u/l": 1.0}, "range": (0, 3000)},
    "potassium":         {"unit": "mmol/L", "factors": {"mmol/l": 1.0, "meq/l": 1.0}, "range": (1, 10)},
    "sodium":            {"unit": "mmol/L", "factors": {"mmol/l": 1.0, "meq/l": 1.0}, "range": (100, 180)},
    "iron":              {"unit": "µmol/L", "factors": {"umol/l": 1.0, "ug/dl": 0.1791}, "range": (0, 90)},
    "tibc":              {"unit": "µmol/L", "factors": {"umol/l": 1.0, "ug/dl": 0.1791}, "range": (0, 200)},
    # ---- vitamins ----
    "vitamin_d":         {"unit": "ng/mL", "factors": {"ng/ml": 1.0, "nmol/l": 0.4006}, "range": (0, 150)},
    "vitamin_d_active":  {"unit": "pg/mL", "factors": {"pg/ml": 1.0, "pmol/l": 0.4166}, "range": (0, 300)},
    "vitamin_b12":       {"unit": "pg/mL", "factors": {"pg/ml": 1.0, "ng/l": 1.0, "pmol/l": 1.355}, "range": (0, 3000)},
    "holotc":            {"unit": "pmol/L", "factors": {"pmol/l": 1.0}, "range": (0, 300)},
    "folate":            {"unit": "ng/mL", "factors": {"ng/ml": 1.0, "nmol/l": 0.4414}, "range": (0, 60)},
    "folate_rbc":        {"unit": "ng/mL", "factors": {"ng/ml": 1.0, "nmol/l": 0.4414}, "range": (0, 2500)},
    "vitamin_b1":        {"unit": "nmol/L", "factors": {"nmol/l": 1.0, "ug/l": 2.965}, "range": (0, 1000)},
    "vitamin_b2":        {"unit": "µg/L", "factors": {"ug/l": 1.0, "nmol/l": 0.3764}, "range": (0, 1000)},
    "vitamin_b3":        {"unit": "µg/L", "factors": {"ug/l": 1.0}, "range": (0, 20000)},
    "vitamin_b5":        {"unit": "µg/L", "factors": {"ug/l": 1.0}, "range": (0, 20000)},
    "vitamin_b6":        {"unit": "nmol/L", "factors": {"nmol/l": 1.0, "ug/l": 4.046}, "range": (0, 1500)},
    "vitamin_b7":        {"unit": "ng/L", "factors": {"ng/l": 1.0}, "range": (0, 3000)},
    "vitamin_a":         {"unit": "µg/dL", "factors": {"ug/dl": 1.0, "umol/l": 28.645}, "range": (0, 200)},
    "vitamin_c":         {"unit": "mg/dL", "factors": {"mg/dl": 1.0, "umol/l": 0.01761}, "range": (0, 6)},
    "vitamin_e":         {"unit": "mg/L", "factors": {"mg/l": 1.0, "mg/dl": 10.0, "umol/l": 0.4307}, "range": (0, 80)},
    "vitamin_k1":        {"unit": "ng/mL", "factors": {"ng/ml": 1.0, "nmol/l": 0.4507}, "range": (0, 50)},
    # ---- minerals / micronutrients ----
    "magnesium":         {"unit": "mmol/L", "factors": {"mmol/l": 1.0, "mg/dl": 0.4114, "meq/l": 0.5}, "range": (0, 5)},
    "zinc":              {"unit": "µmol/L", "factors": {"umol/l": 1.0, "ug/dl": 0.153, "ug/l": 0.0153}, "range": (0, 60)},
    "copper":            {"unit": "µmol/L", "factors": {"umol/l": 1.0, "ug/dl": 0.1574}, "range": (0, 80)},
    "selenium":          {"unit": "µmol/L", "factors": {"umol/l": 1.0, "ug/l": 0.01266, "ug/dl": 0.1266}, "range": (0, 15)},
    "iodine":            {"unit": "µg/L", "factors": {"ug/l": 1.0}, "range": (0, 2000)},
    "homocysteine":      {"unit": "µmol/L", "factors": {"umol/l": 1.0, "mg/l": 7.397}, "range": (0, 200)},
}

_HBA1C_RANGE = (3, 20)  # % (NGSP)

# Derived views (kept as module-level names the tests and callers rely on).
BLOODWORK_TYPES = list(MARKERS) + ["hba1c"]
TARGET_UNITS = {t: m["unit"] for t, m in MARKERS.items()}
TARGET_UNITS["hba1c"] = "%"
_FACTORS = {t: m["factors"] for t, m in MARKERS.items()}
_RANGES = {t: m["range"] for t, m in MARKERS.items()}
_RANGES["hba1c"] = _HBA1C_RANGE

# Register bloodwork units into the shared vocabulary (single source = MARKERS).
CANONICAL_UNITS.update(TARGET_UNITS)

# Qualitative markers — results are categorical, not numeric (stored as text_value).
QUALITATIVE_TYPES = [
    "clostridium_difficile_gdh",
    "clostridium_difficile_toxin_a",
    "clostridium_difficile_toxin_b",
    "quantiferon_tb",
]
_ALLOWED_RESULTS = {"negative", "positive", "indeterminate", "equivocal",
                    "detected", "not_detected", "pending"}


# --------------------------------------------------------------------------- #
# Deterministic unit conversion
# --------------------------------------------------------------------------- #
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
    """Convert + validate model-extracted NUMERIC metrics.

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


def _norm_result(result: str) -> str:
    r = (result or "").strip().lower().replace("(-)", "").replace("(+)", "").strip()
    r = {"neg": "negative", "-": "negative", "pos": "positive", "+": "positive"}.get(r, r)
    return r.replace(" ", "_")


def normalize_qualitative(items: list[dict]) -> tuple[list[dict], list[dict]]:
    """Validate model-extracted QUALITATIVE results (negative/positive/…)."""
    accepted: list[dict] = []
    rejected: list[dict] = []
    for m in items:
        mt = m.get("metric_type")
        res = _norm_result(m.get("result"))
        if mt not in QUALITATIVE_TYPES:
            rejected.append({**m, "reason": f"unknown qualitative metric {mt!r}"})
            continue
        if res not in _ALLOWED_RESULTS:
            rejected.append({**m, "reason": f"unrecognized result {m.get('result')!r}"})
            continue
        accepted.append({"metric_type": mt, "text_value": res, "reported": str(m.get("result"))})
    return accepted, rejected


# --------------------------------------------------------------------------- #
# Model extraction (reading + classification only — no arithmetic)
# --------------------------------------------------------------------------- #
_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["document_type", "document_summary", "report_date", "metrics", "qualitative"],
    "properties": {
        "document_type": {
            "type": "string",
            "enum": ["bloodwork", "health_other", "not_health"],
            "description": "bloodwork = a laboratory report with blood/lab test results; "
                           "health_other = health-related but NOT a lab report with values "
                           "(imaging report, doctor's note, prescription, fitness/nutrition plan); "
                           "not_health = unrelated to health (invoice, receipt, resume, contract).",
        },
        "document_summary": {"type": "string",
                             "description": "At most 12 words naming what this document is."},
        "report_date": {"type": "string",
                        "description": "Sample collection date as YYYY-MM-DD, or '' if none."},
        "metrics": {
            "type": "array",
            "description": "Numeric lab results.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["metric_type", "value", "unit"],
                "properties": {
                    "metric_type": {"type": "string", "enum": BLOODWORK_TYPES},
                    "value": {"type": "number",
                              "description": "the numeric value EXACTLY as printed (do not convert)"},
                    "unit": {"type": "string",
                             "description": "the unit EXACTLY as printed, e.g. 'nmol/L', 'G/l', 'U/l'"},
                },
            },
        },
        "qualitative": {
            "type": "array",
            "description": "Categorical results (microbiology etc.) that are not numbers.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["metric_type", "result"],
                "properties": {
                    "metric_type": {"type": "string", "enum": QUALITATIVE_TYPES},
                    "result": {"type": "string", "enum": sorted(_ALLOWED_RESULTS)},
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
    numeric_list = ", ".join(BLOODWORK_TYPES)
    qual_list = ", ".join(QUALITATIVE_TYPES)
    prompt = (
        "You are processing an uploaded document. First classify it via "
        "document_type (bloodwork / health_other / not_health) and give a short "
        "document_summary naming what it is.\n\n"
        "ONLY if document_type is 'bloodwork', extract results. Reports may be in "
        "any language (e.g. Bulgarian) — map each test to the matching English "
        "metric_type below.\n\n"
        f"NUMERIC metric_types: {numeric_list}.\n"
        "- Report each numeric value and its unit EXACTLY as printed. DO NOT "
        "convert units or do any arithmetic — that happens downstream.\n"
        "- Match each value to its correct test carefully (layouts can be "
        "misaligned; do not grab a neighbouring test's number).\n"
        "- 'hs-CRP' / high-sensitivity CRP is hs_crp, NOT crp. Differential counts "
        "come in both percent (_pct) and absolute (_abs) forms — capture both.\n"
        "- If you cannot confidently identify a test or value, omit it.\n\n"
        f"QUALITATIVE metric_types (result is negative/positive/…, not a number): {qual_list}.\n"
        "- e.g. 'Clostridium difficile Toxin A (-) negative' -> "
        "{metric_type: clostridium_difficile_toxin_a, result: negative}. A pending "
        "result -> 'pending'.\n\n"
        "Set report_date to the sample collection date (YYYY-MM-DD). For non-bloodwork "
        "documents, return metrics: [], qualitative: [], report_date: ''.\n\n"
        f"DOCUMENT TEXT:\n{text}"
    )
    msg = _claude.messages.create(
        model=EXTRACT_MODEL,
        max_tokens=4096,
        output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(b.text for b in msg.content if b.type == "text")
    return json.loads(raw)


def ingest_pdf(user_id: int, pdf_bytes: bytes) -> dict:
    text = extract_text(pdf_bytes)
    if not text.strip():
        return {"status": "unreadable", "metrics_written": 0,
                "message": "I couldn't read any text from that PDF. If it's a scanned "
                           "image, try a text-based export of your lab report."}

    extracted = extract_metrics(text)
    doc_type = extracted.get("document_type", "not_health")
    summary = (extracted.get("document_summary") or "").strip()

    if doc_type != "bloodwork":
        if doc_type == "health_other":
            msg = (f"That looks like a health document ({summary}), but not a lab/bloodwork "
                   "report with test values, so I didn't save anything. Upload a lab report "
                   "(CBC, metabolic panel, vitamin/hormone levels, etc.) to add bloodwork.")
        else:
            msg = (f"That doesn't look like a health document ({summary}), so I didn't save "
                   "anything. This upload is for bloodwork/lab-report PDFs.")
        return {"status": "not_bloodwork", "document_type": doc_type,
                "summary": summary, "metrics_written": 0, "message": msg}

    accepted, rejected = normalize_metrics(extracted.get("metrics", []))
    qual_accepted, qual_rejected = normalize_qualitative(extracted.get("qualitative", []))
    if not accepted and not qual_accepted:
        return {"status": "no_values", "document_type": doc_type, "metrics_written": 0,
                "rejected": rejected + qual_rejected,
                "message": "This looks like a lab report, but I couldn't confidently read "
                           "any of the values I track. It may use an unusual layout or units."}

    date = (extracted.get("report_date") or "").strip() or dt.date.today().isoformat()
    records = [
        {"date": date, "metric_type": m["metric_type"], "value": m["value"], "unit": m["unit"]}
        for m in accepted
    ]
    records += [
        {"date": date, "metric_type": q["metric_type"], "text_value": q["text_value"]}
        for q in qual_accepted
    ]
    written = upsert_metrics(user_id, "bloodwork", records)
    return {
        "status": "ok",
        "source": "bloodwork",
        "report_date": date,
        "metrics_written": written,
        "extracted": accepted,          # canonical values + 'reported' for audit
        "qualitative": qual_accepted,   # categorical results
        "rejected": rejected + qual_rejected,
    }
