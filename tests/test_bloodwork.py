"""Deterministic unit conversion + validation — the logic we deliberately took
away from the LLM. Pure functions, no network."""

import pytest

from app.bloodwork import (
    BLOODWORK_TYPES,
    QUALITATIVE_TYPES,
    TARGET_UNITS,
    _FACTORS,
    _RANGES,
    convert_to_canonical,
    in_range,
    normalize_metrics,
    normalize_qualitative,
)
from app.ingestion import CANONICAL_UNITS


def test_canonical_unit_is_identity():
    assert convert_to_canonical("vitamin_d", 30, "ng/mL") == 30
    assert convert_to_canonical("glucose", 90, "mg/dL") == 90


@pytest.mark.parametrize("metric, value, unit, expected", [
    ("vitamin_d", 75, "nmol/L", 75 * 0.4006),
    ("testosterone", 20, "nmol/L", 20 * 28.84),
    ("testosterone", 5, "ng/mL", 500.0),         # ng/mL x100 = ng/dL
    ("glucose", 5, "mmol/L", 5 * 18.0156),
    ("crp", 0.5, "mg/dL", 5.0),                   # mg/dL x10 = mg/L
    ("hba1c", 53, "mmol/mol", 53 * 0.0915 + 2.15),  # IFCC -> NGSP %
])
def test_conversions(metric, value, unit, expected):
    assert convert_to_canonical(metric, value, unit) == pytest.approx(expected)


def test_micro_sign_is_folded():
    # 'µIU/mL' and 'uIU/mL' must both match tsh's canonical mIU/L (1:1)
    assert convert_to_canonical("tsh", 2.0, "µIU/mL") == 2.0
    assert convert_to_canonical("tsh", 2.0, "uIU/mL") == 2.0


def test_unknown_unit_raises_rather_than_guessing():
    with pytest.raises(ValueError):
        convert_to_canonical("vitamin_d", 30, "pmol/L")
    with pytest.raises(ValueError):
        convert_to_canonical("hba1c", 7, "mg/dL")


def test_in_range_bounds():
    assert in_range("vitamin_d", 30) is True
    assert in_range("vitamin_d", 500) is False   # impossibly high
    assert in_range("testosterone", -5) is False


def test_normalize_metrics_accepts_and_rejects():
    extracted = [
        {"metric_type": "vitamin_d", "value": 75, "unit": "nmol/L"},   # ok -> ~30 ng/mL
        {"metric_type": "glucose", "value": 5000, "unit": "mg/dL"},    # out of range
        {"metric_type": "ldl", "value": 100, "unit": "pmol/L"},        # unknown unit
        {"metric_type": "not_a_test", "value": 1, "unit": "x"},        # unknown metric
    ]
    accepted, rejected = normalize_metrics(extracted)

    assert len(accepted) == 1
    vd = accepted[0]
    assert vd["metric_type"] == "vitamin_d"
    assert vd["unit"] == "ng/mL"
    assert vd["value"] == pytest.approx(75 * 0.4006, abs=1e-3)
    assert vd["reported"] == "75 nmol/L"        # audit trail of what was printed

    assert len(rejected) == 3
    assert all("reason" in r for r in rejected)


def test_bloodwork_types_are_fully_specified():
    # Every bloodwork metric must have a canonical unit, a conversion table
    # (or the hba1c special case), and a plausibility range — no gaps.
    for t in BLOODWORK_TYPES:
        assert t in CANONICAL_UNITS
        assert t in TARGET_UNITS
        assert t in _RANGES
        assert t == "hba1c" or t in _FACTORS


@pytest.mark.parametrize("metric, value, unit, expected", [
    ("hemoglobin", 15.2, "g/dL", 152.0),       # g/dL x10 -> g/L
    ("hematocrit", 44.1, "%", 0.441),          # % x0.01 -> L/L
    ("creatinine", 1.0, "mg/dL", 88.42),       # mg/dL -> µmol/L
    ("wbc", 8.9, "G/l", 8.9),                  # G/l == 10^9/L (identity)
    ("hs_crp", 8.56, "mg/L", 8.56),            # distinct from crp, same units
    ("vitamin_b12", 400, "pmol/L", 400 * 1.355),
])
def test_expanded_conversions(metric, value, unit, expected):
    assert convert_to_canonical(metric, value, unit) == pytest.approx(expected)


def test_hs_crp_is_separate_from_crp():
    assert "hs_crp" in BLOODWORK_TYPES and "crp" in BLOODWORK_TYPES
    assert TARGET_UNITS["hs_crp"] == "mg/L"


def test_normalize_qualitative_accepts_and_rejects():
    accepted, rejected = normalize_qualitative([
        {"metric_type": "clostridium_difficile_toxin_a", "result": "(-) negative"},
        {"metric_type": "quantiferon_tb", "result": "positive"},
        {"metric_type": "not_a_microbe", "result": "negative"},   # unknown metric
        {"metric_type": "clostridium_difficile_gdh", "result": "maybe"},  # bad result
    ])
    by_type = {a["metric_type"]: a for a in accepted}
    assert by_type["clostridium_difficile_toxin_a"]["text_value"] == "negative"
    assert by_type["quantiferon_tb"]["text_value"] == "positive"
    assert len(accepted) == 2
    assert len(rejected) == 2
    assert all(t in QUALITATIVE_TYPES for t in by_type)
