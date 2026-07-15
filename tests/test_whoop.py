"""WHOOP JSON -> canonical metric mapping. Pure parsers, no network."""

from app.whoop import parse_recovery, parse_sleep


def test_parse_recovery_maps_all_score_fields():
    records = [{
        "created_at": "2026-07-10T06:00:00.000Z",
        "score": {
            "recovery_score": 66.4,
            "hrv_rmssd_milli": 48.27,
            "resting_heart_rate": 52.6,
            "spo2_percentage": 97.31,
            "skin_temp_celsius": 33.44,
        },
    }]
    rows = {r["metric_type"]: r for r in parse_recovery(records)}

    assert rows["recovery_score"]["value"] == 66      # rounded to 0 dp
    assert rows["recovery_score"]["date"] == "2026-07-10"
    assert rows["hrv_rmssd"]["value"] == 48.3         # 1 dp
    assert rows["resting_hr"]["value"] == 53
    assert rows["spo2"]["value"] == 97.3
    assert rows["skin_temp"]["value"] == 33.4


def test_parse_recovery_skips_missing_date_and_nulls():
    records = [
        {"score": {"recovery_score": 70}},                       # no created_at
        {"created_at": "2026-07-11T06:00:00Z", "score": {"recovery_score": None}},
    ]
    assert parse_recovery(records) == []


def test_parse_sleep_computes_hours_and_scores():
    records = [{
        "start": "2026-07-10T23:00:00.000Z",
        "score": {
            "stage_summary": {
                "total_light_sleep_time_milli": 10_800_000,   # 3.0 h
                "total_slow_wave_sleep_time_milli": 5_400_000, # 1.5 h
                "total_rem_sleep_time_milli": 5_400_000,       # 1.5 h
            },
            "sleep_efficiency_percentage": 91.37,
            "respiratory_rate": 14.62,
        },
    }]
    rows = {r["metric_type"]: r for r in parse_sleep(records)}

    assert rows["sleep_hours"]["value"] == 6.0        # 21.6M ms / 3.6M
    assert rows["sleep_hours"]["date"] == "2026-07-10"
    assert rows["sleep_efficiency"]["value"] == 91.4
    assert rows["respiratory_rate"]["value"] == 14.6


def test_parsers_handle_empty_input():
    assert parse_recovery([]) == []
    assert parse_sleep([]) == []
