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


def test_parse_sleep_keeps_naps_separate_from_overnight():
    # A nap and the night's sleep can share a calendar date. The nap must NOT
    # overwrite sleep_hours (the real bug: a 22-min nap clobbered a 7h night).
    records = [
        {   # daytime nap, same date as the overnight below
            "start": "2026-07-17T13:00:00.000Z", "nap": True,
            "score": {"stage_summary": {"total_light_sleep_time_milli": 3_600_000}},  # 1.0 h
        },
        {   # the real overnight sleep
            "start": "2026-07-17T22:00:00.000Z", "nap": False,
            "score": {
                "stage_summary": {"total_light_sleep_time_milli": 27_000_000},  # 7.5 h
                "sleep_efficiency_percentage": 90.0,
            },
        },
    ]
    rows = {r["metric_type"]: r for r in parse_sleep(records)}

    assert rows["sleep_hours"]["value"] == 7.5   # overnight preserved, not the nap
    assert rows["nap_hours"]["value"] == 1.0     # nap captured separately
    assert rows["sleep_efficiency"]["value"] == 90.0


def test_parse_sleep_sums_multiple_naps_per_day():
    records = [
        {"start": "2026-07-18T10:00:00.000Z", "nap": True,
         "score": {"stage_summary": {"total_rem_sleep_time_milli": 1_800_000}}},   # 0.5 h
        {"start": "2026-07-18T15:00:00.000Z", "nap": True,
         "score": {"stage_summary": {"total_rem_sleep_time_milli": 2_700_000}}},   # 0.75 h
    ]
    rows = {r["metric_type"]: r for r in parse_sleep(records)}
    assert rows["nap_hours"]["value"] == 1.25    # summed
    assert "sleep_hours" not in rows             # a nap-only day has no overnight


def test_parsers_handle_empty_input():
    assert parse_recovery([]) == []
    assert parse_sleep([]) == []
