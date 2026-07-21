"""WHOOP JSON -> canonical metric mapping. Pure parsers, no network."""

import datetime as dt

from app.whoop import parse_recovery, parse_sleep, parse_workout
from app.workouts import offset_to_tz


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


def test_parse_workout_maps_fields_and_converts_energy():
    records = [{
        "id": "abc-123",
        "start": "2026-07-18T17:00:00.000Z",
        "end": "2026-07-18T18:00:00.000Z",
        "sport_name": "running",
        "score": {
            "strain": 12.34,
            "average_heart_rate": 148.6,
            "max_heart_rate": 181.2,
            "kilojoule": 2000.0,      # -> ~478 kcal
            "distance_meter": 5000.0,
        },
    }]
    row = parse_workout(records)[0]

    assert row["external_id"] == "abc-123"
    assert row["sport"] == "running"
    assert row["workout_date"] == "2026-07-18"
    assert row["duration_min"] == 60.0        # one hour
    assert row["strain"] == 12.3              # 1 dp
    assert row["avg_hr"] == 149               # rounded to a whole bpm
    assert row["max_hr"] == 181
    assert row["calories"] == 478             # 2000 kJ * 0.239006
    assert row["distance_m"] == 5000.0


def test_parse_workout_sport_id_fallback_and_missing_optionals():
    # Older payloads carry a numeric sport_id; a lift has no distance/energy.
    records = [{
        "id": 55, "start": "2026-07-19T07:30:00Z", "end": "2026-07-19T08:15:00Z",
        "sport_id": 45,
        "score": {"strain": 9.0, "average_heart_rate": 130, "max_heart_rate": 160},
    }]
    row = parse_workout(records)[0]

    assert row["external_id"] == "55"         # id stringified for storage
    assert row["sport"] == "45"               # numeric id used when no name
    assert row["duration_min"] == 45.0
    assert row["calories"] is None            # no kilojoule -> no calories
    assert row["distance_m"] is None          # no distance -> None (dropped on read)


def test_parse_workout_skips_records_without_id_or_start():
    assert parse_workout([{"start": "2026-07-19T07:00:00Z"}]) == []   # no id
    assert parse_workout([{"id": "x"}]) == []                          # no start


def test_parse_workout_keeps_utc_instant_and_local_date():
    # A run at 02:30 local (UTC+3) is 23:30 UTC the day before. The stored
    # timestamp stays the true UTC instant; the calendar date is the local one
    # (the day the user actually trained), and the offset is carried through.
    records = [{
        "id": "tz-1",
        "start": "2026-07-19T23:30:00.000Z",
        "end": "2026-07-19T23:55:00.000Z",
        "sport_name": "running",
        "timezone_offset": "+03:00",
        "score": {"strain": 5.0},
    }]
    row = parse_workout(records)[0]

    assert row["start_time"] == "2026-07-19T23:30:00.000Z"   # UTC instant untouched
    assert row["tz_offset"] == "+03:00"
    assert row["workout_date"] == "2026-07-20"               # local calendar date
    assert row["duration_min"] == 25.0                       # offset-independent


def test_offset_to_tz_parses_whoop_shapes():
    assert offset_to_tz("+03:00") == dt.timezone(dt.timedelta(hours=3))
    assert offset_to_tz("-05:00") == dt.timezone(dt.timedelta(hours=-5))
    assert offset_to_tz("+0000") == dt.timezone.utc
    assert offset_to_tz("Z") == dt.timezone.utc
    assert offset_to_tz(None) is None
    assert offset_to_tz("") is None


def test_parsers_handle_empty_input():
    assert parse_recovery([]) == []
    assert parse_sleep([]) == []
    assert parse_workout([]) == []
