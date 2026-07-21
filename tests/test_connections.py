"""Connected data providers: token storage, and the parked-authorization
lifecycle behind "this is a different account — replace or cancel?"."""

import datetime as dt

from app import connections
from app.db import get_connection
from app.ingestion import delete_metrics, query_metrics, upsert_metrics
from app.workouts import delete_workouts, upsert_workouts

WHOOP = "whoop"


def _later(hours: int = 1) -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=hours)


def test_save_and_read_a_connection(temp_user):
    connections.save_tokens(temp_user, WHOOP, "acc-1", "ref-1", _later(), external_user_id="42")
    row = connections.get(temp_user, WHOOP)
    assert row["access_token"] == "acc-1"
    assert row["external_user_id"] == "42"      # which account the tokens belong to
    assert connections.providers_for(temp_user) == [WHOOP]


def test_refresh_keeps_the_refresh_token_and_account_id(temp_user):
    connections.save_tokens(temp_user, WHOOP, "acc-1", "ref-1", _later(), external_user_id="42")
    # a refresh response typically returns neither of those two
    connections.save_tokens(temp_user, WHOOP, "acc-2", None, _later())
    row = connections.get(temp_user, WHOOP)
    assert row["access_token"] == "acc-2"
    assert row["refresh_token"] == "ref-1"      # not nulled out
    assert row["external_user_id"] == "42"      # still the same account


def test_one_connection_per_provider(temp_user):
    connections.save_tokens(temp_user, WHOOP, "acc-1", "ref-1", _later())
    connections.save_tokens(temp_user, WHOOP, "acc-2", "ref-2", _later())
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM provider_connections WHERE user_id = %s", (temp_user,))
        assert cur.fetchone()[0] == 1           # upserted, not duplicated


def test_providers_are_independent(temp_user):
    connections.save_tokens(temp_user, WHOOP, "w", None, _later())
    connections.save_tokens(temp_user, "oura", "o", None, _later())
    assert connections.providers_for(temp_user) == ["oura", WHOOP]
    connections.delete(temp_user, "oura")
    assert connections.providers_for(temp_user) == [WHOOP]


# --- parked authorizations --------------------------------------------------
def test_pending_is_not_a_live_connection(temp_user):
    connections.save_tokens(temp_user, WHOOP, "old", "ref-old", _later(), external_user_id="42")
    connections.save_pending(temp_user, WHOOP, "new", "ref-new", _later(), "99")

    # the live connection is untouched while the decision is outstanding
    assert connections.get(temp_user, WHOOP)["access_token"] == "old"
    assert connections.get_pending(temp_user, WHOOP)["external_user_id"] == "99"


def test_promote_replaces_the_live_connection(temp_user):
    connections.save_tokens(temp_user, WHOOP, "old", "ref-old", _later(), external_user_id="42")
    connections.save_pending(temp_user, WHOOP, "new", "ref-new", _later(), "99")

    assert connections.promote_pending(temp_user, WHOOP) is True
    live = connections.get(temp_user, WHOOP)
    assert live["access_token"] == "new"
    assert live["external_user_id"] == "99"                  # now the new account
    assert connections.get_pending(temp_user, WHOOP) is None  # consumed


def test_cancel_leaves_the_original_connection_alone(temp_user):
    connections.save_tokens(temp_user, WHOOP, "old", "ref-old", _later(), external_user_id="42")
    connections.save_pending(temp_user, WHOOP, "new", "ref-new", _later(), "99")

    assert connections.drop_pending(temp_user, WHOOP) is True
    assert connections.get_pending(temp_user, WHOOP) is None
    assert connections.get(temp_user, WHOOP)["external_user_id"] == "42"   # unchanged


def test_stale_pending_is_ignored(temp_user):
    connections.save_pending(temp_user, WHOOP, "new", "ref-new", _later(), "99")
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("UPDATE pending_connections SET created_at = now() - INTERVAL '2 hours' "
                    "WHERE user_id = %s", (temp_user,))
        conn.commit()
    # an abandoned decision must not resurface days later
    assert connections.get_pending(temp_user, WHOOP) is None


def test_promote_without_anything_pending_is_a_no_op(temp_user):
    connections.save_tokens(temp_user, WHOOP, "old", "ref-old", _later(), external_user_id="42")
    assert connections.promote_pending(temp_user, WHOOP) is False
    assert connections.get(temp_user, WHOOP)["access_token"] == "old"


# --- the destructive half of "replace" --------------------------------------
def test_replacing_whoop_data_leaves_other_sources_untouched(temp_user):
    """Accepting a new WHOOP account clears that provider's rows only — bloodwork
    and anything from another wearable must survive."""
    upsert_metrics(temp_user, WHOOP, [
        {"date": "2026-07-01", "metric_type": "recovery_score", "value": 70},
        {"date": "2026-07-02", "metric_type": "recovery_score", "value": 75},
    ])
    upsert_metrics(temp_user, "bloodwork", [
        {"date": "2026-06-01", "metric_type": "vitamin_d", "value": 32},
    ])
    upsert_metrics(temp_user, "oura", [
        {"date": "2026-07-01", "metric_type": "recovery_score", "value": 61},
    ])
    upsert_workouts(temp_user, WHOOP, [
        {"external_id": "w1", "sport": "running", "workout_date": "2026-07-01"},
    ])

    assert delete_metrics(temp_user, WHOOP) == 2
    assert delete_workouts(temp_user, WHOOP) == 1

    left = {(r["source"], r["metric_type"]) for r in query_metrics(temp_user, limit=50)}
    assert ("bloodwork", "vitamin_d") in left     # lab results are not WHOOP's to delete
    assert ("oura", "recovery_score") in left     # another wearable is independent
    assert not any(src == WHOOP for src, _ in left)


# --- data freshness ("Synced 3 minutes ago") --------------------------------
def test_a_new_connection_has_never_been_synced(temp_user):
    connections.save_tokens(temp_user, WHOOP, "acc-1", "ref-1", _later())
    assert connections.get(temp_user, WHOOP)["last_synced_at"] is None


def test_mark_synced_records_and_returns_the_time(temp_user):
    connections.save_tokens(temp_user, WHOOP, "acc-1", "ref-1", _later())
    returned = connections.mark_synced(temp_user, WHOOP)
    stored = connections.get(temp_user, WHOOP)["last_synced_at"]
    assert returned is not None and stored == returned


def test_mark_synced_moves_forward_on_a_later_sync(temp_user):
    connections.save_tokens(temp_user, WHOOP, "acc-1", "ref-1", _later())
    first = connections.mark_synced(temp_user, WHOOP)
    second = connections.mark_synced(temp_user, WHOOP)
    assert second >= first


def test_mark_synced_is_per_provider(temp_user):
    # syncing WHOOP must not make another wearable look freshly pulled
    connections.save_tokens(temp_user, WHOOP, "acc-1", "ref-1", _later())
    connections.save_tokens(temp_user, "oura", "acc-2", "ref-2", _later())
    connections.mark_synced(temp_user, WHOOP)
    assert connections.get(temp_user, WHOOP)["last_synced_at"] is not None
    assert connections.get(temp_user, "oura")["last_synced_at"] is None


def test_mark_synced_on_a_missing_connection_is_a_no_op(temp_user):
    assert connections.mark_synced(temp_user, WHOOP) is None
