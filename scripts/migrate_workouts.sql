-- Workouts: logged training activities (WHOOP now; other sources later).
-- Unlike health_metrics (one scalar value per day), a workout is a per-activity
-- event — several can share a day, each with a sport, duration, strain, heart
-- rate, calories and distance — so it gets its own table.
CREATE TABLE IF NOT EXISTS workouts (
    id            BIGSERIAL PRIMARY KEY,
    user_id       BIGINT REFERENCES users(id) ON DELETE CASCADE,
    source        TEXT NOT NULL DEFAULT 'whoop',   -- whoop | garmin | manual | ...
    external_id   TEXT,                             -- the source's workout id (idempotency)
    sport         TEXT,                             -- e.g. running, weightlifting
    workout_date  DATE,                             -- calendar date of the session start
    start_time    TIMESTAMPTZ,
    end_time      TIMESTAMPTZ,
    duration_min  DOUBLE PRECISION,
    strain        DOUBLE PRECISION,                 -- WHOOP strain (0–21)
    avg_hr        INT,
    max_hr        INT,
    calories      DOUBLE PRECISION,                 -- kcal (converted from kilojoules)
    distance_m    DOUBLE PRECISION,                 -- cardio only; NULL for e.g. lifting
    tz_offset     TEXT,                             -- local UTC offset at the session, e.g. "+03:00"
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Added after the initial table shipped; safe to re-run.
ALTER TABLE workouts ADD COLUMN IF NOT EXISTS tz_offset TEXT;

-- Idempotent ingestion: one row per (user, source, external workout id).
CREATE UNIQUE INDEX IF NOT EXISTS uq_workouts
    ON workouts (user_id, source, external_id);
