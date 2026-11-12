-- Initial DDL. Existing structures are changed through separate migrations.
BEGIN;

CREATE SCHEMA IF NOT EXISTS mart;

-- One complete scoring snapshot per model version and as_of; each price occurs once.
-- No source/Core FK: historical predictions must survive source edits and deletions.
CREATE TABLE IF NOT EXISTS mart.ml_price_predictions (
    source_price_id       bigint NOT NULL,
    model_version         text NOT NULL,
    as_of_date            date NOT NULL,
    dataset_split         text NOT NULL,
    loading_date          date NOT NULL,
    loading_city_gcid     integer NOT NULL,
    unloading_city_gcid   integer NOT NULL,
    loading_city_name     text NOT NULL,
    unloading_city_name   text NOT NULL,
    body_group            text NOT NULL REFERENCES mart.body_groups (body_group),
    weight_category       varchar(4) NOT NULL,
    volume_category       varchar(4) NOT NULL,

    actual_price_rub      double precision,
    predicted_price_rub   double precision NOT NULL,
    abs_error_rub         double precision GENERATED ALWAYS AS (
        abs(predicted_price_rub - actual_price_rub)
    ) STORED,
    -- Generated expressions cannot reference another generated column in PostgreSQL 17.
    ape_pct               double precision GENERATED ALWAYS AS (
        abs(predicted_price_rub - actual_price_rub) / actual_price_rub * 100.0
    ) STORED,

    model_name            text NOT NULL,
    model_run_id          uuid NOT NULL,
    dataset_run_id        uuid NOT NULL,
    scored_at             timestamptz NOT NULL,
    artifact_sha256       text NOT NULL,
    loaded_at             timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT ml_price_predictions_pkey PRIMARY KEY
        (model_version, as_of_date, source_price_id),
    CONSTRAINT ml_predictions_identity_valid CHECK (
        source_price_id > 0 AND model_version ~ '^[0-9a-f]{16}$'
        AND artifact_sha256 ~ '^[0-9a-f]{64}$'
        AND btrim(model_name) <> ''
    ),
    CONSTRAINT ml_predictions_labels_valid CHECK (
        btrim(loading_city_name) <> '' AND btrim(unloading_city_name) <> ''
        AND btrim(weight_category) <> '' AND btrim(volume_category) <> ''
    ),
    CONSTRAINT ml_predictions_dates_valid CHECK (
        isfinite(as_of_date) AND isfinite(loading_date)
        AND isfinite(scored_at) AND isfinite(loaded_at)
    ),
    CONSTRAINT ml_predictions_partition_valid CHECK (
        (dataset_split = 'test'
         AND loading_date BETWEEN as_of_date - 21 AND as_of_date
         AND actual_price_rub IS NOT NULL)
        OR (dataset_split = 'future' AND loading_date > as_of_date
            AND actual_price_rub IS NULL)
    ),
    CONSTRAINT ml_predictions_prices_valid CHECK (
        predicted_price_rub > 0 AND predicted_price_rub < 'Infinity'::double precision
        AND (actual_price_rub IS NULL
             OR (actual_price_rub > 0 AND actual_price_rub < 'Infinity'::double precision))
    )
);

CREATE INDEX IF NOT EXISTS ml_predictions_route_date_idx
    ON mart.ml_price_predictions (loading_city_gcid, unloading_city_gcid, loading_date);

COMMIT;
