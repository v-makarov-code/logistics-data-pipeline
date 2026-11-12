-- Initial schema creation only; subsequent structural changes require migrations.
BEGIN;

CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS dq;

-- One current, validated record per source prices.id.
CREATE TABLE IF NOT EXISTS core.prices (
    source_price_id      bigint PRIMARY KEY,
    source_created_at    timestamp(0),
    source_updated_at    timestamp(0),
    subscriber_id        integer,
    organization_id      integer,
    source               varchar(255),

    loading_date         date NOT NULL,
    end_loading_date     date,
    loading_city_gcid    integer NOT NULL,
    unloading_city_gcid  integer NOT NULL,
    distance             double precision NOT NULL,
    total_weight         double precision NOT NULL,
    total_volume         double precision NOT NULL,
    body_type            integer,
    weight_category      varchar(4) NOT NULL,
    volume_category      varchar(4) NOT NULL,

    price_original       numeric NOT NULL,
    currency_code        varchar(3) NOT NULL,
    nearest_rate_date    date,
    exchange_rate        numeric NOT NULL,
    price_rub            numeric NOT NULL,

    price_type           smallint NOT NULL,
    is_fact              boolean NOT NULL,
    complex              boolean NOT NULL,
    outlier_mark         boolean NOT NULL,

    CONSTRAINT core_prices_distance_valid
        CHECK (distance > 0 AND distance < 'Infinity'::double precision),
    CONSTRAINT core_prices_weight_valid
        CHECK (total_weight >= 0 AND total_weight < 'Infinity'::double precision),
    CONSTRAINT core_prices_volume_valid
        CHECK (total_volume >= 0 AND total_volume < 'Infinity'::double precision),
    CONSTRAINT core_prices_currency_supported
        CHECK (currency_code IN ('643', '840', '978')),
    CONSTRAINT core_prices_original_valid
        CHECK (price_original > 0 AND price_original < 'Infinity'::numeric
               AND (currency_code <> '643' OR price_original > 1000)),
    CONSTRAINT core_prices_rate_valid
        CHECK (exchange_rate > 0 AND exchange_rate < 'Infinity'::numeric),
    CONSTRAINT core_prices_rub_valid
        CHECK (price_rub >= 0 AND price_rub < 'Infinity'::numeric),
    CONSTRAINT core_prices_currency_rate_consistent
        CHECK ((currency_code = '643' AND exchange_rate = 1)
               OR (currency_code IN ('840', '978') AND nearest_rate_date IS NOT NULL))
);

CREATE INDEX IF NOT EXISTS core_prices_loading_date_idx
    ON core.prices (loading_date);

-- One current rejected source record, with one or more reason codes.
CREATE TABLE IF NOT EXISTS dq.quarantine_prices (
    source_price_id      bigint PRIMARY KEY,
    loading_date         date NOT NULL,
    reason_codes         text[] NOT NULL,
    source_data          jsonb NOT NULL,
    detected_at          timestamptz NOT NULL DEFAULT now(),
    last_run_id          uuid,

    CONSTRAINT quarantine_prices_reasons_valid
        CHECK (cardinality(reason_codes) > 0 AND array_ndims(reason_codes) = 1
               AND array_position(reason_codes, NULL) IS NULL),
    CONSTRAINT quarantine_prices_source_data_object
        CHECK (jsonb_typeof(source_data) = 'object')
);

CREATE INDEX IF NOT EXISTS quarantine_prices_loading_date_idx
    ON dq.quarantine_prices (loading_date);

COMMIT;
