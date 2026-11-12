-- Creates the initial mart schema; structural changes require separate migrations.
BEGIN;

CREATE SCHEMA IF NOT EXISTS mart;

-- Stable analytical codes and display labels, not a replacement for public.body_types.
CREATE TABLE IF NOT EXISTS mart.body_groups (
    body_group text PRIMARY KEY,
    name_ru    text NOT NULL
);

INSERT INTO mart.body_groups (body_group, name_ru) VALUES
    ('TENT', 'Тент'),
    ('REFRIGERATOR', 'Рефрижератор'),
    ('INSULATED', 'Изотерм'),
    ('FLATBED', 'Борт'),
    ('OTHER', 'Прочие'),
    ('UNSPECIFIED', 'Не указан')
ON CONFLICT (body_group) DO NOTHING;

-- One current snapshot: month x direction x body group x weight category.
CREATE TABLE IF NOT EXISTS mart.route_card (
    month_start            date NOT NULL,
    loading_city_gcid      integer NOT NULL,
    unloading_city_gcid    integer NOT NULL,
    body_group             text NOT NULL REFERENCES mart.body_groups (body_group),
    weight_category        varchar(4) NOT NULL,
    loading_city_name      text NOT NULL,
    unloading_city_name    text NOT NULL,

    p25_trip               numeric NOT NULL,
    median_trip            numeric NOT NULL,
    p75_trip               numeric NOT NULL,
    avg_trip               numeric NOT NULL,
    p25_km                 numeric NOT NULL,
    median_km              numeric NOT NULL,
    p75_km                 numeric NOT NULL,

    trips_count            bigint NOT NULL,
    carriers_count         bigint NOT NULL,
    cv_rate                numeric,
    mom_pct                numeric,
    yoy_pct                numeric,

    latest_price_date      date NOT NULL,
    as_of_date             date NOT NULL,
    freshness_days         integer GENERATED ALWAYS AS (as_of_date - latest_price_date) STORED,
    reliability_level      text NOT NULL,
    refreshed_at           timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT route_card_pkey PRIMARY KEY
        (month_start, loading_city_gcid, unloading_city_gcid, body_group, weight_category),
    CONSTRAINT route_card_month_start_valid
        CHECK (isfinite(month_start) AND EXTRACT(DAY FROM month_start) = 1),
    CONSTRAINT route_card_dates_valid
        CHECK (isfinite(latest_price_date) AND isfinite(as_of_date)
               AND latest_price_date >= month_start
               AND latest_price_date < month_start + INTERVAL '1 month'
               AND latest_price_date <= as_of_date),
    CONSTRAINT route_card_trip_metrics_valid
        CHECK (p25_trip >= 0 AND p25_trip <= median_trip AND median_trip <= p75_trip
               AND p75_trip < 'Infinity'::numeric
               AND avg_trip >= 0 AND avg_trip < 'Infinity'::numeric),
    CONSTRAINT route_card_km_metrics_valid
        CHECK (p25_km >= 0 AND p25_km <= median_km AND median_km <= p75_km
               AND p75_km < 'Infinity'::numeric),
    CONSTRAINT route_card_counts_valid
        CHECK (trips_count >= 1 AND carriers_count >= 0 AND carriers_count <= trips_count),
    CONSTRAINT route_card_cv_valid
        CHECK ((trips_count < 2 OR avg_trip = 0) AND cv_rate IS NULL
               OR (trips_count >= 2 AND avg_trip > 0 AND cv_rate IS NOT NULL
                   AND cv_rate >= 0 AND cv_rate < 'Infinity'::numeric)),
    CONSTRAINT route_card_growth_valid
        CHECK ((mom_pct IS NULL OR (mom_pct > '-Infinity'::numeric AND mom_pct < 'Infinity'::numeric))
               AND (yoy_pct IS NULL OR (yoy_pct > '-Infinity'::numeric AND yoy_pct < 'Infinity'::numeric))),
    CONSTRAINT route_card_reliability_valid
        CHECK (reliability_level = CASE
            WHEN trips_count >= 30 AND carriers_count >= 5 AND freshness_days <= 30 THEN 'HIGH'
            WHEN trips_count >= 10 THEN 'MEDIUM'
            ELSE 'LOW_DATA'
        END)
);

COMMIT;
