-- Run with psql -v ON_ERROR_STOP=1 [-v as_of=YYYY-MM-DD].
-- Full replacement of one current snapshot. Dates and transformations use a single snapshot.
\if :{?as_of}
\else
\set as_of 2026-10-26
\endif

BEGIN ISOLATION LEVEL REPEATABLE READ;

-- Same lock as the Core ETL: fail immediately rather than reading while it changes Core.
DO $lock$
BEGIN
    IF NOT pg_try_advisory_xact_lock(2026111101) THEN
        RAISE EXCEPTION 'Core ETL or route card refresh is already running';
    END IF;
END
$lock$;

CREATE TEMP TABLE route_parameters ON COMMIT DROP AS
SELECT :'as_of'::date AS as_of_date;

-- BEGIN SHARED MART REFRESH
-- Stop rather than silently lose otherwise eligible Core prices with missing references.
DO $validate$
BEGIN
    IF EXISTS (
        SELECT 1 FROM core.prices p
        WHERE p.is_fact AND p.price_type IN (1, 2, 3, 4, 5, 9)
          AND NOT p.complex AND NOT p.outlier_mark
          AND (NOT EXISTS (SELECT 1 FROM public.cities c WHERE c.gc_id = p.loading_city_gcid)
               OR NOT EXISTS (SELECT 1 FROM public.cities c WHERE c.gc_id = p.unloading_city_gcid)
               OR NOT EXISTS (SELECT 1 FROM public.price_dates d WHERE d.price_id = p.source_price_id))
    ) THEN
        RAISE EXCEPTION 'Eligible Core prices have missing cities or price_dates; refresh stopped';
    END IF;
END
$validate$;

CREATE TEMP TABLE route_card_result ON COMMIT DROP AS
WITH
-- One canonical source city per gc_id, chosen by minimum id.
cities AS (
    SELECT DISTINCT ON (gc_id) gc_id, name
    FROM public.cities
    ORDER BY gc_id, id
),
-- One row per price and calendar month; retain latest date for freshness.
price_months AS (
    SELECT d.price_id, date_trunc('month', d.price_date)::date AS month_start,
           max(d.price_date) AS latest_price_date
    FROM public.price_dates d CROSS JOIN route_parameters cfg
    WHERE d.price_date <= cfg.as_of_date
    GROUP BY d.price_id, date_trunc('month', d.price_date)::date
),
-- Use the association table when present; only absent associations use the fallback.
source_body_types AS (
    SELECT p.source_price_id, pbt.body_type_id
    FROM core.prices p
    JOIN public.price_body_types pbt ON pbt.price_id = p.source_price_id
    UNION ALL
    SELECT p.source_price_id, p.body_type
    FROM core.prices p
    WHERE NOT EXISTS (SELECT 1 FROM public.price_body_types pbt WHERE pbt.price_id = p.source_price_id)
),
-- Multiple source types in the same standard group represent one observation.
price_body_groups AS (
    SELECT DISTINCT s.source_price_id,
           CASE
               WHEN s.body_type_id IS NULL THEN 'UNSPECIFIED'
               WHEN b.std_id = 56 THEN 'TENT'
               WHEN b.std_id = 4  THEN 'REFRIGERATOR'
               WHEN b.std_id = 2  THEN 'INSULATED'
               WHEN b.std_id = 15 THEN 'FLATBED'
               ELSE 'OTHER'
           END AS body_group
    FROM source_body_types s
    LEFT JOIN public.body_types b ON b.id = s.body_type_id
),
observations AS (
    SELECT p.source_price_id, pm.month_start, pm.latest_price_date,
           p.loading_city_gcid, p.unloading_city_gcid,
           departure.name AS loading_city_name, destination.name AS unloading_city_name,
           bg.body_group, p.weight_category, p.price_rub, p.distance,
           p.price_type, p.organization_id,
           p.price_rub::double precision / NULLIF(p.distance, 0) AS price_per_km
    FROM core.prices p
    JOIN price_months pm ON pm.price_id = p.source_price_id
    JOIN price_body_groups bg ON bg.source_price_id = p.source_price_id
    JOIN cities departure ON departure.gc_id = p.loading_city_gcid
    JOIN cities destination ON destination.gc_id = p.unloading_city_gcid
    WHERE p.is_fact AND p.price_type IN (1, 2, 3, 4, 5, 9)
      AND NOT p.complex AND NOT p.outlier_mark
      AND p.price_original > 0 AND p.distance > 0 AND p.price_rub IS NOT NULL
),
monthly_metrics AS (
    SELECT month_start, loading_city_gcid, unloading_city_gcid,
           loading_city_name, unloading_city_name, body_group, weight_category,
           percentile_cont(0.25) WITHIN GROUP (ORDER BY price_rub::double precision)::numeric AS p25_trip,
           percentile_cont(0.50) WITHIN GROUP (ORDER BY price_rub::double precision)::numeric AS median_trip,
           percentile_cont(0.75) WITHIN GROUP (ORDER BY price_rub::double precision)::numeric AS p75_trip,
           avg(price_rub) AS avg_trip,
           percentile_cont(0.25) WITHIN GROUP (ORDER BY price_per_km)::numeric AS p25_km,
           percentile_cont(0.50) WITHIN GROUP (ORDER BY price_per_km)::numeric AS median_km,
           percentile_cont(0.75) WITHIN GROUP (ORDER BY price_per_km)::numeric AS p75_km,
           count(DISTINCT source_price_id) AS trips_count,
           count(DISTINCT organization_id) FILTER (WHERE price_type IN (3, 9)) AS carriers_count,
           CASE WHEN count(*) >= 2 AND avg(price_rub) <> 0
                THEN stddev_samp(price_rub) / avg(price_rub) ELSE NULL END AS cv_rate,
           max(latest_price_date) AS latest_price_date
    FROM observations
    GROUP BY month_start, loading_city_gcid, unloading_city_gcid,
             loading_city_name, unloading_city_name, body_group, weight_category
)
SELECT current_month.month_start, current_month.loading_city_gcid,
       current_month.unloading_city_gcid, current_month.body_group, current_month.weight_category,
       current_month.loading_city_name, current_month.unloading_city_name,
       current_month.p25_trip, current_month.median_trip, current_month.p75_trip, current_month.avg_trip,
       current_month.p25_km, current_month.median_km, current_month.p75_km,
       current_month.trips_count, current_month.carriers_count, current_month.cv_rate,
       (current_month.median_trip - prev.median_trip) / NULLIF(prev.median_trip, 0) * 100 AS mom_pct,
       (current_month.median_trip - last_year.median_trip) / NULLIF(last_year.median_trip, 0) * 100 AS yoy_pct,
       current_month.latest_price_date, cfg.as_of_date,
       CASE WHEN current_month.trips_count >= 30 AND current_month.carriers_count >= 5
                  AND cfg.as_of_date - current_month.latest_price_date <= 30 THEN 'HIGH'
            WHEN current_month.trips_count >= 10 THEN 'MEDIUM'
            ELSE 'LOW_DATA' END AS reliability_level
FROM monthly_metrics current_month
CROSS JOIN route_parameters cfg
LEFT JOIN monthly_metrics prev
    ON prev.month_start = (current_month.month_start - INTERVAL '1 month')::date
   AND prev.loading_city_gcid = current_month.loading_city_gcid
   AND prev.unloading_city_gcid = current_month.unloading_city_gcid
   AND prev.body_group = current_month.body_group AND prev.weight_category = current_month.weight_category
LEFT JOIN monthly_metrics last_year
    ON last_year.month_start = (current_month.month_start - INTERVAL '1 year')::date
   AND last_year.loading_city_gcid = current_month.loading_city_gcid
   AND last_year.unloading_city_gcid = current_month.unloading_city_gcid
   AND last_year.body_group = current_month.body_group AND last_year.weight_category = current_month.weight_category;

-- No partial refresh is visible: both operations commit together.
DELETE FROM mart.route_card;
INSERT INTO mart.route_card (
    month_start, loading_city_gcid, unloading_city_gcid, body_group, weight_category,
    loading_city_name, unloading_city_name, p25_trip, median_trip, p75_trip, avg_trip,
    p25_km, median_km, p75_km, trips_count, carriers_count, cv_rate, mom_pct, yoy_pct,
    latest_price_date, as_of_date, reliability_level
)
SELECT month_start, loading_city_gcid, unloading_city_gcid, body_group, weight_category,
       loading_city_name, unloading_city_name, p25_trip, median_trip, p75_trip, avg_trip,
       p25_km, median_km, p75_km, trips_count, carriers_count, cv_rate, mom_pct, yoy_pct,
       latest_price_date, as_of_date, reliability_level
FROM route_card_result;

-- END SHARED MART REFRESH

SELECT count(*) AS route_card_rows, min(month_start) AS first_month,
       max(month_start) AS last_month, max(as_of_date) AS as_of_date
FROM mart.route_card;
COMMIT;
