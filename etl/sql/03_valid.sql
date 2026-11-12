CREATE TEMP TABLE valid_prices ON COMMIT DROP AS
SELECT e.id AS source_price_id,
       e.created_at AS source_created_at, e.updated_at AS source_updated_at,
       e.subscriber_id, e.organization_id, e.source,
       e.loading_date, e.end_loading_date, e.loading_city_gcid, e.unloading_city_gcid,
       e.distance, e.total_weight, e.total_volume, e.body_type,
       COALESCE(e.weight_category, CASE
           WHEN round(e.total_weight::numeric, 1) >= 30 THEN 'G20T'
           WHEN round(e.total_weight::numeric, 1) >= 10 THEN 'FTL'
           WHEN round(e.total_weight::numeric, 1) >= 5 THEN '10T'
           WHEN round(e.total_weight::numeric, 1) >= 2.5 THEN '5T'
           WHEN round(e.total_weight::numeric, 1) >= 0.5 THEN 'L5T'
           ELSE 'OTH' END) AS weight_category,
       COALESCE(e.volume_category, CASE
           WHEN e.total_volume > 0 AND e.total_volume <= 120 THEN 'FTL'
           ELSE 'OTH' END) AS volume_category,
       e.price::numeric AS price_original, e.currency_code, e.nearest_rate_date,
       CASE WHEN e.currency_code = '643' THEN 1::numeric
            ELSE e.rate_value::numeric END AS exchange_rate,
       round(e.price::numeric * CASE WHEN e.currency_code = '643' THEN 1::numeric
                                    ELSE e.rate_value::numeric END, 2) AS price_rub,
       e.price_type, e.is_fact, e.complex, e.outlier_mark
FROM evaluated_prices e
WHERE cardinality(e.reason_codes) = 0;
