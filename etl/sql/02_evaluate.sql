CREATE TEMP TABLE evaluated_prices ON COMMIT DROP AS
SELECT p.*,
       rates.rate_count,
       rates.rate_value,
       to_jsonb(p) AS source_data,
       array_remove(ARRAY[
           CASE WHEN p.price <= 0 THEN 'NONPOSITIVE_PRICE' END,
           CASE WHEN p.currency_code = '643' AND p.price <= 1000 THEN 'LOW_RUB_PRICE' END,
           CASE WHEN NOT (p.price < 'Infinity'::float8 AND p.price > '-Infinity'::float8)
                THEN 'NONFINITE_PRICE' END,
           CASE WHEN p.distance <= 0 THEN 'NONPOSITIVE_DISTANCE' END,
           CASE WHEN NOT (p.distance < 'Infinity'::float8 AND p.distance > '-Infinity'::float8)
                THEN 'NONFINITE_DISTANCE' END,
           CASE WHEN p.total_weight < 0 THEN 'NEGATIVE_WEIGHT' END,
           CASE WHEN NOT (p.total_weight < 'Infinity'::float8 AND p.total_weight > '-Infinity'::float8)
                THEN 'NONFINITE_WEIGHT' END,
           CASE WHEN p.total_volume < 0 THEN 'NEGATIVE_VOLUME' END,
           CASE WHEN NOT (p.total_volume < 'Infinity'::float8 AND p.total_volume > '-Infinity'::float8)
                THEN 'NONFINITE_VOLUME' END,
           CASE WHEN NOT EXISTS (SELECT 1 FROM public.cities c WHERE c.gc_id = p.loading_city_gcid)
                THEN 'UNKNOWN_LOADING_CITY' END,
           CASE WHEN NOT EXISTS (SELECT 1 FROM public.cities c WHERE c.gc_id = p.unloading_city_gcid)
                THEN 'UNKNOWN_UNLOADING_CITY' END,
           CASE WHEN p.currency_code NOT IN ('643', '840', '978') THEN 'UNSUPPORTED_CURRENCY' END,
           CASE WHEN p.currency_code IN ('840', '978') AND rates.rate_count = 0
                THEN 'MISSING_EXCHANGE_RATE' END,
           CASE WHEN p.currency_code IN ('840', '978') AND rates.rate_count > 1
                THEN 'DUPLICATE_EXCHANGE_RATE' END,
           CASE WHEN p.currency_code IN ('840', '978') AND rates.rate_count = 1
                     AND NOT (rates.rate_value > 0 AND rates.rate_value < 'Infinity'::float8)
                THEN 'INVALID_EXCHANGE_RATE' END
       ]::text[], NULL) AS reason_codes
FROM candidate_ids ids
JOIN public.prices p ON p.id = ids.source_price_id
LEFT JOIN LATERAL (
    SELECT count(*) AS rate_count, min(er.value) AS rate_value
    FROM public.exchange_rates er
    WHERE p.currency_code IN ('840', '978')
      AND er.num_code = p.currency_code
      AND er.setting_date = p.nearest_rate_date
) rates ON true;
