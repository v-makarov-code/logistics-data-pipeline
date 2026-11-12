-- psycopg parameters: train_start, as_of_date. One row per source_price_id.
-- Do not join price_dates, price_group_stats or the route-card Mart here.
WITH eligible AS (
    SELECT p.*
    FROM core.prices p
    WHERE p.loading_date >= %(train_start)s::date
      AND p.is_fact AND p.price_type IN (1, 2, 3, 4, 5, 9)
      AND NOT p.complex AND p.distance > 0
      -- Future selection must not depend on its (synthetic) target or price anomaly flag.
      AND (p.loading_date > %(as_of_date)s::date
           OR (NOT p.outlier_mark AND p.price_rub > 0))
),
cities AS (
    SELECT DISTINCT ON (gc_id) id, gc_id, name, full_name_local, latitude, longitude
    FROM public.cities
    ORDER BY gc_id, id
),
body_lookup AS (
    SELECT id,
           CASE std_id WHEN 56 THEN 'TENT' WHEN 4 THEN 'REFRIGERATOR'
                       WHEN 2 THEN 'INSULATED' WHEN 15 THEN 'FLATBED'
                       ELSE 'OTHER' END AS body_group
    FROM public.body_types
),
compatible_types AS (
    SELECT p.source_price_id, b.body_type_id
    FROM eligible p JOIN public.price_body_types b ON b.price_id = p.source_price_id
    UNION ALL
    SELECT p.source_price_id, p.body_type
    FROM eligible p
    WHERE NOT EXISTS (
        SELECT 1 FROM public.price_body_types b WHERE b.price_id = p.source_price_id
    )
),
compatible_groups AS (
    SELECT t.source_price_id,
           array_agg(DISTINCT
               CASE WHEN t.body_type_id IS NULL THEN 'UNSPECIFIED'
                    ELSE coalesce(b.body_group, 'OTHER') END
               ORDER BY CASE WHEN t.body_type_id IS NULL THEN 'UNSPECIFIED'
                             ELSE coalesce(b.body_group, 'OTHER') END
           ) AS allowed_body_groups
    FROM compatible_types t LEFT JOIN body_lookup b ON b.id = t.body_type_id
    GROUP BY t.source_price_id
)
SELECT p.source_price_id, p.loading_date,
       p.loading_city_gcid AS origin_city_gcid,
       p.unloading_city_gcid AS destination_city_gcid,
       origin.id AS origin_city_row_id, destination.id AS destination_city_row_id,
       origin.name AS origin_city_name, destination.name AS destination_city_name,
       origin.full_name_local AS origin_full_name,
       destination.full_name_local AS destination_full_name,
       origin.latitude AS origin_latitude, origin.longitude AS origin_longitude,
       destination.latitude AS destination_latitude,
       destination.longitude AS destination_longitude,
       p.distance, p.total_weight, p.total_volume, p.weight_category, p.volume_category,
       CASE WHEN p.body_type IS NULL THEN 'UNSPECIFIED'
            ELSE coalesce(primary_body.body_group, 'OTHER') END AS primary_body_group,
       compatible.allowed_body_groups,
       -- Hide future prices at the SQL boundary, before sending data to Python.
       CASE WHEN p.loading_date <= %(as_of_date)s::date
            THEN p.price_rub::double precision ELSE NULL::double precision END AS target_price_rub
FROM eligible p
LEFT JOIN cities origin ON origin.gc_id = p.loading_city_gcid
LEFT JOIN cities destination ON destination.gc_id = p.unloading_city_gcid
LEFT JOIN body_lookup primary_body ON primary_body.id = p.body_type
JOIN compatible_groups compatible ON compatible.source_price_id = p.source_price_id
ORDER BY p.loading_date, p.source_price_id;
