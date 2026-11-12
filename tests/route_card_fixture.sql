-- Run only in a disposable database with the source, Core and Mart schemas installed.
INSERT INTO public.cities (id, gc_id, name, full_name_local, full_name_gc) VALUES
    (1, 1001, 'Москва', 'Москва', 'Москва'),
    (2, 1001, 'Дубль', 'Дубль', 'Дубль'),
    (3, 1003, 'Казань', 'Казань', 'Казань');
INSERT INTO public.body_types (id, name, "id_from_1C", std_id) VALUES
    (56, 'Тент', 56, 56), (61, 'Тент другой', 61, 56), (15, 'Борт', 15, 15);

INSERT INTO public.prices
    (id, loading_date, loading_city_gcid, unloading_city_gcid, distance,
     total_weight, total_volume, body_type, price, currency_code, price_type,
     is_fact, complex, outlier_mark, organization_id, weight_category, volume_category)
VALUES
    (1, '2026-09-10',1001,1003,100,20,80,56,80000,'643',3,true,false,false,1,'FTL','FTL'),
    (2, '2026-09-15',1001,1003,100,20,80,56,100000,'643',1,true,false,false,3,'FTL','FTL'),
    (3, '2026-09-20',1001,1003,100,20,80,56,120000,'643',9,true,false,false,2,'FTL','FTL'),
    (4, '2025-09-10',1001,1003,100,20,80,56,50000,'643',3,true,false,false,1,'FTL','FTL'),
    (5, '2026-07-10',1001,1003,100,20,80,56,70000,'643',3,true,false,false,1,'FTL','FTL'),
    (6, '2026-10-20',1001,1003,100,20,80,56,200000,'643',3,true,false,false,2,'FTL','FTL'),
    (7, '2026-10-27',1001,1003,100,20,80,56,999000,'643',3,true,false,false,2,'FTL','FTL'),
    (8, '2026-09-20',1001,1003,100,20,80,56,999000,'643',3,true,false,true,2,'FTL','FTL');

INSERT INTO core.prices (
    source_price_id, organization_id, loading_date, loading_city_gcid, unloading_city_gcid,
    distance, total_weight, total_volume, body_type, weight_category, volume_category,
    price_original, currency_code, exchange_rate, price_rub, price_type, is_fact, complex, outlier_mark
)
SELECT id, organization_id, loading_date, loading_city_gcid, unloading_city_gcid,
       distance, total_weight, total_volume, body_type, weight_category, volume_category,
       price::numeric, currency_code, 1, price::numeric, price_type, is_fact, complex, outlier_mark
FROM public.prices;

INSERT INTO public.price_body_types (price_id, body_type_id) VALUES
    (1,56),(1,61),(1,15),(2,56),(6,56),(7,56),(8,56);
INSERT INTO public.price_dates (price_id, price_date)
SELECT id, loading_date FROM public.prices;
-- Same price and month twice, plus a second valid month for the long-term request.
INSERT INTO public.price_dates (price_id, price_date) VALUES
    (1,'2026-09-25'), (1,'2026-10-15');
