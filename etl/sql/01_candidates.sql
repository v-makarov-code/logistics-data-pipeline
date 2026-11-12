CREATE TEMP TABLE candidate_ids ON COMMIT DROP AS
SELECT id AS source_price_id
FROM public.prices
WHERE loading_date >= %(start_date)s AND loading_date < %(end_date)s
UNION
SELECT source_price_id
FROM core.prices
WHERE loading_date >= %(start_date)s AND loading_date < %(end_date)s
UNION
SELECT source_price_id
FROM dq.quarantine_prices
WHERE loading_date >= %(start_date)s AND loading_date < %(end_date)s;
