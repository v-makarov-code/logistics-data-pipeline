"""Five PostgreSQL checks for source -> Core/quarantine -> Mart."""

import os
import unittest
from datetime import date
from decimal import Decimal
from uuid import uuid4

import psycopg

from etl.pipeline import process_window


@unittest.skipUnless(os.environ.get("RUN_ETL_INTEGRATION") == "1", "requires a disposable PostgreSQL test database")
class PipelineIntegrationTests(unittest.TestCase):

    def setUp(self):
        self.conn = psycopg.connect(autocommit=True)
        database = self.conn.execute("SELECT current_database()").fetchone()[0]
        if not database.startswith("de_etl_test_"):
            self.conn.close()
            raise RuntimeError("integration tests require a database named de_etl_test_*")
        self.conn.execute("DROP TRIGGER IF EXISTS mart_test_failure ON mart.route_card")
        self.conn.execute("DROP FUNCTION IF EXISTS mart.mart_test_failure()")
        self.conn.execute("DELETE FROM mart.route_card")
        self.conn.execute("DELETE FROM dq.quarantine_prices")
        self.conn.execute("DELETE FROM core.prices")
        self.conn.execute("DELETE FROM public.prices")
        self.conn.execute("DELETE FROM public.cities")
        self.conn.execute("DELETE FROM public.exchange_rates")
        self.conn.execute("""
            INSERT INTO public.cities (gc_id, name, full_name_local, full_name_gc)
            VALUES (1001, 'A', 'A', 'A'), (1001, 'A duplicate', 'A', 'A'), (1003, 'B', 'B', 'B')
        """)
        self.conn.execute("INSERT INTO public.exchange_rates (num_code, value, setting_date) VALUES ('840', 90, '2026-09-01')")
        self.conn.execute("""
            INSERT INTO public.prices
            (id, loading_date, loading_city_gcid, unloading_city_gcid, distance, total_weight,
             total_volume, price, currency_code, nearest_rate_date, price_type, is_fact, complex, outlier_mark)
            VALUES
            (1, '2026-09-15', 1001, 1003, 800, 9.96, 120, 90000, '643', NULL, 3, true, false, false),
            (2, '2026-09-16', 1001, 1003, 800, 20, 80, 1000, '840', '2026-09-01', 3, true, false, false),
            (3, '2026-09-17', 1001, 1003, 0, -1, 80, 0, '643', NULL, 3, true, false, false),
            (4, '2026-09-18', 1001, 1003, 800, 20, 80, 1000, '840', NULL, 6, false, false, false),
            (5, '2026-10-01', 1001, 1003, 800, 20, 80, 80000, '643', NULL, 3, true, false, true)
        """)
        self.conn.execute("INSERT INTO public.price_dates (price_id, price_date) SELECT id, loading_date FROM public.prices")

    def tearDown(self):
        self.conn.close()

    def run_window(self, start=date(2026, 9, 1), end=date(2026, 10, 1)):
        return process_window(self.conn, start, end, str(uuid4()))

    def test_validation_conversion_boundaries_and_repeat(self):
        result = self.run_window()
        self.assertEqual((result["found"], result["valid"], result["rejected"]), (4, 2, 2))
        self.assertEqual(self.conn.execute("SELECT source_price_id, price_rub, weight_category, volume_category FROM core.prices ORDER BY 1").fetchall(),
                         [(1, Decimal("90000.00"), "FTL", "FTL"), (2, Decimal("90000.00"), "FTL", "FTL")])
        reasons = dict(self.conn.execute("SELECT source_price_id, reason_codes FROM dq.quarantine_prices").fetchall())
        self.assertEqual(set(reasons[3]), {"NONPOSITIVE_PRICE", "LOW_RUB_PRICE", "NONPOSITIVE_DISTANCE", "NEGATIVE_WEIGHT"})
        self.assertEqual(reasons[4], ["MISSING_EXCHANGE_RATE"])
        before = self.conn.execute("SELECT * FROM core.prices ORDER BY source_price_id").fetchall()
        self.run_window()
        self.assertEqual(before, self.conn.execute("SELECT * FROM core.prices ORDER BY source_price_id").fetchall())
        self.assertEqual(self.conn.execute("SELECT count(*) FROM dq.quarantine_prices").fetchone()[0], 2)

    def test_transitions_date_move_and_source_delete(self):
        self.run_window()
        self.conn.execute("UPDATE public.prices SET price = 0 WHERE id = 1")
        self.conn.execute("UPDATE public.prices SET nearest_rate_date = '2026-09-01' WHERE id = 4")
        self.conn.execute("UPDATE public.prices SET loading_date = '2026-10-15' WHERE id = 2")
        self.conn.execute("DELETE FROM public.prices WHERE id = 3")
        result = self.run_window()
        self.assertEqual(result["deleted_at_source"], 1)
        self.assertEqual(self.conn.execute("SELECT source_price_id, loading_date FROM core.prices ORDER BY 1").fetchall(),
                         [(2, date(2026, 10, 15)), (4, date(2026, 9, 18))])
        self.assertEqual(self.conn.execute("SELECT source_price_id FROM dq.quarantine_prices").fetchall(), [(1,)])
        self.run_window(date(2026, 10, 1), date(2026, 11, 1))
        self.assertEqual(self.conn.execute("SELECT outlier_mark FROM core.prices WHERE source_price_id = 5").fetchone(), (True,))
        self.assertEqual(self.conn.execute("SELECT is_fact FROM core.prices WHERE source_price_id = 4").fetchone(), (False,))

    def test_missing_city_unsupported_currency_and_nonfinite_value(self):
        self.conn.execute("UPDATE public.prices SET unloading_city_gcid = 9999, currency_code = '999', total_weight = 'NaN'::float8 WHERE id = 1")
        self.run_window()
        reasons = self.conn.execute("SELECT reason_codes FROM dq.quarantine_prices WHERE source_price_id = 1").fetchone()[0]
        self.assertEqual(set(reasons), {"UNKNOWN_UNLOADING_CITY", "UNSUPPORTED_CURRENCY", "NONFINITE_WEIGHT"})

    def test_complete_chain_refreshes_mart_and_is_repeatable(self):
        # Only types 3/9 count as carriers; type 1 must contribute a trip only.
        self.conn.execute("UPDATE public.prices SET organization_id = id, price_type = CASE WHEN id = 1 THEN 3 ELSE 1 END WHERE id IN (1, 2)")

        def run_chain():
            return process_window(self.conn, date(2026, 9, 1), date(2026, 10, 1),
                                  str(uuid4()), date(2026, 10, 26))

        result = run_chain()
        self.assertEqual(result["mart_rows"], 1)
        metrics_sql = """
            SELECT trips_count, carriers_count, p25_trip, median_trip, p75_trip,
                   p25_km, median_km, p75_km
            FROM mart.route_card
        """
        self.assertEqual(self.conn.execute(metrics_sql).fetchall(), [
            (2, 1, Decimal("90000"), Decimal("90000"), Decimal("90000"),
             Decimal("112.5"), Decimal("112.5"), Decimal("112.5")),
        ])
        self.conn.execute("UPDATE public.prices SET price=95000 WHERE id=1")
        run_chain()
        # Continuous percentiles interpolate between 90,000 and 95,000 RUB.
        self.assertEqual(self.conn.execute(metrics_sql).fetchall(), [
            (2, 1, Decimal("91250"), Decimal("92500"), Decimal("93750"),
             Decimal("114.0625"), Decimal("115.625"), Decimal("117.1875")),
        ])
        before = self.conn.execute("SELECT to_jsonb(r) - 'refreshed_at' FROM mart.route_card r").fetchall()
        run_chain()
        self.assertEqual(before, self.conn.execute("SELECT to_jsonb(r) - 'refreshed_at' FROM mart.route_card r").fetchall())

    def test_mart_failure_rolls_back_core_and_quarantine(self):
        as_of = date(2026, 10, 26)
        process_window(self.conn, date(2026, 9, 1), date(2026, 10, 1), str(uuid4()), as_of)
        core_before = self.conn.execute("SELECT * FROM core.prices ORDER BY source_price_id").fetchall()
        mart_before = self.conn.execute("SELECT * FROM mart.route_card").fetchall()
        quarantine_before = self.conn.execute("SELECT * FROM dq.quarantine_prices ORDER BY source_price_id").fetchall()
        # Exercise both directions between Core and quarantine before Mart fails.
        self.conn.execute("UPDATE public.prices SET price=0 WHERE id=1")
        self.conn.execute("UPDATE public.prices SET nearest_rate_date='2026-09-01' WHERE id=4")
        self.conn.execute("""
            CREATE FUNCTION mart.mart_test_failure() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN RAISE EXCEPTION 'injected mart failure'; END $$
        """)
        self.conn.execute("CREATE TRIGGER mart_test_failure BEFORE INSERT ON mart.route_card FOR EACH STATEMENT EXECUTE FUNCTION mart.mart_test_failure()")
        with self.assertRaisesRegex(psycopg.Error, "injected mart failure"):
            process_window(self.conn, date(2026, 9, 1), date(2026, 10, 1), str(uuid4()), as_of)
        self.assertEqual(core_before, self.conn.execute("SELECT * FROM core.prices ORDER BY source_price_id").fetchall())
        self.assertEqual(mart_before, self.conn.execute("SELECT * FROM mart.route_card").fetchall())
        self.assertEqual(quarantine_before, self.conn.execute("SELECT * FROM dq.quarantine_prices ORDER BY source_price_id").fetchall())


if __name__ == "__main__":
    unittest.main()
