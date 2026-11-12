"""Eight checks for features, SQL extraction and real CatBoost training."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from catboost import CatBoostRegressor
from pandas.testing import assert_frame_equal

from ml.features import FEATURE_COLUMNS, TARGET, DatasetConfig, extract_region, prepare_features
from ml.prepare_dataset import extract_dataset, save_dataset
from ml.train_model import train_and_export
from ml.training import TrainingConfig, load_dataset, make_pool, regression_metrics, split_inner_train


def example_row(price_id=1, loading_date="2026-09-15", **changes):
    row = {
        "source_price_id": price_id, "loading_date": loading_date,
        "origin_city_gcid": 1001, "destination_city_gcid": 1003,
        "origin_city_row_id": 1, "destination_city_row_id": 3,
        "origin_city_name": "Москва", "destination_city_name": "Казань",
        "origin_full_name": "Москва, Москва", "destination_full_name": "Казань, Республика Татарстан",
        "origin_latitude": 55.75, "origin_longitude": 37.62,
        "destination_latitude": 55.80, "destination_longitude": 49.11,
        "distance": 897.0, "total_weight": 20.0, "total_volume": 80.0,
        "weight_category": "FTL", "volume_category": "FTL",
        "primary_body_group": "TENT", "allowed_body_groups": ["TENT", "FLATBED"],
        TARGET: 84500.0,
    }
    return row | changes


def sample_raw():
    dates = ["2026-07-26", "2026-08-10", "2026-09-01", "2026-09-20",
             "2026-09-21", "2026-09-30", "2026-10-04", "2026-10-05", "2026-10-26", "2026-10-27"]
    rows = []
    for index, loading_date in enumerate(dates):
        for offset in range(2):
            number = index * 2 + offset
            rows.append(example_row(number + 1, loading_date, distance=300.0 + 50 * number,
                                    total_weight=10.0 + number % 3, target_price_rub=40000.0 + 5000 * number))
    return pd.DataFrame(rows)


class FeatureTests(unittest.TestCase):

    config = DatasetConfig()

    def prepare(self, *rows):
        return prepare_features(pd.DataFrame(rows), self.config)

    def test_time_boundaries_and_future_mask(self):
        dates = ["2026-07-26", "2026-10-04", "2026-10-05", "2026-10-26", "2026-10-27"]
        parts = self.prepare(*(example_row(i + 1, d) for i, d in enumerate(dates)))
        self.assertEqual(parts["train"].metadata.source_price_id.tolist(), [1, 2])
        self.assertEqual(parts["test"].metadata.source_price_id.tolist(), [3, 4])
        self.assertEqual(parts["future"].metadata.source_price_id.tolist(), [5])
        self.assertTrue(parts["future"].target[TARGET].isna().all())
        with self.assertRaisesRegex(ValueError, "before train_start"):
            self.prepare(example_row(1, "2026-07-25"))

    def test_no_target_or_metadata_leakage_even_with_extra_columns(self):
        base = example_row(1, "2026-10-27")
        changed = base | {TARGET: 999999999, "price_per_km": 999999, "avg_price": 123,
                          "outlier_mark": True, "price_with_vat": 456}
        first = self.prepare(base)["future"]
        second = self.prepare(changed)["future"]
        assert_frame_equal(first.features, second.features)
        assert_frame_equal(first.target, second.target)
        self.assertEqual(tuple(first.features.columns), FEATURE_COLUMNS)
        self.assertFalse(set(first.features.columns) & {TARGET, "source_price_id", "loading_date", "avg_price"})

    def test_regions_and_coordinates(self):
        self.assertEqual(extract_region("Казань, Республика Татарстан"), "Республика Татарстан")
        self.assertEqual(extract_region("Город, Регион, Район"), "Регион, Район")
        for value in (None, "Москва", "Казань,   "):
            self.assertEqual(extract_region(value), "UNKNOWN")
        features = self.prepare(example_row(origin_latitude=91, destination_longitude=np.inf,
                                             origin_full_name=None))["train"].features
        self.assertTrue(pd.isna(features.origin_latitude[0]))
        self.assertTrue(pd.isna(features.destination_longitude[0]))
        self.assertEqual(features.origin_region[0], "UNKNOWN")

    def test_cyclical_encoding_and_deterministic_order(self):
        first, second = example_row(2, "2026-10-05"), example_row(1, "2026-10-05")
        parts = self.prepare(first, second)
        repeated = self.prepare(second, first)
        for name in parts:
            assert_frame_equal(parts[name].features, repeated[name].features)
            assert_frame_equal(parts[name].target, repeated[name].target)
            assert_frame_equal(parts[name].metadata, repeated[name].metadata)
        x = parts["test"].features.iloc[0]
        self.assertEqual(x.weekday, 0)  # 2026-10-05 is Monday.
        self.assertEqual(x.weekday_sin, 0)
        self.assertEqual(x.weekday_cos, 1)
        self.assertAlmostEqual(x.month_sin, -1)
        self.assertEqual(x.year, 2026)
        self.assertEqual(x.day_of_year, 278)
        for name in ("month", "weekday", "day_of_year"):
            self.assertAlmostEqual(x[f"{name}_sin"] ** 2 + x[f"{name}_cos"] ** 2, 1)


class TrainingTests(unittest.TestCase):

    def bundle(self, root, raw=None, run_id="dataset"):
        raw = sample_raw() if raw is None else raw
        config = DatasetConfig()
        parts = prepare_features(raw, config)
        path, _ = save_dataset(parts, raw, config, root, run_id, "test-snapshot")
        return path

    def test_regression_metrics_units_and_undefined_r2(self):
        metrics = regression_metrics(np.array([100.0, 200.0]), np.array([110.0, 180.0]))
        self.assertAlmostEqual(metrics["rmse_rub"], np.sqrt(250))
        self.assertEqual(metrics["mae_rub"], 15)
        self.assertEqual(metrics["mape_pct"], 10)
        self.assertAlmostEqual(metrics["r2"], 0.9)
        self.assertIsNone(regression_metrics(np.array([100.0, 100.0]), np.array([90.0, 110.0]))["r2"])
        with self.assertRaises(ValueError):
            regression_metrics(np.array([0.0]), np.array([1.0]))

    def test_inner_validation_boundaries(self):
        parts = prepare_features(sample_raw(), DatasetConfig())
        fit, validation, start = split_inner_train(parts["train"], "2026-10-05", 14)
        self.assertEqual(start, "2026-09-21")
        self.assertEqual(int(fit.sum()), 8)
        self.assertEqual(int(validation.sum()), 6)
        self.assertLess(parts["train"].metadata.loc[fit, "loading_date"].max(),
                        parts["train"].metadata.loc[validation, "loading_date"].min())
        with self.assertRaises(ValueError):
            split_inner_train(parts["train"], "2026-10-05", 1000)

    def test_training_reload_export_and_test_target_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self.bundle(root)
            config = TrainingConfig(iterations=12, depth=2, thread_count=1, early_stopping_rounds=3)
            output, metrics = train_and_export(dataset, root, config)
            report = json.loads((output / "training_report.json").read_text(encoding="utf-8"))
            self.assertFalse(report["test_used_for_selection"])
            self.assertEqual(report["refit_train_rows"], 14)
            self.assertLessEqual(report["selected_iterations"], 12)
            self.assertEqual(len(report["files_sha256"]), 8)
            importance = pd.read_csv(output / "feature_importance.csv")
            self.assertTrue(np.isfinite(importance.importance).all())
            self.assertAlmostEqual(importance.importance.sum(), 100)
            parts, _ = load_dataset(dataset)
            model = CatBoostRegressor()
            model.load_model(str(output / "model.cbm"))
            exported = pd.read_parquet(output / "predictions_test.parquet")
            np.testing.assert_allclose(model.predict(make_pool(parts["test"])), exported.predicted_price_rub, rtol=0, atol=1e-8)
            np.testing.assert_allclose(exported.abs_error_rub, abs(exported.predicted_price_rub - exported.actual_price_rub))
            future = pd.read_parquet(output / "predictions_future.parquet")
            self.assertTrue(future[["actual_price_rub", "abs_error_rub", "ape_pct"]].isna().all().all())

            # New held-out labels must change evaluation, never the trained model or predictions.
            changed = sample_raw()
            mask = pd.to_datetime(changed.loading_date).between("2026-10-05", "2026-10-26")
            changed.loc[mask, TARGET] *= 10
            changed_dataset = self.bundle(root, changed, "changed-test")
            second, changed_metrics = train_and_export(changed_dataset, root, config)
            second_report = json.loads((second / "training_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["model_version"], second_report["model_version"])
            np.testing.assert_allclose(exported.predicted_price_rub,
                                       pd.read_parquet(second / "predictions_test.parquet").predicted_price_rub, rtol=0, atol=1e-8)
            self.assertNotEqual(metrics["test"]["rmse_rub"], changed_metrics["test"]["rmse_rub"])


@unittest.skipUnless(os.environ.get("RUN_ML_INTEGRATION") == "1", "requires a disposable PostgreSQL test database")
class MlExtractionTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.conn = psycopg.connect(autocommit=True)
        database = cls.conn.execute("SELECT current_database()").fetchone()[0]
        if not database.startswith("de_ml_test_"):
            cls.conn.close()
            raise RuntimeError("ML integration tests require a database named de_ml_test_*")
        root = Path(__file__).resolve().parent.parent
        # Only an empty, disposable database is supported; no destructive fixture reset.
        cls.conn.execute((root / "sql/schema.sql").read_text(encoding="utf-8"))
        cls.conn.execute((root / "sql/03_core_and_quarantine.sql").read_text(encoding="utf-8"))
        cls.conn.execute((root / "tests/route_card_fixture.sql").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def test_extraction_grain_filters_body_fallback_and_future_isolation(self):
        config = DatasetConfig()
        raw, _ = extract_dataset(self.conn, config)
        self.assertEqual(raw.source_price_id.tolist(), [1, 2, 3, 6, 7])
        self.assertTrue(raw.source_price_id.is_unique)
        rows = raw.set_index("source_price_id")
        self.assertEqual(rows.loc[1, "origin_city_row_id"], 1)
        self.assertEqual(rows.loc[1, "origin_city_name"], "Москва")
        self.assertEqual(rows.loc[1, "allowed_body_groups"], ["FLATBED", "TENT"])
        self.assertEqual(rows.loc[3, "allowed_body_groups"], ["TENT"])  # no links: fallback
        self.assertTrue(raw.loc[raw.source_price_id == 7, TARGET].isna().all())
        parts = prepare_features(raw, config)
        self.assertEqual([len(parts[name].features) for name in ("train", "test", "future")], [3, 1, 1])

        # Change the hidden future price and its price-derived flag; the candidate survives.
        self.conn.execute("UPDATE core.prices SET price_rub = 1, outlier_mark = true WHERE source_price_id = 7")
        changed, _ = extract_dataset(self.conn, config)
        changed_parts = prepare_features(changed, config)
        for name in parts:
            assert_frame_equal(parts[name].features, changed_parts[name].features)
            assert_frame_equal(parts[name].target, changed_parts[name].target)
            assert_frame_equal(parts[name].metadata, changed_parts[name].metadata)

        # The primary type and compatibility list carry different information.
        self.conn.execute("INSERT INTO public.body_types (id, name, \"id_from_1C\", std_id) VALUES (4, 'R', 4, 4), (2, 'I', 2, 2)")
        self.conn.execute("UPDATE core.prices SET body_type = NULL WHERE source_price_id = 7")
        self.conn.execute("DELETE FROM public.price_body_types WHERE price_id = 7")
        self.conn.execute("INSERT INTO public.price_body_types (price_id, body_type_id) VALUES (7, 4), (7, 2)")
        changed, _ = extract_dataset(self.conn, config)
        x = prepare_features(changed, config)["future"].features.iloc[0]
        self.assertEqual(x.primary_body_group, "UNSPECIFIED")
        self.assertEqual((x.allows_refrigerator, x.allows_insulated, x.allows_unspecified), (1, 1, 0))


if __name__ == "__main__":
    unittest.main()
