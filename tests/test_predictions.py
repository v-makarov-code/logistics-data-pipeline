"""Three PostgreSQL checks for idempotent, isolated, atomic publication."""

import hashlib
import json
import os
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import psycopg

from ml.load_predictions import publish_predictions, read_predictions


def artifact(root, *, model_version="a" * 16, as_of=date(2026, 10, 26),
             scored_at=datetime(2026, 11, 12, 12, tzinfo=timezone.utc),
             include_future=True, first_prediction=110.0):
    """Minimal complete file contract; publication never deserializes the fixture's model."""
    run_id = str(uuid4())
    output = root / f"model-{run_id}"
    output.mkdir()
    rows = []
    for price_id, loading_date, split, actual, predicted in (
        (1, as_of - timedelta(days=21), "test", 100.0, first_prediction),
        (2, as_of, "test", 200.0, 180.0),
        (3, as_of + timedelta(days=1), "future", np.nan, 310.0),
    ):
        rows.append({
            "source_price_id": price_id, "loading_date": loading_date,
            "origin_city_gcid": 1001, "destination_city_gcid": 1003,
            "origin_city_name": "Москва", "destination_city_name": "Казань",
            "body_group": "TENT", "weight_category": "FTL", "volume_category": "FTL",
            "dataset_split": split, "actual_price_rub": actual, "predicted_price_rub": predicted,
            "abs_error_rub": abs(predicted - actual), "ape_pct": abs(predicted - actual) / actual * 100,
            "model_name": "CatBoostRegressor", "model_version": model_version,
            "model_run_id": run_id, "scored_at": scored_at,
        })
    frame = pd.DataFrame(rows)
    parts = {name: frame.loc[frame.dataset_split.eq(name)].reset_index(drop=True) for name in ("test", "future")}
    if not include_future:
        parts["future"] = parts["future"].iloc[:0].copy()
    for name, table in parts.items():
        table.to_parquet(output / f"predictions_{name}.parquet", index=False)
        table.to_csv(output / f"predictions_{name}.csv", index=False)
    (output / "model.cbm").write_bytes(b"publisher fixture: checksum only, not a CatBoost model")
    for name in ("selection_history.csv", "feature_importance.csv"):
        (output / name).write_text("fixture\n", encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps({"test": {"rows": len(parts["test"])}}), encoding="utf-8")
    report = {
        "run_id": run_id, "dataset_run_id": str(uuid4()), "model_version": model_version,
        "model_name": "CatBoostRegressor", "as_of_date": str(as_of), "created_at": scored_at.isoformat(),
        "files_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir()},
    }
    (output / "training_report.json").write_text(json.dumps(report), encoding="utf-8")
    return output


@unittest.skipUnless(os.environ.get("RUN_ML_PUBLISH_INTEGRATION") == "1", "requires a disposable PostgreSQL test database")
class PredictionPublicationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.conn = psycopg.connect(autocommit=True)
        database = cls.conn.execute("SELECT current_database()").fetchone()[0]
        if not database.startswith("de_ml_publish_test_"):
            cls.conn.close()
            raise RuntimeError("publication tests require a database named de_ml_publish_test_*")
        cls.root = Path(__file__).resolve().parent.parent
        for name in ("04_mart_route_card.sql", "06_mart_ml_price_predictions.sql"):
            cls.conn.execute((cls.root / "sql" / name).read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture_root = Path(self.temporary.name)
        self.conn.execute("DROP TRIGGER IF EXISTS prediction_failure ON mart.ml_price_predictions")
        self.conn.execute("DROP FUNCTION IF EXISTS mart.prediction_failure()")
        self.conn.execute("TRUNCATE mart.ml_price_predictions")

    def batch(self, **kwargs):
        return read_predictions(artifact(self.fixture_root, **kwargs))

    def stored(self):
        return self.conn.execute("SELECT * FROM mart.ml_price_predictions ORDER BY model_version, as_of_date, source_price_id").fetchall()

    def test_repeat_is_a_true_noop_and_errors_are_generated(self):
        batch = self.batch()
        first = publish_predictions(self.conn, batch)
        before = self.stored()
        second = publish_predictions(self.conn, batch)
        self.assertEqual(first["inserted_or_updated"], 3)
        self.assertEqual(second["inserted_or_updated"], 0)
        self.assertEqual(second["unchanged"], 3)
        self.assertEqual(second["deleted"], 0)
        self.assertEqual(before, self.stored())  # Includes loaded_at.
        errors = self.conn.execute("SELECT source_price_id, abs_error_rub, ape_pct FROM mart.ml_price_predictions ORDER BY 1").fetchall()
        self.assertEqual(errors, [(1, 10.0, 10.0), (2, 20.0, 10.0), (3, None, None)])

    def test_new_snapshot_updates_deletes_and_preserves_other_scopes(self):
        publish_predictions(self.conn, self.batch())
        publish_predictions(self.conn, self.batch(model_version="b" * 16))
        publish_predictions(self.conn, self.batch(as_of=date(2026, 11, 26)))
        later = datetime(2026, 11, 12, 13, tzinfo=timezone.utc)
        result = publish_predictions(self.conn, self.batch(scored_at=later, include_future=False, first_prediction=120.0))
        self.assertEqual((result["deleted"], result["inserted_or_updated"]), (1, 2))
        self.assertEqual(len(self.stored()), 8)  # 2 refreshed + 3 other model + 3 other as_of.
        actual = self.conn.execute("SELECT abs_error_rub FROM mart.ml_price_predictions WHERE model_version = %s AND as_of_date = %s AND source_price_id = 1", ("a"*16, date(2026,10,26))).fetchone()[0]
        self.assertEqual(actual, 20.0)

    def test_failure_rolls_back_stale_deletion_and_upsert(self):
        publish_predictions(self.conn, self.batch())
        before = self.stored()
        self.conn.execute("""
            CREATE FUNCTION mart.prediction_failure() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN RAISE EXCEPTION 'forced publication failure'; END $$;
            CREATE TRIGGER prediction_failure BEFORE INSERT OR UPDATE ON mart.ml_price_predictions
            FOR EACH ROW EXECUTE FUNCTION mart.prediction_failure();
        """)
        later = datetime(2026, 11, 12, 13, tzinfo=timezone.utc)
        with self.assertRaisesRegex(psycopg.Error, "forced publication failure"):
            publish_predictions(self.conn, self.batch(scored_at=later, include_future=False))
        self.assertEqual(before, self.stored())


if __name__ == "__main__":
    unittest.main()
