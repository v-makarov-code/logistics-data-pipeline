"""Validate a complete scoring artifact and atomically synchronize its PostgreSQL snapshot."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import UUID

import numpy as np
import pandas as pd
import psycopg

from ml.features import BODY_GROUPS, DatasetConfig

SQL_DIR = Path(__file__).parent / "sql"
PUBLISH_LOCK_ID = 2026111201
LOGGER = logging.getLogger("ml.publish")
ARTIFACT_FILES = (
    "model.cbm", "metrics.json", "selection_history.csv", "feature_importance.csv",
    "predictions_test.parquet", "predictions_test.csv",
    "predictions_future.parquet", "predictions_future.csv",
)
COPY_COLUMNS = (
    "source_price_id", "model_version", "as_of_date", "dataset_split", "loading_date",
    "loading_city_gcid", "unloading_city_gcid", "loading_city_name", "unloading_city_name",
    "body_group", "weight_category", "volume_category", "actual_price_rub", "predicted_price_rub",
    "model_name", "model_run_id", "dataset_run_id", "scored_at", "artifact_sha256",
)


@dataclass(frozen=True)
class PredictionBatch:
    rows: pd.DataFrame
    model_version: str
    as_of_date: date
    scored_at: datetime
    model_run_id: UUID
    artifact_sha256: str


def log_event(event: str, **fields: object) -> None:
    LOGGER.info(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(),
                            "event": event, **fields}, ensure_ascii=False, default=str))


def read_predictions(model_dir: Path) -> PredictionBatch:
    report_bytes = (model_dir / "training_report.json").read_bytes()
    report = json.loads(report_bytes)
    model_version = report["model_version"]
    if not isinstance(model_version, str) or not re.fullmatch(r"[0-9a-f]{16}", model_version):
        raise ValueError("model_version must be a 16-character lowercase hexadecimal string")
    if not isinstance(report["model_name"], str) or not report["model_name"].strip():
        raise ValueError("model_name must be a non-empty string")
    model_run_id, dataset_run_id = UUID(report["run_id"]), UUID(report["dataset_run_id"])
    config = DatasetConfig(date.fromisoformat(report["as_of_date"]))
    scored_at = pd.Timestamp(report["created_at"])
    if pd.isna(scored_at) or scored_at.tzinfo is None or scored_at.value % 1000:
        raise ValueError("scored_at must be timezone-aware with microsecond precision")
    scored_at = scored_at.tz_convert("UTC")
    artifact_sha256 = hashlib.sha256(report_bytes).hexdigest()
    artifacts = {}
    for name in ARTIFACT_FILES:
        payload = (model_dir / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != report["files_sha256"].get(name):
            raise ValueError(f"artifact checksum mismatch: {name}")
        if name.endswith(".parquet"):
            artifacts[name] = pd.read_parquet(io.BytesIO(payload))
        elif name == "metrics.json":
            metrics = json.loads(payload)
    expected_context = {
        "model_version": model_version, "model_name": report["model_name"],
        "model_run_id": str(model_run_id),
    }
    prepared = []
    for split in ("test", "future"):
        frame = artifacts[f"predictions_{split}.parquet"].copy()
        required = {
            "source_price_id", "loading_date", "origin_city_gcid", "destination_city_gcid",
            "origin_city_name", "destination_city_name", "body_group", "weight_category",
            "volume_category", "dataset_split", "actual_price_rub", "predicted_price_rub",
            "abs_error_rub", "ape_pct", "scored_at", *expected_context,
        }
        if not required.issubset(frame.columns):
            raise ValueError(f"missing prediction columns: {sorted(required - set(frame.columns))}")
        if split == "test" and (len(frame) == 0 or len(frame) != metrics["test"]["rows"]):
            raise ValueError("the complete non-empty test partition must match metrics.json")
        for column, expected in (expected_context | {"dataset_split": split}).items():
            if not frame[column].notna().all() or not frame[column].eq(expected).all():
                raise ValueError(f"prediction context mismatch: {column}")
        timestamps = pd.to_datetime(frame["scored_at"], utc=True, errors="raise")
        if not timestamps.notna().all() or not timestamps.eq(scored_at).all():
            raise ValueError("prediction scored_at differs from training report")
        for column in ("source_price_id", "origin_city_gcid", "destination_city_gcid"):
            if frame[column].isna().any() or not pd.api.types.is_integer_dtype(frame[column].dtype):
                raise ValueError(f"integer identifier required: {column}")
        if not frame["source_price_id"].gt(0).all():
            raise ValueError("source_price_id must be positive")
        dates = pd.to_datetime(frame["loading_date"], errors="raise")
        if dates.dt.tz is not None or dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
            raise ValueError("loading_date must be a non-null calendar date")
        valid_dates = dates.between(pd.Timestamp(config.test_start), pd.Timestamp(config.as_of_date)) if split == "test" else dates.gt(pd.Timestamp(config.as_of_date))
        if not valid_dates.all():
            raise ValueError(f"dates outside the {split} scoring window")
        for column in ("origin_city_name", "destination_city_name", "weight_category", "volume_category"):
            if frame[column].isna().any() or not frame[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
                raise ValueError(f"non-empty string required: {column}")
        if not frame["body_group"].isin(BODY_GROUPS).all():
            raise ValueError("unknown body_group")
        for column in ("weight_category", "volume_category"):
            if frame[column].str.len().gt(4).any():
                raise ValueError(f"category exceeds the Core varchar(4) contract: {column}")
        predicted = frame["predicted_price_rub"].to_numpy(dtype="float64")
        if not (np.isfinite(predicted) & (predicted > 0)).all():
            raise ValueError("non-positive or non-finite predictions cannot be published")
        if split == "future":
            if not frame[["actual_price_rub", "abs_error_rub", "ape_pct"]].isna().all().all():
                raise ValueError("future actual price and errors must be NULL")
        else:
            actual = frame["actual_price_rub"].to_numpy(dtype="float64")
            if not (np.isfinite(actual) & (actual > 0)).all():
                raise ValueError("test actual prices must be finite and positive")
            for column, expected in (("abs_error_rub", np.abs(predicted - actual)),
                                     ("ape_pct", np.abs(predicted - actual) / actual * 100)):
                if not np.allclose(frame[column].to_numpy(dtype="float64"), expected, rtol=1e-10, atol=1e-6):
                    raise ValueError(f"inconsistent prediction errors: {column}")
        frame = frame.rename(columns={
            "origin_city_gcid": "loading_city_gcid", "destination_city_gcid": "unloading_city_gcid",
            "origin_city_name": "loading_city_name", "destination_city_name": "unloading_city_name",
        })
        frame["loading_date"] = dates.dt.date
        frame["as_of_date"] = config.as_of_date
        frame["dataset_run_id"] = str(dataset_run_id)
        frame["artifact_sha256"] = artifact_sha256
        frame["scored_at"] = scored_at
        prepared.append(frame.loc[:, COPY_COLUMNS])
    rows = pd.concat(prepared, ignore_index=True)
    if rows["source_price_id"].duplicated().any():
        raise ValueError("each price must occur once across the complete test + future snapshot")
    return PredictionBatch(rows, model_version, config.as_of_date, scored_at.to_pydatetime(), model_run_id, artifact_sha256)


def copy_value(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, np.generic):
        return value.item()
    return value


def publish_predictions(conn: psycopg.Connection, batch: PredictionBatch) -> dict:
    """Full synchronization of one (model_version, as_of_date) in one transaction."""
    if not conn.autocommit:
        raise ValueError("publish_predictions requires an autocommit connection")
    if tuple(batch.rows.columns) != COPY_COLUMNS or not batch.rows.dataset_split.eq("test").any():
        raise ValueError("publication requires the full column contract and a non-empty test partition")
    for column, expected in {
        "model_version": batch.model_version, "as_of_date": batch.as_of_date,
        "model_run_id": str(batch.model_run_id), "artifact_sha256": batch.artifact_sha256,
        "scored_at": batch.scored_at,
    }.items():
        if not batch.rows[column].notna().all() or not batch.rows[column].eq(expected).all():
            raise ValueError(f"batch rows do not match the publication scope: {column}")
    parameters = {"model_version": batch.model_version, "as_of_date": batch.as_of_date}
    with conn.transaction():
        with conn.cursor() as cur:
            # Transaction-scoped: release automatically on success or rollback.
            cur.execute("SELECT pg_try_advisory_xact_lock(%s)", (PUBLISH_LOCK_ID,))
            if not cur.fetchone()[0]:
                raise RuntimeError("another prediction publication is in progress")
            cur.execute("""
                SELECT DISTINCT scored_at, model_run_id, artifact_sha256
                FROM mart.ml_price_predictions
                WHERE model_version = %(model_version)s AND as_of_date = %(as_of_date)s
            """, parameters)
            previous = cur.fetchall()
            if len(previous) > 1:
                raise RuntimeError("published snapshot has mixed artifact metadata")
            if previous:
                old_time, old_run, old_digest = previous[0]
                if old_time > batch.scored_at:
                    raise ValueError("refusing an older scoring artifact for an already newer snapshot")
                if old_time == batch.scored_at and (old_run != batch.model_run_id or old_digest != batch.artifact_sha256):
                    raise ValueError("different artifacts have the same scoring timestamp")
            cur.execute("CREATE TEMP TABLE prediction_stage (LIKE mart.ml_price_predictions INCLUDING ALL) ON COMMIT DROP")
            columns = ", ".join(COPY_COLUMNS)  # Fixed application identifiers, never user input.
            with cur.copy(f"COPY prediction_stage ({columns}) FROM STDIN") as copy:
                for row in batch.rows.itertuples(index=False, name=None):
                    copy.write_row(tuple(copy_value(value) for value in row))
            cur.execute("ANALYZE prediction_stage")
            cur.execute((SQL_DIR / "01_delete_stale_predictions.sql").read_text(encoding="utf-8"), parameters)
            deleted = cur.rowcount
            cur.execute((SQL_DIR / "02_upsert_predictions.sql").read_text(encoding="utf-8"))
            changed = cur.rowcount
            cur.execute("""
                SELECT dataset_split, count(*) FROM mart.ml_price_predictions
                WHERE model_version = %(model_version)s AND as_of_date = %(as_of_date)s
                GROUP BY dataset_split
            """, parameters)
            counts = dict(cur.fetchall())
            expected = batch.rows.dataset_split.value_counts().to_dict()
            if counts != expected:
                raise RuntimeError("published counts differ from the complete artifact; rolling back")
    return {"staged": len(batch.rows), "inserted_or_updated": changed,
            "unchanged": len(batch.rows) - changed, "deleted": deleted, "partitions": counts}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="complete model directory with training_report.json")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    try:
        log_event("prediction_load_started", model_dir=str(args.model_dir.resolve()))
        batch = read_predictions(args.model_dir)
        log_event("prediction_artifact_validated", model_version=batch.model_version,
                  as_of_date=batch.as_of_date, model_run_id=batch.model_run_id, rows=len(batch.rows))
        with psycopg.connect(autocommit=True) as conn:
            result = publish_predictions(conn, batch)
        log_event("prediction_load_succeeded", model_version=batch.model_version,
                  as_of_date=batch.as_of_date, model_run_id=batch.model_run_id, **result)
    except Exception:
        LOGGER.exception(json.dumps({"event": "prediction_load_failed"}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
