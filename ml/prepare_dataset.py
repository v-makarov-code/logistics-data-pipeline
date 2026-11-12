"""Read a consistent Core snapshot and publish versioned Parquet feature datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import psycopg
import pyarrow
from psycopg.rows import dict_row

from ml.features import (
    BODY_GROUPS, CATEGORICAL_FEATURES, CONTRACT_VERSION, FEATURE_COLUMNS, TARGET,
    DatasetConfig, DatasetPart, prepare_features,
)

SQL_PATH = Path(__file__).parent / "sql" / "extract_dataset.sql"
# Same key as Core ETL / Mart. Shared readers can coexist, exclusive writers cannot.
CORE_LOCK_ID = 2026111101
LOGGER = logging.getLogger("ml.features")


def log_event(event: str, **fields: object) -> None:
    LOGGER.info(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(),
                            "event": event, **fields}, ensure_ascii=False, default=str))


def extract_dataset(conn: psycopg.Connection, config: DatasetConfig) -> tuple[pd.DataFrame, str]:
    if not conn.autocommit:
        raise ValueError("extract_dataset requires an autocommit connection")
    locked = conn.execute("SELECT pg_try_advisory_lock_shared(%s)", (CORE_LOCK_ID,)).fetchone()[0]
    if not locked:
        raise RuntimeError("Core ETL or Mart refresh is running; retry after it completes")
    try:
        with conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            snapshot = conn.execute("SELECT pg_current_snapshot()::text").fetchone()[0]
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(SQL_PATH.read_text(encoding="utf-8"), {
                    "train_start": config.train_start, "as_of_date": config.as_of_date,
                })
                columns = [column.name for column in cur.description]
                raw = pd.DataFrame(cur.fetchall(), columns=columns)
        return raw, snapshot
    finally:
        conn.execute("SELECT pg_advisory_unlock_shared(%s)", (CORE_LOCK_ID,))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_summary(parts: dict[str, DatasetPart]) -> dict:
    train = parts["train"].features
    summary = {}
    for name, part in parts.items():
        dates = part.metadata["loading_date"]
        summary[name] = {
            "rows": len(part.features),
            "first_loading_date": str(dates.min().date()) if len(dates) else None,
            "last_loading_date": str(dates.max().date()) if len(dates) else None,
            "missing_features": {column: int(count) for column, count in part.features.isna().sum().items() if count},
            "unknown_regions": {column: int((part.features[column] == "UNKNOWN").sum())
                                for column in ("origin_region", "destination_region")},
            "unseen_category_rows": {
                column: int((~part.features[column].isin(train[column].unique())).sum())
                for column in CATEGORICAL_FEATURES
            },
        }
    return summary


def save_dataset(parts: dict[str, DatasetPart], raw: pd.DataFrame, config: DatasetConfig,
                 output_root: Path, run_id: str, snapshot: str) -> tuple[Path, dict]:
    """Manifest is the completion marker; partial runs must not be consumed."""
    run_dir = output_root / f"dataset-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, part in parts.items():
        partition_dir = run_dir / name
        partition_dir.mkdir()
        for kind, table in (("X", part.features), ("y", part.target), ("metadata", part.metadata)):
            path = partition_dir / f"{kind}.parquet"
            table.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
            files[path.relative_to(run_dir).as_posix()] = sha256_file(path)
    train = parts["train"].features
    manifest = {
        "contract_version": CONTRACT_VERSION, "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": str(config.as_of_date), "train_start_inclusive": str(config.train_start),
        "test_start_inclusive": str(config.test_start), "test_end_inclusive": str(config.as_of_date),
        "future_start_exclusive": str(config.as_of_date),
        "grain": "one row per source_price_id", "target": TARGET,
        "feature_columns": list(FEATURE_COLUMNS), "categorical_features": list(CATEGORICAL_FEATURES),
        "feature_dtypes": {column: str(dtype) for column, dtype in train.dtypes.items()},
        "body_groups": list(BODY_GROUPS),
        "constant_train_features": [column for column in train if train[column].nunique(dropna=False) <= 1],
        "partitions": dataset_summary(parts),
        "database_snapshot": snapshot,
        "extracted_rows_sha256": hashlib.sha256(raw.to_json(orient="records", date_format="iso", double_precision=15).encode()).hexdigest(),
        "code_sha256": {"extract_dataset.sql": sha256_file(SQL_PATH),
                        "features.py": sha256_file(Path(__file__).with_name("features.py")),
                        "prepare_dataset.py": sha256_file(Path(__file__))},
        "files_sha256": files,
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "pyarrow": pyarrow.__version__, "psycopg": psycopg.__version__},
        "limitations": [
            "Core is a current snapshot, not a point-in-time history; source edits cannot be reconstructed.",
            "loading_date and is_fact approximate completion; the source has no completion timestamp.",
            "Future candidates come from Core and inherit its upstream price/currency validation.",
            "Existing outlier_mark is trusted for history; no statistical outlier detector is fitted.",
            "Short 2026 training history cannot establish annual seasonality or a multi-year trend.",
        ],
    }
    manifest_path = run_dir / "manifest.json"
    temporary_manifest = run_dir / "manifest.tmp"
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return run_dir, manifest


def prepare_and_export(config: DatasetConfig, output_dir: Path, run_id: str | None = None) -> Path:
    """Return the exact complete dataset directory for the next pipeline stage."""
    run_id = run_id or str(uuid4())
    log_event("features_started", run_id=run_id, as_of_date=config.as_of_date,
              train_start=config.train_start, test_start=config.test_start)
    with psycopg.connect(autocommit=True) as conn:
        raw, snapshot = extract_dataset(conn, config)
    log_event("features_extracted", run_id=run_id, rows=len(raw), database_snapshot=snapshot)
    parts = prepare_features(raw, config)
    if not len(parts["train"].features) or not len(parts["test"].features):
        raise ValueError("train and test must both be non-empty; load the required Core period")
    run_dir, manifest = save_dataset(parts, raw, config, output_dir, run_id, snapshot)
    log_event("features_succeeded", run_id=run_id, output_dir=str(run_dir.resolve()),
              partitions=manifest["partitions"], constant_train_features=manifest["constant_train_features"])
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of-date", type=date.fromisoformat, default=date(2026, 10, 26))
    parser.add_argument("--output-dir", type=Path, default=Path("ml/artifacts"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    run_id = str(uuid4())
    try:
        prepare_and_export(DatasetConfig(args.as_of_date), args.output_dir, run_id)
    except Exception:
        LOGGER.exception(json.dumps({"event": "features_failed", "run_id": run_id}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
