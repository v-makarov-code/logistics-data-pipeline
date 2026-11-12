"""Run features -> chronological CatBoost training -> atomic prediction publication."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from uuid import uuid4

import psycopg

from ml.features import DatasetConfig
from ml.load_predictions import log_event, publish_predictions, read_predictions
from ml.prepare_dataset import prepare_and_export
from ml.train_model import train_and_export
from ml.training import TrainingConfig


def run_pipeline(as_of_date: date, output_dir: Path, training_config: TrainingConfig) -> dict:
    # Pass exact paths between stages; never pick a directory by modification time.
    dataset_dir = prepare_and_export(DatasetConfig(as_of_date), output_dir)
    model_dir, _ = train_and_export(dataset_dir, output_dir, training_config)
    batch = read_predictions(model_dir)
    with psycopg.connect(autocommit=True) as conn:
        publication = publish_predictions(conn, batch)
    return {"dataset_dir": str(dataset_dir.resolve()), "model_dir": str(model_dir.resolve()),
            "model_version": batch.model_version, "model_run_id": str(batch.model_run_id),
            "publication": publication}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of-date", type=date.fromisoformat, default=date(2026, 10, 26))
    parser.add_argument("--output-dir", type=Path, default=Path("ml/artifacts"))
    parser.add_argument("--thread-count", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    run_id = str(uuid4())
    log_event("ml_pipeline_started", pipeline_run_id=run_id, as_of_date=args.as_of_date)
    try:
        config = TrainingConfig(thread_count=args.thread_count, random_seed=args.random_seed)
        result = run_pipeline(args.as_of_date, args.output_dir, config)
        log_event("ml_pipeline_succeeded", pipeline_run_id=run_id, **result)
    except Exception:
        logging.getLogger("ml.pipeline").exception(json.dumps({"event": "ml_pipeline_failed", "pipeline_run_id": run_id}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
