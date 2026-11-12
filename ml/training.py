"""Training contract, chronological validation, regression metrics and scoring rows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import Pool

from ml.features import (
    CATEGORICAL_FEATURES, CONTRACT_VERSION, FEATURE_COLUMNS, METADATA_COLUMNS,
    TARGET, DatasetConfig, DatasetPart,
)
from ml.prepare_dataset import sha256_file


@dataclass(frozen=True)
class TrainingConfig:
    iterations: int = 1000
    depth: int = 6
    learning_rate: float = 0.05
    random_seed: int = 42
    thread_count: int = 4
    validation_days: int = 14
    early_stopping_rounds: int = 50

    def __post_init__(self) -> None:
        if min(self.iterations, self.thread_count, self.validation_days, self.early_stopping_rounds) < 1:
            raise ValueError("iterations, threads, validation days and stopping rounds must be positive")
        if not 1 <= self.depth <= 16 or not 0 < self.learning_rate <= 1 or self.random_seed < 0:
            raise ValueError("depth must be 1..16, learning rate in (0, 1], seed non-negative")

    def model_params(self, iterations: int | None = None) -> dict:
        return {
            "iterations": self.iterations if iterations is None else iterations,
            "depth": self.depth, "learning_rate": self.learning_rate, "l2_leaf_reg": 3.0,
            "loss_function": "RMSE", "eval_metric": "RMSE", "random_seed": self.random_seed,
            "thread_count": self.thread_count, "task_type": "CPU", "boosting_type": "Plain",
            "has_time": True, "nan_mode": "Min", "allow_writing_files": False,
        }


def load_dataset(dataset_dir: Path) -> tuple[dict[str, DatasetPart], dict]:
    """Consume only complete datasets with intact files and the current feature contract."""
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["contract_version"] != CONTRACT_VERSION:
        raise ValueError("unsupported feature contract version")
    if manifest["feature_columns"] != list(FEATURE_COLUMNS):
        raise ValueError("feature order differs from the current contract")
    if manifest["categorical_features"] != list(CATEGORICAL_FEATURES):
        raise ValueError("categorical features differ from the current contract")
    config = DatasetConfig(pd.Timestamp(manifest["as_of_date"]).date())
    expected_bounds = {
        "train_start_inclusive": str(config.train_start),
        "test_start_inclusive": str(config.test_start),
        "test_end_inclusive": str(config.as_of_date),
        "future_start_exclusive": str(config.as_of_date),
    }
    if any(manifest.get(key) != value for key, value in expected_bounds.items()):
        raise ValueError("dataset time boundaries differ from the current contract")
    parts = {}
    all_ids = []
    for name in ("train", "test", "future"):
        tables = {}
        for kind in ("X", "y", "metadata"):
            relative = f"{name}/{kind}.parquet"
            path = dataset_dir / relative
            if sha256_file(path) != manifest["files_sha256"].get(relative):
                raise ValueError(f"dataset checksum mismatch: {relative}")
            tables[kind] = pd.read_parquet(path)
        x, y, metadata = tables["X"], tables["y"], tables["metadata"]
        if tuple(x.columns) != FEATURE_COLUMNS or list(y.columns) != [TARGET]:
            raise ValueError(f"unexpected feature or target columns: {name}")
        if tuple(metadata.columns) != METADATA_COLUMNS:
            raise ValueError(f"unexpected metadata columns: {name}")
        if not (len(x) == len(y) == len(metadata) == manifest["partitions"][name]["rows"]):
            raise ValueError(f"row counts are not aligned: {name}")
        if name != "future" and len(x) < 2:
            raise ValueError("train and test require at least two rows")
        if metadata.source_price_id.isna().any() or metadata.source_price_id.duplicated().any():
            raise ValueError(f"invalid or duplicate price IDs: {name}")
        all_ids.extend(metadata.source_price_id.tolist())
        dates = pd.to_datetime(metadata.loading_date, errors="raise")
        if dates.isna().any() or (dates != dates.dt.normalize()).any():
            raise ValueError(f"invalid loading dates: {name}")
        in_window = {
            "train": dates.ge(pd.Timestamp(config.train_start)) & dates.lt(pd.Timestamp(config.test_start)),
            "test": dates.ge(pd.Timestamp(config.test_start)) & dates.le(pd.Timestamp(config.as_of_date)),
            "future": dates.gt(pd.Timestamp(config.as_of_date)),
        }[name]
        if not in_window.all():
            raise ValueError(f"dates outside the partition window: {name}")
        expected_order = metadata.sort_values(["loading_date", "source_price_id"]).index
        if not expected_order.equals(metadata.index):
            raise ValueError(f"dataset is not chronologically ordered: {name}")
        labels = y[TARGET].to_numpy(dtype="float64")
        if name == "future":
            if not np.isnan(labels).all():
                raise ValueError("future targets must be masked")
        elif not (np.isfinite(labels) & (labels > 0)).all():
            raise ValueError(f"historical targets must be finite and positive: {name}")
        for column in FEATURE_COLUMNS:
            if column in CATEGORICAL_FEATURES:
                if x[column].isna().any() or not x[column].map(lambda value: isinstance(value, str)).all():
                    raise ValueError(f"categorical values must be non-null strings: {column}")
            else:
                values = x[column].to_numpy(dtype="float64")
                if np.isinf(values).any() or (np.isnan(values).any() and column not in {
                    "origin_latitude", "origin_longitude", "destination_latitude", "destination_longitude",
                }):
                    raise ValueError(f"unexpected missing or infinite feature: {column}")
        parts[name] = DatasetPart(x, y, metadata)
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("price IDs overlap between partitions")
    return parts, manifest


def split_inner_train(train: DatasetPart, test_start: str, validation_days: int) -> tuple[np.ndarray, np.ndarray, str]:
    validation_start = pd.Timestamp(test_start) - timedelta(days=validation_days)
    validation = pd.to_datetime(train.metadata.loading_date).ge(validation_start).to_numpy()
    fit = ~validation
    if not fit.any() or not validation.any():
        raise ValueError("inner fit and validation must both be non-empty; adjust validation days")
    return fit, validation, str(validation_start.date())


def make_pool(part: DatasetPart, mask: np.ndarray | None = None, *, with_target: bool = False) -> Pool:
    features = part.features if mask is None else part.features.loc[mask]
    features = features.copy()
    # Explicit Python strings avoid depending on pandas' internal string storage format.
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].astype(object)
    labels = None
    if with_target:
        target = part.target if mask is None else part.target.loc[mask]
        labels = target[TARGET].to_numpy(dtype="float64")
    return Pool(features, label=labels, cat_features=list(CATEGORICAL_FEATURES))


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype="float64")
    predicted = np.asarray(predicted, dtype="float64")
    if actual.ndim != 1 or predicted.shape != actual.shape or not len(actual):
        raise ValueError("metrics require aligned non-empty one-dimensional arrays")
    if not (np.isfinite(actual) & (actual > 0) & np.isfinite(predicted)).all():
        raise ValueError("metrics require positive finite actuals and finite predictions")
    errors = predicted - actual
    squared_error = float(np.sum(errors ** 2))
    total_variation = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "rows": len(actual),
        "rmse_rub": float(np.sqrt(np.mean(errors ** 2))),
        "mae_rub": float(np.mean(np.abs(errors))),
        "mape_pct": float(np.mean(np.abs(errors) / actual) * 100),
        # R2 is undefined for a constant target or a single observation.
        "r2": 1 - squared_error / total_variation if len(actual) >= 2 and total_variation > 0 else None,
    }


def prediction_rows(part: DatasetPart, predicted: np.ndarray, split: str,
                    model_version: str, run_id: str, scored_at: str) -> pd.DataFrame:
    if split not in ("test", "future"):
        raise ValueError("predictions are exported only for test and future")
    predicted = np.asarray(predicted, dtype="float64")
    if predicted.shape != (len(part.metadata),) or not np.isfinite(predicted).all():
        raise ValueError("invalid prediction shape or non-finite prediction")
    result = part.metadata.copy()
    result["body_group"] = part.features["primary_body_group"].to_numpy()
    result["weight_category"] = part.features["weight_category"].to_numpy()
    result["volume_category"] = part.features["volume_category"].to_numpy()
    result["dataset_split"] = split
    # Never look up a synthetic future price, even if a caller supplies it.
    actual = part.target[TARGET].to_numpy(dtype="float64") if split == "test" else np.full(len(result), np.nan)
    result["actual_price_rub"] = actual
    result["predicted_price_rub"] = predicted
    result["abs_error_rub"] = np.abs(predicted - actual)
    result["ape_pct"] = result["abs_error_rub"] / actual * 100
    result["model_name"] = "CatBoostRegressor"
    result["model_version"] = model_version
    result["model_run_id"] = run_id
    result["scored_at"] = pd.to_datetime(scored_at, utc=True)
    return result
