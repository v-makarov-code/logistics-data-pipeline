"""Train CatBoost using inner chronological validation; evaluate and save file artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from ml.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, TARGET
from ml.prepare_dataset import sha256_file
from ml.training import (
    TrainingConfig, load_dataset, make_pool, prediction_rows, regression_metrics, split_inner_train,
)

LOGGER = logging.getLogger("ml.training")


def log_event(event: str, **fields: object) -> None:
    LOGGER.info(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(),
                            "event": event, **fields}, ensure_ascii=False, default=str))


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def train_and_export(dataset_dir: Path, output_root: Path, config: TrainingConfig) -> tuple[Path, dict]:
    parts, dataset_manifest = load_dataset(dataset_dir)
    run_id = str(uuid4())
    run_dir = output_root / f"model-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    train = parts["train"]
    fit, validation, validation_start = split_inner_train(
        train, dataset_manifest["test_start_inclusive"], config.validation_days,
    )
    log_event("training_started", run_id=run_id, dataset_dir=str(dataset_dir.resolve()), config=asdict(config))
    log_event("selection_started", run_id=run_id, fit_rows=int(fit.sum()),
              validation_rows=int(validation.sum()), validation_start=validation_start)
    selection_model = CatBoostRegressor(**config.model_params())
    selection_model.fit(
        make_pool(train, fit, with_target=True),
        eval_set=make_pool(train, validation, with_target=True),
        early_stopping_rounds=config.early_stopping_rounds, use_best_model=True, verbose=False,
    )
    # CatBoost returns a zero-based iteration index; iterations is a tree count.
    selected_iterations = selection_model.get_best_iteration() + 1
    if selected_iterations < 1:
        raise RuntimeError("CatBoost did not select a valid best iteration")
    validation_predictions = selection_model.predict(make_pool(train, validation))
    validation_metrics = regression_metrics(train.target.loc[validation, TARGET].to_numpy(), validation_predictions)
    history = selection_model.get_evals_result()
    pd.DataFrame({
        "iteration": np.arange(1, len(history["learn"]["RMSE"]) + 1),
        "fit_rmse_rub": history["learn"]["RMSE"],
        "validation_rmse_rub": history["validation"]["RMSE"],
    }).to_csv(run_dir / "selection_history.csv", index=False)
    log_event("selection_succeeded", run_id=run_id, selected_iterations=selected_iterations,
              validation_metrics=validation_metrics)

    final_params = config.model_params(iterations=selected_iterations)
    code_hashes = {name: sha256_file(Path(__file__).with_name(name)) for name in ("train_model.py", "training.py", "features.py")}
    identity = {
        "train_files": {name: digest for name, digest in dataset_manifest["files_sha256"].items() if name.startswith("train/")},
        "params": final_params, "feature_columns": list(FEATURE_COLUMNS),
        "catboost_version": catboost.__version__, "code_sha256": code_hashes,
    }
    # Identity describes training inputs and recipe; it never depends on test labels.
    model_version = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    log_event("refit_started", run_id=run_id, rows=len(train.features), model_version=model_version)
    final_model = CatBoostRegressor(**final_params)
    final_model.fit(make_pool(train, with_target=True), use_best_model=False, verbose=False)
    model_path = run_dir / "model.cbm"
    final_model.save_model(str(model_path), format="cbm")

    restored_model = CatBoostRegressor()
    restored_model.load_model(str(model_path), format="cbm")
    test_pool = make_pool(parts["test"])
    test_predictions = restored_model.predict(test_pool)
    if not np.allclose(test_predictions, final_model.predict(test_pool), rtol=0, atol=1e-8):
        raise RuntimeError("saved model predictions differ after loading")
    test_metrics = regression_metrics(parts["test"].target[TARGET].to_numpy(), test_predictions)
    baseline_price = float(train.target[TARGET].mean())
    baseline_metrics = regression_metrics(
        parts["test"].target[TARGET].to_numpy(), np.full(len(parts["test"].features), baseline_price),
    )
    metrics = {
        "selection_validation": validation_metrics,
        "test": test_metrics,
        "test_baseline_train_mean": baseline_metrics,
        "baseline_price_rub": baseline_price,
    }
    write_json(run_dir / "metrics.json", metrics)
    log_event("test_evaluated", run_id=run_id, model_version=model_version,
              metrics=test_metrics, baseline_metrics=baseline_metrics)

    scored_at = datetime.now(timezone.utc).isoformat()
    nonpositive_predictions = {}
    for name in ("test", "future"):
        part = parts[name]
        predicted = test_predictions if name == "test" else (
            restored_model.predict(make_pool(part)) if len(part.features) else np.array([], dtype="float64")
        )
        rows = prediction_rows(part, predicted, name, model_version, run_id, scored_at)
        rows.to_parquet(run_dir / f"predictions_{name}.parquet", engine="pyarrow", compression="zstd", index=False)
        rows.to_csv(run_dir / f"predictions_{name}.csv", index=False, encoding="utf-8-sig")
        nonpositive_predictions[name] = int((predicted <= 0).sum())
        log_event("predictions_saved", run_id=run_id, partition=name, rows=len(rows),
                  nonpositive_predictions=nonpositive_predictions[name])
    importance = restored_model.get_feature_importance(
        data=make_pool(train, with_target=True), type="PredictionValuesChange",
    )
    if len(importance) != len(FEATURE_COLUMNS) or not np.isfinite(importance).all():
        raise RuntimeError("feature importance is missing or non-finite")
    pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": importance,
    }).sort_values("importance", ascending=False).to_csv(run_dir / "feature_importance.csv", index=False)

    report = {
        "run_id": run_id, "model_version": model_version, "model_name": "CatBoostRegressor",
        "created_at": scored_at, "as_of_date": dataset_manifest["as_of_date"],
        "dataset_run_id": dataset_manifest["run_id"],
        "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        "input_files_sha256": dataset_manifest["files_sha256"],
        "config": asdict(config), "selected_iterations": selected_iterations,
        "selection_iterations_evaluated": len(history["learn"]["RMSE"]),
        "selection_fit_rows": int(fit.sum()), "selection_validation_rows": int(validation.sum()),
        "selection_validation_start_inclusive": validation_start,
        "selection_validation_end_exclusive": dataset_manifest["test_start_inclusive"],
        "refit_train_rows": len(train.features), "final_model_params": restored_model.get_all_params(),
        "feature_columns": list(FEATURE_COLUMNS), "categorical_features": list(CATEGORICAL_FEATURES),
        "target_transform": "none; price_rub in RUB", "prediction_clipping": "none",
        "feature_importance_type": "PredictionValuesChange",
        "test_used_for_selection": False, "saved_model_roundtrip_verified": True,
        "nonpositive_predictions": nonpositive_predictions,
        "constant_train_features": dataset_manifest["constant_train_features"],
        "code_sha256": code_hashes,
        "versions": {"python": platform.python_version(), "catboost": catboost.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "limitations": dataset_manifest["limitations"] + [
            "Validation metrics describe the selection model; test metrics describe the full-train refit.",
            "Metrics on synthetic data do not establish production accuracy or future-period accuracy.",
            "Seed, CPU parameters and versions are fixed; cross-platform bitwise identity is not guaranteed.",
            "Model version identifies inputs and recipe; native model files may contain different run timestamps.",
        ],
        "files_sha256": {path.name: sha256_file(path) for path in sorted(run_dir.iterdir()) if path.is_file()},
    }
    # Completion marker: a run without a final report must not be consumed.
    write_json(run_dir / "training_report.json", report)
    log_event("training_succeeded", run_id=run_id, model_version=model_version, output_dir=str(run_dir.resolve()))
    return run_dir, metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="complete dataset directory containing manifest.json")
    parser.add_argument("--output-dir", type=Path, default=Path("ml/artifacts"))
    defaults = TrainingConfig()
    for name in ("iterations", "depth", "random_seed", "thread_count", "validation_days", "early_stopping_rounds"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    try:
        config = TrainingConfig(**{key: getattr(args, key) for key in asdict(defaults)})
        train_and_export(args.dataset_dir, args.output_dir, config)
    except Exception:
        LOGGER.exception(json.dumps({"event": "training_failed"}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
