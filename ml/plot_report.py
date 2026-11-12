"""Create README figures from a completed training artifact without fitting a model."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = {
    "distance": "Расстояние", "total_weight": "Вес", "total_volume": "Объём",
    "weight_category": "Весовая категория", "volume_category": "Объёмная категория",
    "primary_body_group": "Основной кузов", "allows_tent": "Допуск: тент",
    "allows_refrigerator": "Допуск: рефрижератор", "allows_insulated": "Допуск: изотерм",
    "allows_flatbed": "Допуск: борт", "origin_region": "Регион отправления",
    "destination_region": "Регион назначения", "origin_latitude": "Широта отправления",
    "origin_longitude": "Долгота отправления", "destination_latitude": "Широта назначения",
    "destination_longitude": "Долгота назначения", "month_sin": "Месяц: sin",
    "month_cos": "Месяц: cos", "day_of_year_sin": "День года: sin",
    "day_of_year_cos": "День года: cos", "weekday_sin": "День недели: sin",
    "weekday_cos": "День недели: cos", "weekday": "День недели", "month": "Месяц",
    "day_of_year": "День года", "year": "Год",
}


def create_figures(model_dir: Path, output_dir: Path, seed: int) -> dict:
    report_bytes = (model_dir / "training_report.json").read_bytes()
    report = json.loads(report_bytes)
    sources = {}
    for name in ("metrics.json", "selection_history.csv", "feature_importance.csv",
                 "predictions_test.parquet", "predictions_future.parquet"):
        payload = (model_dir / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != report["files_sha256"][name]:
            raise ValueError(f"artifact checksum mismatch: {name}")
        sources[name] = payload
    metrics = json.loads(sources["metrics.json"])
    history = pd.read_csv(io.BytesIO(sources["selection_history.csv"]))
    importance = pd.read_csv(io.BytesIO(sources["feature_importance.csv"]))
    test = pd.read_parquet(io.BytesIO(sources["predictions_test.parquet"]))
    future = pd.read_parquet(io.BytesIO(sources["predictions_future.parquet"]))
    best = history.loc[history.validation_rmse_rub.idxmin()]
    if int(best.iteration) != report["selected_iterations"] or len(test) != metrics["test"]["rows"]:
        raise ValueError("training history or prediction count differs from the report")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep matplotlib's writable cache outside the published figures and the user profile.
    os.environ.setdefault("MPLCONFIGDIR", str((model_dir / "plot_cache").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})

    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.4), dpi=160)
    axes[0].plot(history.iteration, history.fit_rmse_rub / 1000, label="Fit", color="#2563eb", linewidth=2)
    axes[0].plot(history.iteration, history.validation_rmse_rub / 1000, label="Validation", color="#ea580c", linewidth=2)
    axes[0].axvline(best.iteration, color="#16a34a", linestyle="--")
    axes[0].set(title="Ошибка при добавлении деревьев", xlabel="Число деревьев", ylabel="RMSE, тыс. руб.")
    axes[0].legend(frameon=False)
    zoom = history.loc[history.iteration.ge(max(1, best.iteration - 120))]
    axes[1].plot(zoom.iteration, zoom.validation_rmse_rub, color="#ea580c", linewidth=2)
    axes[1].axvline(best.iteration, color="#16a34a", linestyle="--")
    axes[1].scatter([best.iteration], [best.validation_rmse_rub], color="#16a34a", s=45, zorder=5)
    axes[1].annotate(f"Лучшее число деревьев: {int(best.iteration)}", xy=(best.iteration, best.validation_rmse_rub),
                     xytext=(0.04, 0.86), textcoords="axes fraction",
                     arrowprops={"arrowstyle": "->", "color": "#475569"})
    axes[1].set(title="Validation: область ранней остановки", xlabel="Число деревьев", ylabel="RMSE, руб.")
    for axis in axes:
        axis.grid(alpha=0.18)
    figure.tight_layout(pad=1.8)
    figure.savefig(output_dir / "learning_curve.png", facecolor="white")
    plt.close(figure)

    actual = test.actual_price_rub.to_numpy(dtype="float64")
    predicted = test.predicted_price_rub.to_numpy(dtype="float64")
    indices = np.random.default_rng(seed).choice(len(test), size=min(6000, len(test)), replace=False)
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), dpi=160)
    axes[0].scatter(actual[indices], predicted[indices], s=7, alpha=0.16, color="#2563eb", rasterized=True)
    lower, upper = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    axes[0].plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.5, color="#ea580c", label="Идеальный прогноз")
    if lower > 0:
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
    axes[0].set(title=f"Факт и прогноз: {len(indices):,} точек".replace(",", " "), xlabel="Факт, руб. (log)", ylabel="Прогноз, руб. (log)")
    axes[0].legend(frameon=False)
    ape = np.abs(predicted - actual) / actual * 100
    bins = np.array([0, 5, 10, 15, 20, 30, 50, 100, np.inf])
    counts, _ = np.histogram(ape, bins=bins)
    shares = counts / len(test) * 100
    bars = axes[1].bar(np.arange(len(shares)), shares, color=["#2563eb"] * 7 + ["#ea580c"], width=0.74)
    axes[1].set_xticks(np.arange(len(shares)), ["0–5", "5–10", "10–15", "15–20", "20–30", "30–50", "50–100", "100+"])
    axes[1].tick_params(axis="x", labelrotation=35)
    axes[1].bar_label(bars, labels=[f"{value:.1f}%" for value in shares], padding=3, fontsize=9)
    axes[1].set(title=f"Абсолютная относительная ошибка: все {len(test):,} строк".replace(",", " "),
                xlabel="APE, %", ylabel="Доля перевозок, %", ylim=(0, max(shares) * 1.18))
    axes[0].grid(alpha=0.18)
    axes[1].grid(axis="y", alpha=0.18)
    axes[1].set_axisbelow(True)
    figure.tight_layout(pad=1.8)
    figure.savefig(output_dir / "test_diagnostics.png", facecolor="white")
    plt.close(figure)

    top = importance.sort_values("importance", ascending=False).head(10).sort_values("importance")
    figure, axis = plt.subplots(figsize=(10, 5.1), dpi=160)
    bars = axis.barh([LABELS.get(name, name) for name in top.feature], top.importance, color="#2563eb")
    axis.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
    axis.set(title="Важность признаков: 10 из 28", xlabel="PredictionValuesChange (сумма по всем признакам = 100)",
             xlim=(0, max(top.importance) * 1.18))
    axis.grid(axis="x", alpha=0.18)
    axis.set_axisbelow(True)
    figure.tight_layout(pad=1.8)
    figure.savefig(output_dir / "feature_importance.png", facecolor="white")
    plt.close(figure)

    summary = {
        "as_of_date": report["as_of_date"], "recorded_at": report["created_at"],
        "model_version": report["model_version"], "source_model_run_id": report["run_id"],
        "source_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "train_rows": report["refit_train_rows"], "fit_rows": report["selection_fit_rows"],
        "validation_rows": report["selection_validation_rows"], "test_rows": len(test), "future_rows": len(future),
        "selected_iterations": report["selected_iterations"], "iterations_evaluated": len(history),
        "training_config": report["config"], "versions": report["versions"],
        "metrics": metrics, "plot_seed": seed, "scatter_points": len(indices),
        "feature_importance_type": report["feature_importance_type"],
    }
    (output_dir / "report_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("assets/ml"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = create_figures(args.model_dir, args.output_dir, args.seed)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "model_version": result["model_version"], "test_metrics": result["metrics"]["test"]}))


if __name__ == "__main__":
    main()
