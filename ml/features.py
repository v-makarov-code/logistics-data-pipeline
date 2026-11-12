"""Pure feature transformations: no database access, fitting or target statistics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

CONTRACT_VERSION = 1
BODY_GROUPS = ("TENT", "REFRIGERATOR", "INSULATED", "FLATBED", "OTHER", "UNSPECIFIED")
CATEGORICAL_FEATURES = (
    "origin_region", "destination_region", "primary_body_group",
    "weight_category", "volume_category",
)
PHYSICAL_FEATURES = (
    "distance", "total_weight", "total_volume", "origin_latitude", "origin_longitude",
    "destination_latitude", "destination_longitude",
)
BODY_FEATURES = tuple(f"allows_{group.lower()}" for group in BODY_GROUPS)
CALENDAR_FEATURES = (
    "year", "month", "weekday", "day_of_year", "month_sin", "month_cos",
    "weekday_sin", "weekday_cos", "day_of_year_sin", "day_of_year_cos",
)
# An allowlist is intentional: adding a SQL column never adds a model feature.
FEATURE_COLUMNS = PHYSICAL_FEATURES + CATEGORICAL_FEATURES + BODY_FEATURES + CALENDAR_FEATURES
METADATA_COLUMNS = (
    "source_price_id", "loading_date", "origin_city_gcid", "destination_city_gcid",
    "origin_city_row_id", "destination_city_row_id", "origin_city_name", "destination_city_name",
    "origin_full_name", "destination_full_name",
)
RAW_COLUMNS = METADATA_COLUMNS + PHYSICAL_FEATURES + (
    "weight_category", "volume_category", "primary_body_group", "allowed_body_groups", "target_price_rub",
)
TARGET = "target_price_rub"


@dataclass(frozen=True)
class DatasetConfig:
    as_of_date: date = date(2026, 10, 26)

    @property
    def train_start(self) -> date:
        return (pd.Timestamp(self.as_of_date) - pd.DateOffset(months=3)).date()

    @property
    def test_start(self) -> date:
        return self.as_of_date - timedelta(days=21)


@dataclass
class DatasetPart:
    features: pd.DataFrame
    target: pd.DataFrame
    metadata: pd.DataFrame


def extract_region(value: object) -> str:
    """Keep everything after the first comma; never infer a region from a city name."""
    if pd.isna(value):
        return "UNKNOWN"
    _, separator, region = str(value).partition(",")
    return region.strip() if separator and region.strip() else "UNKNOWN"


def category_values(series: pd.Series) -> pd.Series:
    return series.fillna("UNKNOWN").astype(str).str.strip().replace("", "UNKNOWN")


def prepare_features(raw: pd.DataFrame, config: DatasetConfig) -> dict[str, DatasetPart]:
    """Validate the extraction contract and split by loading date, preserving row alignment."""
    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"missing extraction columns: {sorted(missing)}")
    data = raw.copy()
    if data["source_price_id"].isna().any() or data["source_price_id"].duplicated().any():
        raise ValueError("expected exactly one row per non-null source_price_id")
    data["source_price_id"] = data["source_price_id"].astype("int64")
    data["loading_date"] = pd.to_datetime(data["loading_date"], errors="raise")
    dates = data["loading_date"]
    if dates.isna().any() or (dates != dates.dt.normalize()).any():
        raise ValueError("loading_date must be a non-null calendar date")
    if (dates < pd.Timestamp(config.train_start)).any():
        raise ValueError("extraction contains rows before train_start")
    for column in ("origin_city_row_id", "destination_city_row_id"):
        if data[column].isna().any():
            raise ValueError(f"missing city reference: {column}; refresh Core and source references")
    for column in ("source_price_id", "origin_city_gcid", "destination_city_gcid",
                   "origin_city_row_id", "destination_city_row_id"):
        data[column] = data[column].astype("int64")
    data = data.sort_values(["loading_date", "source_price_id"]).reset_index(drop=True)
    dates = data["loading_date"]
    future = dates > pd.Timestamp(config.as_of_date)
    # A second boundary protects callers passing a DataFrame without SQL masking.
    data.loc[future, TARGET] = np.nan
    target = pd.to_numeric(data[TARGET], errors="raise").astype("float64")
    history_target = target[~future]
    if (~np.isfinite(history_target) | (history_target <= 0)).any():
        raise ValueError("historical targets must be finite and positive")

    features = pd.DataFrame(index=data.index)
    for column in PHYSICAL_FEATURES:
        features[column] = pd.to_numeric(data[column], errors="raise").astype("float64")
    for column in ("distance", "total_weight", "total_volume"):
        values = features[column]
        if (~np.isfinite(values) | (values <= 0 if column == "distance" else values < 0)).any():
            raise ValueError(f"invalid physical feature: {column}")
    # Missing/bad coordinates remain missing; no statistics are learned from test/future.
    for side in ("origin", "destination"):
        for coordinate, limit in (("latitude", 90), ("longitude", 180)):
            column = f"{side}_{coordinate}"
            values = features[column]
            features[column] = values.where(np.isfinite(values) & values.between(-limit, limit))
        features[f"{side}_region"] = data[f"{side}_full_name"].map(extract_region).astype(str)
    for column in ("primary_body_group", "weight_category", "volume_category"):
        features[column] = category_values(data[column])
    if not features["primary_body_group"].isin(BODY_GROUPS).all():
        raise ValueError("unexpected primary_body_group")
    for groups in data["allowed_body_groups"]:
        if not isinstance(groups, (list, tuple, np.ndarray)) or len(groups) == 0:
            raise ValueError("each price needs a non-empty list of compatible body groups")
        if not set(groups).issubset(BODY_GROUPS):
            raise ValueError(f"unexpected compatible body groups: {groups}")
    for group, column in zip(BODY_GROUPS, BODY_FEATURES):
        features[column] = data["allowed_body_groups"].map(lambda groups: group in groups).astype("uint8")
    features["year"] = dates.dt.year.astype("int16")
    features["month"] = dates.dt.month.astype("uint8")
    features["weekday"] = dates.dt.day_of_week.astype("uint8")  # Monday = 0.
    features["day_of_year"] = dates.dt.day_of_year.astype("uint16")
    for name, offset, period in (("month", 1, 12), ("weekday", 0, 7), ("day_of_year", 1, 365.25)):
        angle = 2 * np.pi * (features[name].astype("float64") - offset) / period
        features[f"{name}_sin"] = np.sin(angle)
        features[f"{name}_cos"] = np.cos(angle)
    features = features.loc[:, FEATURE_COLUMNS]
    masks = {
        "train": dates < pd.Timestamp(config.test_start),
        "test": (dates >= pd.Timestamp(config.test_start)) & ~future,
        "future": future,
    }
    return {
        name: DatasetPart(
            features.loc[mask].reset_index(drop=True),
            target.loc[mask].to_frame(TARGET).reset_index(drop=True),
            data.loc[mask, METADATA_COLUMNS].reset_index(drop=True),
        )
        for name, mask in masks.items()
    }
