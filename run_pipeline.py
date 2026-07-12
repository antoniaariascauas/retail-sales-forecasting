"""
End-to-end retail sales forecasting pipeline.

Loads the Walmart weekly sales data, engineers time-series features, splits by
time (most recent weeks held out for validation), and compares two baselines
against a LightGBM model on RMSE / MAE / MAPE.

Usage:
    python run_pipeline.py --data data/train.csv --val-weeks 12

Expected input columns (Kaggle "Walmart Recruiting - Store Sales Forecasting",
train.csv): Store, Dept, Date, Weekly_Sales[, IsHoliday]. Column names are
normalized case-insensitively, so store/dept/date/weekly_sales also work.
"""

import argparse

import numpy as np
import pandas as pd

from src.features import (
    create_date_features,
    create_growth_features,
    create_lag_features,
    create_rolling_features,
)
from src.models import (
    evaluate_forecast,
    naive_baseline,
    seasonal_baseline,
    train_lightgbm,
)

TARGET = "weekly_sales"
GROUP = ["store", "dept"]
# Columns that must never be fed to the model as features.
NON_FEATURES = {TARGET, "date", "store", "dept", "isholiday"}


def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV and normalize schema to store/dept/date/weekly_sales."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    rename = {"store": "store", "dept": "dept", "date": "date", "weekly_sales": TARGET}
    missing = [c for c in rename if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing required columns {missing}. "
            f"Got columns: {list(df.columns)}"
        )
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(GROUP + ["date"]).reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply calendar, lag, rolling, and growth features on the full panel."""
    df = create_date_features(df, date_col="date")
    df = create_lag_features(df, target_col=TARGET, group_cols=GROUP)
    df = create_rolling_features(df, target_col=TARGET, group_cols=GROUP)
    df = create_growth_features(df, target_col=TARGET, group_cols=GROUP)
    return df


def time_split(df: pd.DataFrame, val_weeks: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the most recent `val_weeks` dates as the validation set."""
    cutoff = df["date"].drop_duplicates().sort_values().iloc[-val_weeks]
    train = df[df["date"] < cutoff].copy()
    val = df[df["date"] >= cutoff].copy()
    return train, val


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric engineered columns, excluding identifiers and the target."""
    return [
        c
        for c in df.columns
        if c not in NON_FEATURES and pd.api.types.is_numeric_dtype(df[c])
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/train.csv", help="Path to train CSV")
    parser.add_argument(
        "--val-weeks", type=int, default=12, help="Most recent weeks held out for validation"
    )
    args = parser.parse_args()

    print(f"Loading {args.data} ...")
    raw = load_data(args.data)
    print(f"  {len(raw):,} rows | {raw['store'].nunique()} stores | {raw['dept'].nunique()} depts")

    print("Engineering features ...")
    feat = build_features(raw)

    train, val = time_split(feat, args.val_weeks)
    print(f"  train: {len(train):,} rows (< {val['date'].min().date()}) | val: {len(val):,} rows")

    y_val = val[TARGET].to_numpy()
    results = []

    # --- Baselines -------------------------------------------------------
    results.append(evaluate_forecast(y_val, naive_baseline(train, val), "Naive (last value)"))
    results.append(evaluate_forecast(y_val, seasonal_baseline(train, val), "Seasonal (same week last year)"))

    # --- LightGBM --------------------------------------------------------
    cols = feature_columns(feat)
    print(f"Training LightGBM on {len(cols)} features ...")
    model, val_pred = train_lightgbm(
        train, train[TARGET].to_numpy(), val, y_val, feature_cols=cols
    )
    results.append(evaluate_forecast(y_val, val_pred, "LightGBM"))

    # --- Report ----------------------------------------------------------
    report = pd.DataFrame(results)[["model", "rmse", "mae", "mape"]]
    print("\n=== Validation results (lower is better) ===")
    print(report.to_string(index=False))

    best = report.loc[report["rmse"].idxmin(), "model"]
    print(f"\nBest by RMSE: {best}")


if __name__ == "__main__":
    main()
