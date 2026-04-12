"""
Feature engineering for retail sales forecasting.
Creates time-based, lag, and rolling window features from raw sales data.
"""

import numpy as np
import pandas as pd


def create_date_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Extract calendar features from date column."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["day_of_month"] = df[date_col].dt.day
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df[date_col].dt.is_month_start.astype(int)
    df["is_month_end"] = df[date_col].dt.is_month_end.astype(int)
    df["quarter"] = df[date_col].dt.quarter

    # Cyclical encoding for month and day_of_week
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df


def create_lag_features(
    df: pd.DataFrame,
    target_col: str = "weekly_sales",
    group_cols: list[str] = None,
    lags: list[int] = None,
) -> pd.DataFrame:
    """Create lagged features for time series."""
    df = df.copy()
    if lags is None:
        lags = [1, 2, 4, 8, 12, 26, 52]
    if group_cols is None:
        group_cols = ["store", "dept"]

    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(group_cols)[target_col].shift(lag)

    return df


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str = "weekly_sales",
    group_cols: list[str] = None,
    windows: list[int] = None,
) -> pd.DataFrame:
    """Create rolling statistics features."""
    df = df.copy()
    if windows is None:
        windows = [4, 8, 12, 26]
    if group_cols is None:
        group_cols = ["store", "dept"]

    for w in windows:
        shifted = df.groupby(group_cols)[target_col].shift(1)
        rolling = shifted.rolling(window=w, min_periods=1)
        df[f"rolling_mean_{w}"] = rolling.mean().values
        df[f"rolling_std_{w}"] = rolling.std().values
        df[f"rolling_min_{w}"] = rolling.min().values
        df[f"rolling_max_{w}"] = rolling.max().values

    return df


def create_growth_features(
    df: pd.DataFrame,
    target_col: str = "weekly_sales",
    group_cols: list[str] = None,
) -> pd.DataFrame:
    """Create week-over-week and year-over-year growth rates."""
    df = df.copy()
    if group_cols is None:
        group_cols = ["store", "dept"]

    prev_week = df.groupby(group_cols)[target_col].shift(1)
    prev_year = df.groupby(group_cols)[target_col].shift(52)

    df["wow_growth"] = (df[target_col] - prev_week) / prev_week.abs().clip(lower=1)
    df["yoy_growth"] = (df[target_col] - prev_year) / prev_year.abs().clip(lower=1)

    return df
