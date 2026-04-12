"""
Forecasting models for retail sales prediction.
Implements baseline, Prophet, and LightGBM approaches.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb


def naive_baseline(train: pd.DataFrame, test: pd.DataFrame, target_col: str = "weekly_sales") -> np.ndarray:
    """Baseline: predict last known value (random walk)."""
    last_values = train.groupby(["store", "dept"])[target_col].last()
    preds = test.set_index(["store", "dept"]).index.map(
        lambda x: last_values.get(x, train[target_col].mean())
    )
    return np.array(preds)


def seasonal_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "weekly_sales",
) -> np.ndarray:
    """Baseline: predict same-week-last-year average."""
    weekly_avg = train.groupby(["store", "dept", "week_of_year"])[target_col].mean()
    preds = []
    for _, row in test.iterrows():
        key = (row["store"], row["dept"], row["week_of_year"])
        if key in weekly_avg.index:
            preds.append(weekly_avg[key])
        else:
            preds.append(train[target_col].mean())
    return np.array(preds)


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    feature_cols: list[str],
) -> tuple:
    """Train LightGBM regressor with early stopping."""
    train_data = lgb.Dataset(X_train[feature_cols], label=y_train)
    val_data = lgb.Dataset(X_val[feature_cols], label=y_val, reference=train_data)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 30,
        "verbose": -1,
        "random_state": 42,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    return model, model.predict(X_val[feature_cols])


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "") -> dict:
    """Compute forecasting metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # MAPE with protection against zero sales
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    return {
        "model": model_name,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2) if not np.isnan(mape) else None,
    }
