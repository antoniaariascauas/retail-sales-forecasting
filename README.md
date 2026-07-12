# Retail Sales Forecasting

Forecasting pipeline for weekly store-department sales. Establishes baseline methods, then compares them against a LightGBM model trained on engineered time-series features, evaluated with business-relevant metrics.

## Problem

A retailer needs to forecast weekly sales by store and department to optimize inventory, staffing, and markdown decisions. Accurate forecasts reduce overstock waste and stockout losses across thousands of SKUs.

**Dataset:** [Walmart Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting) — 45 stores, 99 departments, ~420K weekly observations over ~2.5 years.

## Approach

### Feature Engineering (`src/features.py`)
- **Calendar features**: year, month, quarter, day of week, week of year, weekend / month-start / month-end flags, plus cyclical (sin/cos) encoding of month and day of week
- **Lag features**: 1, 2, 4, 8, 12, 26, 52-week lags per store-department
- **Rolling statistics**: mean, std, min, max over 4/8/12/26-week windows (shifted to avoid leakage)
- **Growth rates**: week-over-week and year-over-year growth

### Models (`src/models.py`)

| Model | Description |
|-------|-------------|
| Naive baseline | Last observed value per store-department (random walk) |
| Seasonal baseline | Same-week-last-year average |
| LightGBM | Gradient boosting on engineered features, with early stopping |

Baselines are established first: if LightGBM cannot beat same-week-last-year, the added complexity is not justified. (Prophet is included in `requirements.txt` as an option for extending the comparison.)

### Evaluation (`evaluate_forecast`)
- **RMSE** — penalizes large errors (important for high-volume stores)
- **MAE** — average absolute error in dollars
- **MAPE** — percentage error for cross-store comparison (guarded against zero sales)

## Exploratory Analysis

`notebooks/01_eda.ipynb` explores overall sales trend, monthly and holiday seasonality (holiday weeks show a measurable sales uplift), and store-level heterogeneity — motivating the calendar, lag, and rolling features above.

## Project Structure

```
├── README.md
├── requirements.txt
├── run_pipeline.py         # End-to-end runner: features → baselines + LightGBM → comparison
├── notebooks/
│   └── 01_eda.ipynb        # Exploratory data analysis
├── src/
│   ├── features.py         # Time-series feature engineering
│   └── models.py           # Baseline + LightGBM models and metrics
└── data/                   # Download from Kaggle (not tracked)
```

## How to Run

```bash
pip install -r requirements.txt

# Download the Walmart dataset from Kaggle and place train.csv in data/
# https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting

# Run the full pipeline: engineers features, holds out the most recent
# weeks, and compares the two baselines against LightGBM on RMSE/MAE/MAPE
python run_pipeline.py --data data/train.csv --val-weeks 12
```

`run_pipeline.py` ties `src/features.py` and `src/models.py` into a single
time-aware train/validation run and prints a model comparison table. The
`notebooks/01_eda.ipynb` notebook covers the exploratory analysis that motivates
the engineered features.

## Key Design Decisions

1. **Baselines first** — quantify what "good" means with naive and seasonal baselines before adding model complexity.
2. **Store-department granularity** — features are computed per store-department, since sales patterns differ fundamentally across departments.
3. **Lag-based features** — explicit lags and rolling windows let a gradient-boosting model learn non-linear seasonal interactions (e.g. holiday × department type).

## Tech Stack

Python, LightGBM, scikit-learn, pandas, NumPy, matplotlib, seaborn
