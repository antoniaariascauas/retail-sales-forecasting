# Retail Sales Forecasting

Multi-model forecasting pipeline for weekly store-department sales. Compares baseline methods, Prophet, and LightGBM with engineered time series features, evaluated with business-relevant metrics.

## Problem

A retailer needs to forecast weekly sales by store and department to optimize inventory, staffing, and markdown decisions. Accurate forecasts reduce overstock waste and stockout losses — even a 1% MAPE improvement across thousands of SKUs translates to significant savings.

**Dataset:** [Walmart Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting) — 45 stores, 99 departments, ~420K weekly observations over 2.5 years.

## Approach

### Feature Engineering (`src/features.py`)
- **Calendar features**: Day of week, month, week of year, holiday flags, with cyclical encoding
- **Lag features**: 1, 2, 4, 8, 12, 26, 52-week lags per store-department
- **Rolling statistics**: Mean, std, min, max over 4/8/12/26-week windows
- **Growth rates**: Week-over-week and year-over-year growth

### Models Compared

| Model | Description |
|-------|-------------|
| Naive baseline | Last observed value (random walk) |
| Seasonal baseline | Same-week-last-year average |
| Prophet | Facebook's decomposition model with holidays |
| LightGBM | Gradient boosting with engineered features |

### Evaluation
- **RMSE** — Penalizes large errors (important for high-volume stores)
- **MAE** — Average absolute error in dollars
- **MAPE** — Percentage error for cross-store comparison

## Project Structure

```
├── README.md
├── requirements.txt
├── notebooks/
│   └── 01_eda.ipynb              # Exploratory data analysis
├── src/
│   ├── features.py               # Time series feature engineering
│   └── models.py                 # Baseline, Prophet, LightGBM implementations
└── data/                         # Download from Kaggle
```

## How to Run

```bash
pip install -r requirements.txt

# Download data from Kaggle and place in data/
# Run EDA
jupyter notebook notebooks/01_eda.ipynb
```

## Key Design Decisions

1. **Baselines first** — Before using complex models, establish what "good" means with naive and seasonal baselines. If LightGBM can't beat same-week-last-year, the extra complexity isn't justified.

2. **Store-department granularity** — Rather than one global model, features are computed at the store-department level. Sales patterns for electronics differ fundamentally from groceries.

3. **Lag-based features over decomposition** — For LightGBM, explicit lags and rolling windows outperform STL decomposition features because the model can learn non-linear seasonal interactions (e.g., holiday + department type).

## Tech Stack

Python, LightGBM, Prophet, scikit-learn, pandas, matplotlib, seaborn
