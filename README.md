# EU Day-Ahead Electricity Price Forecasting

End-to-end time series project: data cleaning → EDA → feature engineering → model training → evaluation → BI dashboard.
Forecasts Germany's hourly day-ahead electricity price for 2022 using its own price history and four
neighboring markets (France, Italy, Belgium, Spain), validated with walk-forward cross-validation rather
than a single train/test split.



## Why this dataset is a real test, not a toy problem

2022 was the year of the European energy crisis — Germany's price roughly **quadrupled** from spring lows
(~€100/MWh) to a late-August peak (~€800+/MWh), then collapsed back down by year-end. Any model has to cope
with a genuine regime shift mid-year, not a stable, easy signal.



## Results

| Model | Holdout MAE | Holdout RMSE | Holdout WAPE | Walk-forward CV MAE (mean ± std) |
|---|---|---|---|---|
| Naive (t-24) | 45.15 | 61.01 | 22.68% | — |
| **Linear Regression** | **14.34** | **20.09** | **7.21%** | **19.34 ± 5.74** |
| Random Forest | 16.51 | 23.34 | 8.30% | 23.50 ± 12.86 |
| XGBoost | 17.39 | 23.92 | 8.73% | 26.65 ± 20.31 |

All three models beat the naive baseline by a wide margin. **Linear Regression wins on both accuracy and
stability** across folds — a useful reminder that more complex models aren't automatically better, especially
for a strongly autoregressive series like this one, and that conclusion only became visible by testing with
walk-forward CV instead of trusting a single split.


## Two correctness bugs this project explicitly fixes (and shows the fix for)

An earlier draft of this pipeline had two mistakes that are common enough to be worth calling out directly,
with evidence, rather than glossing over:

1. **Target leakage in rolling features.** `pandas.rolling()` is right-inclusive by default, so
   `df[target].rolling(24).mean()` at time *t* includes the value being predicted. The fix is a `.shift(1)`
   before `.rolling()`. The notebook (§4) measures the leak's effect directly — both as a feature-level
   correlation gap and as an end-to-end holdout MAE comparison — and reports the honest result: the leak is
   a real correctness issue, concentrated in the most volatile periods, but its measured impact on *this*
   model is small because a single dominant lag feature already captures most of the signal. The fix is
   kept regardless of measured size, because a feature that can see its own target is not a performance
   trade-off to weigh — it's a bug that happens to look fine in a backtest.
2. **MAPE on a series that crosses zero.** Electricity prices can be at, near, or below zero (real, due to
   renewable oversupply), so dividing by `y_true` explodes — the original version of this project reported
   MAPE values over 3,000,000%. Replaced with MAE, RMSE, and WAPE, none of which break down near zero.


## What's covered, end to end

- **Data cleaning**: timestamp parsing from a raw string range, EU DST duplicate-hour resolution, reindexing
  onto a complete hourly grid, short-gap interpolation vs. long-gap imputation (with an explicit flag column
  for the latter — UK is missing 1,442 hours across 9 separate blocks, including all of July), and outlier
  capping (France has a 2-hour, ~3,000 EUR/MWh spike, ~3.5x its own 99.9th percentile).
- **EDA**: per-country distributions, a cross-country correlation heatmap (Spain stands out as decoupled —
  consistent with the EU gas-price-cap mechanism it had for part of 2022), a missingness calendar, an
  annotated time series with the energy-crisis period marked, hour/day/month seasonality, and ACF/PACF
  plots used to justify the lag choices in feature engineering rather than picking them arbitrarily.
- **Feature engineering**: cyclical time encodings, lags chosen from the ACF/PACF evidence (1, 2, 3, 24, 48,
  168h), leak-free rolling statistics, and lagged cross-country features (UK deliberately excluded — see
  the cleaning section for why).
- **Modeling**: Naive baseline, Linear Regression, Random Forest, XGBoost.
- **Evaluation**: a chronological holdout *and* 5-fold walk-forward cross-validation, residual diagnostics
  (over time and distribution), and feature importance (standardized LR coefficients alongside RF
  importances, so they're actually comparable).
- **Dashboard**: five purpose-built CSV exports plus a chart-by-chart Tableau/Power BI build guide.

## Limitations & next steps

- Single year of data — no validation across multiple yearly cycles or crisis/non-crisis regimes.
- No exogenous fundamentals (gas prices, wind/solar generation, demand forecasts) — the real economic
  drivers of electricity prices aren't in this dataset; cross-country lags act as an indirect proxy.
- UK excluded from modeling due to a genuine ~16% data gap; a production version would need a real UK feed.
- Natural next steps: add weather/generation-mix features, try a quantile-regression or SARIMAX model to get
  uncertainty bands (residual variance clearly scales with price level — see §7), and extend to multi-country,
  multi-horizon forecasting using the cross-country correlation structure already confirmed in the EDA.

# Author
Rabiya Farheen
Master’s in Data Science (Germany)
