# Electricity Price Forecasting Using Machine Learning

## Overview

This project develops a machine learning-based forecasting system for predicting hourly electricity prices in Germany using historical market prices from neighboring European countries and engineered time-series features.

The workflow includes:

* Data preprocessing and cleaning
* Exploratory Data Analysis (EDA)
* Feature engineering
* Baseline forecasting
* Random Forest modeling
* Hyperparameter tuning
* Walk-forward validation
* Model explainability using SHAP
* XGBoost benchmarking
* Performance evaluation and comparison

The objective is to build an accurate and interpretable forecasting model for short-term electricity price prediction.

---

# Dataset

The dataset contains hourly electricity prices for multiple European countries:

* Germany (Target Variable)
* France
* Belgium
* Italy
* Spain
* United Kingdom

### Dataset Characteristics

* Hourly observations
* One year of data
* 8,760 rows (365 × 24 hours)
* No missing values after preprocessing

---

# Methodology

## 1. Data Preprocessing

The following preprocessing steps were performed:

### Datetime Conversion

Timestamp columns were converted to datetime format and sorted chronologically.

### Frequency Standardization

Data was resampled to an hourly frequency.

### Missing Value Handling

Missing values were handled using:

```python
ffill()
bfill()
```

After preprocessing:

```text
Missing Values = 0
```

---

# Exploratory Data Analysis

Several analyses were performed to understand market behavior.

## Price Distribution

Distribution plots revealed:

* Right-skewed distributions
* High volatility
* Occasional price spikes

These characteristics are typical of electricity markets.

---

## Correlation Analysis

A correlation heatmap was generated to identify relationships between countries.

Key findings:

| Country Pair      | Correlation |
| ----------------- | ----------- |
| Germany – Belgium | ~0.95       |
| Germany – France  | ~0.86       |
| Germany – Italy   | ~0.84       |
| Germany – Spain   | ~0.17       |

Belgium and France showed the strongest relationships with Germany.

---

## Temporal Analysis

Hourly and weekly price patterns were analyzed.

Observed patterns:

* Morning demand peaks
* Evening demand peaks
* Lower weekend prices
* Seasonal variations

---

# Feature Engineering

Several feature groups were created to improve predictive performance.

---

## Time Features

```python
hour
dayofweek
month
is_weekend
```

These features capture seasonality and calendar effects.

---

## Lag Features

```python
germany_lag_1
germany_lag_2
germany_lag_24
germany_lag_168
```

Where:

* Lag 1 = Previous hour
* Lag 2 = Two hours ago
* Lag 24 = Same hour previous day
* Lag 168 = Same hour previous week

---

## Rolling Statistics

```python
germany_roll_mean_24
germany_roll_std_24
```

These capture short-term trends and volatility.

To avoid data leakage:

```python
rolling(...).shift(1)
```

was used.

---

## Cross-Country Features

Lagged values from neighboring countries were added:

```python
france_lag_1
belgium_lag_1
italy_lag_1
spain_lag_1
uk_lag_1
```

These features help capture interconnected European electricity markets.

---

# Baseline Model

A simple persistence model was used as a benchmark.

## Naive Forecast

```text
Prediction(t) = Price(t-24)
```

The previous day's same-hour price was used as the forecast.

This baseline provides a meaningful comparison against machine learning models.

---

# Random Forest Model

A Random Forest Regressor was trained using all engineered features.

## Advantages

* Handles non-linear relationships
* Robust to noise
* Supports feature importance analysis
* Strong performance on tabular datasets

---

# Hyperparameter Tuning

Randomized Search with TimeSeriesSplit was used.

## Search Space

Parameters optimized:

```python
n_estimators
max_depth
min_samples_split
min_samples_leaf
max_features
```

### Best Parameters

```python
{
    'n_estimators': 800,
    'min_samples_split': 2,
    'min_samples_leaf': 2,
    'max_features': 1.0,
    'max_depth': 15
}
```

---

# Walk-Forward Validation

Time-series cross-validation was implemented using:

```python
TimeSeriesSplit
```

This approach ensures:

* No future information leakage
* Realistic forecasting evaluation
* Robust performance estimates

---

# SHAP Explainability

SHAP (SHapley Additive exPlanations) was used to interpret model decisions.

## Most Important Features

| Feature       | Importance |
| ------------- | ---------- |
| germany_lag_1 | 79.0%      |
| belgium       | 17.5%      |
| france        | 1.5%       |
| germany_lag_2 | 0.3%       |
| belgium_lag_1 | 0.3%       |

### Interpretation

The model relies primarily on:

1. Previous German electricity prices
2. Belgian electricity prices
3. French electricity prices

This aligns with correlation analysis and domain knowledge.

---

# Model Performance

## Tuned Random Forest

| Metric | Value  |
| ------ | ------ |
| MAE    | 10.898 |
| RMSE   | 19.021 |
| WAPE   | 5.47%  |

### Performance Interpretation

A WAPE below 6% indicates strong forecasting performance for electricity price prediction.

---

# XGBoost Benchmark

An XGBoost model was trained as a benchmark.

## Results

| Metric | Value  |
| ------ | ------ |
| MAE    | 11.570 |
| RMSE   | 17.708 |
| WAPE   | 5.81%  |

---

## Comparison

| Model                 | MAE    | RMSE   | WAPE  |
| --------------------- | ------ | ------ | ----- |
| Random Forest (Tuned) | 10.898 | 19.021 | 5.47% |
| XGBoost               | 11.570 | 17.708 | 5.81% |

### Conclusion

* Random Forest achieved the lowest MAE and WAPE.
* XGBoost achieved the lowest RMSE.
* Random Forest was selected as the final model due to superior overall forecasting accuracy.

---

# Key Findings

### Germany's Previous Hour Price is the Strongest Predictor

```text
Importance = 79%
```

### Belgium Has Significant Influence

```text
Importance = 17.5%
```

### France Provides Additional Predictive Power

```text
Importance = 1.5%
```

### Time Features Contribute Less Than Expected

Hourly and seasonal information are largely captured indirectly through lagged prices.

---

# Technologies Used

## Programming Language

* Python 3.x

## Libraries

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
```

---

# Future Improvements

Potential enhancements include:

* LightGBM implementation
* Deep learning models (LSTM, GRU, Transformer)
* Holiday and weather features
* Renewable energy production data
* Electricity demand forecasting integration
* Probabilistic forecasting
* Multi-step forecasting horizons

---

# Conclusion

This project demonstrates a complete end-to-end machine learning workflow for electricity price forecasting.

By combining historical market prices, lag-based features, and ensemble learning methods, the final Random Forest model achieved:

* MAE: 10.898
* RMSE: 19.021
* WAPE: 5.47%

The results show that neighboring market prices and recent German price history are the most influential factors in short-term electricity price prediction. The project provides a strong foundation for further research and real-world energy market forecasting applications.

# Author
Rabiya Farheen MSc Data Science, Technical University of Braunschweig, Germany



