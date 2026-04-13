# French Champagne Sales Forecasting

End-to-end time series forecasting project on monthly French champagne sales (Perrin Frères, January 1964 – September 1972). Three model families are developed and compared — ARIMA, SARIMA, and Holt-Winters — each following a consistent pipeline: manual proof-of-concept → grid search → residual analysis → bias correction → held-out validation.

---

## Dataset

| File | Description | Observations |
|---|---|---|
| `data/champagne.csv` | Full raw series | 105 |
| `data/dataset.csv` | Training set (Jan 1964 – Sep 1971) | 93 |
| `data/validation.csv` | Held-out validation set (Oct 1971 – Sep 1972) | 12 |
| `data/stationary.csv` | Seasonally differenced series (lag-12) | 81 |

The original data is sourced from Jason Brownlee's dataset repository: [github.com/jbrownlee/Datasets](https://github.com/jbrownlee/Datasets/tree/master).

The validation set is split before any modelling or analysis to prevent data leakage. It is never examined during development and is used only for final evaluation.

---

## Project Structure

```
.
├── data/
│   ├── champagne.csv
│   ├── dataset.csv
│   ├── stationary.csv
│   └── validation.csv
├── models/                        # Serialised model artefacts (produced at runtime)
│   ├── model.pkl                  # Final ARIMA(1,0,0) ARIMAResults object
│   ├── model_bias.npy             # ARIMA bias correction constant
│   ├── sarima_model.pkl           # Final SARIMA(2,0,1)(1,1,0,12) SARIMAXResults object
│   └── sarima_bias.npy            # SARIMA bias correction constant
├── validation_dataset.ipynb       # Train / validation split
├── eda.ipynb                      # Exploratory data analysis
├── stationary_dataset.ipynb       # Stationarity testing and differenced series
├── persistence_model.ipynb        # Naïve baseline
├── manual_arima.ipynb             # ARIMA proof-of-concept
├── grid_arima.ipynb               # ARIMA grid search
├── arima_residuals.ipynb          # ARIMA residual analysis
├── bias_corrected_arima.ipynb     # Bias-corrected ARIMA + validation
├── manual_sarima.ipynb            # SARIMA proof-of-concept
├── grid_sarima.ipynb              # SARIMA grid search
├── sarima_residuals.ipynb         # SARIMA residual analysis
├── bias_corrected_sarima.ipynb    # Bias-corrected SARIMA + validation
├── manual_hw.ipynb                # Holt-Winters proof-of-concept
├── grid_hw.ipynb                  # Holt-Winters grid search
└── bias_corrected_hw.ipynb        # Bias-corrected Holt-Winters + validation
```

---

## Pipeline

### 1. Data Preparation — `validation_dataset.ipynb`
Splits `champagne.csv` into training (93 obs) and held-out validation (12 obs, one full year) sets. Must be run before any other notebook.

### 2. Exploratory Data Analysis — `eda.ipynb`
Examines the raw training series for trend, seasonality, and variance behaviour. Produces a line plot, descriptive statistics, and annual box plots across 1964–1970.

### 3. Stationarity Analysis — `stationary_dataset.ipynb`
Tests the raw series for stationarity using the Augmented Dickey-Fuller (ADF) test, applies seasonal differencing (lag-12) to remove the dominant seasonal component, confirms stationarity on the differenced series, and plots ACF/PACF to inform ARIMA parameter selection. Saves `data/stationary.csv`.

### 4. Persistence Baseline — `persistence_model.ipynb`
Establishes a naïve walk-forward baseline: the forecast at each step is the previous observed value. Any candidate model must beat this to demonstrate genuine predictive skill.

**Baseline RMSE: 3186.501**

---

## Models

Each model family follows the same four-stage process.

### ARIMA

| Stage | Notebook | Detail |
|---|---|---|
| Manual | `manual_arima.ipynb` | ARIMA(1,1,1) with manual lag-12 differencing; RMSE 961.619 |
| Grid search | `grid_arima.ipynb` | Exhaustive search over p ∈ [0,6], d ∈ [0,2], q ∈ [0,6] |
| Residual analysis | `arima_residuals.ipynb` | Residual statistics and distribution plots for ARIMA(1,0,0) |
| Bias correction + validation | `bias_corrected_arima.ipynb` | Bias estimated from training residuals and applied to all forecasts |

**Grid search best (converged): ARIMA(1,0,0) — Training RMSE 945.107**

A critical convergence guard is applied in the grid search: statsmodels raises a `ConvergenceWarning` rather than an exception when L-BFGS-B optimisation fails, so non-converged fits pass silently through a bare `except` block. The `mle_retvals['converged']` flag is checked explicitly after every fit; any non-converged configuration is skipped and excluded from evaluation.

**Final validation RMSE (bias-corrected): 390.870**

---

### SARIMA

| Stage | Notebook | Detail |
|---|---|---|
| Manual | `manual_sarima.ipynb` | SARIMA(1,0,0)(1,1,0,12); raw series passed directly to SARIMAX |
| Grid search | `grid_sarima.ipynb` | Two-pass search: broad unrestricted grid, then D fixed to 1 |
| Residual analysis | `sarima_residuals.ipynb` | Two candidates evaluated: SARIMA(1,0,0)(2,0,0,12) and SARIMA(2,0,1)(1,1,0,12) |
| Bias correction + validation | `bias_corrected_sarima.ipynb` | Both candidates bias-corrected and evaluated on validation set |

**Grid search best (D=1 pass): SARIMA(2,0,1)(1,1,0,12) — Training RMSE 902.148**

Unlike the ARIMA pipeline, no manual pre-differencing is required — `SARIMAX` handles seasonal integration internally via `D=1`. The second grid search pass fixes `D=1` given the strong seasonal structure present in the data.

**Final validation RMSE (bias-corrected, SARIMA(2,0,1)(1,1,0,12)): 722.929**

---

### Holt-Winters Exponential Smoothing

| Stage | Notebook | Detail |
|---|---|---|
| Manual | `manual_hw.ipynb` | Single configuration using `statsmodels ExponentialSmoothing` |
| Grid search | `grid_hw.ipynb` | All combinations of trend ∈ {add, mul}, damped ∈ {True, False}, seasonal ∈ {add, mul} |
| Bias correction + validation | `bias_corrected_hw.ipynb` | Bias correction applied; convergence checked via smoothing parameter bounds |

`ExponentialSmoothing` does not expose `mle_retvals` like ARIMA. Convergence is validated empirically by checking that all estimated smoothing parameters are finite and within [0, 1].

**Grid search best: HW(trend=add, damped=False, seasonal=mul) — Training RMSE 840.194**

**Final validation RMSE (bias-corrected): 358.853**

---

## Results Summary

| Model | Training RMSE | Validation RMSE |
|---|---|---|
| Persistence (baseline) | 3186.501 | — |
| ARIMA(1,0,0) | 945.107 | 390.870 |
| SARIMA(2,0,1)(1,1,0,12) | 902.148 | 722.929 |
| **HW(trend=add, damped=False, seasonal=mul)** | **840.194** | **358.853** |

All three models substantially outperform the naïve baseline. **Holt-Winters achieves the lowest validation RMSE** and is the recommended model for this series.

---

## Dependencies

```
pandas
numpy
matplotlib
statsmodels
scikit-learn
```

Install with:

```bash
pip install pandas numpy matplotlib statsmodels scikit-learn
```

**Python version:** 3.14.3

---

## Execution Order

Notebooks must be run in the following order due to data dependencies:

```
1. validation_dataset.ipynb
2. eda.ipynb
3. stationary_dataset.ipynb
4. persistence_model.ipynb
5. manual_arima.ipynb
6. grid_arima.ipynb
7. arima_residuals.ipynb
8. bias_corrected_arima.ipynb
9. manual_sarima.ipynb
10. grid_sarima.ipynb
11. sarima_residuals.ipynb
12. bias_corrected_sarima.ipynb
13. manual_hw.ipynb
14. grid_hw.ipynb
15. bias_corrected_hw.ipynb
```
