"""
Hybrid SARIMAX + LSTM CO2 emissions forecasting pipeline.

This module is a direct refactor of the FYP notebooks
(02_sarimax_model.ipynb, 03_lstm_hybrid_model.ipynb) from:
https://github.com/alishbanaaz/A-Hybrid-Deep-Learning-Approach-For-Modelling-Global-Carbon-Emission

Same methodology, same default hyperparameters — just wrapped as functions so
a FastAPI service can call them on demand for any country, and on any dataset
the user supplies (as long as it has the required columns).
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Required schema — any dataset uploaded by a user must contain these columns
# --------------------------------------------------------------------------
TARGET_COL = "Value_co2_emissions_kt_by_country"
EXOG_COLS = [
    "gdp_per_capita",
    "gdp_growth",
    "Primary energy consumption per capita (kWh/person)",
    "Renewable energy share in the total final energy consumption (%)",
]
REQUIRED_COLS = ["Entity", "Year", TARGET_COL] + EXOG_COLS

# LSTM hyperparameters (best config found during FYP tuning)
LOOKBACK = 3
LSTM_UNITS = 32
DROPOUT_RATE = 0.2
LSTM_EPOCHS = 200
LSTM_BATCH = 4
PATIENCE = 20


class InsufficientDataError(Exception):
    pass


def validate_schema(df: pd.DataFrame) -> list[str]:
    """Return a list of missing required columns (empty list = valid)."""
    return [c for c in REQUIRED_COLS if c not in df.columns]


def top_countries(df: pd.DataFrame, n: int = 10) -> list[str]:
    return (
        df.groupby("Entity")[TARGET_COL]
        .mean()
        .dropna()
        .sort_values(ascending=False)
        .head(n)
        .index.tolist()
    )


def _country_series(df: pd.DataFrame, country: str) -> pd.DataFrame:
    sub = (
        df[df["Entity"] == country][["Year", TARGET_COL] + EXOG_COLS]
        .dropna()
        .sort_values("Year")
        .set_index("Year")
    )
    return sub


def _get_d_order(series: pd.Series) -> int:
    if adfuller(series.dropna())[1] <= 0.05:
        return 0
    if adfuller(series.diff().dropna())[1] <= 0.05:
        return 1
    return 2


def _extrapolate_exog(exog_df: pd.DataFrame, steps: int) -> pd.DataFrame:
    """Linear-trend extrapolation of exogenous variables into the future,
    same approach as the FYP notebook."""
    future = {}
    x = np.arange(len(exog_df))
    for col in exog_df.columns:
        coeffs = np.polyfit(x, exog_df[col].values, 1)
        future_x = np.arange(len(exog_df), len(exog_df) + steps)
        future[col] = np.polyval(coeffs, future_x)
    return pd.DataFrame(future, index=range(exog_df.index[-1] + 1, exog_df.index[-1] + 1 + steps))


def _create_sequences(data: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i])
    return np.array(X).reshape(-1, lookback, 1), np.array(y)


def _build_lstm_model(lookback: int, units: int, dropout_rate: float):
    # Imported lazily: tensorflow is heavy and only needed at train time.
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

    model = Sequential([
        Input(shape=(lookback, 1)),
        LSTM(units, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(max(units // 2, 8)),
        Dropout(dropout_rate),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def _fit_sarimax(train: pd.DataFrame):
    d = _get_d_order(train[TARGET_COL])
    aa = auto_arima(
        train[TARGET_COL], X=train[EXOG_COLS],
        start_p=0, max_p=3, start_q=0, max_q=3, d=d,
        information_criterion="aic", stepwise=True,
        suppress_warnings=True, error_action="ignore",
    )
    order = aa.order
    model = SARIMAX(
        train[TARGET_COL], exog=train[EXOG_COLS],
        order=order, seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    fit = model.fit(disp=False, maxiter=500)
    return fit, order


def _lstm_residual_forecast(resid: np.ndarray, steps: int):
    """Train an LSTM on residuals and autoregressively forecast `steps` ahead.
    Returns (forecast_array, trained_model, fitted_scaler)."""
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping

    if len(resid) <= LOOKBACK:
        raise InsufficientDataError(
            f"Need more than {LOOKBACK} years of data for this country to train the LSTM residual model."
        )

    scaler = MinMaxScaler(feature_range=(-1, 1))
    resid_scaled = scaler.fit_transform(resid.reshape(-1, 1)).flatten()

    X_seq, y_seq = _create_sequences(resid_scaled, LOOKBACK)
    split = max(1, int(len(X_seq) * 0.8))
    X_tr, X_val = X_seq[:split], X_seq[split:]
    y_tr, y_val = y_seq[:split], y_seq[split:]
    val_data = (X_val, y_val) if len(X_val) > 0 else None

    model = _build_lstm_model(LOOKBACK, LSTM_UNITS, DROPOUT_RATE)
    es = EarlyStopping(
        monitor="val_loss" if val_data else "loss",
        patience=PATIENCE, restore_best_weights=True, verbose=0,
    )
    model.fit(
        X_tr, y_tr, validation_data=val_data,
        epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH,
        callbacks=[es], verbose=0,
    )

    seed_seq = resid_scaled[-LOOKBACK:].tolist()
    pred_scaled = []
    for _ in range(steps):
        inp = np.array(seed_seq[-LOOKBACK:]).reshape(1, LOOKBACK, 1)
        pred = model.predict(inp, verbose=0)[0, 0]
        pred_scaled.append(pred)
        seed_seq.append(pred)

    forecast = scaler.inverse_transform(np.array(pred_scaled).reshape(-1, 1)).flatten()
    tf.keras.backend.clear_session()
    return forecast


def train_and_forecast(
    df: pd.DataFrame,
    country: str,
    test_years: int = 5,
    forecast_years: int = 5,
) -> dict:
    """
    Full hybrid pipeline for one country:
      1. Fit SARIMAX + LSTM-residual model on a train split, evaluate on a held-out
         test split (this gives the honest accuracy metrics).
      2. Refit on the FULL series and forecast `forecast_years` beyond the last
         available year (this is what powers the actual forecast the user sees).
    """
    sub = _country_series(df, country)
    n = len(sub)
    if n < 8:
        raise InsufficientDataError(
            f"Only {n} years of usable data for {country}. Need at least 8 to fit a "
            "hybrid SARIMAX+LSTM model."
        )

    test_years = max(1, min(test_years, n // 3))
    train = sub.iloc[: n - test_years]
    test = sub.iloc[n - test_years:]

    # ---- Step 1: evaluate on held-out years ----
    sarimax_fit, order = _fit_sarimax(train)
    sarimax_test_preds = sarimax_fit.predict(
        start=len(train), end=len(train) + len(test) - 1, exog=test[EXOG_COLS]
    )
    sarimax_test_preds = pd.Series(sarimax_test_preds.values, index=test.index)

    resid_forecast = _lstm_residual_forecast(sarimax_fit.resid.dropna().values, len(test))
    hybrid_test_preds = pd.Series(sarimax_test_preds.values + resid_forecast, index=test.index)

    actual = test[TARGET_COL].values
    metrics = {
        "order": list(order),
        "test_years": test.index.tolist(),
        "sarimax": {
            "mape": float(np.mean(np.abs((actual - sarimax_test_preds.values) / actual)) * 100),
            "rmse": float(np.sqrt(mean_squared_error(actual, sarimax_test_preds.values))),
            "r2": float(r2_score(actual, sarimax_test_preds.values)),
            "predictions": sarimax_test_preds.round(1).tolist(),
        },
        "hybrid": {
            "mape": float(np.mean(np.abs((actual - hybrid_test_preds.values) / actual)) * 100),
            "rmse": float(np.sqrt(mean_squared_error(actual, hybrid_test_preds.values))),
            "r2": float(r2_score(actual, hybrid_test_preds.values)),
            "predictions": hybrid_test_preds.round(1).tolist(),
        },
        "actual": actual.round(1).tolist() if hasattr(actual, "round") else list(actual),
    }

    # ---- Step 2: refit on full data, forecast into the future ----
    full_fit, full_order = _fit_sarimax(sub)
    future_exog = _extrapolate_exog(sub[EXOG_COLS], forecast_years)
    future_years = future_exog.index.tolist()

    sarimax_future = full_fit.predict(
        start=len(sub), end=len(sub) + forecast_years - 1, exog=future_exog
    )
    resid_future = _lstm_residual_forecast(full_fit.resid.dropna().values, forecast_years)
    hybrid_future = sarimax_future.values + resid_future

    forecast = {
        "years": future_years,
        "sarimax": [round(float(v), 1) for v in sarimax_future.values],
        "hybrid": [round(float(v), 1) for v in hybrid_future],
    }

    history = {
        "years": sub.index.tolist(),
        "actual": [round(float(v), 1) for v in sub[TARGET_COL].values],
    }

    return {
        "country": country,
        "metrics": metrics,
        "forecast": forecast,
        "history": history,
    }
