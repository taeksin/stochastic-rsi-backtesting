from __future__ import annotations

import pandas as pd

from src.utils.validators import ensure_columns


def _wilder_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = avg_loss.where(avg_loss != 0, other=1e-10)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate(
    df: pd.DataFrame,
    column: str = "close",
    rsi_period: int = 14,
    stoch_period: int = 14,
    k: int = 3,
    d: int = 3,
) -> pd.DataFrame:
    ensure_columns(df, [column])
    result = df.copy()
    rsi = _wilder_rsi(result[column], rsi_period)

    rsi_min = rsi.rolling(window=stoch_period, min_periods=stoch_period).min()
    rsi_max = rsi.rolling(window=stoch_period, min_periods=stoch_period).max()
    stochastic_rsi = (rsi - rsi_min) / (rsi_max - rsi_min)
    stochastic_rsi = stochastic_rsi.clip(lower=0, upper=1)

    k_line = stochastic_rsi.rolling(window=k, min_periods=k).mean()
    d_line = k_line.rolling(window=d, min_periods=d).mean()

    result["rsi"] = rsi
    result["stoch_rsi"] = stochastic_rsi * 100
    result["stoch_k"] = k_line * 100
    result["stoch_d"] = d_line * 100
    return result


def detect_crossover(df: pd.DataFrame) -> pd.DataFrame:
    ensure_columns(df, ["stoch_k", "stoch_d"])
    result = df.copy()
    prev_k = result["stoch_k"].shift(1)
    prev_d = result["stoch_d"].shift(1)
    result["golden_cross"] = (prev_k < prev_d) & (result["stoch_k"] >= result["stoch_d"])
    result["dead_cross"] = (prev_k > prev_d) & (result["stoch_k"] <= result["stoch_d"])
    return result
