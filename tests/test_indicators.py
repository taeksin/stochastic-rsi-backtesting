from __future__ import annotations

import pandas as pd

from src.indicators import moving_average, stochastic_rsi


def sample_price_data(size: int = 250) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=size, freq="1T")
    prices = pd.Series(range(size), dtype=float) + 100
    return pd.DataFrame({"timestamp": dates, "close": prices})


def test_calculate_sma_returns_series():
    df = sample_price_data()
    sma = moving_average.calculate_sma(df, period=10)
    assert len(sma) == len(df)
    assert sma.iloc[9] == df["close"].iloc[:10].mean()


def test_stochastic_rsi_columns_present():
    df = sample_price_data()
    result = stochastic_rsi.calculate(df)
    expected_cols = {"rsi", "stoch_rsi", "stoch_k", "stoch_d"}
    assert expected_cols.issubset(result.columns)


def test_detect_crossover_flags_crosses():
    df = sample_price_data()
    stoch_df = stochastic_rsi.calculate(df)

    stoch_df.loc[20, "stoch_k"] = 30
    stoch_df.loc[20, "stoch_d"] = 40
    stoch_df.loc[21, "stoch_k"] = 50
    stoch_df.loc[21, "stoch_d"] = 45

    cross = stochastic_rsi.detect_crossover(stoch_df)
    assert cross.loc[21, "golden_cross"] is True
