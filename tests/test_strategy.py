from __future__ import annotations

import pandas as pd

from src.strategy.ma200_stochrsi import MA200StochRSIStrategy, StrategyParameters


def sample_price_data(size: int = 500) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=size, freq="5T")
    price = pd.Series(range(size), dtype=float) + 100
    return pd.DataFrame({"timestamp": dates, "open": price, "high": price, "low": price, "close": price})


def test_strategy_generates_signal_column():
    strategy = MA200StochRSIStrategy()
    df = sample_price_data()
    result = strategy.generate_signals(df)
    assert "signal" in result.columns
    assert "position" in result.columns


def test_position_size_uses_leverage():
    strategy = MA200StochRSIStrategy()
    size = strategy.calculate_position_size(1000, leverage=5)
    assert size == 5000
